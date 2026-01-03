from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from tqdm import tqdm
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--model', default='../models/ToolACE-2-Llama-3.1-8b', type=str)
    parser.add_argument('--datatype', default='xlam', type=str)
    parser.add_argument('--input_file', default='../data/xlam/xlam_function_calling_60k.json', type=str)
    parser.add_argument('--output_dir', default='../data/xlam/embedding', type=str)
    # parser.add_argument('--datatype', default='BFCL', type=str)
    # parser.add_argument('--input_file', default='../data/BFCL/question/BFCL_v3_parallel_multiple.json', type=str)
    # parser.add_argument('--output_dir', default='../data/BFCL/embedding', type=str)
    return parser.parse_args()


args = parse_args()
model_name = args.model
data_type = args.datatype
input_file = args.input_file
output_dir = args.output_dir


system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    device_map='auto'
)


def read_json_file(file_path):
    assert data_type in ['BFCL', 'xlam']
    if data_type == 'BFCL': # read BFCL format
        # 存储所有字典的列表
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data = json.loads(line)
                    data_list.append(data)
        with open(file_path.replace('question', 'answer'), 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:  # 跳过空行
                    data = json.loads(line)
                    assert data['id'] == data_list[idx]['id']
                    data_list[idx]['ground_truth'] = data['ground_truth']
    elif data_type == 'xlam': # read xlam format
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    return data_list


def get_embedding(query_id, query, tools, func_answer, func_answer_info):
    start_time = time.time()

    messages = [
        {'role': 'system', 'content': system_prompt.format(functions=tools)},
        {'role': 'user', 'content': query}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        full_outputs = model.model(inputs, output_hidden_states=True)
        input_hidden_states = full_outputs.hidden_states[-1]
    
    # 提取 tool 的 embedding
    tool_embeddings = []
    start_idx = 0
    for tool in tools:
        tool_ids = tokenizer.encode(str(tool), add_special_tokens=False, return_tensors="pt").to(model.device)[:, 1:-2]
        tool_len = tool_ids.shape[1]
        for i in range(start_idx, inputs.shape[1] - tool_len + 1):
            if torch.equal(inputs[0, i:i+tool_len], tool_ids[0, :]):
                emb = input_hidden_states[0, i:i+tool_len, :].mean(dim=0)
                tool_embeddings.append(emb)
                start_idx = i + tool_len
                break
    assert len(tool_embeddings) == len(tools)

    with torch.no_grad():            
        # 第一个生成 token 的 embedding（最后一层）
        logits = model(inputs).logits
        next_token_id = torch.argmax(logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
        gen_inputs = torch.cat([inputs, next_token_id], dim=1)
        gen_outputs = model.model(gen_inputs, output_hidden_states=True)
        first_gen_token_emb = gen_outputs.hidden_states[-1][0, -1, :]  # [hidden_dim], hidden_dim=4096
    
    end_time = time.time()

    return {'id': query_id,
            'query': query,
            'true_func': func_answer,
            'true_func_info': func_answer_info,
            'func_list': [tools[tool_index]['name'] for tool_index in range(len(tools))],
            'func_list_info': tools,
            'func_embeddings': [tool_embeddings[tool_index].detach().to(torch.float).cpu().numpy().tolist() for tool_index in range(len(tools))],
            'first_gen_token_embedding': first_gen_token_emb.detach().to(torch.float).cpu().numpy().tolist(),
            'time_cost': end_time - start_time,
            'token_cost': inputs.shape[1] + 1}


question_jsons = read_json_file(input_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for idx, question_json in tqdm(enumerate(question_jsons), total=len(question_jsons)):

    query_id = question_json['id'] if isinstance(question_json['id'], str) else str(question_json['id'])
    query = question_json['question'][0][0]['content'] if data_type == 'BFCL' else question_json['query']
    tools = question_json['function'] if data_type == 'BFCL' else json.loads(question_json['tools'])
    if data_type == 'BFCL':
        func_answer = [list(func_json.keys())[0] for func_json in question_json['ground_truth']]
        func_answer_info = question_json['ground_truth']
    elif data_type == 'xlam':
        func_answer = [tool['name'] for tool in json.loads(question_json['answers'])]
        func_answer_info = json.loads(question_json['answers'])
    
    data = get_embedding(query_id, query, tools, func_answer, func_answer_info)

    output_file = os.path.join(output_dir, os.path.basename(input_file).split('.')[0] + '_' + query_id + '.json')
    with open(output_file, 'w') as f:
        json.dump(data, f)
