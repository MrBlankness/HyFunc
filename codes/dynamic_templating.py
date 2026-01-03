import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import ast
import os
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict
from collections import Counter
import time


class SoftTokenProjector(nn.Module):
    def __init__(self, soft_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(soft_dim, llm_dim, bias=False)

    def forward(self, soft):
        return self.proj(soft)


PARAM_PLACEHOLDER = "<param></param>"


def replace_values_with_param(obj: Any) -> Any:
    """
    递归地将所有 value 替换为 <param></param>
    """
    if isinstance(obj, dict):
        return {k: replace_values_with_param(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_values_with_param(v) for v in obj]
    else:
        return PARAM_PLACEHOLDER


def convert_function_json(input_json: Dict) -> str:
    """
    将
    {"func_name": {...}}
    转换为
    {"name": "func_name", "arguments": {...}}
    并替换 arguments 中所有 value
    """
    if len(input_json) != 1:
        raise ValueError("输入 JSON 必须且只能包含一个函数")

    # 取函数名和参数
    func_name, arguments = next(iter(input_json.items()))

    # 替换 arguments 中的所有 value
    arguments_with_param = replace_values_with_param(arguments)

    result = {
        "name": func_name,
        "arguments": arguments_with_param
    }

    # 返回字符串形式
    return json.dumps(result, ensure_ascii=False)



class FunctionCallPredictor:
    def __init__(self, model_name, projector_path):
        """
        初始化模型和tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
        self.model.eval()
        self.projector = SoftTokenProjector(
            soft_dim=4096,
            llm_dim=self.model.config.hidden_size  # = 1024
        ).to(self.model.device)
        self.projector.load_state_dict(torch.load(projector_path))
        self.projector.eval()
        
        # 设置特殊token
        self.soft_token_start = "<soft_token>"
        self.soft_token_end = "</soft_token>"
        self.param_start = "<param>"
        self.param_end = "</param>"

        self.key_list = ['properties', 'enum', 'parameters', 'required', 'name', 'type', 'optional', 'format', 'description', 'default', 'items', 'maximum']
        self.type_list = ['any', 'float', 'string', 'tuple', 'array', 'dict', 'integer', 'boolean']
    
    def gen_complete_template(self, embedding_json, retriever_json):
        """
        使用控制生成的方式分析任务
        """
        question = embedding_json["query"]
        soft_embedding = embedding_json["first_gen_token_embedding"]
        selected_functions_name = retriever_json['function']
        functions_desc = [func for func in embedding_json["func_list_info"] if func['name'] in selected_functions_name]
        pred_function_template = embedding_json['true_func_info']
        
        # 构建分析prompt
        TEMPLATE = """
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{functions}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags, and your answer for the function parameters must be enclosed within <param></param> tags:
<tool_call>
{{"name": <function-name>, "arguments": <param><args-json-object></param>}}
</tool_call><|im_end|>
<|im_start|>user
{query}<|im_end|>
<soft_token>
</soft_token>
<|im_start|>assistant
<think>

</think>
<tool_call>
{response}
</tool_call>
"""     

        function_template_all = ""
        for i, func in enumerate(pred_function_template):
            if list(func.keys())[0] in selected_functions_name:
                function_template = convert_function_json(func)
                function_template_all += function_template + "\n"
        TEMPLATE = TEMPLATE.format(functions=functions_desc, query=question, response=function_template_all)
        return TEMPLATE, soft_embedding
    

    def run_template_with_param_prediction(
        self,
        template: str,
        soft_token_emb: torch.Tensor,
        max_param_tokens: int = 32
    ):
        device = self.model.device

        past_key_values = None
        generated_text = ""

        i = 0
        while i < len(template):

            # ---------- 1. soft_token ----------
            if template.startswith(self.soft_token_start, i):
                i = template.index(self.soft_token_end, i) + len(self.soft_token_end)

                soft_emb = self.projector(soft_token_emb.to(device).unsqueeze(0)).unsqueeze(0)
                print(soft_emb.shape)

                outputs = self.model(
                    inputs_embeds=soft_emb,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                past_key_values = outputs.past_key_values
                continue

            # ---------- 2. param prediction ----------
            if template.startswith(self.param_start, i):
                i += len(self.param_start)

                param_tokens = []

                for _ in range(max_param_tokens):
                    # outputs = self.model(
                    #     input_ids=input_ids,
                    #     past_key_values=past_key_values,
                    #     use_cache=True
                    # )

                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1)

                    token_str = self.tokenizer.decode(next_token_id)

                    # 终止条件
                    if self.param_end in token_str:
                        break

                    param_tokens.append(token_str)

                    past_key_values = outputs.past_key_values

                    # 将预测 token 再送回模型
                    outputs = self.model(
                        input_ids=next_token_id.unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values

                predicted_value = "".join(param_tokens)
                generated_text += predicted_value

                # 跳过 </param>
                if template.startswith(self.param_end, i):
                    i += len(self.param_end)

                continue

            # ---------- 3. normal text ----------
            next_special = min(
                [
                    template.find(tag, i)
                    for tag in [self.param_start, self.soft_token_start]
                    if template.find(tag, i) != -1
                ] + [len(template)]
            )

            chunk = template[i:next_special]
            i = next_special

            input_ids = self.tokenizer(chunk, return_tensors="pt").input_ids.to(device)

            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            generated_text += chunk

        return generated_text
    

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    function_call_predictor = FunctionCallPredictor(
        model_name="../models/Qwen3-0.6B",  # 替换为实际模型路径
        projector_path="../models/LMS_SFT/soft_token_projector.pth"  # 替换为实际投影器路径
    )
    function_embedding_data = read_json_file("../data/BFCL/embedding/BFCL_v3_parallel_multiple_parallel_multiple_0.json")
    retriever_results = read_json_file("../data/BFCL/retrieval_results/BFCL_v3_parallel_multiple_parallel_multiple_0.json")
    template, soft_embedding = function_call_predictor.gen_complete_template(function_embedding_data, retriever_results)
    result = function_call_predictor.run_template_with_param_prediction(template, torch.Tensor(soft_embedding))
    print(result)


if __name__ == "__main__":
    main()
