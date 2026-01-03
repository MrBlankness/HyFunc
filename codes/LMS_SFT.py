import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

IGNORE_INDEX = -100
INSTRUCTION_TEMPLATE = """
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


class SoftTokenProjector(nn.Module):
    def __init__(self, soft_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(soft_dim, llm_dim, bias=False)

    def forward(self, soft):
        return self.proj(soft)


class FunctionCallDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=2048):
        self.data_dir = data_dir
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.param_start = tokenizer.convert_tokens_to_ids("<param>")
        self.param_end = tokenizer.convert_tokens_to_ids("</param>")

    def _build_labels(self, input_ids):
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        inside = False
        for i, t in enumerate(input_ids):
            if t == self.param_start:
                inside = True
                continue
            if inside:
                labels[i] = input_ids[i]
            if t == self.param_end:
                inside = False
        return labels

    def __getitem__(self, idx):
        file_dir = self.file_list[idx]
        with open(file_dir, "r") as f:
            item = json.load(f)

        function_str = ""
        for func_idx, func in enumerate(item["func_list"]):
            if func in item["true_func"]:
                function_str += str(item['func_list_info'][func_idx]) + '\n'
        response_str = ""
        for func in enumerate(item["true_func_info"]):
            response_str += str(func) + '\n'

        prompt = INSTRUCTION_TEMPLATE.format(
            functions=function_str,
            query=item["query"],
            response=response_str
        )
        
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = self._build_labels(input_ids)
        soft = torch.Tensor(item['first_gen_token_embedding'])

        # 备注1：代码撰写时发现只能传input_ids、attention_mask、labels三项，无法传入soft embeddings
        # 因此将input_ids, attention_mask存入在input_ids中，soft存入在labels中
        # 后续的代码将从labels中提取soft embeddings
        return {
            "input_ids": torch.cat([input_ids, labels], dim=0),
            "attention_mask": attention_mask,
            "labels": soft,
        }

    def __len__(self):
        return len(self.file_list)


class LMSTrainer(Trainer):
    def __init__(self, soft_adapter, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_adapter = soft_adapter
        self.tokenizer = tokenizer
        self.query_end_id = tokenizer.convert_tokens_to_ids("</soft_token>")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        # 备注2: 参加上方备注1
        input_ids, labels = torch.chunk(inputs["input_ids"], chunks=2, dim=1)
        attention_mask = inputs["attention_mask"]
        softs = inputs["labels"]

        B = input_ids.size(0)
        device = input_ids.device

        # LLM token embeddings
        token_embeds = model.get_input_embeddings()(input_ids)

        # soft token -> linear -> llm space
        soft_embeds = self.soft_adapter(softs).to(device)

        new_embeds, new_masks, new_labels = [], [], []

        for i in range(B):
            qe = (input_ids[i] == self.query_end_id).nonzero()[0].item() + 1
            new_embeds.append(torch.cat([
                token_embeds[i, :qe],
                soft_embeds[i].unsqueeze(0),
                token_embeds[i, qe:]
            ], dim=0))

            new_masks.append(torch.cat([
                attention_mask[i, :qe],
                torch.ones(soft_embeds.size(0), device=device),
                attention_mask[i, qe:]
            ], dim=0))

            new_labels.append(torch.cat([
                labels[i, :qe],
                torch.full(
                    (soft_embeds.size(0),),
                    IGNORE_INDEX,
                    device=device
                ),
                labels[i, qe:]
            ], dim=0))

        outputs = model(
            inputs_embeds=torch.stack(new_embeds),
            attention_mask=torch.stack(new_masks),
            labels=torch.stack(new_labels)
        )

        return (outputs.loss, outputs) if return_outputs else outputs.loss


model_name = "../models/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

special_tokens = ["<param>", "</param>", "<soft_token>", "</soft_token>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.resize_token_embeddings(len(tokenizer))

train_data_dir = "../data/xlam/embedding"
train_dataset = FunctionCallDataset(train_data_dir, tokenizer)


soft_token_projector = SoftTokenProjector(
    soft_dim=4096,
    llm_dim=model.config.hidden_size  # = 1024
).to(model.device)


training_args = TrainingArguments(
    output_dir="../models/LMS_SFT",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=1,
    bf16=True,
    save_steps=500,
    logging_steps=50,
    save_total_limit=2,
    report_to="none"
)

trainer = LMSTrainer(
    soft_adapter=soft_token_projector,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
torch.save(soft_token_projector.state_dict(), "../models/LMS_SFT/soft_token_projector.pth")