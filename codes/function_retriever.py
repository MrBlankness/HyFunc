import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FunctionEmbeddingDataset(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = os.listdir(data_path)
        self.data_files = [os.path.join(data_path, f) for f in self.data_files if f.endswith('.json')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(self.data_files[idx], "r") as f:
            raw_data = json.load(f)

        query_emb = raw_data["first_gen_token_embedding"]
        func_list = raw_data["func_list"]
        func_embs = raw_data["func_embeddings"]
        true_funcs = set(raw_data["true_func"])

        pairs = []
        for fn, fn_emb in zip(func_list, func_embs):
            if fn in true_funcs:
                pairs.append(
                    (
                        torch.tensor(query_emb, dtype=torch.float),
                        torch.tensor(fn_emb, dtype=torch.float),
                    )
                )
        
        return pairs[random.randint(0, len(pairs)-1)]   # (query_emb, key_emb)


class DualEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.query_encoder = nn.Linear(embed_dim, proj_dim)
        self.key_encoder = nn.Linear(embed_dim, proj_dim)

    def encode_query(self, q):
        return F.normalize(self.query_encoder(q), dim=-1)

    def encode_key(self, k):
        return F.normalize(self.key_encoder(k), dim=-1)

    def forward(self, q, k):
        return self.encode_query(q), self.encode_key(k)


def info_nce_loss(q, k, temperature=0.07):
    """
    q: (B, D)
    k: (B, D)
    """
    logits = torch.matmul(q, k.T) / temperature  # (B, B)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def train(
    model,
    dataloader,
    optimizer,
    device,
    epochs=10
):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for query_emb, key_emb in dataloader:
            query_emb = query_emb.to(device)  # (256, D)
            key_emb = key_emb.to(device)      # (256, D)

            q, k = model(query_emb, key_emb)
            loss = info_nce_loss(q, k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {total_loss / len(dataloader):.4f}"
        )


@torch.no_grad()
def retrieve(
    model,
    query_embedding,
    key_embeddings,
    func_list,
    threshold=0.5,
    device="cpu"
):
    model.eval()

    q = model.encode_query(
        torch.tensor(query_embedding, dtype=torch.float)
        .unsqueeze(0)
        .to(device)
    )  # (1, D)

    k = model.encode_key(
        torch.tensor(key_embeddings, dtype=torch.float)
        .to(device)
    )  # (N, D)

    sim = torch.matmul(q, k.T).squeeze(0)  # cosine similarity

    hit = sim > threshold

    if hit.any():
        idxs = torch.where(hit)[0].tolist()
        return {
            "type": "threshold_hit",
            "functions": [func_list[i] for i in idxs],
            "scores": sim[idxs].tolist()
        }

    max_idx = torch.argmax(sim).item()
    return {
        "type": "fallback_max",
        "function": [func_list[max_idx]],
        "score": sim[max_idx].item()
    }


def main():
    
    train_data_dir = "../data/xlam/embedding"
    embed_dim = 4096           # 原始 embedding 维度
    proj_dim = 1024            # 投影维度
    batch_size = 256
    epochs = 100
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = FunctionEmbeddingDataset(train_data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,   # 保证 batch 内严格对齐
    )

    model = DualEncoder(embed_dim, proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )

    eval_data_dir = "../data/BFCL/embedding"
    output_dir = "../data/BFCL/retrieval_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for eval_file in os.listdir(eval_data_dir):
        if not eval_file.endswith('.json'):
            continue
        json_path = os.path.join(eval_data_dir, eval_file)
        with open(json_path, "r") as f:
            sample = json.load(f)

        result = retrieve(
            model=model,
            query_embedding=sample["first_gen_token_embedding"],
            key_embeddings=sample["func_embeddings"],
            func_list=sample["func_list"],
            threshold=0.5,
            device=device,
        )

        output_file = os.path.join(output_dir, eval_file)
        with open(output_file, 'w') as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
