import json
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

import argparse
import os


###############################
# Encoder
###############################

class Encoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, padding_side="left", trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cuda:0"
        )
        self.encoder.eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode_batch(self, texts):
        """Return (batch, dim) numpy array"""
        marker = "\nuser\n"
        processed_texts = []
        for text in texts:
            pos = text.find(marker)
            processed_text = text[pos + len(marker):]
            processed_text = processed_text[-1300:]
            processed_texts.append(processed_text)
        inputs = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.encoder.device)
        attention_mask = inputs["attention_mask"].to(self.encoder.device)

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state[:, -1, :]  # CLS or last token embedding
        emb = torch.nn.functional.normalize(emb, dim=-1)
        vec = emb.detach().cpu().numpy()

        del inputs, input_ids, attention_mask, out, emb
        torch.cuda.empty_cache()
        return vec


###############################
# Main function
###############################

def build_faiss_index(corpus_path, index_path, batch_size=5000):
    print(f"Loading rollout corpus: {corpus_path}")

    corpus = []
    with open(corpus_path, "r") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            item = json.loads(line.strip())
            corpus.append(item["key"])

    print(f"Rollout corpus loaded. Total documents = {len(corpus)}")
    print("Example:", corpus[0])

    # Load encoder
    print("Loading model...")
    encoder = Encoder()

    # Encode corpus
    all_embs = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding"):
        batch_texts = corpus[i: i + batch_size]
        emb = encoder.encode_batch(batch_texts)
        all_embs.append(emb)

    all_embs = np.vstack(all_embs).astype("float32")
    print("All embeddings shape:", all_embs.shape)

    dim = all_embs.shape[1]

    # HNSW parameters
    M = 32  # graph degree, usually 16–64

    index = faiss.IndexHNSWFlat(
        dim,
        M,
        faiss.METRIC_INNER_PRODUCT
    )

    # Higher → better recall, slower build
    index.hnsw.efConstruction = 200

    # Higher → better recall, slower query
    index.hnsw.efSearch = 128

    print("Adding vectors to index...")
    index.add(all_embs)

    print(f"Total vectors in index: {index.ntotal}")

    faiss.write_index(index, index_path)
    print(f"Saved faiss index to: {index_path}")


###############################
# CLI
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default='./rollout/Step_Trace_Buffer.jsonl', help="Input JSONL path, e.g., /data/wiki18.jsonl")
    parser.add_argument("--index_path", type=str, default='./rollout/Step_Trace_Buffer.index', help="Output index file path, e.g., /data/wiki18.index")
    parser.add_argument("--batch_size", type=int, default=3000)
    args = parser.parse_args()

    build_faiss_index(
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        batch_size=args.batch_size,
    )
