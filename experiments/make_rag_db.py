import argparse
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from experiments.eval_bracket_structure import build_model_from_ckpt
import numpy as np
import faiss

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/dn_baseline_30L_217M_for_oracle.pt")
    p.add_argument("--n_chunks", type=int, default=10000, help="Number of code chunks to index.")
    p.add_argument("--out_dir", type=str, default="data/rag_db")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    
    print(f"Loading encoder model for RAG: {args.ckpt}")
    model, _ = build_model_from_ckpt(args.ckpt)
    model = model.to(device)
    model.eval()
    
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
    
    # 1. Collect and Embed Chunks
    embeddings = []
    metadata = []
    
    print("Embedding code chunks...")
    count = 0
    for x in ds:
        content = x["content"]
        # Split into rough chunks (e.g., by lines or fixed length)
        # For simplicity, just take the first 256 tokens of each file
        ids = tok(content, truncation=True, max_length=256, return_tensors="pt")["input_ids"].to(device)
        
        if ids.shape[1] < 50: continue
        
        with torch.no_grad():
            # Get hidden state from final layer before LM head
            _, hidden = model(ids, return_hidden=True)
            # Pool hidden state (e.g., mean) to get chunk representation
            emb = hidden.mean(dim=1).cpu().numpy()
            
        embeddings.append(emb)
        metadata.append(content[:500]) # Store snippet for inspection
        
        count += 1
        if count % 100 == 0:
            print(f"  Embedded {count}/{args.n_chunks}")
        if count >= args.n_chunks:
            break
            
    embeddings = np.concatenate(embeddings, axis=0)
    
    # 2. Build FAISS Index
    print(f"Building FAISS index (dim={embeddings.shape[1]})...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # Inner product for cosine similarity (if normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # 3. Save
    faiss.write_index(index, os.path.join(args.out_dir, "index.faiss"))
    np.save(os.path.join(args.out_dir, "metadata.npy"), np.array(metadata))
    print(f"RAG Database saved to {args.out_dir}")

if __name__ == "__main__":
    main()
