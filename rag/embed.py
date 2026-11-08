"""Paragraph chunks + dense embeddings for retrieval (cached to disk; rerun only when paragraphs change).

Long paragraphs are split into ~200-word windows (40-word overlap) so embeddings and BM25 see comparable units;
every chunk keeps its paragraph id, and answers always cite the paragraph.

Writes data/rag/chunks.parquet (chunk_id, pid, case_id, opinion, para_no, text) and data/rag/emb.npy (float16,
L2-normalised, row i = chunk i).

Usage: python -m rag.embed [--model BAAI/bge-base-en-v1.5] [--batch 64]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

OUT = "data/rag"
MODEL = "BAAI/bge-base-en-v1.5"
WIN, OVERLAP, MIN_WORDS = 200, 40, 5


def chunk(text: str) -> list[str]:
    words = text.split()
    if len(words) <= WIN + OVERLAP:
        return [" ".join(words)]
    out, step = [], WIN - OVERLAP
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i + WIN]))
        if i + WIN >= len(words):
            break
    return out


def build_chunks() -> pd.DataFrame:
    p = pd.read_parquet("data/graph/nodes_paragraph.parquet")
    rows = []
    for r in p.itertuples():
        if not isinstance(r.text, str) or len(r.text.split()) < MIN_WORDS:
            continue
        for j, t in enumerate(chunk(r.text)):
            rows.append({"pid": r.pid, "case_id": r.case_id, "opinion": r.opinion, "para_no": r.para_no,
                         "part": j, "text": t})
    df = pd.DataFrame(rows)
    df.insert(0, "chunk_id", range(len(df)))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=16, help="M3 / 8 GB: 16 fastest (fp16)")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    import torch

    os.makedirs(OUT, exist_ok=True)
    chunks = build_chunks()
    chunks.to_parquet(f"{OUT}/chunks.parquet", index=False)
    print(f"chunks {len(chunks):,} from {chunks['pid'].nunique():,} paragraphs")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    if device == "mps":
        model = model.half()  # 1.6x faster on M3; embeddings are stored as float16 anyway
    # longest first: similar lengths per batch, and an out-of-memory error shows up immediately
    order = chunks["text"].str.len().sort_values(ascending=False).index.to_numpy()
    t0 = time.time()
    emb = model.encode(chunks["text"].to_numpy()[order].tolist(), batch_size=args.batch, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    out = np.empty_like(emb)
    out[order] = emb
    np.save(f"{OUT}/emb.npy", out.astype(np.float16))
    with open(f"{OUT}/embed_meta.txt", "w") as f:
        f.write(f"model={args.model}\nchunks={len(chunks)}\ndim={out.shape[1]}\nseconds={time.time() - t0:.0f}\n")
    print(f"embedded {out.shape} in {time.time() - t0:.0f}s on {device}")


if __name__ == "__main__":
    main()
