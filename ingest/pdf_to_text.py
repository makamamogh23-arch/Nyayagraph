"""PDF -> cached line geometry (one pass; downstream never reopens PDFs).

data/interim/lines/{path}.json.gz:
  [{"w", "h", "lines": [{"x0", "y0", "y1", "spans": [[text, size], ...]}]}]   one entry per page

Usage: python -m ingest.pdf_to_text [--workers 8]
"""
import argparse
import gzip
import json
import os
from multiprocessing import Pool

import pandas as pd
import pymupdf

RAW = "data/raw"
OUT = "data/interim/lines"


def extract(pdf_path: str) -> list[dict]:
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            lines = []
            for b in page.get_text("dict")["blocks"]:
                for l in b.get("lines", []):
                    spans = [[s["text"], round(s["size"], 1)] for s in l["spans"] if s["text"].strip()]
                    if spans:
                        x0, y0, _, y1 = l["bbox"]
                        lines.append({"x0": round(x0, 1), "y0": round(y0, 1), "y1": round(y1, 1), "spans": spans})
            pages.append({"w": round(page.rect.width, 1), "h": round(page.rect.height, 1), "lines": lines})
    return pages


def cache_path(path: str) -> str:
    return f"{OUT}/{path}.json.gz"


def load(path: str) -> list[dict]:
    with gzip.open(cache_path(path), "rt") as f:
        return json.load(f)


def work(path: str) -> tuple[str, str | None]:
    dest = cache_path(path)
    if os.path.exists(dest):
        return path, None
    try:
        pages = extract(f"{RAW}/pdf/{path}.pdf")
    except Exception as e:
        return path, repr(e)
    tmp = dest + ".part"
    with gzip.open(tmp, "wt") as f:
        json.dump(pages, f, ensure_ascii=False)
    os.replace(tmp, dest)
    return path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_parquet(f"{RAW}/meta.parquet")
    paths = meta.loc[meta["ok"], "path"].tolist()
    with Pool(args.workers) as pool:
        errors = [(p, e) for p, e in pool.imap_unordered(work, paths, chunksize=8) if e]
    for p, e in errors:
        print(f"FAILED {p}: {e}")
    print(f"cached {len(paths) - len(errors)}/{len(paths)} -> {OUT}")


if __name__ == "__main__":
    main()
