"""Download SC criminal-appeal judgments from the public AWS Open Data bucket (no auth).

  s3://indian-supreme-court-judgments
    metadata/parquet/year=Y/metadata.parquet
    data/pdf/year=Y/english/{path}_EN.pdf

Writes:
  data/raw/meta/{Y}.parquet   cached per-year metadata (all case types)
  data/raw/pdf/{path}.pdf     one per criminal appeal (skipped if present; atomic .part -> rename)
  data/raw/meta.parquet       filtered metadata + case_no, pdf_file, ok

Usage: python -m ingest.fetch [--years 2016 2025] [--workers 16]
"""
import argparse
import html
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BUCKET = "https://indian-supreme-court-judgments.s3.amazonaws.com"
OUT = "data/raw"


def download(url: str, dest: str, retries: int = 3) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    tmp = dest + ".part"
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, dest)
            return True
        except Exception:
            continue
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def case_no(raw_html: str) -> str:
    txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw_html or "")))
    m = re.search(r"Case No\s*:\s*([^|]+)", txt)
    return m.group(1).strip() if m else ""


def load_meta(year: int) -> pd.DataFrame:
    dest = f"{OUT}/meta/{year}.parquet"
    if not download(f"{BUCKET}/metadata/parquet/year={year}/metadata.parquet", dest):
        raise RuntimeError(f"metadata download failed for {year}")
    return pd.read_parquet(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2016, 2025])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(f"{OUT}/meta", exist_ok=True)
    os.makedirs(f"{OUT}/pdf", exist_ok=True)

    df = pd.concat([load_meta(y) for y in range(args.years[0], args.years[1] + 1)], ignore_index=True)
    df["case_no"] = df["raw_html"].map(case_no)
    df = df[df["case_no"].str.upper().str.startswith("CRIMINAL APPEAL")].drop_duplicates("path").reset_index(drop=True)
    df["pdf_file"] = df["path"] + ".pdf"
    print(f"criminal appeals: {len(df)}  per year: {df['year'].value_counts().sort_index().to_dict()}")

    jobs = {(r.year, r.path): f"{OUT}/pdf/{r.pdf_file}" for r in df.itertuples()}
    todo = {k: v for k, v in jobs.items() if not (os.path.exists(v) and os.path.getsize(v) > 0)}
    print(f"cached: {len(jobs) - len(todo)}  to download: {len(todo)}")
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download, f"{BUCKET}/data/pdf/year={y}/english/{p}_EN.pdf", dest): p
                for (y, p), dest in todo.items()}
        for i, f in enumerate(as_completed(futs), 1):
            if not f.result():
                print(f"  FAILED {futs[f]}")
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}")

    df["ok"] = df["pdf_file"].map(lambda p: os.path.exists(f"{OUT}/pdf/{p}"))
    df.to_parquet(f"{OUT}/meta.parquet", index=False)
    print(f"ok: {df['ok'].sum()}/{len(df)}  -> {OUT}/meta.parquet")


if __name__ == "__main__":
    main()
