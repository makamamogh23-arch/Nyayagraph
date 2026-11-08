"""IPC -> BNS section correspondence from two independent sources, merged with agreement flags.

  pdf  BNS->IPC correspondence table, CAPT Bhopal (A.K. Yadav, IPS), hosted by keralaprisons.gov.in
       data/ref/src_pdf/comparison_summary_BNS_to_IPC.pdf
  hf   nandhakumarg/IPC_and_BNS_transformation (Apache-2.0), IPC->BNS pairs inside prompt/response rows
       data/ref/src_hf/ipc_bns_transformation.csv

Section granularity matches graph Provision nodes: base section ("115"), sub-section kept separately ("(2)").
The PDF wins where both sources cover an IPC section (spot checks show the hf set off by one on several
sections, e.g. IPC 341 -> 127 instead of 126); hf only fills IPC sections the PDF lacks.
Writes data/ref/ipc_bns.csv: ipc, bns, bns_sub, in_pdf, in_hf, confidence (agree | pdf | hf_only)

Usage: python -m link.ipc_bns
"""
import ast
import re

import pandas as pd
import pymupdf

PDF = "data/ref/src_pdf/comparison_summary_BNS_to_IPC.pdf"
HF = "data/ref/src_hf/ipc_bns_transformation.csv"
OUT = "data/ref/ipc_bns.csv"
SEC = re.compile(r"\b(\d{1,3}[A-Z]{0,2})\b")
BNS_CELL = re.compile(r"^(\d{1,3})\s*((?:\(\s*\w{1,4}\s*\)\s*)*)")


def split_bns(cell: str) -> tuple[str, str] | None:
    m = BNS_CELL.match(cell.replace(" ", "")) if cell else None
    return (m[1], m[2]) if m else None


def ipc_sections(cell: str) -> list[str]:
    """'29 and 29A' -> [29, 29A]; '23 Clause-1' -> [23]; 'New' -> []. Drops clause numbers after 'Clause'."""
    cell = re.sub(r"(?i)(?:clause|explanation|illustration|proviso|para)\s*[-–]?\s*\w+", " ", cell or "")
    cell = re.sub(r"\(\s*\w{1,4}\s*\)", " ", cell)  # "376(1)" -> "376", not 376 + 1
    if re.search(r"(?i)\bnew\b", cell) and not SEC.search(cell):
        return []
    return [s for s in SEC.findall(cell) if 0 < int(re.match(r"\d+", s)[0]) <= 511]


def from_pdf() -> pd.DataFrame:
    """Table rows -> (bns, bns_sub, ipc). Rows with an empty BNS cell continue the previous row."""
    rows, cur = [], None
    with pymupdf.open(PDF) as doc:
        for page in doc:
            for t in page.find_tables().tables:
                for r in t.extract():
                    cells = [(c or "").replace("\n", " ").strip() for c in r]
                    cells = [c for c in cells] + [""] * 4
                    bns_cell, ipc_cell = cells[0], cells[2]
                    b = split_bns(bns_cell)
                    if b:
                        cur = {"bns": b[0], "bns_sub": b[1], "ipc_raw": ipc_cell}
                        rows.append(cur)
                    elif cur is not None and not bns_cell and ipc_cell and not ipc_cell.lower().startswith("ipc"):
                        cur["ipc_raw"] += " " + ipc_cell
    out = [{"ipc": i, "bns": r["bns"], "bns_sub": r["bns_sub"]} for r in rows for i in ipc_sections(r["ipc_raw"])]
    return pd.DataFrame(out)


def from_hf() -> pd.DataFrame:
    out = []
    for resp in pd.read_csv(HF)["response"]:
        d = ast.literal_eval(resp)
        for i in ipc_sections(str(d["IPC Section"])):
            for b in re.split(r"\s*(?:,|and|&|/)\s*", str(d["BNS Section"])):
                sb = split_bns(b)
                if sb:
                    out.append({"ipc": i, "bns": sb[0], "bns_sub": sb[1]})
    return pd.DataFrame(out)


def main():
    pdf, hf = from_pdf().drop_duplicates(), from_hf().drop_duplicates()
    pairs = pd.concat([pdf.assign(src="pdf"), hf.assign(src="hf")])
    base = pairs.groupby(["ipc", "bns"])["src"].agg(set).reset_index()
    subs = pairs.groupby(["ipc", "bns"])["bns_sub"].agg(lambda s: ";".join(sorted({x for x in s if x}))).reset_index()
    df = base.merge(subs, on=["ipc", "bns"])
    df["in_pdf"], df["in_hf"] = df["src"].map(lambda s: "pdf" in s), df["src"].map(lambda s: "hf" in s)
    agree = df["in_pdf"] & df["in_hf"]
    pdf_covers = df["ipc"].isin(set(pdf["ipc"]))
    df["confidence"] = agree.map({True: "agree", False: ""})
    df.loc[~agree & df["in_pdf"], "confidence"] = "pdf"
    df.loc[~df["in_pdf"] & ~pdf_covers, "confidence"] = "hf_only"
    df = df[df["confidence"] != ""]  # hf pairs contradicting the PDF are dropped
    df = df.drop(columns="src").sort_values(["ipc", "bns"], key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
    df.to_csv(OUT, index=False)
    print("confidence:", df["confidence"].value_counts().to_dict(), f" IPC sections mapped: {df['ipc'].nunique()}")

    ipc_pdf, ipc_hf = set(pdf["ipc"]), set(hf["ipc"])
    both = ipc_pdf & ipc_hf
    same = sum(set(pdf.loc[pdf.ipc == i, "bns"]) == set(hf.loc[hf.ipc == i, "bns"]) for i in both)
    print(f"pairs: pdf {len(pdf)}  hf {len(hf)}  kept {len(df)}")
    print(f"IPC sections: pdf {len(ipc_pdf)}  hf {len(ipc_hf)}  in both {len(both)}  "
          f"identical BNS targets {same}/{len(both)} ({same / max(len(both), 1):.3f})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
