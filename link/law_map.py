"""CrPC -> BNSS and Indian Evidence Act -> BSA section correspondence (IPC -> BNS is link/ipc_bns.py).

Source: the CAPT Bhopal comparison tables (A.K. Yadav, IPS) hosted by keralaprisons.gov.in, the same series as the
IPC -> BNS table, so all three mappings share one method:
  data/ref/src_pdf/comparison_summary_BNSS_to_CrPC.pdf   columns: BNSS | subject | CrPC | summary
  data/ref/src_pdf/comparison_summary_BSA_to_IEA.pdf     columns: BSA | IEA | subject | summary

Writes data/ref/crpc_bnss.csv and data/ref/iea_bsa.csv: old, new, new_sub, confidence (= "pdf"; one source).
Section granularity matches Provision nodes: base section ("528"), sub-section of the new Act kept separately.

Usage: python -m link.law_map
"""
import re

import pandas as pd
import pymupdf

SRC = "data/ref/src_pdf"
MAPS = {
    "crpc_bnss": {"pdf": f"{SRC}/comparison_summary_BNSS_to_CrPC.pdf", "new_col": 0, "old_col": 2, "old_max": 484,
                  "old_prefix": r"Cr\.?\s*P\.?\s*C\.?"},
    "iea_bsa": {"pdf": f"{SRC}/comparison_summary_BSA_to_IEA.pdf", "new_col": 0, "old_col": 1, "old_max": 167,
                "old_prefix": r"IEA|I\.E\.A\.?|Evidence\s+Act"},
}
NEW_CELL = re.compile(r"^(\d{1,3}[A-Z]?)\s*((?:\(\s*\w{1,4}\s*\)\s*)*)")
SEC = re.compile(r"(?<![\w(])(\d{1,3}[A-Z]{0,2})(?![\w)])")

# well-known correspondences (Ministry of Home Affairs / bare-act tables); the parse must reproduce them
CHECKS = {
    "crpc_bnss": {"482": "528", "438": "482", "439": "483", "437": "480", "164": "183", "161": "180", "313": "351",
                  "154": "173", "156": "175", "167": "187", "125": "144", "41A": "35", "190": "210", "200": "223",
                  "374": "415", "389": "430", "320": "359", "173": "193"},
    "iea_bsa": {"25": "23", "27": "23", "32": "26", "65B": "63", "113B": "118", "106": "109", "133": "138",
                "45": "39", "114": "119", "8": "6", "60": "55", "145": "148", "137": "142", "3": "2"},
}


def split_new(cell: str) -> tuple[str, str] | None:
    """'23(2)' -> ('23', '(2)'); 'Proviso to section 23' -> ('23', 'proviso')."""
    if not cell:
        return None
    p = re.match(r"(?i)proviso\s+to\s+section\s+(\d{1,3}[A-Z]?)\s*((?:\(\s*\w{1,4}\s*\)\s*)*)", cell)
    if p:
        return p[1], (p[2].replace(" ", "") + " proviso").strip()
    m = NEW_CELL.match(cell.replace(" ", ""))
    return (m[1], m[2]) if m else None


def old_sections(cell: str, old_max: int, prefix: str) -> list[str]:
    """'41& 41A' -> [41, 41A]; '2(a)]' -> [2]; 'IEA3 Interpretation clause' -> [3]; '3 para1 ,' -> [3]; '-' -> []."""
    cell = re.sub(prefix, " ", cell or "", flags=re.I)
    cell = re.sub(r"(?i)(?:clause|explanation|illustration|proviso|para(?:graph)?|schedule|form)\s*[-–]?\s*\w+", " ", cell)
    cell = re.sub(r"\(\s*\w{1,4}\s*\)", " ", cell)
    if re.search(r"(?i)\bnew\b|newly", cell) and not SEC.search(cell):
        return []
    out = []
    for s in SEC.findall(cell):
        n = int(re.match(r"\d+", s)[0])
        if 0 < n <= old_max and s not in out:
            out.append(s)
    return out


def parse(spec: dict) -> pd.DataFrame:
    rows, cur = [], None
    with pymupdf.open(spec["pdf"]) as doc:
        for page in doc:
            for t in page.find_tables().tables:
                for r in t.extract():
                    cells = [(c or "").replace("\n", " ").strip() for c in r] + [""] * 4
                    new_cell, old_cell = cells[spec["new_col"]], cells[spec["old_col"]]
                    b = split_new(new_cell)
                    if b:
                        cur = {"new": b[0], "new_sub": b[1], "old_raw": old_cell}
                        rows.append(cur)
                    elif cur is not None and not new_cell and old_cell and not re.match(r"(?i)crpc|iea|ipc", old_cell):
                        cur["old_raw"] += " " + old_cell
    out = [{"old": o, "new": r["new"], "new_sub": r["new_sub"]}
           for r in rows for o in old_sections(r["old_raw"], spec["old_max"], spec["old_prefix"])]
    df = pd.DataFrame(out)
    df = df.groupby(["old", "new"])["new_sub"].agg(lambda s: ";".join(sorted({x for x in s if x}))).reset_index()
    return df.assign(confidence="pdf")


def main():
    for name, spec in MAPS.items():
        df = parse(spec)
        key = lambda s: s.str.extract(r"(\d+)")[0].astype(int)
        df = df.sort_values(["old", "new"], key=key)
        df.to_csv(f"data/ref/{name}.csv", index=False)
        fails = []
        for old, new in CHECKS[name].items():
            got = sorted(df.loc[df["old"] == old, "new"])
            if new not in got:
                fails.append(f"{old}->{new} (got {got})")
        multi = df.groupby("old")["new"].nunique()
        print(f"{name}: {len(df)} pairs, {df['old'].nunique()} old sections -> {df['new'].nunique()} new; "
              f"old sections with >1 target: {int((multi > 1).sum())}; "
              f"checks {len(CHECKS[name]) - len(fails)}/{len(CHECKS[name])}" + (f"  FAIL: {fails}" if fails else ""))


if __name__ == "__main__":
    main()
