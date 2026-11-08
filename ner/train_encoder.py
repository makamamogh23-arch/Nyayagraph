"""InLegalBERT token-classification baseline on InLegalNER. Single file: paste into a Kaggle notebook/script.

Kaggle setup: GPU T4 x2, Internet on, secret HF_TOKEN (read access to the gated dataset; write access if
--hub_repo is used), then:
  !pip install -q "transformers>=4.46" accelerate datasets
  !python train_encoder.py --hub_repo <user>/inlegalbert-ner
or as a notebook: run the pip line in a first cell, set os.environ["HUB_REPO"] = "<user>/inlegalbert-ner",
then run this file's contents in the next cell.

Long preambles use overlapping windows; each token's prediction comes from the window where it sits most
centrally. Labels impossible for the row type (preamble / judgment) are masked at decoding.

Writes to --out: preds_dev.jsonl, preds_test.jsonl ({id, spans}), metrics.json.
Score locally: python -m eval.ner_f1 data/instruct/test.jsonl preds_test.jsonl

Smoke test (CPU/MPS, no Hub): python -m ner.train_encoder --smoke --out data/encoder_smoke
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

DATASET = "opennyaiorg/InLegalNER"
MODEL = "law-ai/InLegalBERT"
ENTITIES = ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "DATE", "ORG", "GPE",
            "STATUTE", "PROVISION", "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"]
PREAMBLE = {"COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER"}
LABELS = ["O"] + [f"{p}-{e}" for e in ENTITIES for p in "BI"]
L2I = {l: i for i, l in enumerate(LABELS)}
IGNORE = -100


def is_preamble(source: str) -> bool:
    return "preamble" in (source or "").lower()


def parse(ex: dict) -> dict:
    """Same output as ner/common.py parse_example: gold spans trimmed of edge whitespace."""
    text = ex["data"]["text"]
    spans = []
    for r in (ex["annotations"][0]["result"] if ex.get("annotations") else []) or []:
        v = r["value"]
        s, e = v["start"], v["end"]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        spans += [{"start": s, "end": e, "label": lab} for lab in v["labels"]]
    return {"id": str(ex["id"]), "source": (ex.get("meta") or {}).get("source", ""), "text": text,
            "spans": sorted(spans, key=lambda x: x["start"])}


def windows(docs: list[dict], tok, max_len: int, stride: int) -> list[dict]:
    """Tokenize with overflow; labels are BIO from char overlap. Each row keeps doc index + offsets."""
    enc = tok([d["text"] for d in docs], truncation=True, max_length=max_len, stride=stride,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    rows = []
    for w, doc_i in enumerate(enc["overflow_to_sample_mapping"]):
        offsets = enc["offset_mapping"][w]
        labels = [IGNORE if a == b else L2I["O"] for a, b in offsets]
        for sp in docs[doc_i]["spans"]:
            first = True
            for t, (a, b) in enumerate(offsets):
                if a < b and a < sp["end"] and sp["start"] < b:
                    labels[t] = L2I[("B-" if first else "I-") + sp["label"]]
                    first = False
        rows.append({"input_ids": enc["input_ids"][w], "attention_mask": enc["attention_mask"][w],
                     "labels": labels, "doc": doc_i, "offsets": offsets})
    return rows


def to_words(tokens: list[tuple[int, int, str]], text: str) -> list[tuple[int, int, str]]:
    """Word-pieces -> words, tagged by the first piece. Pieces join only when they meet between two
    alphanumerics ("My|so|re"); BERT splits punctuation into its own tokens, which stay separate ("Kumar" ","),
    so entity edges at punctuation are kept. Stops "Mysore" -> "s", "e" fragment entities."""
    words = []
    for a, b, tag in tokens:
        if words and a == words[-1][1] and a > 0 and text[a - 1].isalnum() and text[a].isalnum():
            words[-1] = (words[-1][0], b, words[-1][2])
        else:
            words.append((a, b, tag))
    return words


def decode(tokens: list[tuple[int, int, str]], text: str) -> list[dict]:
    """Sorted [(start, end, BIO tag)] -> entity spans. An I- tag with a new type (or after O) opens an entity."""
    out = []
    for a, b, tag in to_words(tokens, text):
        if tag == "O":
            out.append(None)
            continue
        kind, ent = tag.split("-", 1)
        if kind == "I" and out and out[-1] is not None and out[-1]["label"] == ent:
            out[-1]["end"] = b
        else:
            out.append({"start": a, "end": b, "label": ent})
    return [s for s in out if s]


def predict(trainer: Trainer, docs: list[dict], rows: list[dict]) -> list[dict]:
    keep = ["input_ids", "attention_mask"]
    logits = trainer.predict([{k: r[k] for k in keep} for r in rows]).predictions  # padded to longest in batch
    masks = {}
    for pre in (True, False):
        allowed = PREAMBLE if pre else set(ENTITIES) - {"LAWYER"}
        masks[pre] = np.array([l == "O" or l.split("-", 1)[1] in allowed for l in LABELS])

    best = [{} for _ in docs]  # doc -> {(start, end): (centrality, label id)}
    for w, r in enumerate(rows):
        mask = masks[is_preamble(docs[r["doc"]]["source"])]
        n = len(r["input_ids"])
        pred = np.where(mask, logits[w][:n], -np.inf).argmax(-1)
        for t, (a, b) in enumerate(r["offsets"]):
            if a < b:
                centrality = min(t, n - 1 - t)
                if (a, b) not in best[r["doc"]] or centrality > best[r["doc"]][(a, b)][0]:
                    best[r["doc"]][(a, b)] = (centrality, int(pred[t]))
    return [{"id": d["id"], "spans": decode([(a, b, LABELS[v[1]]) for (a, b), v in sorted(b_.items())], d["text"])}
            for d, b_ in zip(docs, best)]


def strict_f1(docs: list[dict], preds: list[dict]) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    for d, p in zip(docs, preds):
        gold = {(s["start"], s["end"], s["label"]) for s in d["spans"]}
        pred = {(s["start"], s["end"], s["label"]) for s in p["spans"]}
        for x in gold & pred:
            tp[x[2]] += 1
        for x in pred - gold:
            fp[x[2]] += 1
        for x in gold - pred:
            fn[x[2]] += 1

    def f1(t, a, b):
        return 2 * t / (2 * t + a + b) if t else 0.0

    per = {e: round(f1(tp[e], fp[e], fn[e]), 4) for e in ENTITIES}
    return {"micro_f1": round(f1(sum(tp.values()), sum(fp.values()), sum(fn.values())), 4),
            "macro_f1": round(sum(per.values()) / len(per), 4), "per_label": per}


def window_entities(ids: np.ndarray) -> set:
    ents, start, ent = set(), None, None
    for t, i in enumerate(list(ids) + [0]):
        tag = LABELS[i]
        if ent and not (tag == f"I-{ent}"):
            ents.add((start, t, ent))
            ent = None
        if tag != "O" and ent is None:
            start, ent = t, tag.split("-", 1)[1]
    return ents


def compute_metrics(p) -> dict:
    """Window-level strict entity F1, for model selection on dev."""
    preds, labels = p.predictions.argmax(-1), p.label_ids
    tp = fp = fn = 0
    for a, b in zip(preds, labels):
        keep = b != IGNORE
        g, h = window_entities(b[keep]), window_entities(a[keep])
        tp, fp, fn = tp + len(g & h), fp + len(h - g), fn + len(g - h)
    return {"entity_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def hf_login() -> str:
    """HF token from env / `hf auth login` cache, else the Kaggle secret 'HF_TOKEN'. Fails loudly: the dataset is gated."""
    from huggingface_hub import get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    err = None
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:
            err = e
    if not token:
        raise SystemExit(
            f"No Hugging Face token found ({err!r}). On Kaggle: Add-ons -> Secrets -> add 'HF_TOKEN' "
            "and tick the checkbox to attach it to this notebook; locally: run `hf auth login` or set HF_TOKEN.")
    os.environ["HF_TOKEN"] = token
    return token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/inlegalbert")
    ap.add_argument("--hub_repo", default=os.environ.get("HUB_REPO"),
                    help="private HF repo for the model + predictions (or env HUB_REPO)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32, help="total across GPUs")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="60 rows, 1 epoch")
    args, _ = ap.parse_known_args()  # notebook kernels pass their own argv (-f kernel.json)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    token = hf_login() if not os.environ.get("HF_HUB_OFFLINE") else None

    ds = load_dataset(DATASET, token=token)
    train, dev, test = ([parse(x) for x in ds[s]] for s in ("train", "dev", "test"))
    if args.smoke:
        train, dev, test = train[:60], dev[:30], test[:30]
        args.epochs, args.batch = 1, 8

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, token=token)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, token=token, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I,
        ignore_mismatched_sizes=True)
    tr, dv, te = (windows(x, tok, args.max_len, args.stride) for x in (train, dev, test))
    print(f"windows: train {len(tr)} dev {len(dv)} test {len(te)}")
    strip = lambda rows: [{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in rows]

    cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=f"{args.out}/ckpt", num_train_epochs=args.epochs, learning_rate=args.lr, weight_decay=0.01,
        warmup_steps=int(0.1 * args.epochs * len(tr) / args.batch), per_device_train_batch_size=args.batch // max(torch.cuda.device_count(), 1),
        per_device_eval_batch_size=args.batch // max(torch.cuda.device_count(), 1), fp16=cuda,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="entity_f1", greater_is_better=True, logging_steps=50, report_to="none",
        seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=strip(tr), eval_dataset=strip(dv), processing_class=tok,
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()

    metrics = {"args": vars(args), "best_dev_window_f1": trainer.state.best_metric}
    for name, docs, rows in (("dev", dev, dv), ("test", test, te)):
        preds = predict(trainer, docs, rows)
        with open(f"{args.out}/preds_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        metrics[name] = strict_f1(docs, preds)
        print(name, metrics[name])
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)

    if args.hub_repo:
        from huggingface_hub import HfApi
        trainer.save_model(f"{args.out}/model")
        tok.save_pretrained(f"{args.out}/model")
        api = HfApi(token=token)
        api.create_repo(args.hub_repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.hub_repo, folder_path=f"{args.out}/model")
        for name in ("preds_dev.jsonl", "preds_test.jsonl", "metrics.json"):
            api.upload_file(path_or_fileobj=f"{args.out}/{name}", path_in_repo=name, repo_id=args.hub_repo)
        print("pushed to", args.hub_repo)


if __name__ == "__main__":
    main()
