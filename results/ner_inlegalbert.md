# InLegalBERT NER baseline (InLegalNER)

Encoder baseline for step 2: `law-ai/InLegalBERT` fine-tuned for token classification (BIO, 14 entity types),
trained on Kaggle T4 x2 with `ner/train_encoder.py`, scored locally with `eval/ner_f1.py`.

## Setup

- Data: InLegalNER train 10,995 / dev 1,074 / test 4,501 rows (gold spans trimmed of edge whitespace).
- Inputs: 256-token windows with 64-token overlap (long preambles); each token is predicted from the window
  where it sits most centrally. Labels impossible for the row type (preamble vs judgment) are masked.
- Optimisation: lr 3e-5, batch 32, fp16, early stopping on dev window-level strict F1 (patience 2).
  Best epoch 5 of 7; dev loss bottoms out at epoch 4, so later epochs overfit.
- Pipeline ceiling (gold labels pushed through windowing + decoding): test strict F1 0.995.

| epoch | train loss | dev loss | dev window F1 |
|---|---|---|---|
| 1 | 0.391 | 0.333 | 0.766 |
| 2 | 0.230 | 0.239 | 0.832 |
| 3 | 0.180 | 0.245 | 0.837 |
| 4 | 0.130 | 0.229 | 0.868 |
| 5 | 0.103 | 0.238 | **0.869** |
| 6 | 0.066 | 0.272 | 0.861 |
| 7 | 0.060 | 0.276 | 0.865 |

## Results (test, 13,365 gold entities)

Strict = exact span and label. Lenient = any overlap with the same label, one-to-one matching.
"Word-level" re-decodes the same predictions so sub-word pieces of one word cannot form separate entities
(`ner/postprocess.py`; built into `ner/train_encoder.py` for future runs).

| decoding | strict micro | strict macro | lenient micro | lenient macro |
|---|---|---|---|---|
| token-level (as trained) | 0.847 | 0.836 | 0.922 | 0.916 |
| word-level | **0.856** | **0.846** | **0.925** | **0.918** |

Dev, word-level: strict micro 0.881, macro 0.867.

Per label (test, word-level):

| label | support | strict F1 | lenient F1 |
|---|---|---|---|
| LAWYER | 1798 | 0.958 | 0.981 |
| JUDGE | 590 | 0.943 | 0.962 |
| STATUTE | 975 | 0.925 | 0.963 |
| DATE | 1112 | 0.918 | 0.981 |
| WITNESS | 413 | 0.915 | 0.932 |
| PROVISION | 1220 | 0.901 | 0.959 |
| COURT | 1221 | 0.898 | 0.949 |
| OTHER_PERSON | 1109 | 0.868 | 0.906 |
| PETITIONER | 862 | 0.822 | 0.903 |
| RESPONDENT | 1113 | 0.814 | 0.929 |
| CASE_NUMBER | 667 | 0.769 | 0.865 |
| GPE | 721 | 0.752 | 0.815 |
| ORG | 914 | 0.688 | 0.801 |
| PRECEDENT | 650 | 0.667 | 0.911 |

## Error analysis (`python -m eval.ner_errors data/instruct/test.jsonl <preds>`)

- **PRECEDENT is a boundary problem, not a detection problem:** lenient recall 0.998, strict 0.73.
  Spans are long ("party v. party (year) vol SCC page") and annotators differ on where they end.
  The pipeline extracts precedents with the citation parser (`link/citations.py`, 88% recall vs headnotes),
  so the encoder's PRECEDENT boundaries matter less downstream.
- **ORG / GPE confuse each other** (GPE->ORG 41, ORG->GPE 27): "GMC Bank, Bengaluru", "Indian State of Mysore".
- **Party roles swap** (PETITIONER<->RESPONDENT 60 combined) and WITNESS->OTHER_PERSON (30): the span is
  right, the role is wrong; role needs context beyond the 256-token window.
- **Sub-word fragments** ("Mysore" -> "s", "e") were a decoding artefact; word-level decoding removes them.

Published InLegalNER baselines are around 0.90 strict F1. The remaining gap sits in the four weak labels.

## Next levers

1. 512-token windows for preambles (party lists cut across windows).
2. Retrain labelling only the first word-piece (standard practice) instead of B/I on every piece.
3. LR 2e-5; keep early stopping.
4. Compare with the QLoRA LLM extractor on the same `eval/ner_f1.py` (gold-as-LLM ceiling 0.999).
