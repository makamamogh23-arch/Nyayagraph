# QLoRA LLM extractor vs InLegalBERT (InLegalNER)

Generative NER: `Qwen/Qwen2.5-7B-Instruct` fine-tuned with QLoRA (`ner/train_qlora.py`) to emit
`{"entities": [{"label", "text"}]}`; strings are aligned back to character spans and scored with the same
`eval/ner_f1.py` as the encoder baseline (`results/ner_inlegalbert.md`). Inference: vLLM 0.33, greedy, fp16,
tensor-parallel over Kaggle T4 x2 (`ner/infer_vllm.py`).

## Setup

- Data: same InLegalNER split, 10,982 train rows (13 rows > 2,048 tokens dropped), 400 dev rows for eval loss,
  full 4,501-row test set. Prompt = system + label inventory (preamble or judgment label set) + text.
- QLoRA: 4-bit NF4 base with double quantisation, fp16 compute (no bf16 on T4), LoRA on all 7 linear
  projections, alpha = 2r, dropout 0.05. Loss on the assistant JSON only.
- Optimisation: 1 epoch (687 optimiser steps), lr 2e-4 cosine, batch 1 x grad-accum 16, paged AdamW 8-bit,
  best dev-loss checkpoint (eval every 100 steps). r8 and r16 trained in parallel, one T4 each.

| rank | steps | hours (one T4) | best dev loss (per answer token) |
|---|---|---|---|
| 8 | 687 | 6.56 | 0.0199 |
| 16 | 687 | 6.80 | 0.0183 |

Both runs completed the full epoch (no time-limit stop). The dev-loss ordering matches the test F1 ordering.
- Batch 4 ran out of memory on the first (longest) batch: fp32 logits over Qwen's 152k vocabulary for
  4 x 2,048 tokens alone need about 5 GB.

## Results (test, 13,365 gold entities)

| model | strict micro | strict macro | lenient micro | lenient macro |
|---|---|---|---|---|
| Qwen2.5-7B zero-shot (same prompt) | 0.334 | 0.318 | 0.592 | 0.572 |
| InLegalBERT, word-level decoding | 0.856 | 0.846 | 0.925 | 0.918 |
| Qwen2.5-7B QLoRA r8 | 0.872 | 0.861 | 0.933 | 0.927 |
| Qwen2.5-7B QLoRA r16 | **0.879** | **0.868** | **0.938** | **0.932** |

Paired bootstrap over documents (1,000 resamples, strict micro F1):

- r16 - InLegalBERT: **+0.023**, 95% CI [+0.015, +0.030]
- r16 - r8: +0.007, 95% CI [+0.003, +0.011]

By row type (strict micro): preambles r16 0.909 vs encoder 0.899; judgment sentences r16 0.862 vs 0.833.

Output health (r16): 99.7% of generations are valid JSON, 5 of 4,501 hit the 1,536-token cap, 4 off-schema
labels, and 72 of 13,796 predicted strings (0.5%) fail to align, almost all from whitespace the model
normalised inside line-broken names and addresses.

### Per label (strict F1, test)

| label | support | InLegalBERT | QLoRA r16 | diff |
|---|---|---|---|---|
| CASE_NUMBER | 667 | 0.769 | 0.825 | +0.056 |
| PRECEDENT | 650 | 0.667 | 0.718 | +0.051 |
| PETITIONER | 862 | 0.822 | 0.872 | +0.050 |
| ORG | 914 | 0.688 | 0.733 | +0.045 |
| PROVISION | 1220 | 0.901 | 0.943 | +0.042 |
| OTHER_PERSON | 1109 | 0.868 | 0.891 | +0.023 |
| DATE | 1112 | 0.918 | 0.934 | +0.016 |
| RESPONDENT | 1113 | 0.814 | 0.827 | +0.013 |
| GPE | 721 | 0.752 | 0.765 | +0.013 |
| COURT | 1221 | 0.898 | 0.908 | +0.010 |
| STATUTE | 975 | 0.925 | 0.932 | +0.007 |
| JUDGE | 590 | 0.943 | 0.944 | +0.001 |
| WITNESS | 413 | 0.915 | 0.905 | -0.010 |
| LAWYER | 1798 | 0.958 | 0.947 | -0.011 |

## Reading the numbers

- **Fine-tuning is the whole story.** Zero-shot, the same model scores 0.33 strict: it invents labels
  (PERSON, STATE, SECTION; 295 off-schema), wraps half its answers in code fences, paraphrases spans
  (13% of its strings cannot be found in the text) and misses most PRECEDENTs. One epoch of QLoRA fixes all of it.
- **The LLM wins where context matters.** Biggest gains are party roles (PETITIONER +0.05), CASE_NUMBER,
  ORG and PROVISION: the LLM reads the whole preamble or sentence, while the encoder sees 256-token windows.
  Exact-span PETITIONER/RESPONDENT swaps fall from 60 (encoder) to 13 (r16).
- **Boundaries are still the ceiling.** PRECEDENT is 0.72 strict but 0.97 lenient, and RESPONDENT 0.83 vs 0.96:
  the remaining error is where a span ends (long citations, party names with addresses), not whether it is found.
- **Remaining type confusions** (gold -> pred): GPE->ORG 53, ORG->GPE 51, OTHER_PERSON->WITNESS 36,
  GPE->COURT 21. Same ORG/GPE ambiguity as the encoder.
- **Rank:** r16 beats r8 by a small but consistent margin (CI excludes 0). r64 is not worth the extra run
  unless the r16 to r64 step is wanted for the write-up.

## Cost

The encoder is far cheaper: 110M parameters, one forward pass, CPU-feasible. The LLM needs a 7B model on two
T4s and about 6.5-7 GPU-hours of training per rank. For the pipeline, the LLM extractor is the default for the
judgment corpus (better roles and case numbers, one pass that also yields structured JSON), and InLegalBERT
stays as the fast fallback. PRECEDENT spans come from the citation parser either way (`link/citations.py`).

## Reproduce

```bash
python -m eval.ner_f1 data/instruct/test.jsonl data/preds/preds_r16.jsonl
python -m eval.ner_errors data/instruct/test.jsonl data/preds/preds_r16.jsonl
```
Adapters: `Makamamogh23/nyaya-ner-qwen7b-r8`, `Makamamogh23/nyaya-ner-qwen7b-r16` (private, Hugging Face).
