# Retrieval benchmark for the GraphRAG assistant

`eval/rag_bench.py`. Measures `rag/retrieve.py` alone (no LLM in the loop).

## Gold labels

SCR headnotes list "HELD" points, each a proposition with the judgment paragraphs it comes from
("1.3 ... the evidence of a child witness must find adequate corroboration ... [Paras 22, 23]"). 5,205 points with
paragraph references parse from 1,106 headnotes. Headnotes are not in the retrieval index.

- Sample: 600 points, seeded, at most 2 per case (464 cases).
- Claude Haiku 5.5 labels each point "proposition" (a rule other cases could apply) or "facts", and rewrites
  propositions as a research question of at most 30 words with no party names, dates or copied wording.
  439 propositions kept. Cost $0.062 (600 calls, cached in `data/bench/haiku.jsonl`).
- Split by case: dev 160 questions (tuning), test 279 (reported).

Query styles:

| style | query | n (test) |
|---|---|---|
| held | the editor's text, first 80 words | 279 |
| question | Haiku's research question | 279 |
| newact | questions naming IPC / CrPC / Evidence Act sections, with the numbers rewritten to BNS / BNSS / BSA ("Section 405 IPC" to "Section 316 of the BNS"); no LLM | 77 |

Metrics over the top 10 paragraphs: para@k (a gold paragraph in the top k), case@k (any paragraph of the source
case), MRR. Only the source case counts as correct, though other judgments often state the same rule, so the
numbers are a lower bound and the benchmark is for comparing setups. 95% CIs from a paired bootstrap that
resamples cases (2,000 resamples).

## Results (test)

| config | question MRR | para@1 | para@10 | case@10 | newact MRR | newact para@10 |
|---|---|---|---|---|---|---|
| deployed before (BM25 + dense + graph boosts) | 0.522 | 0.412 | 0.746 | 0.860 | 0.556 | 0.805 |
| dense only (bge-base) | 0.385 | 0.290 | 0.609 | 0.742 | | |
| hybrid (BM25 + dense, no graph) | 0.522 | 0.412 | 0.746 | 0.860 | | |
| BM25 only | 0.640 | 0.548 | 0.817 | 0.903 | 0.573 | 0.766 |
| BM25 + cross-encoder rerank (top 30) | 0.633 | 0.534 | 0.817 | 0.914 | | |
| **BM25 + graph bridge (new default)** | **0.640** | **0.548** | **0.817** | **0.903** | **0.597** | 0.779 |

The `held` style is near ceiling for every setup (MRR 0.89-0.93 on dev): editors reuse the judgment's wording, so it
does not separate configurations.

Paired differences (test):

- New default vs deployed before, question: MRR +0.118 [+0.077, +0.159], para@10 +0.072 [+0.032, +0.114],
  case@10 +0.043 [+0.011, +0.079].
- Dense only vs BM25: MRR -0.255 [-0.304, -0.203]. Adding dense to BM25 at any weight lowered MRR on dev
  (weight 0.25: 0.614 vs 0.636).
- Cross-encoder (`BAAI/bge-reranker-base`) vs BM25: para@10 +0.000 [-0.039, +0.039]. It looked like +5 points
  para@10 on dev; it did not replicate, and costs 1.7 s per question.
- Graph bridge vs BM25, newact: MRR +0.024 [-0.025, +0.082] on test (n = 77); +0.057 [+0.009, +0.121] on dev
  (n = 41). Identical to BM25 on old-section questions, by construction.

## What changed in the retriever

1. Dense retrieval off by default (`use_dense=False`); the embeddings stay for picking a cited precedent's best
   paragraph and a named case's best chunks.
2. Provision boosts apply one way only: to paragraphs invoking the old-Act counterpart of a new-Act section the
   question names (BNSS 528 to CrPC 482). Boosting the literally named section (BM25 already matches it) cost
   -0.038 MRR on provision questions; boosting new-Act counterparts of an old section pulled up the few 2024-25
   judgments over better older matches (-0.031 MRR).
3. PageRank prior off (no effect at any scale on dev).
4. Named-case boost and precedent expansion kept: they serve questions the benchmark does not contain (asking about
   a named precedent; following citations).

## Caveats

- The test set was looked at three times: the first run (all configs), the run that showed the two-way bridge
  hurting, and the final run. The bridge direction was fixed and validated on dev before the final run, but the
  final test numbers for the graph bridge are not fully held out.
- Questions are written from the editor's text, which shares legal terms with the gold paragraph; this favours
  BM25. Real users may phrase questions further from the judgment's wording, where dense retrieval might help. A
  stronger or legal-domain embedding model was not tried (re-embedding takes 46 min on this machine).
- The newact slice is small (77 test / 41 dev questions).

## Reproduce

```bash
python -m eval.rag_bench build --n 600                                      # $0.06 Haiku, cached
python -m eval.rag_bench run --split dev  --styles question,newact --configs bm25,bm25+graph
python -m eval.rag_bench run --split test --styles question,newact --configs full,bm25,bm25+graph
```
