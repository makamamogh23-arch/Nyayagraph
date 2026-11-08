# GraphRAG research assistant (first version)

`rag/` answers questions about Indian Supreme Court criminal judgments from the corpus, citing paragraph and case.

## Pipeline

| step | what | where it runs |
|---|---|---|
| chunks | 57,314 paragraphs to 87,738 chunks of at most ~200 words (40-word overlap); citations stay at paragraph level | `rag/embed.py`, local |
| embeddings | `BAAI/bge-base-en-v1.5`, fp16 on Apple M3 GPU (MPS), 46 min, cached in `data/rag/emb.npy` | local |
| query analysis | provisions in the question (`link/provisions.py`) expanded IPC <-> BNS via `MAPS_TO`; case names matched against every Case node, corpus or external | `rag/retrieve.py` + Neo4j |
| text search | BM25 (`bm25s`), top 100 chunks; paragraph = best chunk. Dense (bge-base) is optional and off: it lowered every metric on the benchmark (`results/rag_bench.md`) | local |
| graph re-rank | old-Act counterpart of a new-Act section in the question: paragraph invokes it (+0.020), case invokes it (+0.008); paragraph cites a named case (+0.030); max 3 paragraphs per case, 12 total | Neo4j |
| precedents | corpus cases cited *from* the retrieved paragraphs (`CITES.paras`), their paragraph closest to the question; up to 4 | Neo4j |
| answer | Claude, each paragraph a `search_result` block (source = `case_id ¶para`), native citations; verbatim cited spans get quotation marks | `rag/ask.py`, Anthropic API |

Every Q&A is logged to `data/rag/answers.jsonl` with the retrieved paragraphs, cited sources, raw answer blocks
and usage.

## Smoke test (`python -m rag.smoke`, 10 questions in `rag/questions.txt`, free)

- Provisions detected and mapped: 304B IPC <-> BNS 80, BNS 103 <-> IPC 302, CrPC 482 <-> BNSS 528, UAPA 43D, NDPS 50.
  Paragraphs invoking them rank first.
- Precedent expansion added cited judgments for 8 of 10 questions.
- Named external precedents (Sharad Birdhichand Sarda, Bhajan Lal, Arnesh Kumar) are found through the
  paragraphs that cite them; without this the Sarda question returned only text matches.
- Retrieval: 0.1-0.3 s per question after a 20 s load.

## Answers (Claude Opus 5.5, effort medium)

| question | cited / retrieved | in / out tokens | cost | time |
|---|---|---|---|---|
| conviction solely on circumstantial evidence | 7 / 13 | 7,657 / 2,334 | $0.077 | 36 s (incl. first load) |
| presumption of dowry death, s.304B IPC | 10 / 13 | 7,376 / 2,296 | $0.075 | 20 s |
| BNS s.103: when culpable homicide is murder | 9 / 14 | 9,512 / 2,943 | $0.097 | 25 s |
| Sharad Sarda's panchsheel | 7 / 12 | 7,955 / 1,976 | $0.071 | 18 s |

What the answers do well: every proposition cites a paragraph; holdings are separated from counsel's submissions
and from passages quoted to be rejected; truncated extracts are flagged instead of filled from memory; a source's
slip ("soon after her death") is noticed; BNS numbers are not guessed. On the BNS question the model pointed out
that the graph maps BNS 103 to IPC 302 (punishment), while the test it was asked about lives in IPC 300.

Claude Haiku 5.5 (effort low) on the last-seen question: comparable structure and citations, $0.002, 26 s.

### Re-run on the benchmark-chosen retriever (BM25 + one-way graph bridge, no dense)

| question | retrieved shared with first run | cited (shared) | cost | time |
|---|---|---|---|---|
| circumstantial evidence | 4 / 13 | 7 (2) | $0.073 | 32 s (incl. load) |
| dowry death, s.304B | 8 / 13 | 10 (6) | $0.073 | 19 s |
| BNS s.103 | 11 / 14 | 9 (4) | $0.088 | 23 s |
| Sharad Sarda's panchsheel | 5 / 14 | 9 (3) | $0.086 | 21 s |
| quashing on settlement, s.528 BNSS | 7 / 14 | 7 (3) | $0.099 | 26 s |

Read side by side (not scored): Sarda is better, all five conditions now come from one paragraph quoting
it in full (Suresh v. State of Haryana, 2018) instead of being stitched together; the s.528 BNSS answer is better,
finding a 2025 judgment that states the CrPC 482 / BNSS 528 correspondence itself (Khursheed Ahmad Chohan) and the
three-Judge Parbatbhai Aahir line; dowry death is equal (Jogendra's four prerequisites quoted verbatim); the
circumstantial-evidence answer is equal but different (gains Satye Singh, loses the Padala Veera Reddy four-part
test); BNS s.103 is unchanged in substance. No answer invented a provision number or cited outside the paragraphs.

## Old Code -> new Sanhita mappings (MAPS_TO)

All three come from one source series, the CAPT Bhopal comparison tables (keralaprisons.gov.in):

| mapping | parser | pairs | checks | old sections in the corpus mapped |
|---|---|---|---|---|
| IPC -> BNS | `link/ipc_bns.py` (+ a second dataset; 491 pairs agree) | 549 | - | 373 |
| CrPC -> BNSS | `link/law_map.py` | 524 | 18/18 | 356 |
| Evidence Act -> BSA | `link/law_map.py` | 177 | 14/14 | 103 |

The checks are well-known correspondences (CrPC 482 -> 528, 438 -> 482, 439 -> 483, 164 -> 183, 313 -> 351;
IEA 25/26 -> 23(1)/(2), 27 -> proviso to 23, 32 -> 26, 65B -> 63, 113B -> 118, 106 -> 109, 133 -> 138). Pairs far
from the section-order trend were reviewed: only CrPC 166A/166B -> BNSS 112/113, which are real moves.

The retriever passes the matched pairs to the model as a correspondence note ("Code of Criminal Procedure, 1973
s.482 = Bharatiya Nagarik Suraksha Sanhita, 2023 s.528"), so a question that names only a new section retrieves the
judgments applying the old one. Tested: "quash an FIR under Section 528 BNSS on settlement" returned the CrPC 482 line
(Gian Singh, Narinder Singh, Parbatbhai Aahir, Laxmi Narayan) and said the BNSS link came from the table; $0.091.

Fix found on the way: in `link/acts.py` the BNS pattern preceded BNSS, so "Section 528 BNSS" parsed as BNS 528
(ordered alternation). Corpus now: BNSS 227 mentions / 43 cases, BNS 116 / 10, BSA 11 / 5.

## Known gaps

- The mapping is section to section. Where one old section was split across several new ones, sub-section detail
  comes from the table's sub-section column only.
- Judgments quote old precedents in truncated form; when the full quotation is not retrieved the answer says so.
- Retrieval is benchmarked in `results/rag_bench.md` (BM25 + graph bridge: MRR 0.640, para@10 0.817 on 279 test questions; +0.118 MRR over the first setup). The re-run table above uses the new retriever.
- Sarda exists as two external nodes (AIR 1984 and a mis-cited AIR 1994); both are used.

## Reproduce

```bash
python -m rag.embed                      # once; 46 min on M3 / 8 GB (stop Neo4j first to free memory)
python -m rag.smoke                      # retrieval only
python -m rag.ask "question"             # Opus 5.5; --model claude-sonnet-5-5 or claude-haiku-5-5
python -m rag.ask --show-context "..."   # also print the retrieved paragraphs
```
