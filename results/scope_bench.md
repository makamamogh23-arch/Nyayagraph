# Scope benchmark: lower court and counsel named in the question

`python -m eval.scope_bench`: 10 questions, retrieval only (BM25 + graph), each run with scope off and on. Top 10 paragraphs. Gold = in-scope judgments that invoke the named provision (graph-defined, see the module docstring). No model calls.

Scope detected for 10 of 10 questions.

## Summary (mean over questions)

| questions | scope | in_scope@10 | gold@10 | cases@10 | rr |
|---|---|---|---|---|---|
| court | off | 0.360 | 0.300 | 2.6 | 0.510 |
| court | on | 1.000 | 0.900 | 5.6 | 1.000 |
| lawyer | off | 0.240 | 0.100 | 1.0 | 0.099 |
| lawyer | on | 1.000 | 0.960 | 6.8 | 1.000 |
| **all** | off | 0.300 | 0.200 | 1.8 | 0.304 |
| **all** | on | 1.000 | 0.930 | 6.2 | 1.000 |

## Per question

| kind | named | provision | in scope | gold | gold@10 off → on | gold cases@10 off → on | RR off → on |
|---|---|---|---|---|---|---|---|
| court | High Court of Judicature at Allahabad | Indian Penal Code, 1860 s.304B | 247 | 13 | 0.1 → 1.0 | 1 → 5 | 0.10 → 1.00 |
| court | High Court of Punjab and Haryana | Narcotic Drugs and Psychotropic Substances Act, 1985 s.50 | 201 | 9 | 0.2 → 0.7 | 2 → 4 | 0.20 → 1.00 |
| court | High Court of Judicature at Bombay | Indian Evidence Act, 1872 s.27 | 186 | 18 | 0.1 → 0.8 | 1 → 6 | 0.25 → 1.00 |
| court | High Court of Judicature at Madras | Prevention of Corruption Act, 1988 s.13 | 170 | 26 | 0.7 → 1.0 | 7 → 7 | 1.00 → 1.00 |
| court | High Court of Delhi | Code of Criminal Procedure, 1973 s.439 | 160 | 19 | 0.4 → 1.0 | 2 → 6 | 1.00 → 1.00 |
| lawyer | Sidharth Luthra | Code of Criminal Procedure, 1973 s.482 | 63 | 22 | 0.3 → 1.0 | 3 → 9 | 0.25 → 1.00 |
| lawyer | Tushar Mehta | Prevention of Corruption Act, 1988 s.13 | 59 | 12 | 0.0 → 1.0 | 0 → 6 | 0.00 → 1.00 |
| lawyer | R Basant | Indian Penal Code, 1860 s.307 | 58 | 15 | 0.1 → 0.8 | 1 → 6 | 0.14 → 1.00 |
| lawyer | V N Raghupathy | Indian Penal Code, 1860 s.498A | 54 | 9 | 0.1 → 1.0 | 1 → 5 | 0.10 → 1.00 |
| lawyer | Mukesh Kumar Maroria | Indian Penal Code, 1860 s.420 | 128 | 35 | 0.0 → 1.0 | 0 → 8 | 0.00 → 1.00 |

## How the scope got here

| run | gold@10 | gold cases@10 | MRR |
|---|---|---|---|
| scope off | 0.200 | 1.8 | 0.304 |
| scope on, first version (filter only) | 0.600 | 4.3 | 0.820 |
| scope on, scope words dropped from the BM25 query (current) | **0.930** | **6.2** | **1.000** |

The first version restricted the search but still matched the scope's own words. Inside "appeals from the Bombay
High Court" nearly every judgment's opening paragraph says "appeal", "High Court" and "Bombay", so those ¶1s crowded
out the Section 27 paragraphs (gold@10 0.1 even with the filter on). Once the search is restricted, the words that
only name the scope (court place names, "appeals from", the lawyer's name, "argued", "appeared", "counsel") are
dropped from the BM25 query (`Scope.drop_words`); the topic words then decide the ranking.

## Caveats

- **Tuned on the test.** The fix came from reading these 10 questions' results, and the current numbers are on the
  same 10 questions. The failure it fixes is generic (scope words match opening paragraphs), but there is no
  held-out set.
- **Gold is case-level.** A gold paragraph comes from a judgment that invokes the provision somewhere. Spot check of
  the top 3 for three questions: Bombay s.27 returns Section 27 recovery memos and arguments; Maroria s.420 returns
  the cheating / dishonesty analysis; R. Basant s.307 returns a s.307 conviction that does not discuss intention.
- **Scope off is a weak baseline by design:** the deployed retriever has no other way to use the court or counsel.
- **The scope rests on NER output** (lower court from the headnote, counsel from the appearance lists); a judgment
  missing from those tables cannot be found by its scope.
- Only scoped questions change. Unscoped questions (the 600-question benchmark in `rag_bench.md`) never trigger a
  scope and are unaffected.
