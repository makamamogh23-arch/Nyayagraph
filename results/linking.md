# Name-based precedent linking: fuzzy matching + Claude Haiku tie-break

Citation regexes (`link/citations.py`) link a citation only when the reporter cite is printed. Judgments also
refer back to cases by name ("Leela Ram (supra)"), and cite outside cases that are really judgments in the corpus
under a different reporter. `link/resolve.py` links these by party name:

1. **Fuzzy matching** (rapidfuzz) of the name printed before a citation or "(supra)" against every known name of each
   candidate. Candidates for "(supra)" = cases the same judgment cites; if none fits, every named case sharing a rare
   party token, under stricter thresholds. Accept when the top score is high and clearly ahead of the second.
2. **Claude Haiku 5.5** (`claude-haiku-5-5`, effort low, structured output `{choice: int}`) for the rest: the mention,
   400 characters of surrounding text and the top 5 candidates; it answers a candidate number or 0 for none.
3. **Duplicate merge** of outside cases with the same year and matching full names (no conflicting SCC cite).

Haiku decisions are cached in `data/cache/link_llm.jsonl`, keyed by a hash of model + prompt. The graph build
only reads that cache and never calls the API, so rebuilds give identical output.

## Accuracy: hidden-citation test

600 citations whose printed name has both parties. Each is hidden and only the first party's name kept (how
"(supra)" refers back); it must be resolved among everything the same judgment cites (median 14 candidates).
Correct = the citation's own case, counting cases merged as duplicates once (as in the graph).

| method | n | correct |
|---|---|---|
| fuzzy accept | 510 | 509 |
| Haiku picked a candidate | 49 | 39 |
| Haiku answered none | 37 | - |
| no candidate above floor | 4 | - |

| | precision | recall |
|---|---|---|
| fuzzy only | 0.998 | 0.848 |
| fuzzy + Haiku | 0.980 | 0.913 |

Haiku adds 6.5 points of recall for 1.8 points of precision. The test understates it:

- Most of the 10 "wrong" picks name the same judgment as the gold, stored under a second citation node the duplicate
  merge did not join (Kathi Kalu Oghad as AIR 1961 and SCR 1962; Hardeep Singh as SCC and SCC (Cri)). Two are clear
  errors, both on bare mentions ("State", "State (Delhi Admn.)").
- The test gives Haiku only `... X (supra) ...` as context; real calls get the surrounding paragraph.

Spot check of real decisions: 15 random "(supra)" picks, 13 right (including OCR names: "As/wk Narain" to Ashok
Narain, "Virupaxappa" to Virupakshappa), 1 wrong ("Loraiya" to Army Headquarters v. CBI), 1 unsure. 11 outside-to-corpus
picks: 9 right, 2 unsure ("Shri Bhagwan" to Bhagwan Sahai; truncated "Singh v. State of M.P." to Phool Singh).

## Effect on the graph

| | regex only | + fuzzy | + fuzzy + Haiku |
|---|---|---|---|
| citation-paragraph links | 16,605 | 19,536 | **20,059** |
| CITES edges to corpus judgments | 1,614 | 1,782 | **1,796** |
| corpus judgments cited at least once | 643 | 723 | **732** |
| outside cases with a name | 48.7% | 78.3% | 78.3% |
| case nodes | 8,390 | 8,230 | 8,219 |

"(supra)" mentions (6,169): 4,337 fuzzy, 843 Haiku, 713 Haiku "none", 276 no candidate.
Outside-to-corpus candidates (645): 108 fuzzy, 11 Haiku, 8 Haiku "none", 518 no candidate.

Most Haiku "(supra)" links attach paragraphs to CITES edges that already exist, so new paragraph links (+523)
are fewer than picks (843).

## Cost

`python -m link.cost` (from per-call usage stored in the cache). Rates: $0.10 / MTok input, $0.50 / MTok output
(Haiku 5.5, prompts up to 100K tokens; output includes thinking).

| | calls | input tokens | output tokens | cost |
|---|---|---|---|---|
| cached decisions | 1,740 | 1,177,115 | 192,706 | $0.214 |
| superseded (hit 1,024-token cap, re-asked) | 7 | | | $0.004 |
| **billed total** | 1,747 | | | **$0.218** |

- Mean call: 677 input / 111 output tokens, $0.00012. About $0.12 per 1,000 decisions, $0.00022 per link made.
- Output costs nearly as much as input (45% of spend) despite a sixth of the tokens: 5x rate, plus thinking.
- 1,661 of the cached calls are live (1,556 supra, 19 outside-to-corpus, 86 test). The other 79 were asked
  before candidate ordering was made deterministic (below) and are no longer referenced.

## Fixes made during the run

- `.env` loader accepts `KEY = value` and `export KEY=value`.
- `max_tokens` 1,024 to 4,096: 7 calls spent the whole budget thinking and returned no answer (recorded as none).
  Re-asked, 4 of 7 picked a candidate.
- Candidate ties are now broken by node id. The global fallback pool is a Python set, whose order changes per
  process, so prompt text, and with it the cache key, changed between runs and ~40 decisions came back
  "unresolved" on rebuild. Verified identical output under two different `PYTHONHASHSEED`s.

## Reproduce

```bash
python -m link.resolve --eval --llm   # hidden-citation test (fills cache for its 86 calls)
python -m link.resolve --llm          # Haiku pass over the corpus (cached; reruns are free)
python -m link.cost
python -m graph.build                 # reads the cache only
```

## Act names (provisions)

`link/acts.py` + `link/provisions.py`, rebuilt into the graph and Neo4j.

- **Section suffix parsing.** "Section 498 A of the Code" was read as section 498 of an Act named "A of the Code";
  "Section 304, Part II of the IPC" and "149 to 152 of Chapter XI of the Code" likewise. A spaced suffix now joins
  the section (498A), Part / Chapter / Schedule / Order qualifiers are skipped, and a lone OCR "I" after the digits
  reads as 1 ("Section 40 I" = 401; IPC 304 Part I/II stays 304).
- **Prefix stripping** in `normalize()` for debris the generic Act pattern swallows ("(C) of the", "E Indian Penal
  Code", "In the", "Sec.4", "Rule of").
- **Spelling / OCR variant merge** (`merge_variants`): same kind (Act vs Rules) always; different years never
  merge (Limitation Act 1908 / 1963) except a reviewed `YEAR_TYPOS` list; abbreviations merge only on identical
  letters; an undated name joins a dated one only when the corpus has one year for it. 275 surface names merged.
- **Patterns** for SC/ST (PA / POA / "Prevention of Atrocities"), TADA, and the UP Gangsters Act.

| | before | after |
|---|---|---|
| Act nodes | 816 | 598 |
| Provision nodes | 3,533 | 3,247 |
| INVOKES edges | 19,999 | 19,825 (duplicate spellings collapsed into one edge) |

Remaining near-duplicates of the 40 most-used Acts are different instruments (Rules vs Act, RP Act 1950 / 1951,
Contempt of Courts Act 1926 / 1971). "Gangster(s) Act" without a state is assumed to be the UP Act.

### Bare "the Code"

An undefined "the Code" defaults to CrPC (3,779 mentions in 1,025 judgments; source `default`). Where the same
judgment cites that section number with an explicit Code, the default agrees 95% of the time (2,184 of 2,303).
Fixes:

- The judgment's own explicit cite wins when it names another Code (87 mentions moved, mostly to IPC).
- Lettered sections that exist only in the IPC (498A, 304B, 120B, 376D ... 37 in `IPC_ONLY`) are IPC whatever the
  text says, including "498A of the CrPC" and a "Code" defined as CrPC then cited for 304B. Plain numbers are not
  used: 485-565 are also sections of the 1898 CrPC still cited (561A).
- Default rows no longer count as evidence in the case-level pass for bare sections (that was circular); they
  feed a second, weaker pass only for rows the explicit cites leave unresolved, which keeps coverage unchanged.

Result: CrPC provisions with an IPC-only section 6 cases to 0; Provision nodes 3,247 to 3,235; INVOKES 19,825 to 19,761.
