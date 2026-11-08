# Free checks on logged answers

`python -m eval.answer_checks`: 10 answers that kept their raw blocks (6 Opus 5.5, 4 Haiku 5.5). Deterministic text checks, no model calls.

| answer | model | verbatim cited passages | paraphrased cited passages | quotes exact / checked | quotes not found | cited prose | sections stated | unsupported sections |
|---|---|---|---|---|---|---|---|---|
| Is the last seen together theory enough on its own to convic… | haiku | 13 | 2 | 0 / 0 | 0 | 52% | 2 | - |
| Can the High Court quash an FIR under Section 528 BNSS when … | opus | 19 | 0 | 0 / 0 | 0 | 60% | 5 | - |
| When can a conviction rest solely on circumstantial evidence… | opus | 11 | 0 | 0 / 0 | 0 | 39% | 0 | - |
| What must the prosecution prove to raise the presumption of … | opus | 17 | 0 | 0 / 0 | 0 | 45% | 6 | - |
| Under Section 103 of the BNS, when does culpable homicide am… | opus | 16 | 0 | 2 / 2 | 0 | 50% | 5 | - |
| What did the Court hold in Sharad Birdhichand Sarda v. State… | opus | 10 | 3 | 2 / 2 | 0 | 60% | 1 | - |
| Can the High Court quash an FIR under Section 528 BNSS when … | opus | 14 | 0 | 0 / 0 | 0 | 54% | 5 | - |
| Can the High Court quash an FIR under Section 528 BNSS on se… | haiku | 9 | 1 | 0 / 0 | 0 | 31% | 2 | - |
| Is a disclosure statement admissible under Section 23 of the… | haiku | 11 | 0 | 1 / 1 | 0 | 28% | 4 | - |
| What is the test for bail under Section 43D(5) UAPA? | haiku | 0 | 5 | 0 / 0 | 0 | 10% | 4 | - |

**Cited passages:** 131; 120 are word-for-word text of the paragraph they cite, 0 are word-for-word text of a different retrieved paragraph (misattributed), 11 are paraphrase or commentary (not checkable without a model).

**Explicit quotation marks:** 5 quoted spans in cited passages: 5 exact in the cited paragraph (100.0%), 0 found only in another retrieved paragraph, 0 not found word for word. 2 quotes in uncited passages, all found in the retrieved paragraphs.
