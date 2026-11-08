# Product

## Register

product

## Users

- **Law students** studying criminal-law doctrine (circumstantial evidence, dowry death, bail, quashing). They want an
  explanation they can trust and the authorities behind it, often while preparing assignments or moots, and are
  still learning which judgments matter.
- **Researchers** working on Indian criminal jurisprudence and the 2024 transition from the IPC, CrPC and Evidence
  Act to the BNS, BNSS and BSA. They need to move between old and new section numbers and trace how a rule developed.
- **Recruiters and reviewers** evaluating Nyayagraph as portfolio work. They want to see within a minute what it
  does, that its answers are grounded, and how it was built and measured.

The job on the main screen: ask a legal question, read a short answer, and check every proposition against the
paragraph of the judgment it came from.

## Product Purpose

Nyayagraph answers questions about Indian Supreme Court criminal-appeal judgments (about 2,000, 2016-2025) from the
judgments themselves. Retrieval runs over a Neo4j graph of cases, paragraphs, precedents and statutory provisions;
Claude writes an answer in which every claim cites a case and paragraph number.

Success on a single visit:
- **Trust and verify:** every citation opens the exact paragraph, and the quoted words can be seen in it.
- **Find authorities to cite:** the user leaves with cases and paragraphs they can use, with citations they can copy.
- **Bridge old and new law:** a question in BNS / BNSS / BSA terms returns the IPC / CrPC / Evidence Act case law,
  and the section correspondence used is shown.

## Brand Personality

Rigorous, calm, clear. It reads like a well-edited law report that also explains itself: sourced and authoritative,
never shouting, and approachable enough that a second-year student is not intimidated. Rooted in Indian law and
legal language (case names, "(supra)", section citations) while looking like a modern tool.

References: Perplexity and Elicit for the answer-with-sources pattern (inline numbered citations, a source panel
beside the answer). Indian Kanoon and SCC Online as the familiar baseline Indian lawyers know: keep their
conventions (case titles, neutral citations, paragraph numbers), improve on their readability.

## Anti-references

- **Generic AI chatbot:** chat bubbles, purple gradients, sparkle icons, "Ask me anything!" copy, a typing persona.
- **Dated legal portals:** dense blue link walls, tiny text, banner-heavy court and government sites.
- **SaaS marketing page:** hero metrics, identical card grids, gradients, testimonials.

## Design Principles

1. **Every claim is one click from its source.** Citations are the core interaction, not a footnote: visible,
   precise, and instant to open.
2. **Show the reasoning trail.** Surface what the system did (provisions detected, old/new correspondence used,
   precedents followed) so a user can judge the answer, not just accept it.
3. **Honest about limits.** When the paragraphs do not answer the question, or a quotation is truncated, say so
   plainly; never dress up uncertainty.
4. **Reading comes first.** Answers and judgment text are long-form legal prose; type, measure and contrast are
   chosen for reading, and chrome stays out of the way.
5. **Speak the law's language.** Use Indian legal conventions (case titles, neutral citations, section notation)
   exactly, so lawyers recognise them and students learn them.

## Accessibility & Inclusion

WCAG 2.2 AA: text contrast at least 4.5:1 (3:1 for large text), full keyboard operation including opening and
moving between citations, visible focus states, screen-reader labels for citation markers and the source panel,
and `prefers-reduced-motion` respected for any motion.
