# The Dimensions of Approximate Matching

[← hierarchical correction](./hierarchical-correction.md)

Each correction concern is an independent **dimension** — its own similarity or
plausibility axis, engine, and weight. This page details each dimension, how its
weight enters the semiring, and how the language-model dimensions fuse two
sub-dimensions inside `HybridLanguageModel`.

> **Notation.** Mathematical prose uses MathJax delimited for GitHub-flavored
> Markdown: inline math is a backtick span wrapped in dollar signs, and display
> math is a fenced block whose info-string is `math`. Bare dollar delimiters are
> never used.

## Semirings — the common currency

The dimensions are **heterogeneous**: dimension 1 is a true string *metric*,
dimension 4 a *likelihood field*, dimension 3 a *hard constraint*. They are made
composable by mapping every weight into one **semiring**
$`(K, \oplus, \otimes, \bar{0}, \bar{1})`$, where $`\otimes`$ combines factors along a
path and $`\oplus`$ selects among competing paths.

| Semiring | $`\oplus`$ | $`\otimes`$ | $`\bar{0}`$ | $`\bar{1}`$ | Decodes to |
|----------|----------|-----------|-----------|-----------|------------|
| Tropical | $`\min`$ | $`+`$ | $`+\infty`$ | $`0`$ | shortest path (edit distance) |
| Log | $`-\log(e^{-x}+e^{-y})`$ | $`+`$ | $`+\infty`$ | $`0`$ | max-probability path (log space) |
| Probability | $`+`$ | $`\times`$ | $`0`$ | $`1`$ | total probability mass |

## Dimension 1 — Orthographic (edit distance)

- **Engine:** a liblevenshtein Levenshtein automaton intersected with the
  vocabulary trie (see [dictionary-backend.md](./dictionary-backend.md) for how
  that automaton walks the persistent vocabulary in place).
- **Metric:** Levenshtein / Damerau–Levenshtein edit distance — a true metric on
  strings.
- **Weight:** the candidate's edit distance $`d`$, entered as a `TropicalWeight`
  of value $`d`$ (Damerau–Levenshtein by default, so a transposition typo costs
  $`d = 1`$, not $`2`$).
- **Layer:** `LevenshteinCorrectionLayer` (this integration).

## Dimension 2 — Phonetic (sound-alike)

- **Engine:** liblevenshtein's formally-verified **Zompist** spelling-to-sound
  rewrite rules, driven as an articulatory-weighted product automaton
  (`PhoneticTransducerChar`).
- **Metric:** edit distance over *normalized pronunciations* — recovers
  `knight` $`\leftrightarrow`$ `night`, `fone` $`\leftrightarrow`$ `phone`.
- **Weight:** an articulatory cost $`\le 0`$ (a sound-alike *discount*), **fused
  into** the orthographic edit cost so a homophone competes directly with an
  orthographic neighbor — not applied as a separate rescore.
- **Realization:** the shipped path is the cascade's $`T_{\text{lex}}`$ (feature
  `phonetic-correction`), scoring each candidate by edit **and** articulatory cost
  in one query (see [hierarchical-correction.md](./hierarchical-correction.md) §6.3).
  lling-llang offers a `PhoneticRescoreLayer` and libgrammstein a standalone
  `PhoneticEmbedding` utility, but neither is wired into a shipped corrector: a
  downstream rescore would only re-rank candidates the edit automaton already
  surfaced, never recovering a sound-alike it ranked out — which is why fusion, not
  rescoring, is the principled design.

## Dimension 3 — Syntactic (grammaticality)

- **Engine:** lling-llang `CfgFilterLayer` — an Earley parser run *over the
  lattice* rather than a single string.
- **Metric:** membership in a context-free grammar (a hard/soft constraint).
- **Weight:** a $`\{\bar{0}, \bar{1}\}`$ filter, or a soft PCFG log-probability
  penalty.
- **Note:** English is not context-free and no general English CFG ships, so this
  layer is *optional* — the spelling $`\circ`$ LM pipeline is complete without it,
  and the layer is inserted only when a grammar is supplied (e.g. for a formal
  language). This is a supported, first-class optional slot — wired and ready
  for whatever grammar you supply.

## Dimension 4 — Contextual (n-gram fluency)

- **Engine:** libgrammstein's n-gram model with **Modified Kneser-Ney** smoothing
  (order-specific discounts $`D_1, D_2, D_{3+}`$, interpolation via continuation
  counts; [Chen & Goodman 1999](https://doi.org/10.1006/csla.1999.0128)).
- **Field:** $`-\log P(w \mid \text{context})`$ — a likelihood, not a metric.
- **Weight:** interpolated in cost space,
  $`w' = (1-\lambda)\,w + \lambda\,\bigl(-\log P(w \mid \text{context})\bigr)`$.
- **Layer:** lling-llang `LanguageModelLayer` wrapping libgrammstein's
  `GrammsteinLanguageModel` (which implements lling-llang's `LanguageModel`
  trait: `score_sequence`, `score_continuation`).

## Dimension 5 — Semantic (embeddings / OOV)

- **Engine:** libgrammstein's `SubwordEmbedding` (FastText-style hashed character
  n-grams; [Bojanowski et al. 2017](https://doi.org/10.1162/tacl_a_00051)).
- **Field:** cosine similarity of the candidate to the context vector, mapped to
  a pseudo-probability; degrades gracefully for out-of-vocabulary words the
  n-gram cannot score.

### The hybrid fusion (dimensions 4 $`\oplus`$ 5)

`HybridLanguageModel` fuses the two LM sub-dimensions. Let $`P_n`$ be the n-gram
probability and $`P_e`$ the embedding probability; the vocabulary is $`V`$ and the
context is $`h`$:

| Strategy | Combination |
|----------|-------------|
| Linear | $`P = \alpha\,P_n + (1-\alpha)\,P_e`$ |
| Log-linear | $`\log P = \alpha \log P_n + (1-\alpha) \log P_e`$ |
| N-gram-with-embedding-fallback | $`P = P_n`$ if $`w \in V`$, else $`P_e`$ |
| Dynamic | $`\alpha(h)`$ grows with the context length $`\lvert h \rvert`$ |

So the "language-model dimension" is itself two sub-dimensions fused by a tunable
$`\alpha`$ — which is why "2+" is exact and the true dimensionality is a design
knob.

## How weights compose

Along any path $`p`$ through the lattice, the joint weight is the semiring product
$`w(p) = \bigotimes_i w(e_i)`$ of the per-edge weights each dimension contributed.
The interpolation weights ($`\lambda`$ for the LM layer, $`\alpha`$ inside the hybrid
model) are how the heterogeneous dimensions are balanced within the single
semiring — the tunable "aspect ratio" of the multi-dimensional match. The best
correction is the $`\oplus`$-optimal path, extracted by `viterbi` (single best) or
`nbest` (k-best).

## The term-id cascade — a new axis and a second arrangement

The five dimensions above are arranged as a **lattice** that the
`HierarchicalCorrector` expands and reweights. libgrammstein's second corrector, the
`GrammarCorrector` (§6 of
[hierarchical-correction.md](./hierarchical-correction.md)), re-arranges the *same*
concerns as a **cascade over a shared term-id alphabet** — and in doing so promotes
a concern the lattice leaves implicit into a first-class dimension. Its pipeline is
literally **characters $`\rightarrow`$ term-ids $`\rightarrow`$ n-gram windows**:

1. **$`T_{\text{lex}}`$ (characters $`\rightarrow`$ term-ids)** folds Dimensions 1 and 2 into one
   stage. A Levenshtein / articulatory automaton maps a token's *characters* to
   candidate vocabulary **term-ids**, scoring each with
   $`\text{cost} = \text{edit_distance} + \text{phonetic_cost}`$: the sound-alike
   discount of Dimension 2 ($`\text{phonetic_cost} \le 0`$) is *fused into* the
   orthographic cost rather than applied as a separate rescore. Same Damerau–Levenshtein
   metric as Dimension 1, over the character alphabet.

2. **$`T_{\text{gram}}`$ (term-ids $`\rightarrow`$ n-gram windows) — the new axis.** Treating each
   term-id as a single alphabet symbol, a *word-level* Levenshtein automaton edits
   whole *sequences* of term-ids against the stored n-grams. Its metric is
   Damerau–Levenshtein over **words** — insertion, deletion, and substitution of
   entire words at unit cost. This is the axis the lattice pipeline has no
   counterpart for: it is what lets the cascade *insert* a missing word or *delete*
   an extraneous one, not merely substitute per slot.

3. **Source model (windows $`\rightarrow`$ score).** The candidate n-gram windows are
   scored by **stupid backoff** over the raw counts,
   $`-\log S(w \mid h)`$ — a likelihood field like Dimension 4, but estimated from
   *plain* counts rather than Modified Kneser-Ney, and with **no** embedding
   sub-dimension (Dimension 5 is absent on this path).

| Axis | Alphabet | Metric / field | Stage |
|---|---|---|---|
| Dimensions 1–2 (folded) | characters | Damerau edit $`+`$ phonetic discount | $`T_{\text{lex}}`$ |
| **word-level edit (new)** | **term-ids** | **Damerau edit over words** | $`T_{\text{gram}}`$ |
| Dimension 4 (counts only) | n-gram windows | $`-\log S`$ (stupid backoff) | source model |

The term-ids $`T_{\text{lex}}`$ emits are exactly the vocabulary indices the n-gram store is
keyed by, so the two stages share one coherent alphabet with no translation layer.
Because both stages are Damerau–Levenshtein automata differing *only* in their
alphabet — characters for $`T_{\text{lex}}`$, term-ids for $`T_{\text{gram}}`$ — the cascade is approximate
matching at two granularities of one engine, closed by a single source-model
rescore. Its joint objective is the minimum-cost decode of §6.1 in
[hierarchical-correction.md](./hierarchical-correction.md) — the term-id analogue of
the semiring-product path score above.

The *same* cascade runs unchanged over either an in-memory count store
(`GrammarCorrector`) or the sharded Google-Books corpus (`ShardedGrammarCorrector`, §6.9
of [hierarchical-correction.md](./hierarchical-correction.md); full design record in
[multi-shard-grammar-corrector.md](../multi-shard-grammar-corrector.md)) — only the
n-gram *view source* differs. That is a storage-backend swap, not a modeling change: it
adds **no** new dimension and removes none, so the axis count above is invariant to how
the counts are physically stored.
