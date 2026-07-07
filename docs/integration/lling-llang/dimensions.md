# The Dimensions of Approximate Matching

[← hierarchical correction](./hierarchical-correction.md)

Each correction concern is an independent **dimension** — its own similarity or
plausibility axis, engine, and weight. This page details each dimension, how its
weight enters the semiring, and how the language-model dimensions fuse two
sub-dimensions inside `HybridLanguageModel`.

> Notation: mathematical prose uses MathJax/LaTeX (`$…$`).

## Semirings — the common currency

The dimensions are **heterogeneous**: dimension 1 is a true string *metric*,
dimension 4 a *likelihood field*, dimension 3 a *hard constraint*. They are made
composable by mapping every weight into one **semiring**
$(K, \oplus, \otimes, \bar{0}, \bar{1})$, where $\otimes$ combines factors along a
path and $\oplus$ selects among competing paths.

| Semiring | $\oplus$ | $\otimes$ | $\bar{0}$ | $\bar{1}$ | Decodes to |
|----------|----------|-----------|-----------|-----------|------------|
| Tropical | $\min$ | $+$ | $+\infty$ | $0$ | shortest path (edit distance) |
| Log | $-\log(e^{-x}+e^{-y})$ | $+$ | $+\infty$ | $0$ | max-probability path (log space) |
| Probability | $+$ | $\times$ | $0$ | $1$ | total probability mass |

## Dimension 1 — Orthographic (edit distance)

- **Engine:** a liblevenshtein Levenshtein automaton intersected with the
  vocabulary trie (see [dictionary-backend.md](./dictionary-backend.md) for how
  that automaton walks the persistent vocabulary in place).
- **Metric:** Levenshtein / Damerau–Levenshtein edit distance — a true metric on
  strings.
- **Weight:** the candidate's edit distance $d$, entered as a `TropicalWeight`
  of value $d$ (Damerau–Levenshtein by default, so a transposition typo costs
  $d = 1$, not $2$).
- **Layer:** `LevenshteinCorrectionLayer` (this integration).

## Dimension 2 — Phonetic (sound-alike)

- **Engine:** liblevenshtein's formally-verified **Zompist** spelling-to-sound
  rewrite rules; on the LM side, libgrammstein's `PhoneticEmbedding`.
- **Metric:** edit distance over *normalized pronunciations* — recovers
  `knight` $\leftrightarrow$ `night`, `fone` $\leftrightarrow$ `phone`.
- **Weight:** a fixed or $\lambda$-interpolated phonetic cost.
- **Layer:** lling-llang `PhoneticRescoreLayer`.

## Dimension 3 — Syntactic (grammaticality)

- **Engine:** lling-llang `CfgFilterLayer` — an Earley parser run *over the
  lattice* rather than a single string.
- **Metric:** membership in a context-free grammar (a hard/soft constraint).
- **Weight:** a $\{\bar{0}, \bar{1}\}$ filter, or a soft PCFG log-probability
  penalty.
- **Note:** English is not context-free and no general English CFG ships, so this
  layer is *optional* — the spelling $\circ$ LM pipeline is complete without it,
  and the layer is inserted only when a grammar is supplied (e.g. for a formal
  language). This is a supported slot, not a stub.

## Dimension 4 — Contextual (n-gram fluency)

- **Engine:** libgrammstein's n-gram model with **Modified Kneser-Ney** smoothing
  (order-specific discounts $D_1, D_2, D_{3+}$, interpolation via continuation
  counts; [Chen & Goodman 1999](https://doi.org/10.1006/csla.1999.0128)).
- **Field:** $-\log P(w \mid \text{context})$ — a likelihood, not a metric.
- **Weight:** interpolated in cost space,
  $w' = (1-\lambda)\,w + \lambda\,\bigl(-\log P(w \mid \text{context})\bigr)$.
- **Layer:** lling-llang `LanguageModelLayer` wrapping libgrammstein's
  `GrammsteinLanguageModel` (which implements lling-llang's `LanguageModel`
  trait: `score_sequence`, `score_continuation`).

## Dimension 5 — Semantic (embeddings / OOV)

- **Engine:** libgrammstein's `SubwordEmbedding` (FastText-style hashed character
  n-grams; [Bojanowski et al. 2017](https://doi.org/10.1162/tacl_a_00051)).
- **Field:** cosine similarity of the candidate to the context vector, mapped to
  a pseudo-probability; degrades gracefully for out-of-vocabulary words the
  n-gram cannot score.

### The hybrid fusion (dimensions 4 $\oplus$ 5)

`HybridLanguageModel` fuses the two LM sub-dimensions. Let $P_n$ be the n-gram
probability and $P_e$ the embedding probability; the vocabulary is $V$ and the
context is $h$:

| Strategy | Combination |
|----------|-------------|
| Linear | $P = \alpha\,P_n + (1-\alpha)\,P_e$ |
| Log-linear | $\log P = \alpha \log P_n + (1-\alpha) \log P_e$ |
| N-gram-with-embedding-fallback | $P = P_n$ if $w \in V$, else $P_e$ |
| Dynamic | $\alpha(h)$ grows with the context length $\lvert h \rvert$ |

So the "language-model dimension" is itself two sub-dimensions fused by a tunable
$\alpha$ — which is why "2+" is exact and the true dimensionality is a design
knob.

## How weights compose

Along any path $p$ through the lattice, the joint weight is the semiring product
$w(p) = \bigotimes_i w(e_i)$ of the per-edge weights each dimension contributed.
The interpolation weights ($\lambda$ for the LM layer, $\alpha$ inside the hybrid
model) are how the heterogeneous dimensions are balanced within the single
semiring — the tunable "aspect ratio" of the multi-dimensional match. The best
correction is the $\oplus$-optimal path, extracted by `viterbi` (single best) or
`nbest` (k-best).
