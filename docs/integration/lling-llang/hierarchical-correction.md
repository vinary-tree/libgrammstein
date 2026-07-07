# Hierarchical Spelling & Grammar Correction

[← lling-llang integration index](./overview.md)

This document explains, end to end, how libgrammstein's Google-Books language
models plug into [lling-llang](https://github.com/f1r3fly-io/lling-llang)'s
weighted finite-state transducer (WFST) framework — together with the
[liblevenshtein](https://github.com/universal-automata/liblevenshtein-rust)
edit-distance automaton and the
[libdictenstein](https://github.com/vinary-tree/libdictenstein) dictionary
substrate — to perform **hierarchical spelling and grammar correction**.

The one-sentence thesis: correction is the **noisy-channel model** realized as a
**search for the best path through a semiring-weighted lattice**, where each
correction concern is one **independent dimension of approximate ("fuzzy")
matching**, and WFST composition fuses those dimensions into a single joint
score. The minimal system is two-dimensional (edit-distance $\times$
language-model); the production system is genuinely $N$-dimensional.

> Notation: mathematical prose uses MathJax/LaTeX (`$…$`). PlantUML diagram
> labels use Unicode glyphs, since SVG cannot render MathJax.

---

## Table of contents

1. [The governing equation](#1-the-governing-equation)
2. [Component & data-flow map](#2-component--data-flow-map)
3. [The dimensions](#3-the-dimensions)
4. [Why the search is N-dimensional](#4-why-the-search-is-n-dimensional)
5. [The pipeline, stage by stage](#5-the-pipeline-stage-by-stage)
6. [Dependency contract](#6-dependency-contract)
7. [References](#7-references)

See also: **[dimensions.md](./dimensions.md)** (per-dimension detail),
**[dictionary-backend.md](./dictionary-backend.md)** (how the Levenshtein
automaton is fed), and **[pipeline-assembly.md](./pipeline-assembly.md)** (the
concrete `HierarchicalCorrector` API).

---

## 1. The governing equation

Given an observed, corrupted input $x$ (a misspelled/misgrammatical token
sequence), recover the intended clean text $\hat{w}$. This is Bayesian decoding —
the **noisy-channel model** ([Shannon 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x);
[Kernighan, Church & Gale 1990](https://doi.org/10.3115/997939.997975)):

$$
\hat{w} \;=\; \arg\max_{w} P(w \mid x)
        \;=\; \arg\max_{w} \; \underbrace{P(x \mid w)}_{\text{channel}} \cdot \underbrace{P(w)}_{\text{prior}}
$$

- $P(x \mid w)$ — the **channel / error model**: how the surface form was
  corrupted. Realized by the **orthographic** (edit-distance) and **phonetic**
  (sound-alike) dimensions.
- $P(w)$ — the **language prior**: how plausible $w$ is as language. Realized by
  the **contextual** (n-gram), **semantic** (embedding), and **syntactic**
  (grammar) dimensions.

**Every independent factor in that product is one dimension of approximate
matching** — the precise content of "2+ dimensions."

---

## 2. Component & data-flow map

Five crates collaborate. Each contributes one or more *dimensions*; lling-llang
is the shared WFST substrate that composes them.

![Five-crate component and data-flow map: libdictenstein supplies the Dictionary substrate, liblevenshtein the edit-distance automaton, libgrammstein the Modified Kneser-Ney n-gram plus subword embeddings, lling-llang the WFST lattice host, and duallity the WFST adapters; arrows show the acyclic dependency DAG and the artifacts flowing into a correction.](../../diagrams/correction-component-map.svg)

- **libgrammstein** — the *factory*: from the Google-Books corpora it builds the
  vocabulary, the **Modified Kneser-Ney** n-gram model, and subword embeddings.
- **libdictenstein** — the *dictionary substrate*: the `Dictionary` /
  `DictionaryNode` traversal traits and `PersistentVocabARTrie`.
- **liblevenshtein** — the *edit-distance engine*: a Levenshtein automaton
  (`Transducer`) that intersects a query with any `Dictionary`, plus phonetic
  encoders.
- **lling-llang** — the *WFST host*: `Lattice`, semirings, the `CorrectionLayer`
  stack, and the `viterbi`/`nbest` decoders.
- **duallity** — the *WFST adapters* (`LevenshteinWfst`, `compose`) that bridge
  liblevenshtein automata into lling-llang composition.

---

## 3. The dimensions

A *dimension* is an independent similarity/plausibility axis with its own metric
and engine. They are **heterogeneous** — one is a true string metric, another a
likelihood field, another a hard constraint — and a **semiring** is what unifies
them.

![The N heterogeneous dimensions of approximate matching — orthographic edit distance, phonetic sound-alike, syntactic CFG membership, contextual n-gram likelihood, and semantic embedding similarity — each fused into one joint score by a semiring whose product combines factors along a path and whose sum selects among competing paths.](../../diagrams/correction-dimensions.svg)

| # | Dimension | Question | Engine · type |
|---|-----------|----------|---------------|
| 1 | **Orthographic** | "what real word is this a typo of?" | liblevenshtein automaton $\cap$ dictionary → `LevenshteinCorrectionLayer` |
| 2 | **Phonetic** | "what real word does this *sound* like?" | liblevenshtein Zompist rules → `PhoneticRescoreLayer`; libgrammstein `PhoneticEmbedding` |
| 3 | **Syntactic** | "is this token sequence grammatical?" | lling-llang `CfgFilterLayer` (Earley-over-lattice) |
| 4 | **Contextual** | "which word fits its neighbors?" | libgrammstein n-gram, **Modified Kneser-Ney** → `LanguageModelLayer` |
| 5 | **Semantic** | "which candidate means the right thing / handles OOV?" | libgrammstein `SubwordEmbedding`, fused in `HybridLanguageModel` |

**Definitions.** *Modified Kneser-Ney (MKN)* — an n-gram smoothing estimator
with order-specific absolute discounts $D_1, D_2, D_{3+}$ and interpolation to
lower orders via continuation counts ([Chen & Goodman 1999](https://doi.org/10.1006/csla.1999.0128)).
*Subword embedding* — word vectors composed from hashed character n-grams,
degrading gracefully for out-of-vocabulary words ([Bojanowski et al. 2017](https://doi.org/10.1162/tacl_a_00051)).
*Levenshtein automaton* — a finite automaton whose language is exactly the
strings within edit distance $k$ of a query, simulated on the fly and intersected
with a dictionary ([Schulz & Mihov 2002](https://doi.org/10.1007/s10032-002-0082-8)).

Full per-dimension detail — weights, semirings, and the `HybridLanguageModel`
interpolation strategies — is in **[dimensions.md](./dimensions.md)**.

---

## 4. Why the search is N-dimensional

A **weighted finite-state transducer** is
$T = (\Sigma, \Delta, Q, I, F, E, \lambda, \rho)$ over a semiring $K$
([Mohri, Pereira & Riley 2002](https://doi.org/10.1006/csla.2001.0184)).
**Composition** $T = T_1 \circ T_2$ builds a machine whose **states are pairs**
$(q_1, q_2)$ — a Cartesian product of the two state spaces. Searching it is
dynamic programming over a 2-D grid; stacking $T_n$ makes it an $n$-D grid.

![WFST composition builds a product automaton whose states are tuples (q1,q2); the path weight is the semiring product of the per-dimension weights, and the best correction is the shortest path (Tropical) or max-probability path (Log), extracted by Viterbi or n-best with beam pruning.](../../diagrams/correction-wfst-composition.svg)

- **Path weight** $w(p) = \bigotimes_{i} w(e_i)$ — the semiring product of every
  dimension's contribution along the path.
- **Best correction** — the shortest path (Tropical, $\oplus = \min,\ \otimes = +$)
  or most-probable path (Log, $\oplus = \text{logsumexp},\ \otimes = +$),
  computed by the single generalized shortest-distance algorithm; `viterbi`,
  `nbest`, and `beam_search` are its specializations
  ([Mohri 2009](https://doi.org/10.1007/978-3-642-01492-5_6)).
- **Tractability** — lazy composition + beam search keep the product
  $\prod_i |T_i|$ of dimension sizes from exploding. Global optimality does **not**
  compose from per-layer optima, which is why beam / k-best is used — a
  Pareto-frontier phenomenon.

---

## 5. The pipeline, stage by stage

![The correction pipeline: tokenize into an input lattice, expand it with edit-distance and phonetic candidates, prune with the CFG filter, reweight with the language-model layer, then collapse to the joint argmax with Viterbi.](../../diagrams/correction-pipeline.svg)

The mechanism in one line: **layer 1 expands the lattice, layers 2–3
reweight/prune it, Viterbi collapses it** — multi-dimensional dynamic
programming over a product automaton.

1. **Tokenize → input lattice.** Each token becomes one original edge
   $i \rightarrow i{+}1$.
2. **`LevenshteinCorrectionLayer` (DIM 1).** For every out-of-vocabulary token,
   the liblevenshtein automaton emits candidates within edit distance $k$, each
   added as a weighted arc. The automaton walks the **persistent vocabulary trie
   in place** — no in-RAM materialization — because the vocabulary's node handle
   descends the full depth of the trie; see
   [dictionary-backend.md](./dictionary-backend.md).
3. **`CfgFilterLayer` (DIM 3, optional).** An Earley parse over the lattice drops
   or penalizes ungrammatical paths. Active only when a grammar is supplied.
4. **`LanguageModelLayer` (DIM 4+5).** libgrammstein's `GrammsteinLanguageModel`
   rescores each arc: $w' = (1-\lambda)\,w + \lambda\,\bigl(-\log P(\text{word} \mid \text{context})\bigr)$,
   where the score fuses the MKN n-gram with the subword embedding.
5. **`viterbi` / `nbest`.** The joint-optimal correction is the shortest path.

The concrete assembly (`HierarchicalCorrector::from_ngram_model(…).correct(text)`)
is documented in **[pipeline-assembly.md](./pipeline-assembly.md)**.

---

## 6. Dependency contract

The graph is **one-directional and acyclic**:

```
libgrammstein → { lling-llang(opt), liblevenshtein, libdictenstein }
duallity      → { liblevenshtein, lling-llang }
lling-llang   → { liblevenshtein(opt), libdictenstein(opt) }
```

lling-llang has **no** libgrammstein dependency. libgrammstein feeds the LM
dimensions to lling-llang two ways: (a) the string-typed `LanguageModel` trait
(`score_sequence`, `score_continuation`) consumed by `LanguageModelLayer`; and
(b) a composable `NgramTransducer` exported via `src/integration/wfst_export.rs`
(with backoff $\varepsilon$-arcs) for ASR-style cascade composition. Note that
**Modified Kneser-Ney lives in libgrammstein**; lling-llang's own `asr/ngram.rs`
uses Katz backoff — a different estimator.

---

## 7. References

- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System
  Technical Journal. DOI: [10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
- Kernighan, Church & Gale (1990). *A Spelling Correction Program Based on a
  Noisy Channel Model.* COLING. DOI: [10.3115/997939.997975](https://doi.org/10.3115/997939.997975)
- Schulz, K. & Mihov, S. (2002). *Fast String Correction with Levenshtein
  Automata.* IJDAR. DOI: [10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)
- Mohri, Pereira & Riley (2002). *Weighted Finite-State Transducers in Speech
  Recognition.* Computer Speech & Language. DOI: [10.1006/csla.2001.0184](https://doi.org/10.1006/csla.2001.0184)
- Mohri, M. (2009). *Weighted Automata Algorithms.* In *Handbook of Weighted
  Automata*, Springer. DOI: [10.1007/978-3-642-01492-5_6](https://doi.org/10.1007/978-3-642-01492-5_6)
- Chen, S. & Goodman, J. (1999). *An Empirical Study of Smoothing Techniques for
  Language Modeling.* Computer Speech & Language. DOI: [10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)
- Bojanowski et al. (2017). *Enriching Word Vectors with Subword Information.*
  TACL. DOI: [10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)
