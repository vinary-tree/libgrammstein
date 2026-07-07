# Pipeline Assembly — the `HierarchicalCorrector` API

[← hierarchical correction](./hierarchical-correction.md)

This page documents the concrete, batteries-included corrector that assembles the
dimensions of [hierarchical-correction.md](./hierarchical-correction.md) into a
runnable pipeline. It lives in
[`src/integration/corrector.rs`](../../../src/integration/corrector.rs) behind the
`lling-llang-integration` feature.

The design has two layers of API:

- a **generic, reusable** `LevenshteinCorrectionLayer<W, B, D>` — a first-class
  lling-llang `CorrectionLayer` that turns a liblevenshtein automaton into
  lattice edges; and
- a **concrete façade** `HierarchicalCorrector` — fixing the semiring to
  `TropicalWeight` and the backend to `HashMapBackend`, wiring the spelling layer,
  an optional CFG filter, and an n-gram/hybrid language-model rescorer, then
  decoding.

---

## The generic layer

```rust
pub struct LevenshteinCorrectionLayer<W, B, D> { /* one Transducer<D> + EditConfig */ }

impl<W, B, D> LevenshteinCorrectionLayer<W, B, D> {
    pub fn new(dictionary: D, config: EditConfig) -> Self;             // builds the transducer
    pub fn from_transducer(transducer: Transducer<D>, config: EditConfig) -> Self;
}
```

It implements `CorrectionLayer<W, B>` (name `"levenshtein-correction"`). The
generic bounds are exactly those liblevenshtein's `Transducer` requires — copied
verbatim from `lling-llang`'s `fuzzy_lookup`:

```rust
where
    W: Semiring + From<f64>,                       // (on the apply impl)
    B: LatticeBackend,
    D: libdictenstein::Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as libdictenstein::DictionaryNode>::Unit:
        Into<char> + TryFrom<char> + Copy + Send + Sync,
```

`apply` builds a fresh lattice, copies every original edge, and for each
out-of-vocabulary token of length $\ge$ `min_word_length` adds one deduplicated
correction edge per candidate returned by `query_with_distance`, with weight
`W::from(distance)` and `EdgeMetadata::correction(distance)`. It never mutates the
input lattice. Because it is generic over `D`, it works over *any* fully
traversable `Dictionary` — the persistent `SharedVocabARTrie` the façade walks in
place by default, or a materialized `DynamicDawgChar` for a small static word list
(see [dictionary-backend.md](./dictionary-backend.md)).

---

## The façade

```rust
pub struct CorrectorConfig {
    pub max_edit_distance: usize,          // default 2
    pub algorithm: Algorithm,              // default Damerau–Levenshtein (Transposition)
    pub max_corrections_per_word: usize,   // default 8
    pub min_word_length: usize,            // default 2
    pub keep_original: bool,               // default true
    pub k_best: usize,                     // default 1 (Viterbi); >1 runs n-best
    pub lm_weight: f64,                    // default 0.75  (the λ below)
    pub vocabulary_filename: String,       // default "vocabulary"
    pub model_filename: String,            // default "model.bin"
}

pub struct CorrectionResult { pub text: String, pub score: f64 }

impl HierarchicalCorrector {
    // in-memory, pure n-gram
    pub fn from_ngram_model<D>(vocab: SharedVocabARTrie, model: NgramModel<D>, cfg: CorrectorConfig) -> Self;
    // in-memory, n-gram or hybrid
    pub fn from_parts<D>(vocab: SharedVocabARTrie, lm: GrammsteinLanguageModel<D>, cfg: CorrectorConfig) -> Self;
    // load from a Google-Books checkpoint directory (requires the `serde-extras` feature)
    pub fn from_checkpoint(dir: &Path, cfg: CorrectorConfig) -> Result<Self, CorrectorError>;

    pub fn with_grammar(self, grammar: Grammar) -> Self;   // optional CFG filter
    pub fn correct(&self, text: &str) -> Vec<CorrectionResult>;
}
```

`correct` whitespace-tokenizes the input into a one-original-edge-per-token
lattice, applies **spelling** $\to$ (optional) **CFG filter** $\to$ **LM rescore**,
then decodes with `viterbi` (`k_best == 1`) or `nbest` (`k_best > 1`), returning
each hypothesis as `CorrectionResult { text, score }` (lower `score` is better
under the tropical semiring).

### The scoring balance

The LM rescorer interpolates the correction cost with fluency in cost space:

$$
w' \;=\; (1-\lambda)\,w \;+\; \lambda\,\bigl(-\log P(\text{word} \mid \text{context})\bigr),
\qquad \lambda = \texttt{lm\_weight}.
$$

A correction edge carries a positive edit-distance cost $w$ relative to the
zero-cost original out-of-vocabulary token, so $\lambda$ must be large enough for
fluency to overturn a wrong-but-cheap original. The default $\lambda = 0.75$ lets
fluency dominate (the desired behavior for spelling correction) while edit
distance still breaks ties among comparably fluent candidates. At production
scale the out-of-vocabulary log-probability gap is many nats, so this is
comfortably robust; the default also picks Damerau–Levenshtein so a transposition
typo (`teh`) costs $d = 1$, not $2$.

### The optional CFG filter

`with_grammar` attaches a `CfgFilterLayer` that runs between the spelling and LM
stages. Because `CfgFilterLayer<'g>` borrows its grammar (it cannot live in a
`'static` pipeline), the façade owns the `Grammar` and constructs the CFG layer
per `correct()` call. With no grammar the CFG stage is simply skipped — a
genuinely optional slot, since there is no general-purpose English CFG to default
to (English is not context-free).

---

## Worked example

From [`examples/correct_sentence.rs`](../../../examples/correct_sentence.rs) — a
tiny in-memory vocabulary and n-gram model trained on a corpus containing *"the
quick brown fox …"*:

```rust
let corrector = HierarchicalCorrector::from_ngram_model(vocab, model, CorrectorConfig::default());
let results = corrector.correct("teh quikc brwon fox");
assert_eq!(results[0].text, "the quick brown fox");
```

```
$ cargo run --example correct_sentence --features google-books,lling-llang-integration
input:  "teh quikc brwon fox"
output: "the quick brown fox"  (score 4.9835)
```

![The correction pipeline: tokenize into an input lattice, expand it with edit-distance candidates, optionally prune with the CFG filter, reweight with the language-model layer, then collapse to the joint argmax with Viterbi.](../../diagrams/correction-pipeline.svg)

---

## Feature flags & tests

- Enable with `--features lling-llang-integration` (the corrector) plus
  `google-books` if loading real checkpoints. No **new** lling-llang features are
  required: `lm-rerank` (already enabled) provides `LanguageModelLayer`, and the
  lattice / `CorrectionLayer` / `CfgFilterLayer` / `viterbi` / `nbest` /
  `TropicalWeight` / `HashMapBackend` types are un-gated.
- Verified: `cargo check --features google-books,lling-llang-integration
  --all-targets` is clean (0 warnings); the corrector unit + end-to-end tests
  pass (5/5); the full library suite is unchanged (540 passed).
