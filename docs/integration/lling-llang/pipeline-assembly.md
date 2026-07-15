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
out-of-vocabulary token of length $`\ge`$ `min_word_length` adds one deduplicated
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
lattice, applies **spelling** $`\to`$ (optional) **CFG filter** $`\to`$ **LM rescore**,
then decodes with `viterbi` (`k_best == 1`) or `nbest` (`k_best > 1`), returning
each hypothesis as `CorrectionResult { text, score }` (lower `score` is better
under the tropical semiring).

### The scoring balance

The LM rescorer interpolates the correction cost with fluency in cost space:

```math
w' \;=\; (1-\lambda)\,w \;+\; \lambda\,\bigl(-\log P(\text{word} \mid \text{context})\bigr),
\qquad \lambda = \texttt{lm_weight}.
```

A correction edge carries a positive edit-distance cost $`w`$ relative to the
zero-cost original out-of-vocabulary token, so $`\lambda`$ must be large enough for
fluency to overturn a wrong-but-cheap original. The default $`\lambda = 0.75`$ lets
fluency dominate (the desired behavior for spelling correction) while edit
distance still breaks ties among comparably fluent candidates. At production
scale the out-of-vocabulary log-probability gap is many nats, so this is
comfortably robust; the default also picks Damerau–Levenshtein so a transposition
typo (`teh`) costs $`d = 1`$, not $`2`$.

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
$ cargo run --example correct_sentence --features lling-llang-integration
input:  "teh quikc brwon fox"
output: "the quick brown fox"  (score 4.9835)
```

![The lattice correction pipeline: tokenize into an input lattice, expand it with edit-distance candidates, optionally prune with the CFG filter, reweight with the language-model layer, then collapse to the joint argmax with Viterbi.](../../diagrams/correction-pipeline.svg)

**Figure.** The lattice `correct()` pipeline this façade assembles — the same
expand → prune → reweight → collapse sequence as
[Figure 4](./hierarchical-correction.md#5-the-pipeline-stage-by-stage) in the
correction reference. *(Rendered from `docs/diagrams/correction-pipeline.puml`.)*

---

## The `GrammarCorrector` — the term-id cascade façade

A second, independent corrector in the same module realizes the noisy channel as a
$`T_{\text{lex}} \circ T_{\text{gram}}`$ **cascade over a term-id alphabet** rather than a lattice (the theory
is in [hierarchical-correction.md](./hierarchical-correction.md) §6). It lives in
[`src/integration/grammar_corrector.rs`](../../../src/integration/grammar_corrector.rs)
and shares nothing with the lattice corrector at run time — no `Lattice`, no
`CorrectionLayer`, no `viterbi` — driving liblevenshtein's `Transducer` directly and
decoding with a history-indexed beam.

### The type and its store

```rust
pub struct GrammarCorrector<D>
where
    D: MappedDictionary<Value = u64> + Clone + Send + Sync + 'static,
    <D as Dictionary>::Node: MappedDictionaryNode<Value = u64>,
    <<D as Dictionary>::Node as DictionaryNode>::Unit: VarintByteUnit,
{ /* vocabulary · count store · order · total_count · bos/eos ids · config */ }
```

`D` is the n-gram **count** store: keyed by concatenated LEB128 varints and valued by
the raw `u64` frequency — the artifact the Google-Books importer builds alongside the
vocabulary. Its node `Unit` is a `VarintByteUnit` carrier, so both the importer's
`PersistentARTrie<u64>` (`Unit = u8`) and an in-memory `DynamicDawgChar<u64>`
(`Unit = char`) qualify; the store is presented to the word-level automaton as a
`Unit = u64` dictionary by [`U64NgramView`](../../../src/ngram/u64_view.rs).

### Configuration

```rust
pub struct GrammarCorrectorConfig {
    pub max_char_edit_distance: usize,        // default 2   — T_lex radius
    pub max_word_edit_distance: usize,        // default 1   — T_gram radius
    pub max_lex_candidates_per_token: usize,  // default 8
    pub max_insertions_per_gap: usize,        // default 1
    pub algorithm: Algorithm,                 // default Transposition (Damerau)
    pub min_word_length: usize,               // default 2   — shorter tokens kept verbatim
    pub deletion_penalty: f64,                // default 3.0
    pub insertion_penalty: f64,               // default 3.0
    pub oov_penalty: f64,                     // default 6.0
    pub lm_weight: f64,                       // default 1.0 — the λ on −log S
    pub backoff_alpha: f64,                   // default 0.4 — stupid-backoff α
    pub beam_width: usize,                    // default 16
    pub k_best: usize,                        // default 1
    pub use_phonetics: bool,                  // default = cfg!(feature = "phonetic-correction")
    pub bos_token: String,                    // default "<s>"
    pub eos_token: String,                    // default "</s>"
}
```

The aggressive word-level moves (insertion, deletion) are enabled but penalized, so
they win only when the source model clearly prefers them.

### Constructors and queries

```rust
impl<D> GrammarCorrector<D> /* bounds as above */ {
    // supply the corpus token total N = Σ f(v) explicitly
    pub fn new(vocab: SharedVocabARTrie, store: D, order: usize, total_count: u64,
               cfg: GrammarCorrectorConfig) -> Self;
    // derive N by summing the stored unigram counts (scans depth-1 edges once)
    pub fn from_store(vocab: SharedVocabARTrie, store: D, order: usize,
                      cfg: GrammarCorrectorConfig) -> Self;

    pub fn lex_candidates(&self, token: &str) -> Vec<LexCandidate>;                 // T_lex
    pub fn grammar_neighbors(&self, ids: &[u64], k: usize) -> Vec<GrammarNeighbor>; // T_gram
    pub fn correct(&self, text: &str) -> Vec<GrammarCorrection>;                    // beam decode
}
```

`from_store` is convenient for tests and small stores; prefer `new` with a
precomputed `total_count` at web scale, since `from_store` scans every unigram edge
once. The result types are `LexCandidate { term, id, cost }`,
`GrammarNeighbor { ids, distance, frequency }`, and `GrammarCorrection { text, cost }`
(lower `cost` is better). Boundary markers are resolved against the vocabulary at
construction; a store trained without `<s>` / `</s>` silently skips boundary modeling.

### Worked example

From [`examples/grammar_correct.rs`](../../../examples/grammar_correct.rs) — a tiny
in-memory count store (a `DynamicDawg<u64>` of LEB128-varint keys) over a corpus
containing *"the quick brown fox …"*:

```rust
let corrector =
    GrammarCorrector::from_store(vocabulary, store, order, GrammarCorrectorConfig::default());
let best = corrector.correct("the quick fox");        // a missing word
assert_eq!(best[0].text, "the quick brown fox");
```

```
$ cargo run --example grammar_correct --features phonetic-correction
  [missing word]
    in:  "the quick fox"
    out: "the quick brown fox"  (cost 4.070)
  [extraneous word]
    in:  "the the quick brown fox"
    out: "the quick brown fox"  (cost 4.070)
  [phonetic sound-alike]
    in:  "please answer the fone"
    out: "please answer the phone"  (cost 3.616)
```

The complete error taxonomy and the model (stupid backoff, the beam decode) are in
[hierarchical-correction.md](./hierarchical-correction.md) §6.

---

## The `ShardedGrammarCorrector` — the sharded cascade façade

The same $`T_{\text{lex}} \circ T_{\text{gram}}`$ cascade, scored against the **sharded** Google-Books n-gram
corpus instead of a single in-memory store. It is a thin delegating newtype over the
*identical* `GrammarCore` beam decoder that backs `GrammarCorrector` (above), differing
only in its n-gram **view source**: an `NgramViewSource` that routes each lookup to the one
shard physically holding it and opens that shard **read-only**. The model, the routing /
co-location argument, and the read-only-mode safety are in
[hierarchical-correction.md](./hierarchical-correction.md) §6.9; the full design record is
[multi-shard-grammar-corrector.md](../multi-shard-grammar-corrector.md). It lives in
[`src/integration/sharded_grammar_corrector.rs`](../../../src/integration/sharded_grammar_corrector.rs).

### Constructors and queries

```rust
impl ShardedGrammarCorrector {
    // the corpus token total N = Σ f(v) is INJECTED — there is NO `from_store`, because a
    // sharded corpus persists no single total and its unigrams span every shard
    pub fn new(coordinator: Arc<ShardCoordinator>, vocabulary: SharedVocabARTrie,
               order: usize, total_count: u64, config: GrammarCorrectorConfig) -> Self;

    pub fn lex_candidates(&self, token: &str) -> Vec<LexCandidate>;                 // T_lex
    // T_gram, anchored — the query's own first-token/length shard (the hot-path default)
    pub fn grammar_neighbors(&self, ids: &[u64], k: usize) -> Vec<GrammarNeighbor>;
    // T_gram, fanout — every shard, merged by min distance / max frequency (batch / offline)
    pub fn grammar_neighbors_fanout(&self, ids: &[u64], k: usize) -> Vec<GrammarNeighbor>;
    pub fn correct(&self, text: &str) -> Vec<GrammarCorrection>;                    // beam decode
}
```

The `GrammarCorrectorConfig`, `LexCandidate`, `GrammarNeighbor`, and `GrammarCorrection`
types are shared verbatim with the single-store corrector — only the store backend and the
extra `grammar_neighbors_fanout` differ. There is **no** `from_store`: `total_count` must be
passed to `new` (the importer, which accumulates it, supplies it).

### Deployment notes

- Open the coordinator with **`max_open_shards = 0`** (all shards resident). The read-only
  query path never evicts, so a bounded cap would only accumulate touched shards without
  reclaiming them; the constructor logs a warning if it is handed a positive cap.
- Serve a **cleanly-finalized** corpus — WAL replay on shard open is a no-op only when the
  corpus is finalized — and query with the **same `ShardConfig` and CPU count** used at
  import, or `CpuProportional` routing lands on the wrong shard. See
  [hierarchical-correction.md](./hierarchical-correction.md) §6.9 (deployment contract) and
  [multi-shard-grammar-corrector.md](../multi-shard-grammar-corrector.md) §13.

### No runnable snippet

Unlike the single-store corrector, the sharded façade needs an on-disk shard store, so there
is no inline example. Its end-to-end behavior — fanout soundness + completeness, anchored
narrowing, decoder parity with the single store, and write-free read-only mode — is exercised
by [`tests/sharded_grammar_corrector_proptest.rs`](../../../tests/sharded_grammar_corrector_proptest.rs)
and reconstructed in [multi-shard-grammar-corrector.md](../multi-shard-grammar-corrector.md).

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
- The `GrammarCorrector` is gated by the same `lling-llang-integration` feature; its
  articulatory $`T_{\text{lex}}`$ path additionally requires `phonetic-correction`. It uses
  neither lling-llang types nor `google-books` — its count store is any
  `VarintByteUnit` dictionary. Its behavior is covered by the inline tests in
  [`src/integration/grammar_corrector.rs`](../../../src/integration/grammar_corrector.rs)
  and the soundness / completeness properties in
  [`tests/grammar_corrector_proptest.rs`](../../../tests/grammar_corrector_proptest.rs)
  (`cargo test --features lling-llang-integration --test grammar_corrector_proptest`).
- The **`ShardedGrammarCorrector`** additionally requires `google-books` (for the shard
  coordinator) on top of `lling-llang-integration`; it uses no lling-llang lattice types.
  Its property / differential suite — fanout soundness + completeness, anchored narrowing,
  single-store parity, and write-free read-only mode — is
  [`tests/sharded_grammar_corrector_proptest.rs`](../../../tests/sharded_grammar_corrector_proptest.rs)
  (`cargo test --features "lling-llang-integration google-books" --test sharded_grammar_corrector_proptest`).
