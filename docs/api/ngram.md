# NgramModel API Reference

`NgramModel<D>` is libgrammstein's n-gram language model: **Modified Kneser-Ney** smoothing over
a **trie-backed** count store, queried in log space. It is trained by `TrainerBuilder`, queried
with `log_prob` / `sentence_log_prob` / `count`, and persisted in four different formats
depending on how portable you need the result to be.

> **Scope.** Source of truth: [`src/ngram/mod.rs`](../../src/ngram/mod.rs),
> [`src/ngram/model.rs`](../../src/ngram/model.rs),
> [`src/ngram/trainer.rs`](../../src/ngram/trainer.rs),
> [`src/ngram/entry.rs`](../../src/ngram/entry.rs). For the smoothing mathematics see
> [Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md); for the key encoding see
> [Trie Storage](../components/ngram/trie-storage.md); for the training workflow see the
> [N-gram Training guide](../training/ngram.md).

## Exports

```rust
use libgrammstein::ngram::{
    // model + statistics
    NgramModel, NgramEntry, NgramEntrySnapshot, NgramTrie, IterableDictionary,
    // training
    TrainerBuilder, NgramTrainer, TrainingConfig, TrainingProgress, TrainingStats, VocabularyMode,
    // convenience aliases
    SerializableNgramModel, PathMapNgramModel,
};
use libgrammstein::ngram::smoothing::KneserNeySmoothing;

#[cfg(feature = "serde-extras")]
use libgrammstein::ngram::{PortableNgramModel, PortableVocabulary};
```

> **There is no `NgramModel::train`.** Training always goes through `TrainerBuilder` (or
> `NgramTrainer`), which owns the counting passes and hands back a finished model.
> `NgramModel::new(trie, smoothing, vocab_size, total_count)` exists but is the low-level
> constructor the trainer itself calls.

## The type parameter `D`, and its bounds

`D` is the dictionary backend that stores the n-gram counts. **The bound is not uniform across
the API** — it tightens as you ask for more:

| Operation | Required bound on `D` |
|---|---|
| Query (`log_prob`, `count`, `order`, …) | `MappedDictionary<Value = NgramEntry>` |
| Training (`TrainerBuilder`, `NgramTrainer`) | `MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync + 'static` |
| `to_portable`, `save_portable` | `MappedDictionary<Value = NgramEntry> + IterableDictionary` |
| `load_portable` | `MutableMappedDictionary<Value = NgramEntry>` |
| `save`, `load` (direct bincode) | `MappedDictionary<Value = NgramEntry> + Serialize + DeserializeOwned` |

The consequence worth internalizing: the **read-only** `DoubleArrayTrieChar` backend satisfies
the *query* bound (and `IterableDictionary`) but **not** `MutableMappedDictionary`, so it can be
queried and exported but never trained into or `load_portable`d — it is bulk-built by
`from_portable_static` / `load_static_portable` instead.

| Backend (from `libdictenstein`) | Train | Query | Portable I/O | Best for |
|---|---|---|---|---|
| `dynamic_dawg::char::DynamicDawgChar<NgramEntry>` | yes | yes | yes (+ direct `save`/`load`) | general purpose; the only backend with full serde |
| `pathmap::PathMapDictionary<NgramEntry>` | yes | yes | yes | memory sharing; lling-llang lattice integration |
| `persistent_artrie::char::SharedCharARTrie<NgramEntry>` | yes | yes | yes | crash-safe, disk-backed training over huge corpora |
| `double_array_trie::char::DoubleArrayTrieChar<NgramEntry>` | **no** | yes | export only | fast, immutable inference |

Two aliases cover the common cases:

```rust
pub type SerializableNgramModel = NgramModel<DynamicDawgChar<NgramEntry>>;
pub type PathMapNgramModel      = NgramModel<PathMapDictionary<NgramEntry>>;
```

## Training

`TrainerBuilder` is the fluent entry point. **`train` consumes both the builder and the
reader** (the reader is moved, not borrowed, so the trainer can stream it across Rayon workers).

```rust
impl<D> TrainerBuilder<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync + 'static,
{
    pub fn new(dictionary: D) -> Self;
    pub fn order(self, order: usize) -> Self;                       // default 5
    pub fn batch_size(self, size: usize) -> Self;                   // default 10_000
    pub fn min_word_freq(self, freq: u64) -> Self;                  // default 1
    pub fn tokenizer(self, tokenizer: Tokenizer) -> Self;
    pub fn with_vocabulary_path(self, path: PathBuf) -> Self;       // varint-indexed keys, persisted
    pub fn with_vocabulary(self, vocab: SharedVocabARTrie) -> Self; // reuse an existing vocabulary
    pub fn build(self) -> NgramTrainer<D>;
    pub fn train<R: CorpusReader + 'static>(self, reader: R) -> Result<NgramModel<D>>;
}
```

```rust
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libgrammstein::corpus::PlaintextReader;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
    .order(5)             // 5-gram model
    .batch_size(10_000)   // sentences per parallel batch
    .train(PlaintextReader::from_file("corpus.txt")?)?;   // reader is MOVED
# Ok::<(), libgrammstein::Error>(())
```

Training runs three phases: parallel n-gram counting over prefetched batches, a continuation-count
pass ($`N_{1+}(\bullet, w)`$ and $`N_{1+}(h, \bullet)`$, the statistics MKN needs), and estimation
of the discounts from the corpus count-of-counts.

### `TrainingConfig` and `VocabularyMode`

`TrainerBuilder` is sugar over `NgramTrainer::new(dictionary, config)`. Reach for the config
directly when you want progress reporting (`train_with_progress` lives on `NgramTrainer`, not on
the builder):

```rust
pub struct TrainingConfig {
    pub order: usize,                    // default 5
    pub batch_size: usize,               // default 10_000
    pub min_word_freq: u64,              // default 1
    pub vocabulary_mode: VocabularyMode, // default Legacy
}

pub enum VocabularyMode {
    Legacy,                  // pipe-separated keys "the|quick|brown"  (default; deprecated encoding)
    Create(PathBuf),         // build a varint-indexed vocabulary, persisted at this path
    Shared(SharedVocabARTrie), // reuse an existing vocabulary (e.g. the Google Books importer's)
}
```

Progress is reported over a `crossbeam_channel`; `train_with_progress` also consumes `self` and
the reader:

```rust
use crossbeam_channel::bounded;
use libgrammstein::ngram::{NgramTrainer, TrainingConfig, TrainingProgress};

let (tx, rx) = bounded::<TrainingProgress>(100);
std::thread::spawn(move || {
    while let Ok(p) = rx.recv() {
        println!(
            "sentences: {}  n-grams: {}  elapsed: {:.1}s",
            p.sentences_processed, p.ngrams_counted, p.elapsed_secs
        );
    }
});

let trainer = NgramTrainer::new(DynamicDawgChar::<NgramEntry>::new(), TrainingConfig::new(5));
let model = trainer.train_with_progress(PlaintextReader::from_file("corpus.txt")?, tx)?;
# Ok::<(), libgrammstein::Error>(())
```

`TrainingStats` (atomic counters, readable during training via `NgramTrainer`) exposes
`sentences_processed()`, `ngrams_counted()`, and `tokens_processed()`.

## Query methods

```rust
impl<D: MappedDictionary<Value = NgramEntry>> NgramModel<D> {
    pub fn log_prob(&self, word: &str, context: &[&str]) -> f64;
    pub fn sentence_log_prob(&self, tokens: &[&str]) -> f64;
    pub fn count(&self, tokens: &[&str]) -> u64;
    pub fn in_vocabulary(&self, word: &str) -> bool;
    pub fn order(&self) -> usize;
    pub fn vocab_size(&self) -> usize;
    pub fn total_count(&self) -> u64;
    pub fn ngram_count(&self) -> usize;
    pub fn oov_log_prob(&self) -> f64;
    pub fn smoothing(&self) -> &KneserNeySmoothing;
    pub fn trie(&self) -> &NgramTrie<D>;
}
```

| Method | Returns | Description |
|---|---|---|
| `log_prob(word, context)` | `f64` | $`\log \mathbb{P}_{\mathrm{MKN}}(w \mid h)`$ in **nats**. Always finite and $`\leq 0`$. Uses at most `order - 1` context words (the *rightmost* ones). |
| `sentence_log_prob(tokens)` | `f64` | $`\sum_i \log \mathbb{P}(w_i \mid h_i)`$, sliding the context window over the sentence. `0.0` for an empty slice. |
| `count(tokens)` | `u64` | Raw training count $`c(\text{tokens})`$ of that exact n-gram; `0` if unseen. |
| `in_vocabulary(word)` | `bool` | Whether the *unigram* was seen in training. |
| `order()` | `usize` | Maximum n-gram order $`n`$. |
| `vocab_size()` | `usize` | $`\lvert V \rvert`$ — distinct unigrams. |
| `total_count()` | `u64` | Total tokens processed during training. |
| `ngram_count()` | `usize` | Number of n-grams stored (all orders). |
| `oov_log_prob()` | `f64` | $`-\log \lvert V \rvert`$, the uniform floor. $`-\infty`$ iff `vocab_size() == 0`. |
| `smoothing()` | `&KneserNeySmoothing` | The fitted discounts and the $`N_{1+}(\bullet,\bullet)`$ denominator. |

`NgramModel` is `Clone`, `Send`, and `Sync`; the query path takes no locks, so one model can be
scored from many threads at once.

```rust
// P(fox | quick brown) — the context is the words BEFORE the target.
let log_p = model.log_prob("fox", &["quick", "brown"]);

// P(the) — the unigram case is an empty context, not a special method.
let unigram = model.log_prob("the", &[]);

// Counts, vocabulary membership, and the OOV floor.
let bigram_count = model.count(&["quick", "brown"]);
let known = model.in_vocabulary("fox");
let floor = model.oov_log_prob();          // = -(vocab_size as f64).ln()
```

### Smoothing: backoff, not zeros

`log_prob` never returns $`-\infty`$. An unseen n-gram backs off — dropping the **oldest**
(leftmost) context word at each level — down to a unigram base case whose OOV branch returns the
strictly-positive $`1/\lvert V \rvert`$.

![Modified Kneser-Ney backoff recursion](../diagrams/mkn-backoff.svg)

The discounts are fitted from the corpus count-of-counts at the end of training; when the counts
are too sparse to fit them, the model falls back to the fixed defaults
$`D_1 = 0.75,\ D_2 = 0.85,\ D_{3+} = 0.95`$ (`KneserNeySmoothing::default_discounts`). The
struct's fields are private; `total_bigram_types()` is the one public accessor, and
`from_counts(n1, n2, n3, n4)` builds the parameters directly. The full derivation is in
[Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md).

## `NgramEntry` — the stored statistics

Each n-gram maps to one `NgramEntry`. Its three fields are **atomics**, so parallel corpus
workers update them without locks; the fields are private and accessed through methods.

```rust
pub struct NgramEntry { /* count: AtomicU64, continuation_count: AtomicU32, unique_continuations: AtomicU32 */ }

impl NgramEntry {
    pub fn new(count: u64) -> Self;
    pub fn with_stats(count: u64, continuation_count: u32, unique_continuations: u32) -> Self;

    pub fn count(&self) -> u64;                   // c(h·w)          — raw corpus count
    pub fn continuation_count(&self) -> u32;      // N1+(•, ngram)   — distinct preceding contexts
    pub fn unique_continuations(&self) -> u32;    // N1+(ngram, •)   — distinct following words

    pub fn increment(&self);                      // all four take &self: atomic, lock-free
    pub fn increment_by(&self, amount: u64);
    pub fn increment_continuation(&self);
    pub fn increment_unique_continuations(&self);
    pub fn set_continuation_count(&self, value: u32);
    pub fn set_unique_continuations(&self, value: u32);
}
```

`NgramEntrySnapshot` is the plain (`Copy`) counterpart — public `count`, `continuation_count`,
and `unique_continuations` fields — used for crossing thread boundaries and for serialization.
`From` converts both ways.

## Persistence (feature `serde-extras`)

Four save/load paths, differing in what they require of `D`:

| Method | Bound on `D` | Notes |
|---|---|---|
| `save(path)` / `load(path)` | `+ Serialize + DeserializeOwned` | Direct bincode of the whole model. In practice: `DynamicDawgChar` only. |
| `save_portable(path)` | `+ IterableDictionary` | Backend-agnostic: dumps `(key, snapshot)` pairs. |
| `load_portable(path, factory)` | `MutableMappedDictionary` | Rebuilds into a fresh backend produced by the `FnOnce() -> D` factory. |
| `save_portable_with_vocabulary(path, &vocab)` | `+ IterableDictionary` | Self-contained: embeds the varint vocabulary so keys decode back to words. |
| `load_static_portable(path)` | *(inherent on `DoubleArrayTrieChar`)* | Bulk-builds the fast, immutable backend from a portable snapshot. |

```rust
use libgrammstein::ngram::{NgramEntry, NgramModel};
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;

// Direct bincode (requires D: Serialize + DeserializeOwned).
model.save("model.bin")?;
let model: NgramModel<DynamicDawgChar<NgramEntry>> = NgramModel::load("model.bin")?;

// Portable: works with ANY backend; the factory supplies the empty target.
model.save_portable("model.portable.bin")?;
let model: NgramModel<DynamicDawgChar<NgramEntry>> =
    NgramModel::load_portable("model.portable.bin", DynamicDawgChar::new)?;

// Static: bulk-build the read-only Double-Array Trie for fast inference.
let fast: NgramModel<DoubleArrayTrieChar<NgramEntry>> =
    NgramModel::load_static_portable("model.portable.bin")?;
# Ok::<(), libgrammstein::Error>(())
```

`to_portable()` returns the intermediate `PortableNgramModel { entries, max_order, vocab_size,
total_count, smoothing, vocabulary }` if you want to route the snapshot somewhere other than a
file; `from_portable_static(portable)` is the in-memory counterpart of `load_static_portable`.

## Complete workflow

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libgrammstein::scoring::Perplexity;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

fn main() -> libgrammstein::Result<()> {
    // 1. Train a 5-gram Modified Kneser-Ney model (the reader is moved).
    let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
        .order(5)
        .train(PlaintextReader::from_file("corpus.txt")?)?;

    // 2. Query a conditional log-probability.
    println!("log P(world | hello) = {:.4}", model.log_prob("world", &["hello"]));

    // 3. Evaluate on held-out text.
    let dev = PlaintextReader::from_file("dev.txt")?;
    let result = Perplexity::new(&model).corpus_perplexity(&dev)?;
    println!("{result}");

    // 4. Persist (feature: serde-extras).
    model.save("model.bin")?;
    Ok(())
}
```

## Performance notes

- **Backend choice dominates.** `DynamicDawgChar` for training and general use;
  `DoubleArrayTrieChar` for read-only inference (bulk-built, faster look-ups, no writes);
  `SharedCharARTrie` when the corpus exceeds RAM and training must be crash-safe.
- **A query is $`O(n)`$ look-ups** for an order-$`n`$ model — one per backoff level, each linear
  in the key length. Deeper orders cost proportionally more per query *and* store far more
  n-grams.
- **`batch_size` tunes the parallel granularity**, not correctness; the counting pass is
  lock-free (atomics in `NgramEntry`), so throughput scales with cores.
- **Vocabulary-indexed keys** (`with_vocabulary_path`) produce compact varint keys and avoid the
  legacy pipe-separator's corruption hazard when a token itself contains `|`.

## See also

- [N-gram overview](../components/ngram/overview.md) — concepts and the query surface
- [Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md) — the smoothing mathematics
- [Trie Storage](../components/ngram/trie-storage.md) — key encoding and the vocabulary
- [Scoring API](scoring.md) — perplexity and sentence ranking over this model
- [HybridLanguageModel API](hybrid.md) — combining this model with subword embeddings
- [Traits API](traits.md) — `CorpusReader`, `MappedDictionary`, `IterableDictionary`
- [N-gram Training guide](../training/ngram.md) — the full training workflow
