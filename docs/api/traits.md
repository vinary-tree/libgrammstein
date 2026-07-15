# Traits API Reference

libgrammstein is generic over two things: **where the corpus comes from** (`CorpusReader`) and
**where the counts are stored** (the `libdictenstein` dictionary traits). Everything else — the
n-gram model, the trainer, the hybrid scorer, perplexity — is written against those two
abstractions. This page documents them, the exact bound each API demands, and the
feature-gated `LanguageModel` trait that lets a libgrammstein model rescore an
[lling-llang](https://github.com/f1r3fly-io/lling-llang) WFST lattice.

> **Scope.** Source of truth: [`src/lib.rs`](../../src/lib.rs) (the prelude and `Error`),
> [`src/corpus/reader.rs`](../../src/corpus/reader.rs) (`CorpusReader`, `Document`),
> [`src/ngram/trie.rs`](../../src/ngram/trie.rs) (`IterableDictionary`),
> [`src/integration/lling_llang.rs`](../../src/integration/lling_llang.rs)
> (`GrammsteinLanguageModel`), and `libdictenstein`'s `src/lib.rs` for the dictionary traits.

## The prelude

`libgrammstein::prelude` is deliberately small — it carries the types you need to *query* a
model, not to *build* one:

```rust
use libgrammstein::prelude::*;
// brings into scope:
//   CorpusReader                    (trait)
//   Error, Result                   (error handling)
//   NgramEntry, NgramModel          (the statistical core)
//   Perplexity                      (evaluation)
```

Plus, when the corresponding feature is enabled: `GrammsteinLanguageModel`
(`lling-llang-integration`); the acoustic, neural, RAG, topic, code, and LaTeX type sets
(`acoustic`, `candle-model`, `neural-rescore`, `rag`, `code`, `latex`).

> **What the prelude does *not* export.** `HybridLanguageModel`, `HybridConfig`,
> `InterpolationStrategy`, `TrainerBuilder`, `NgramTrainer`, `EmbeddingTrainerBuilder`,
> `SubwordEmbedding`, and `SentenceScorer` are **not** in it. Import them from their modules:

```rust
use libgrammstein::ngram::{TrainerBuilder, NgramTrainer, TrainingConfig};
use libgrammstein::embedding::{EmbeddingTrainerBuilder, SubwordEmbedding};
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy};
use libgrammstein::scoring::SentenceScorer;
```

The dictionary backends live in the **`libdictenstein`** crate, which you depend on directly:

```rust
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::{MappedDictionary, MutableMappedDictionary};   // the traits themselves
```

## `CorpusReader`

The streaming corpus abstraction. It has **four** methods (two required, two with defaults), and
every returned iterator is `Send` so a trainer can hand batches to Rayon workers.

```rust
pub trait CorpusReader: Send + Sync {
    /// Iterate over documents (articles, chapters, …). Yielded once, in order, never cached.
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + Send + '_>;

    /// Iterate over sentences across all documents.
    fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_>;

    /// Estimated total tokens, for progress reporting. `None` if not cheaply computable.
    fn estimated_tokens(&self) -> Option<usize> { None }

    /// Number of documents, if known.
    fn document_count(&self) -> Option<usize> { None }
}
```

`CorpusReader` is implemented for `Box<dyn CorpusReader>`, so a boxed trait object is itself a
reader — which is what makes the ownership-based trainer APIs usable with dynamic dispatch.

> **Trainers take the reader by value.** `TrainerBuilder::train(reader)` and
> `EmbeddingTrainerBuilder::train(reader)` **move** it (`R: CorpusReader + 'static`). To train two
> models from one corpus, construct two readers. `Perplexity::corpus_perplexity(&reader)`, by
> contrast, only borrows.

### `Document`

```rust
pub struct Document {
    pub id: Option<String>,        // optional identifier
    pub title: Option<String>,     // optional title
    pub content: String,           // the text itself — NOT pre-split into sentences
    pub source: Option<PathBuf>,   // originating file, if any
}

impl Document {
    pub fn new(content: String) -> Self;
    pub fn with_title(title: String, content: String) -> Self;
}
```

### Bundled implementations

| Reader | Constructors |
|---|---|
| `PlaintextReader` | `from_file(path)`, `from_directory(path)`, `from_paths(Vec<PathBuf>)` — each returns `std::io::Result<Self>` (`from_paths` is infallible). Builders: `with_normalizer`, `with_tokenizer`, `with_extensions`. |
| `GutenbergReader` | `from_file(path)`, `from_directory(dir)`, `from_paths(Vec<PathBuf>)`. Strips Project Gutenberg header/footer boilerplate. |
| `WikipediaReader` | `new(path)`, `with_config(path, WikipediaConfig)`; and, with feature `http-corpus`, `from_url(url, config)` and `from_url_with_strategy(…)`. Streams bz2-compressed XML dumps. |

```rust
use libgrammstein::corpus::{CorpusReader, PlaintextReader, WikipediaConfig, WikipediaReader};

// A file, or every matching file under a directory.
let reader = PlaintextReader::from_file("corpus.txt")?;
let reader = PlaintextReader::from_directory("corpus/")?;

// Wikipedia, filtered.
let config = WikipediaConfig {
    namespace_filter: vec![0],   // main namespace only (the default)
    skip_redirects: true,        // default
    max_articles: Some(10_000),
    min_text_length: 100,        // default: skip near-empty pages
};
let reader = WikipediaReader::with_config("enwiki-latest.xml.bz2", config)?;

for sentence in reader.sentences() {
    println!("{sentence}");
}
# Ok::<(), std::io::Error>(())
```

`PlaintextReader::sentences()` runs each document's text through its `Tokenizer`, which
**lowercases by default** — worth knowing when you compare tokens against a trained model. There
is no `from_string` constructor; write to a file, or implement the trait yourself:

```rust
use libgrammstein::corpus::{CorpusReader, Document};

struct InMemoryCorpus {
    sentences: Vec<String>,
}

impl CorpusReader for InMemoryCorpus {
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + Send + '_> {
        Box::new(std::iter::once(Document::new(self.sentences.join(" "))))
    }

    fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
        Box::new(self.sentences.clone().into_iter())
    }

    fn document_count(&self) -> Option<usize> {
        Some(1)
    }
}
```

## The dictionary traits (from `libdictenstein`)

Three layered traits. Note that the mutating methods take **`&self`**, not `&mut self`: every
backend uses interior mutability, which is what allows lock-free parallel counting.

```rust
pub trait Dictionary {
    type Node: DictionaryNode;
    fn root(&self) -> Self::Node;
    fn contains(&self, term: &str) -> bool;      // has a default (traverses from the root)
    fn len(&self) -> Option<usize>;              // NOTE: Option — not every backend can count cheaply
    fn is_empty(&self) -> bool;                  // default
    fn sync_strategy(&self) -> SyncStrategy;     // default: ExternalSync
    fn is_suffix_based(&self) -> bool;           // default: false (prefix matching)
}

/// A "fuzzy map": terms carry values.
pub trait MappedDictionary: Dictionary {
    type Value: DictionaryValue;
    fn get_value(&self, term: &str) -> Option<Self::Value>;
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool;             // default: get_value + test
}

/// Value-aware writes.
pub trait MutableMappedDictionary: MappedDictionary {
    fn insert_with_value(&self, term: &str, value: Self::Value) -> bool;  // true if NEW
    fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize
    where
        F: Fn(&Self::Value, &Self::Value) -> Self::Value;
    fn union_replace(&self, other: &Self) -> usize;
    fn update_or_insert<F>(&self, term: &str, default_value: Self::Value, update_fn: F) -> bool
    where
        F: Fn(&mut Self::Value);
}
```

`Self::Value` must implement `DictionaryValue`, which requires
`Clone + Default + Send + Sync + Unpin + 'static` (plus `Serialize + DeserializeOwned` when
libdictenstein's serialization feature is on). `NgramEntry` satisfies all of them — it derives
`Default`, and hand-writes `Clone`, `Serialize`, and `Deserialize` because its fields are
atomics — and asserts membership with a one-line `impl DictionaryValue for NgramEntry {}`.

### `IterableDictionary` (libgrammstein's own)

Portable serialization needs to walk every stored n-gram, which the dictionary traits do not
offer. libgrammstein adds a narrow trait for exactly that:

```rust
pub trait IterableDictionary: MappedDictionary<Value = NgramEntry> {
    fn iter_all(&self) -> Box<dyn Iterator<Item = (String, NgramEntry)> + '_>;
}
```

Its supertrait is the **read-only** `MappedDictionary`, deliberately: that is what lets the
immutable `DoubleArrayTrieChar` back an `NgramModel` and be exported, even though it can never be
trained into.

### Bound lattice — which API demands what

| API | Bound on `D` |
|---|---|
| `NgramModel<D>` query surface | `MappedDictionary<Value = NgramEntry>` |
| `HybridLanguageModel<D>` scoring surface | `MappedDictionary<Value = NgramEntry> + Send + Sync` |
| `NgramModel::to_portable` / `save_portable` | `+ IterableDictionary` |
| `NgramModel::save` / `load` | `+ Serialize + DeserializeOwned` |
| `NgramModel::load_portable` | `MutableMappedDictionary<Value = NgramEntry>` |
| `TrainerBuilder<D>` / `NgramTrainer<D>` | `MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync + 'static` |
| `scoring::Perplexity` / `SentenceScorer` | `MutableMappedDictionary<Value = NgramEntry>` (stricter than it needs — see [Scoring](scoring.md#type-parameter-and-bound)) |
| `GrammsteinLanguageModel<D>` | `MutableMappedDictionary<Value = NgramEntry> + Send + Sync` |

### Backend capability matrix

| Backend | `MappedDictionary` | `MutableMappedDictionary` | `IterableDictionary` | Train | Query | `Perplexity` | Best for |
|---|---|---|---|---|---|---|---|
| `DynamicDawgChar<NgramEntry>` | yes | yes | yes | yes | yes | yes | general purpose; the only backend with full serde (`save`/`load`) |
| `PathMapDictionary<NgramEntry>` | yes | yes | yes | yes | yes | yes | memory sharing; lling-llang lattice integration |
| `SharedCharARTrie<NgramEntry>` | yes | yes | yes | yes | yes | yes | crash-safe, disk-backed training over huge corpora |
| `DoubleArrayTrieChar<NgramEntry>` | yes | **no** | yes | **no** | yes | **no** | fast, immutable inference (bulk-built via `load_static_portable`) |

## `LanguageModel` (feature `lling-llang-integration`)

This is the seam that lets a libgrammstein model rescore a WFST correction lattice.
`GrammsteinLanguageModel<D>` is an enum over the two model kinds, wrapped in `Arc` so it is cheap
to clone across threads, and it implements lling-llang's `LanguageModel` trait:

```rust
pub enum GrammsteinLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    Ngram(Arc<NgramModel<D>>),
    Hybrid(Arc<HybridLanguageModel<D>>),
}

impl<D> GrammsteinLanguageModel<D> {
    pub fn from_ngram(model: NgramModel<D>) -> Self;
    pub fn from_ngram_arc(model: Arc<NgramModel<D>>) -> Self;
    pub fn from_hybrid(model: HybridLanguageModel<D>) -> Self;
    pub fn from_hybrid_arc(model: Arc<HybridLanguageModel<D>>) -> Self;
    pub fn from_components(ngram: NgramModel<D>, embedding: SubwordEmbedding) -> Self;
    pub fn from_components_with_config(
        ngram: NgramModel<D>, embedding: SubwordEmbedding, config: HybridConfig,
    ) -> Self;

    pub fn is_hybrid(&self) -> bool;
    pub fn ngram_model(&self) -> Option<&NgramModel<D>>;   // Some in BOTH variants
    pub fn hybrid_model(&self) -> Option<&HybridLanguageModel<D>>;  // None for Ngram
}

// lling_llang::layers::LanguageModel
impl<D> LanguageModel for GrammsteinLanguageModel<D> {
    fn score_sequence(&self, tokens: &[&str]) -> f64;             // whole-sequence log-prob
    fn score_continuation(&self, prefix: &[&str], next: &str) -> f64;  // one-step log-prob
    fn vocab_size(&self) -> usize;
}
```

`from_components(ngram, embedding)` is the shortcut for "build a hybrid with default config, then
wrap it". Note that **`ngram_model()` returns `Some` for both variants** — a hybrid model still
has an n-gram inside it, and this is how you reach it.

Dispatch is by variant: `score_continuation` calls `NgramModel::log_prob` for `Ngram` and
`HybridLanguageModel::score` for `Hybrid`; `score_sequence` calls the respective
`sentence_log_prob`. `Clone` is an `Arc::clone` — the model itself is never copied.

```rust
use libgrammstein::integration::GrammsteinLanguageModel;
use lling_llang::layers::LanguageModel;

let lm = GrammsteinLanguageModel::from_hybrid(hybrid);
let seq = lm.score_sequence(&["the", "quick", "brown", "fox"]);
let next = lm.score_continuation(&["the", "quick"], "brown");

// Clone is Arc-cheap: share one model across rescoring threads.
let lm2 = lm.clone();
```

The same feature also exports the WFST-export surface (`NgramWfstExport`, `FromLogProb`,
`NgramTransducerBuilder`, `NgramLazyWfst`, …) and the `HierarchicalCorrector` —
see the [lling-llang integration docs](../integration/lling-llang/overview.md).

## Corpus utilities (structs, not traits)

For completeness, `libgrammstein::corpus` also exports `Tokenizer`, `Normalizer`,
`QualityFilter`, `Deduplicator`, `TextPreprocessor`, and `PreprocessingPipeline`. These are
**concrete builder-configured structs**, not traits, and are documented with the corpus
component: see [Corpus overview](../components/corpus/overview.md) and
[Streaming](../components/corpus/streaming.md).

## See also

- [NgramModel API](ngram.md) — the model the dictionary bounds serve
- [HybridLanguageModel API](hybrid.md) — the `Send + Sync` scoring surface
- [Scoring API](scoring.md) — where the `MutableMappedDictionary` bound bites
- [Errors](errors.md) — `Error`, `Result`, and their `From` conversions
- [Corpus formats](../components/corpus/formats.md) — what each reader accepts
- [liblevenshtein backend selection](../integration/liblevenshtein/backend-selection.md) — choosing a trie
- [lling-llang integration](../integration/lling-llang/overview.md) — WFST lattice rescoring
