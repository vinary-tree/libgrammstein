# libgrammstein Documentation

**libgrammstein** is a multi-paradigm, formally-verified language-modeling toolkit in Rust: a
Modified Kneser-Ney n-gram core, FastText-style subword embeddings, a hybrid scorer, and a fan
of higher-level capabilities (neural rescoring, retrieval, topic modeling, code & LaTeX
correction) plus a lock-free Google Books importer. It implements
[lling-llang](https://github.com/f1r3fly-io/lling-llang)'s `LanguageModel` trait so any model
here can rescore a WFST correction lattice.

> New here? The root [`README.md`](../README.md) is the narrative overview (with the four
> flagship deep-dives). This page is the **map** of the full documentation set.

## Architecture at a glance

libgrammstein is organized as layers: a persistent-storage foundation, a statistical +
embedding core, a hybrid scorer, and à-la-carte capabilities enabled through Cargo features.
Everything is `Send + Sync`; the heavy data structures live in memory-mapped, crash-safe tries.

![Layered architecture of libgrammstein](diagrams/architecture.svg)

## Quick Start

Train a 5-gram model and FastText subword embeddings, then combine them into a hybrid scorer.
All APIs below match the current source.

```rust
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel};
use libgrammstein::corpus::PlaintextReader;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

// 1. Train a Modified Kneser-Ney n-gram model over a serializable trie backend.
let ngram = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
    .order(5)
    .train(PlaintextReader::from_file("corpus.txt")?)?;

// 2. Train subword embeddings (a word's vector is the sum of its character-n-gram vectors).
let embedding = EmbeddingTrainerBuilder::new()
    .dim(100).window_size(5).min_count(5).epochs(10)
    .train(PlaintextReader::from_file("corpus.txt")?)?;

// 3. Combine into a hybrid model and score a word in context (returns a log-probability).
let hybrid = HybridLanguageModel::new(ngram, embedding, HybridConfig::default());
let log_p = hybrid.score("fox", &["the", "quick", "brown"]);
println!("log P(fox | the quick brown) = {log_p:.3}");
# Ok::<(), libgrammstein::Error>(())
```

Measure fit with perplexity, or wrap the model as an lling-llang `LanguageModel` to rescore a
correction lattice:

```rust
use libgrammstein::scoring::Perplexity;
use libgrammstein::integration::GrammsteinLanguageModel;   // feature: lling-llang-integration

let ppl = Perplexity::new(&ngram).corpus_perplexity(&PlaintextReader::from_file("dev.txt")?)?;
let lm = GrammsteinLanguageModel::from_hybrid(hybrid);      // score_sequence / score_continuation
```

> Persistence (`save`/`load`, `save_portable`/`load_portable`) is behind the `serde-extras`
> feature; see the [Hybrid API reference](api/hybrid.md).

## Conventions

These docs follow the pgmcp documentation guidelines. When contributing:

- **Mathematics** uses MathJax, delimited for GitHub-flavored Markdown. **Inline** math is a
  backtick span *wrapped in* dollar signs — the dollars go on the **outside**; **display** math is
  a fenced block whose info-string is `math`. Bare dollar delimiters are never used (GitHub's
  CommonMark pass strips backslash escapes before MathJax parses them, silently corrupting the
  expression), and formulae are never spelled out with unicode literals:

  ```text
  inline    dollar, backtick, LaTeX, backtick, dollar:   $`\mathbb{P}(w \mid h)`$
  display   a fenced block whose info-string is exactly: math
            (number equations inside it with \tag{M3})
  ```

  The backticks shield the LaTeX from CommonMark; the outer dollars mark it as math. Note the
  ordering carefully: a code span *containing* dollars renders as inert monospace, **not** math —
  which is also why a literal dollar sign is written as inline code. Keep each inline span on a
  single line, and never let an ASCII letter abut the opening delimiter.

- **Algorithms** are presented as literate pseudocode in plain code fences, using ASCII
  identifiers and operators (typeset mathematics stays in math spans).
- **Diagrams** are PlantUML rendered to SVG under [`diagrams/`](diagrams/README.md), with a
  semantic color palette (one hue per concept) and a documented regeneration loop.
- **Citations** are numbered with DOI links.

## Documentation map

### Architecture

- [Overview](architecture/overview.md) — high-level design and layering
- [Data Flow](architecture/data-flow.md) — how data moves through the system
- [Threading Model](architecture/threading.md) — concurrency and parallelism
- [Memory Optimization](architecture/memory-optimization.md) — the bounded-heap importer techniques
- [Google Books Importer](architecture/google-books-importer.md) — design of the sharded, lock-free importer
- [Google Books Shard Routing](architecture/google-books-shard-routing.md) — term-id keys, the single input boundary, and the delimiter-collision migration

### Components

**N-gram model** — [Overview](components/ngram/overview.md) ·
[Modified Kneser-Ney](components/ngram/modified-kneser-ney.md) ·
[Trie Storage](components/ngram/trie-storage.md) · [Query API](components/ngram/query-api.md)

**Subword embeddings** — [Overview](components/embedding/overview.md) ·
[BPE](components/embedding/bpe.md) · [Skip-gram](components/embedding/skip-gram.md) ·
[Similarity](components/embedding/similarity.md) · [Phonetic](components/embedding/phonetic.md) ·
[Acoustic word embeddings](components/embedding/acoustic-word.md)

**Hybrid model** — [Overview](components/hybrid/overview.md) ·
[Interpolation](components/hybrid/interpolation.md) ·
[OOV Handling](components/hybrid/oov-handling.md)

**Scoring & generation** — [Scoring overview](components/scoring/overview.md) (perplexity,
sentence scoring) · [Text Generation](components/generation/text-generation.md) (sampling
strategies)

**Corpus & dictionary** — [Corpus overview](components/corpus/overview.md) ·
[Streaming](components/corpus/streaming.md) · [Formats](components/corpus/formats.md) ·
[Dictionary overview](components/dictionary/overview.md) ·
[Building](components/dictionary/building.md) · [Extraction](components/dictionary/extraction.md)

**Language detection** — [Overview](components/language/overview.md) (whatlang detection +
language-aware tokenization)

**Acoustic** — [Overview](components/acoustic/overview.md) ·
[Feature Extraction](components/acoustic/features.md) ·
[Acoustic Models](components/acoustic/models.md)

**Neural (ModernBERT)** — [Overview](components/neural/overview.md) ·
[Model](components/neural/model.md) · [Embedder](components/neural/embedder.md) ·
[Rescorer](components/neural/rescorer.md) · [Summarizer](components/neural/summarizer.md) ·
[Cache](components/neural/cache.md)

**RAG (retrieval)** — [Overview](components/rag/overview.md) ·
[Document](components/rag/document.md) · [Backend](components/rag/backend.md) ·
[Index](components/rag/index.md) · [Retriever](components/rag/retriever.md) ·
[Builder](components/rag/builder.md)

**Topic modeling** — [Overview](components/topic/overview.md) ·
[Clustering](components/topic/clustering.md) · [c-TF-IDF](components/topic/ctfidf.md) ·
[Dendrogram](components/topic/dendrogram.md)

**Paradigm detection** — [Overview](components/paradigm/overview.md) ·
[Detection](components/paradigm/detection.md) · [Indicators](components/paradigm/indicators.md) ·
[API Patterns](components/paradigm/api-patterns.md) ·
[Domain Patterns](components/paradigm/domain-patterns.md)

**Code correction** — [Overview](components/code/overview.md) ·
[Language trait](components/code/language.md) · [Languages](components/code/languages.md) ·
[AST](components/code/ast.md) · [Tokenizer](components/code/tokenizer.md) ·
[CPG](components/code/cpg.md) · [Correction framework](components/code/correction.md) ·
[Pipeline](components/code/pipeline.md) · [PCFG](components/code/pcfg.md) ·
[GNN](components/code/gnn.md) · [Code embeddings](components/code/embeddings.md) ·
[Constrained decoding](components/code/constrained-decoding.md) ·
[WFST export](components/code/wfst-export.md) ·
[Subtree mining](components/code/subtree-mining.md)
  - Correctors: [Overview](components/code/correctors/overview.md) ·
    [Lexical](components/code/correctors/lexical.md) ·
    [Grammar](components/code/correctors/grammar.md) ·
    [Semantic](components/code/correctors/semantic.md) ·
    [Ensemble](components/code/correctors/ensemble.md)

**Neural code embeddings** — [Overview](components/code-embeddings/overview.md) ·
[CodeT5+](components/code-embeddings/codet5.md) ·
[UniXcoder](components/code-embeddings/unixcoder.md) ·
[GraphCodeBERT](components/code-embeddings/graphcodebert.md) ·
[Ensemble](components/code-embeddings/ensemble.md) ·
[Caching](components/code-embeddings/caching.md)

**Subtree mining** — [Overview](components/subtree/overview.md) ·
[TreeMinerD](components/subtree/treeminer-d.md)

**LaTeX-aware modeling** — [Overview](components/latex/overview.md) ·
[Tokenizer](components/latex/tokenizer.md) · [N-gram](components/latex/ngram.md) ·
[Embedding](components/latex/embedding.md) · [Scorer](components/latex/scorer.md) ·
[Rescorer](components/latex/rescorer.md) · [RAG](components/latex/rag.md)

**Sources** — [PDF extraction](components/sources/pdf.md) (Marker/Nougat PDF→LaTeX)

### Integration

**lling-llang** — [Overview](integration/lling-llang/overview.md) ·
[Hierarchical Correction](integration/lling-llang/hierarchical-correction.md) ·
[Dimensions](integration/lling-llang/dimensions.md) ·
[Dictionary Backend](integration/lling-llang/dictionary-backend.md) ·
[Pipeline Assembly](integration/lling-llang/pipeline-assembly.md) ·
[Pipeline Usage](integration/lling-llang/pipeline-usage.md) ·
[Multi-shard grammar corrector](integration/multi-shard-grammar-corrector.md)

**liblevenshtein** — [Overview](integration/liblevenshtein/overview.md) ·
[Backend Selection](integration/liblevenshtein/backend-selection.md) ·
[PathMap Synergy](integration/liblevenshtein/pathmap-synergy.md)

### CLI

- [CLI overview](cli/README.md) — the `grammstein` binary (train, eval, query, corpus, models)
- [Google Books import](cli/import-google-books.md) — the sharded importer command + flags

### Training

- [N-gram Training](training/ngram.md) — count collection and smoothing
- [Embedding Training](training/embedding.md) — skip-gram training workflow
- [Hybrid Training](training/hybrid.md) — combining and tuning the two models
- [Hyperparameters](training/hyperparameters.md) — tuning guide
- [Large Corpora](training/large-corpora.md) — streaming, memory, and throughput

### API reference

- [NgramModel](api/ngram.md) · [SubwordEmbedding](api/embedding.md) ·
  [HybridLanguageModel](api/hybrid.md) · [Scoring](api/scoring.md) ·
  [Traits](api/traits.md) · [Errors](api/errors.md)

### Examples

- [Train and Evaluate](examples/train-and-evaluate.md) — end-to-end workflow
- [Perplexity Scoring](examples/perplexity-scoring.md) — text-quality filtering
- [Domain Adaptation](examples/domain-adaptation.md) — adapting a model to a new domain
- [Spell Correction](examples/spell-correction.md) — lling-llang integration

### Utilities

- [Cron Scheduler](util/cron/cron-manager.md) — the checkpoint scheduler used by the importer

### Formal verification & diagrams

- [Formal verification](../formal/README.md) — TLA+ / TLAPS / Apalache specs + Rocq proofs
- [Diagram authoring guide](diagrams/README.md) — the PlantUML house style + regeneration loop

### Archive

- [Archived documentation](archive/README.md) — historical reports and superseded designs

## Prerequisites

- **Rust** 1.70+ (edition 2021) — see `rust-version` in [`Cargo.toml`](../Cargo.toml)
- **Sibling crates** (path dependencies): `liblevenshtein-rust`, `libdictenstein`, `PathMap`,
  and (for the integration feature) `lling-llang`
- **Corpus data** — Wikipedia dumps, Project Gutenberg, or plaintext files

## Feature flags

The default build is the always-on statistical + embedding + hybrid core; everything else is
feature-gated so you compile only what you use. `serde` is a hard dependency (not a feature);
binary model serialization is behind `serde-extras`.

```toml
[dependencies]
libgrammstein = { path = "../libgrammstein", features = ["cli", "google-books", "rag"] }
```

| Feature | Unlocks |
|---|---|
| *(default)* | n-gram (MKN), subword embeddings, hybrid model, perplexity, corpus streaming |
| `google-books` | the sharded, checkpointed Google Books importer (+ `mimalloc`) |
| `pdf-extraction` | PDF→LaTeX source extraction (Marker/Nougat) |
| `neural-rescore` | ModernBERT embeddings, MLM rescoring, MMR summarization (Candle) |
| `rag` / `rag-hnsw` | retrieval index — exact cosine / HNSW; topic modeling |
| `code` · `code-{python,rust,javascript,rholang,metta}` · `code-neural` | code correction (+ per-language, + neural code embeddings) |
| `code-mainstream` / `code-dsl` / `code-full` | convenience aggregates over the `code-*` flags |
| `latex` / `latex-neural` / `latex-rag` / `latex-full` | mode-aware LaTeX modeling |
| `acoustic` / `candle-model` / `gpu` | MFCC features / neural acoustic models / GPU kernels |
| `subword` | BPE subword tokenization |
| `lling-llang-integration` / `wfst-export` | `LanguageModel` trait + WFST export for lattice rescoring |
| `serde-extras` | binary model serialization (bincode `save`/`load`) |
| `async` / `http-corpus` | async corpus streaming (Tokio) / streaming corpora over HTTP |
| `ner` / `ocr` / `latex-ocr` | named-entity recognition / OCR / OCR for LaTeX inputs |
| `language-full` | the full `whatlang` language set |
| `mimalloc-alloc` | the `mimalloc` global allocator |
| `cli` | the `grammstein` binary + ratatui TUI |
| `loom-tests` | the `loom` concurrency-test harness (development) |

## Related projects

- [lling-llang](https://github.com/f1r3fly-io/lling-llang) — WFST framework for text correction
- [liblevenshtein-rust](https://github.com/f1r3fly-io/liblevenshtein-rust) — fuzzy matching & trie dictionaries
- **libdictenstein** — persistent adaptive-radix trie with lock-free overlay & eviction
- [F1R3FLY.io](https://f1r3fly.io) — distributed computing platform

## License

Licensed under **Apache-2.0** (declared in [`Cargo.toml`](../Cargo.toml)).
