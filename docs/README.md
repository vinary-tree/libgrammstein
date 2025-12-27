# libgrammstein Documentation

**libgrammstein** is a hybrid language model library combining N-gram models with subword embeddings. It is designed to integrate with [lling-llang](https://github.com/f1r3fly-io/lling-llang) for WFST-based text correction and normalization.

## What is libgrammstein?

libgrammstein provides:

- **N-gram Language Model**: Statistical word sequence prediction using Modified Kneser-Ney smoothing
- **Subword Embeddings**: FastText-style embeddings for handling out-of-vocabulary words
- **Hybrid Model**: Combines both approaches for robust scoring
- **WFST Integration**: Implements lling-llang's `LanguageModel` trait for lattice rescoring

```
┌─────────────────────────────────────────────────────────────────┐
│                        libgrammstein                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐   ┌─────────────────┐                     │
│  │   N-gram Model  │   │    Subword      │                     │
│  │                 │   │   Embeddings    │                     │
│  │  Modified KN    │   │   FastText-     │                     │
│  │  smoothing      │   │   style         │                     │
│  └────────┬────────┘   └────────┬────────┘                     │
│           │                     │                               │
│           └──────────┬──────────┘                               │
│                      ▼                                          │
│           ┌─────────────────────┐                               │
│           │   HybridLanguage    │                               │
│           │       Model         │                               │
│           │                     │                               │
│           │  Implements         │                               │
│           │  LanguageModel      │                               │
│           │  trait              │                               │
│           └─────────────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use libgrammstein::prelude::*;

// Load a trained hybrid model
let model = HybridLanguageModel::load("model.bin")?;

// Score a token sequence
let log_prob = model.score_sequence(&["the", "quick", "brown", "fox"]);

// Score a continuation
let next_prob = model.score_continuation(&["the", "quick"], "brown");

println!("Sequence log probability: {}", log_prob);
println!("P(brown | the quick): {}", next_prob.exp());
```

### Training a Model

```rust
use libgrammstein::prelude::*;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::TrainerBuilder;

// Stream corpus from directory
let reader = PlaintextReader::from_directory("./corpus")?;

// Train N-gram model
let ngram = TrainerBuilder::new()
    .order(5)
    .train(&reader)?;

// Train subword embeddings
let embedding = EmbeddingTrainer::new()
    .dimension(100)
    .epochs(20)
    .train(&reader)?;

// Combine into hybrid model
let hybrid = HybridLanguageModel::new(ngram, embedding);
hybrid.save("model.bin")?;
```

## Documentation Structure

### Architecture

- [Overview](architecture/overview.md) - High-level design and principles
- [Data Flow](architecture/data-flow.md) - How data flows through the system
- [Threading Model](architecture/threading-model.md) - Concurrency and parallelism

### Components

#### N-gram Model

- [Overview](components/ngram/overview.md) - What N-gram models are and how they work
- [Modified Kneser-Ney](components/ngram/modified-kneser-ney.md) - The smoothing algorithm
- [Trie Storage](components/ngram/trie-storage.md) - Dictionary backend storage
- [Query API](components/ngram/query-api.md) - Probability computation

#### Subword Embeddings

- [Overview](components/embedding/overview.md) - Word embeddings with subword enrichment
- [BPE Tokenizer](components/embedding/bpe-tokenizer.md) - Byte-Pair Encoding
- [Skip-gram](components/embedding/skip-gram.md) - Training with negative sampling
- [Similarity](components/embedding/similarity.md) - Cosine similarity and nearest neighbors

#### Hybrid Model

- [Overview](components/hybrid/overview.md) - Combining N-gram and embeddings
- [Interpolation](components/hybrid/interpolation.md) - Score combination strategies
- [OOV Handling](components/hybrid/oov-handling.md) - Out-of-vocabulary word handling

#### Corpus Processing

- [Overview](components/corpus/overview.md) - Streaming corpus architecture
- [Streaming](components/corpus/streaming.md) - Memory-efficient processing
- [Formats](components/corpus/formats.md) - Wikipedia, Gutenberg, plaintext

### Integration

#### lling-llang

- [Overview](integration/lling-llang/overview.md) - Integration architecture
- [LanguageModel Trait](integration/lling-llang/language-model-trait.md) - Implementing the trait
- [Pipeline Usage](integration/lling-llang/pipeline-usage.md) - Using in correction pipelines
- [PathMap Synergy](integration/lling-llang/pathmap-synergy.md) - Shared infrastructure

#### liblevenshtein

- [Overview](integration/liblevenshtein/overview.md) - Dictionary backend integration
- [Backend Selection](integration/liblevenshtein/backend-selection.md) - Choosing the right backend

### Training

- [N-gram Training](training/ngram-training.md) - Count collection and smoothing
- [Embedding Training](training/embedding-training.md) - Skip-gram training workflow
- [Hyperparameters](training/hyperparameters.md) - Tuning guide

### API Reference

- [NgramModel](api/ngram-reference.md) - N-gram model API
- [SubwordEmbedding](api/embedding-reference.md) - Embedding API
- [HybridLanguageModel](api/hybrid-reference.md) - Hybrid model API
- [Traits](api/trait-reference.md) - Key traits and interfaces

### Examples

- [Train and Evaluate](examples/train-and-evaluate.md) - End-to-end workflow
- [Perplexity Filter](examples/perplexity-filter.md) - Text quality filtering
- [Spell Correction](examples/spell-correction.md) - lling-llang integration

## Prerequisites

- **Rust**: 1.75+ (2024 edition)
- **liblevenshtein-rust**: Dictionary backends
- **Corpus data**: Wikipedia dumps, Project Gutenberg, or custom text files

## Features

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["lling-llang-integration", "serde"] }
```

| Feature | Description |
|---------|-------------|
| `default` | Core N-gram and embedding functionality |
| `lling-llang-integration` | Implements lling-llang's `LanguageModel` trait |
| `serde` | Model serialization/deserialization |
| `async` | Async corpus streaming (Tokio) |

## Related Projects

- [lling-llang](https://github.com/f1r3fly-io/lling-llang): WFST framework for text correction
- [liblevenshtein-rust](https://github.com/f1r3fly-io/liblevenshtein-rust): Fuzzy string matching and trie dictionaries
- [F1R3FLY.io](https://f1r3fly.io): Distributed computing platform

## License

MIT OR Apache-2.0
