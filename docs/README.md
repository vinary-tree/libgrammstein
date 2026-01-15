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
- [Phonetic Embeddings](components/embedding/phonetic.md) - Combining orthographic and phonetic similarity
- [Acoustic Word Embeddings](components/embedding/acoustic-word.md) - Fixed-dimensional audio representations

#### Hybrid Model

- [Overview](components/hybrid/overview.md) - Combining N-gram and embeddings
- [Interpolation](components/hybrid/interpolation.md) - Score combination strategies
- [OOV Handling](components/hybrid/oov-handling.md) - Out-of-vocabulary word handling

#### Text Generation

- [Text Generation](components/generation/text-generation.md) - Autoregressive text generation with sampling strategies

#### Corpus Processing

- [Overview](components/corpus/overview.md) - Streaming corpus architecture
- [Streaming](components/corpus/streaming.md) - Memory-efficient processing
- [Formats](components/corpus/formats.md) - Wikipedia, Gutenberg, plaintext

#### Acoustic Processing

- [Overview](components/acoustic/overview.md) - Audio feature extraction module
- [Feature Extraction](components/acoustic/features.md) - MFCC, filterbank, streaming extraction
- [Acoustic Models](components/acoustic/models.md) - Candle-based neural acoustic models

#### Neural Module

- [Overview](components/neural/overview.md) - ModernBERT-based neural components
- [Model](components/neural/model.md) - ModernBERT model wrapper and inference
- [Embedder](components/neural/embedder.md) - Document and query embedding
- [Rescorer](components/neural/rescorer.md) - Neural rescoring for beam search
- [Summarizer](components/neural/summarizer.md) - Extractive summarization with MMR
- [Cache](components/neural/cache.md) - KV cache for efficient inference

#### RAG (Retrieval-Augmented Generation)

- [Overview](components/rag/overview.md) - Document indexing and retrieval
- [Document](components/rag/document.md) - Document structures and metadata
- [Backend](components/rag/backend.md) - Retrieval backends (Exact, HNSW)
- [Index](components/rag/index.md) - RagIndex with topic integration
- [Retriever](components/rag/retriever.md) - Query and retrieval pipeline
- [Builder](components/rag/builder.md) - Index construction workflow

#### Topic Extraction

- [Overview](components/topic/overview.md) - BERTopic-style topic modeling
- [Clustering](components/topic/clustering.md) - Hierarchical agglomerative clustering
- [c-TF-IDF](components/topic/ctfidf.md) - Keyword extraction algorithm
- [Dendrogram](components/topic/dendrogram.md) - Topic hierarchy navigation

#### Paradigm Detection

- [Overview](components/paradigm/overview.md) - Programming paradigm analysis
- [Detection](components/paradigm/detection.md) - ParadigmDetector usage and configuration
- [Indicators](components/paradigm/indicators.md) - OOP, FP, reactive, procedural indicators
- [API Patterns](components/paradigm/api-patterns.md) - PrefixSpan sequence mining
- [Domain Patterns](components/paradigm/domain-patterns.md) - Rholang and MeTTa pattern catalogs

#### Code Embeddings

- [Overview](components/code-embeddings/overview.md) - Neural code embedding models
- [CodeT5+](components/code-embeddings/codet5.md) - CodeT5+ model integration
- [UniXcoder](components/code-embeddings/unixcoder.md) - UniXcoder model integration
- [GraphCodeBERT](components/code-embeddings/graphcodebert.md) - GraphCodeBERT with data flow
- [Ensemble](components/code-embeddings/ensemble.md) - Multi-model ensemble strategies
- [Caching](components/code-embeddings/caching.md) - Embedding cache management

#### Subtree Mining

- [Overview](components/subtree/overview.md) - Frequent subtree pattern discovery
- [TreeminerD](components/subtree/treeminer-d.md) - TreeminerD algorithm details

#### Code Correction

- [Overview](components/code/overview.md) - Multi-language code correction module
- [Language Trait](components/code/language.md) - CodeLanguage trait and token types
- [Languages](components/code/languages.md) - Python, Rust, JavaScript, Rholang, MeTTa support
- [AST](components/code/ast.md) - Tree-sitter integration and parsing
- [Tokenizer](components/code/tokenizer.md) - Code tokenization system
- [CPG](components/code/cpg.md) - Code Property Graphs (AST + CFG + DFG)
- [Correction Framework](components/code/correction.md) - Correction types and traits
- [Correctors](components/code/correctors/overview.md) - Lexical, grammar, semantic correctors
  - [Lexical](components/code/correctors/lexical.md) - Fuzzy matching with Levenshtein distance
  - [Grammar](components/code/correctors/grammar.md) - PCFG + Earley parsing
  - [Semantic](components/code/correctors/semantic.md) - GNN + CPG analysis
  - [Ensemble](components/code/correctors/ensemble.md) - Multi-source aggregation
- [Pipeline](components/code/pipeline.md) - End-to-end correction workflow
- [PCFG](components/code/pcfg.md) - Probabilistic context-free grammars
- [GNN](components/code/gnn.md) - Graph neural networks for code
- [Embeddings](components/code/embeddings.md) - Code embeddings (UniXcoder, GraphCodeBERT)
- [Constrained Decoding](components/code/constrained-decoding.md) - Grammar-constrained generation
- [WFST Export](components/code/wfst-export.md) - PCFG to WFST approximation
- [Subtree Mining](components/code/subtree-mining.md) - TreeminerD frequent pattern mining

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
| `acoustic` | Audio feature extraction (MFCC, filterbank) |
| `candle-model` | Candle-based neural acoustic models |
| `phonetic` | Phonetic-aware embeddings with Zompist rules |
| `neural-rescore` | ModernBERT embeddings, rescoring, and summarization |
| `rag` | RAG indexing with topic extraction |
| `code` | Core code correction (lexical corrector only) |
| `code-python` | Python language support |
| `code-rust` | Rust language support |
| `code-javascript` | JavaScript language support |
| `code-rholang` | Rholang language support |
| `code-metta` | MeTTa language support |
| `code-neural` | Neural code embeddings (UniXcoder, GraphCodeBERT) |

## Related Projects

- [lling-llang](https://github.com/f1r3fly-io/lling-llang): WFST framework for text correction
- [liblevenshtein-rust](https://github.com/f1r3fly-io/liblevenshtein-rust): Fuzzy string matching and trie dictionaries
- [F1R3FLY.io](https://f1r3fly.io): Distributed computing platform

## License

MIT OR Apache-2.0
