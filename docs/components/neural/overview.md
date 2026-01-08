# Neural Module Overview

The neural module provides ModernBERT-based components for semantic understanding, embedding generation, rescoring, and summarization in libgrammstein.

## What is the Neural Module?

The neural module wraps [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), a state-of-the-art encoder model optimized for semantic understanding tasks. It provides:

- **Embeddings**: Dense vector representations for documents and queries
- **Rescoring**: Neural language model scoring for beam search paths
- **Summarization**: Extractive summarization using Maximal Marginal Relevance (MMR)
- **Caching**: Efficient inference with KV cache and embedding cache

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Neural Module                                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ModernBertModel                               │    │
│  │  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐  │    │
│  │  │  Tokenizer    │  │ Transformer  │  │   MLM Head           │  │    │
│  │  │  (WordPiece)  │  │ (12 layers)  │  │   (optional)         │  │    │
│  │  └───────────────┘  └──────────────┘  └──────────────────────┘  │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│         ┌───────────────────────┼───────────────────────┐               │
│         │                       │                       │               │
│         ▼                       ▼                       ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ ModernBert      │  │   Summarizer    │  │ ModernBertRescorer      │  │
│  │ Embedder        │  │   (extractive)  │  │ (beam search)           │  │
│  │                 │  │                 │  │                         │  │
│  │ • embed_query   │  │ • extractive    │  │ • score_sentence        │  │
│  │ • embed_document│  │ • create_synopsis│ │ • rescore_paths         │  │
│  │ • embed_batch   │  │                 │  │                         │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────────────┘  │
│           │                    │                                        │
│           ▼                    ▼                                        │
│  ┌─────────────────┐  ┌─────────────────┐                              │
│  │ EmbeddingCache  │  │    Synopsis     │                              │
│  │ (lock-free)     │  │ (Explicit/Gen)  │                              │
│  └─────────────────┘  └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## ModernBERT Model

ModernBERT is a 149M parameter encoder model with:

| Property | Value |
|----------|-------|
| Hidden size | 768 |
| Attention heads | 12 |
| Layers | 12 |
| Max sequence length | 8,192 tokens |
| Vocabulary size | 50,368 |
| Training objective | Masked Language Modeling (MLM) |

The model uses the [Candle](https://github.com/huggingface/candle) framework for inference, supporting CPU, CUDA, and Metal backends.

## Components

### ModernBertEmbedder

Generates dense vector embeddings for semantic similarity:

```rust
use libgrammstein::neural::{ModernBertEmbedder, EmbeddingConfig};

let config = EmbeddingConfig::default();
let embedder = ModernBertEmbedder::new(config)?;

// Embed a query (optimized for retrieval)
let query_embedding = embedder.embed_query("What is machine learning?")?;

// Embed a document
let doc_embedding = embedder.embed_document("Machine learning is a branch of AI...")?;

// Compute similarity
let similarity = embedder.cosine_similarity(&query_embedding, &doc_embedding);
```

See [Embedder](embedder.md) for details.

### ModernBertRescorer

Rescores n-gram beam search outputs using neural language modeling:

```rust
use libgrammstein::neural::{ModernBertRescorer, RescoringConfig, ScoredPath};

let config = RescoringConfig {
    ngram_weight: 0.3,
    neural_weight: 0.7,
    top_k: 10,
    ..Default::default()
};
let rescorer = ModernBertRescorer::new(config)?;

// Rescore beam search paths
let paths: Vec<ScoredPath<f32>> = vec![/* ... */];
let result = rescorer.rescore_paths(&paths)?;
println!("Best: {}", result.best_path.text());
```

See [Rescorer](rescorer.md) for details.

### Summarizer

Extracts representative sentences using MMR diversity selection:

```rust
use libgrammstein::neural::{Summarizer, SummarizerConfig};

let config = SummarizerConfig {
    num_sentences: 3,
    diversity_threshold: 0.3,
    ..Default::default()
};
let summarizer = Summarizer::new(config)?;

let text = "Long document text here...";
let synopsis = summarizer.create_synopsis(text, None)?;
println!("{}", synopsis.text);
```

See [Summarizer](summarizer.md) for details.

### Caching

Efficient caching for inference and embeddings:

- **KvCache**: Key-value cache for transformer layer outputs
- **SlidingWindowCache**: Bounded memory cache for long sequences
- **EmbeddingCache**: Lock-free cache for sentence embeddings

See [Cache](cache.md) for details.

## Feature Flags

Enable the neural module with the `neural-rescore` feature:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["neural-rescore"] }
```

## Device Selection

The model supports multiple compute backends:

```rust
use libgrammstein::neural::{ModernBertConfig, Device};

// CPU (default)
let config = ModernBertConfig::default();

// CUDA (requires CUDA toolkit)
let config = ModernBertConfig {
    device: Device::Cuda(0),  // GPU 0
    ..Default::default()
};

// Metal (macOS)
let config = ModernBertConfig {
    device: Device::Metal,
    ..Default::default()
};
```

## Integration with RAG

The neural module integrates with the [RAG module](../rag/overview.md) for:

1. **Document embedding**: `ModernBertEmbedder` generates embeddings stored in `RagIndex`
2. **Synopsis generation**: `Summarizer` creates document summaries for display
3. **Query embedding**: Convert text queries to vectors for similarity search

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

// IndexBuilder uses ModernBertEmbedder internally
let config = IndexBuilderConfig::default();
let builder = IndexBuilder::new(config)?;
let index = builder.build_from_directory("./docs", None)?;
```

## Thread Safety

All neural components support concurrent access:

- `ModernBertModel` is wrapped in `Arc` for shared ownership
- Embedder, rescorer, and summarizer use `&self` (immutable) API
- `EmbeddingCache` uses lock-free `DashMap` for concurrent access

```rust
use std::sync::Arc;
use std::thread;

let embedder = Arc::new(ModernBertEmbedder::new(config)?);

// Multiple threads can embed concurrently
let handles: Vec<_> = texts.iter().map(|text| {
    let embedder = Arc::clone(&embedder);
    let text = text.clone();
    thread::spawn(move || embedder.embed_query(&text))
}).collect();
```

## See Also

- [Model](model.md) - ModernBERT model wrapper details
- [Embedder](embedder.md) - Document and query embedding
- [Rescorer](rescorer.md) - Neural rescoring for beam search
- [Summarizer](summarizer.md) - Extractive summarization
- [Cache](cache.md) - Caching strategies
- [RAG Overview](../rag/overview.md) - RAG integration
