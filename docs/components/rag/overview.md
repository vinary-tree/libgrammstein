# RAG Module Overview

The RAG (Retrieval-Augmented Generation) module provides document indexing and semantic retrieval capabilities for libgrammstein.

## What is RAG?

RAG combines retrieval and generation to provide contextually relevant information:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                                     │
│                                                                          │
│  Documents              Index                    Query                   │
│  ┌──────────┐          ┌──────────┐             ┌──────────┐            │
│  │ Doc 1    │          │          │             │ "What is │            │
│  │ Doc 2    │  ────►   │ RagIndex │   ◄────     │  ML?"    │            │
│  │ Doc 3    │          │          │             │          │            │
│  │ ...      │          └────┬─────┘             └──────────┘            │
│  └──────────┘               │                                           │
│                             │                                           │
│                             ▼                                           │
│                    ┌────────────────┐                                   │
│                    │ Top-K Results  │                                   │
│                    │                │                                   │
│                    │ 1. Doc 3 (0.95)│                                   │
│                    │ 2. Doc 1 (0.82)│                                   │
│                    │ 3. Doc 7 (0.76)│                                   │
│                    └────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG Module                                       │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Document Layer                                 │  │
│  │                                                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │  │
│  │  │ Document    │  │ DocumentMeta│  │ DocumentBuilder         │   │  │
│  │  │ (full)      │  │ (metadata)  │  │ (fluent API)            │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                 │                                        │
│                                 ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Index Layer                                    │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │                    RagIndex<B>                              │ │  │
│  │  │                                                             │ │  │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │  │
│  │  │  │ Backend<B>  │  │ Metadata    │  │ TopicModel          │ │ │  │
│  │  │  │ (embeddings)│  │ (HashMap)   │  │ (optional)          │ │ │  │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                 │                                        │
│                                 ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Backend Layer                                  │  │
│  │                                                                   │  │
│  │  ┌─────────────────────┐     ┌─────────────────────────────────┐ │  │
│  │  │ ExactCosineBackend  │     │ HnswBackend                     │ │  │
│  │  │                     │     │                                 │ │  │
│  │  │ • Dense retrieval   │     │ • Approximate NN                │ │  │
│  │  │ • O(n) query       │     │ • O(log n) query               │ │  │
│  │  │ • Best < 1M docs   │     │ • Best > 1M docs               │ │  │
│  │  └─────────────────────┘     └─────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                 │                                        │
│                                 ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Retrieval Layer                                │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Retriever<B>                             │ │  │
│  │  │                                                             │ │  │
│  │  │  Query → ModernBertEmbedder → Index Query → Results        │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Building an Index

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

// Create builder with default configuration
let config = IndexBuilderConfig::default();
let builder = IndexBuilder::new(config)?;

// Build index from directory of documents
let index = builder.build_from_directory("./documents", Some(&|current, total| {
    println!("Processing {}/{}", current, total);
}))?;

// Save index for later use
index.save("./index")?;
```

### Querying an Index

```rust
use libgrammstein::rag::{RagIndex, Retriever, RetrievalConfig};
use libgrammstein::neural::{ModernBertEmbedder, EmbeddingConfig};

// Load existing index
let index = RagIndex::load("./index")?;

// Create retriever with embedder
let embedder = ModernBertEmbedder::new(EmbeddingConfig::default())?;
let retriever = Retriever::new(
    Arc::new(index),
    embedder,
    RetrievalConfig::default(),
);

// Query the index
let results = retriever.query("What is machine learning?")?;

for result in results {
    println!("{}. {} (score: {:.2})",
        result.rank,
        result.display_title(),
        result.score
    );
    println!("   {}", result.synopsis);
}
```

## Components

### Document

Represents a document with content and metadata:

```rust
use libgrammstein::rag::{Document, DocumentBuilder, LanguageTag};

let doc = DocumentBuilder::new("file:///docs/intro.md")
    .title("Introduction to ML")
    .content("Machine learning is...")
    .language(LanguageTag::english_us())
    .build()?;
```

See [Document](document.md) for details.

### RagIndex

Central index combining backend and metadata:

```rust
use libgrammstein::rag::{RagIndex, RagIndexConfig, ExactCosineBackend};

let config = RagIndexConfig::default();
let mut index = RagIndex::<ExactCosineBackend>::new(config);

// Add documents
index.add_document(doc)?;

// Query
let results = index.query(&query_embedding, 10);
```

See [Index](index.md) for details.

### Backend

Pluggable retrieval backends:

| Backend | Query Time | Best For |
|---------|------------|----------|
| `ExactCosineBackend` | O(n) | < 1M documents |
| `HnswBackend` | O(log n) | > 1M documents |

See [Backend](backend.md) for details.

### Retriever

High-level query interface:

```rust
use libgrammstein::rag::{Retriever, RetrievalConfig};

let config = RetrievalConfig {
    top_k: 10,
    min_similarity: 0.5,
    ..Default::default()
};
```

See [Retriever](retriever.md) for details.

### IndexBuilder

Constructs indices from document collections:

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

let config = IndexBuilderConfig {
    auto_synopsis: true,  // Generate summaries
    ..Default::default()
};
```

See [Builder](builder.md) for details.

## Feature Flags

Enable the RAG module with the `rag` feature:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["rag"] }
```

This also enables:
- `neural-rescore` - For embeddings and summarization
- `topic` - For topic extraction (optional)

## Integration with Neural Module

The RAG module uses the [Neural Module](../neural/overview.md) for:

1. **Embeddings**: `ModernBertEmbedder` generates document and query embeddings
2. **Summarization**: `Summarizer` creates synopses for display
3. **Thread safety**: Shared `Arc<ModernBertModel>` across components

## Integration with Topic Module

The RAG module integrates with the [Topic Module](../topic/overview.md) for:

1. **Topic extraction**: `index.extract_topics()` clusters documents
2. **Topic storage**: `TopicModel` stored in index
3. **Topic display**: Show topics in query results

```rust
use libgrammstein::topic::TopicConfig;

// Extract topics from indexed documents
let topic_config = TopicConfig::default();
let embeddings = index.get_all_embeddings();
let texts: Vec<_> = index.iter().map(|(_, meta)| meta.synopsis.clone()).collect();

index.extract_topics(&topic_config, &embeddings, &texts)?;

// Query with topic information
for (meta, score) in index.query(&embedding, 5) {
    println!("{}: {}", meta.title.unwrap_or_default(), meta.synopsis);
    if !meta.topic_ids.is_empty() {
        let topics: Vec<_> = meta.topic_ids.iter()
            .filter_map(|id| index.topic_model().and_then(|m| m.get(*id)))
            .map(|t| t.keyword_summary(3))
            .collect();
        println!("   Topics: {}", topics.join(", "));
    }
}
```

## Persistence

The RAG index persists to a directory structure:

```
index/
├── config.json          # RagIndexConfig
├── state.json           # Index state (next_id)
├── metadata.json        # Document metadata
├── topic_model.json     # TopicModel (optional)
└── backend/             # Backend-specific data
    ├── embeddings.bin   # Embedding matrix
    └── doc_ids.bin      # Document ID mapping
```

## Error Handling

```rust
use libgrammstein::rag::RagError;

match index.add_document(doc) {
    Ok(id) => println!("Added document {}", id),
    Err(RagError::EmbeddingError(msg)) => {
        eprintln!("Failed to embed: {}", msg);
    }
    Err(RagError::IndexError(msg)) => {
        eprintln!("Index error: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## See Also

- [Document](document.md) - Document structures
- [Backend](backend.md) - Retrieval backends
- [Index](index.md) - RagIndex operations
- [Retriever](retriever.md) - Query interface
- [Builder](builder.md) - Index construction
- [Neural Overview](../neural/overview.md) - Embedding and summarization
- [Topic Overview](../topic/overview.md) - Topic extraction
