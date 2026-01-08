# RAG Index

The `RagIndex` combines a retrieval backend with document metadata for semantic search.

## Index Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RagIndex<B>                                       │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        Config                                     │  │
│  │  embedding_dim: 768                                               │  │
│  │  max_documents: None (unlimited)                                  │  │
│  │  store_content: false                                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Backend<B>                                     │  │
│  │                                                                   │  │
│  │  DocumentId → Embedding                                           │  │
│  │  [0] → [0.12, -0.34, ..., 0.56]                                  │  │
│  │  [1] → [0.23, 0.45, ..., -0.67]                                  │  │
│  │  [2] → [-0.11, 0.22, ..., 0.33]                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Metadata HashMap                               │  │
│  │                                                                   │  │
│  │  DocumentId → DocumentMeta                                        │  │
│  │  [0] → { uri: "doc1.md", title: "Intro", synopsis: "..." }       │  │
│  │  [1] → { uri: "doc2.md", title: "Guide", synopsis: "..." }       │  │
│  │  [2] → { uri: "doc3.md", title: "API", synopsis: "..." }         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    TopicModel (optional)                          │  │
│  │                                                                   │  │
│  │  Topics with keywords and descriptions                            │  │
│  │  Document → Topic mappings                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

```rust
use libgrammstein::rag::RagIndexConfig;

let config = RagIndexConfig {
    // Embedding dimension (must match embedder)
    embedding_dim: 768,

    // Maximum documents (None = unlimited)
    max_documents: Some(1_000_000),

    // Store document content (increases memory)
    store_content: false,
};
```

## Creating an Index

### Empty Index

```rust
use libgrammstein::rag::{RagIndex, RagIndexConfig, ExactCosineBackend};

let config = RagIndexConfig::default();
let index: RagIndex<ExactCosineBackend> = RagIndex::new(config);
```

### With Custom Backend

```rust
use libgrammstein::rag::{RagIndex, RagIndexConfig, HnswBackend, HnswConfig};

let hnsw_config = HnswConfig::default();
let backend = HnswBackend::new(768, hnsw_config);
let index = RagIndex::with_backend(RagIndexConfig::default(), backend);
```

### From Directory

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

let builder = IndexBuilder::new(IndexBuilderConfig::default())?;
let index = builder.build_from_directory("./documents", None)?;
```

## Adding Documents

```rust
use libgrammstein::rag::{Document, DocumentMeta};

// Add a full document (with embedding)
let id = index.add_document(document)?;

// ID is auto-assigned sequentially
println!("Added document {}", id.0);
```

## Querying

### Basic Query

```rust
// Query with pre-computed embedding
let results = index.query(&query_embedding, 10);

for (meta, score) in results {
    println!("{}: {:.4}", meta.title.unwrap_or_default(), score);
}
```

### Query Results

Returns `Vec<(DocumentMeta, f32)>` sorted by descending score:

```rust
let results = index.query(&embedding, 5);

for (meta, score) in &results {
    // Score is cosine similarity (0.0 to 1.0 for normalized vectors)
    println!("Score: {:.4}", score);
    println!("Title: {}", meta.title.as_deref().unwrap_or("Untitled"));
    println!("Synopsis: {}", meta.synopsis);
    println!("Topics: {:?}", meta.topic_ids);
    println!();
}
```

## Document Operations

### Get Document Metadata

```rust
if let Some(meta) = index.get(DocumentId::new(42)) {
    println!("Found: {}", meta.uri);
}
```

### Check Existence

```rust
if index.contains(DocumentId::new(42)) {
    println!("Document exists");
}
```

### Remove Document

```rust
let removed = index.remove(DocumentId::new(42))?;
if removed {
    println!("Document removed");
}
```

### Iterate Documents

```rust
// Iterate all documents
for (id, meta) in index.iter() {
    println!("{}: {}", id.0, meta.uri);
}

// Get all document IDs
let ids: Vec<_> = index.document_ids().collect();
```

### Index Size

```rust
println!("Documents: {}", index.len());
println!("Empty: {}", index.is_empty());
```

## Topic Integration

### Extract Topics

```rust
use libgrammstein::topic::TopicConfig;

// Get embeddings and texts
let embeddings = index.get_all_embeddings();
let texts: Vec<_> = index.iter()
    .map(|(_, meta)| meta.synopsis.clone())
    .collect();

// Extract topics
let config = TopicConfig::default();
index.extract_topics(&config, &embeddings, &texts)?;
```

### Access Topic Model

```rust
if let Some(topic_model) = index.topic_model() {
    for topic in topic_model.topics() {
        println!("Topic {}: {}", topic.id.0, topic.keyword_summary(5));
    }
}
```

### Document Topics

```rust
// Get topics for a document
if let Some(topic_ids) = index.document_topics(DocumentId::new(0)) {
    for topic_id in topic_ids {
        if let Some(topic) = index.topic_model().and_then(|m| m.get(*topic_id)) {
            println!("Topic: {}", topic.description);
        }
    }
}
```

### Clear Topics

```rust
index.clear_topic_model();
```

## Persistence

### Save Index

```rust
index.save("./my_index")?;
```

Creates directory structure:
```
my_index/
├── config.json          # RagIndexConfig
├── state.json           # Index state (next_id, etc.)
├── metadata.json        # All DocumentMeta
├── topic_model.json     # TopicModel (if extracted)
└── backend/             # Backend-specific data
    ├── embeddings.bin
    └── doc_ids.bin
```

### Load Index

```rust
let index: RagIndex<ExactCosineBackend> = RagIndex::load("./my_index")?;
```

### Extend Existing Index

```rust
// Load existing index
let mut index = RagIndex::load("./my_index")?;

// Add new documents
for doc in new_documents {
    index.add_document(doc)?;
}

// Save updated index
index.save("./my_index")?;
```

## ID Allocation

Document IDs are allocated sequentially:

```rust
// Allocate ID for manual document creation
let id = index.allocate_id();

// Or let add_document allocate automatically
let id = index.add_document(doc)?;
```

## Get All Embeddings

For operations like topic extraction:

```rust
// Get all embeddings (Vec<Vec<f32>>)
let embeddings = index.get_all_embeddings();

// Embeddings are in same order as document_ids()
let ids: Vec<_> = index.document_ids().collect();
for (id, emb) in ids.iter().zip(embeddings.iter()) {
    println!("Doc {}: {} dims", id.0, emb.len());
}
```

## Clear Index

```rust
// Remove all documents
index.clear();

assert!(index.is_empty());
```

## Thread Safety

The index uses interior mutability for safe concurrent access:

```rust
use std::sync::Arc;
use std::thread;

let index = Arc::new(RagIndex::load("./index")?);

// Multiple threads can query concurrently
let handles: Vec<_> = queries.iter().map(|q| {
    let index = Arc::clone(&index);
    let q = q.clone();
    thread::spawn(move || index.query(&q, 10))
}).collect();
```

## Error Handling

```rust
use libgrammstein::rag::RagError;

match index.add_document(doc) {
    Ok(id) => println!("Added: {}", id.0),
    Err(RagError::IndexError(msg)) => {
        eprintln!("Index error: {}", msg);
    }
    Err(RagError::EmbeddingError(msg)) => {
        eprintln!("Embedding dimension mismatch: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Best Practices

### 1. Choose Backend by Scale

```rust
// < 1M documents: ExactCosineBackend (default)
let index: RagIndex<ExactCosineBackend> = RagIndex::new(config);

// > 1M documents: HnswBackend
let index: RagIndex<HnswBackend> = RagIndex::with_backend(config, backend);
```

### 2. Save Periodically for Large Indices

```rust
const SAVE_INTERVAL: usize = 10_000;

for (i, doc) in documents.iter().enumerate() {
    index.add_document(doc.clone())?;

    if (i + 1) % SAVE_INTERVAL == 0 {
        index.save("./index")?;
        println!("Checkpoint at {} documents", i + 1);
    }
}
```

### 3. Extract Topics After Building

```rust
// Build index first
let index = builder.build_from_directory("./docs", None)?;

// Then extract topics (requires embeddings)
let config = TopicConfig::default();
let embeddings = index.get_all_embeddings();
let texts: Vec<_> = index.iter().map(|(_, m)| m.synopsis.clone()).collect();
index.extract_topics(&config, &embeddings, &texts)?;

// Save with topics
index.save("./index")?;
```

## See Also

- [Overview](overview.md) - RAG module introduction
- [Backend](backend.md) - Backend implementations
- [Retriever](retriever.md) - High-level query interface
- [Builder](builder.md) - Index construction
- [Topic Overview](../topic/overview.md) - Topic extraction
