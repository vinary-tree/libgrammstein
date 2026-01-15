# LaTeX Embeddings

The embedding module provides dense vector representations for LaTeX commands, equations, and documents, enabling semantic similarity search and neural scoring.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   LaTeX Embedding System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ ModernBERT      │        │ Pre-trained     │            │
│  │ Embedder        │        │ Command Emb.    │            │
│  └────────┬────────┘        └────────┬────────┘            │
│           │                          │                      │
│           ▼                          ▼                      │
│  ┌─────────────────────────────────────────┐               │
│  │           Embedding Cache               │               │
│  └─────────────────────────────────────────┘               │
│                      │                                      │
│           ┌──────────┴──────────┐                          │
│           ▼                     ▼                          │
│  ┌─────────────────┐   ┌─────────────────┐                 │
│  │ Document Emb.   │   │ Equation Emb.   │                 │
│  └─────────────────┘   └─────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Embedding Types

### 1. Document Embeddings

Full document representations using ModernBERT:

```rust
use libgrammstein::neural::{ModernBertEmbedder, EmbeddingConfig};

let config = EmbeddingConfig {
    pooling: PoolingStrategy::Cls,
    normalize: true,
    cache_size: 10000,
    batch_size: 32,
    ..Default::default()
};

let embedder = ModernBertEmbedder::new(config)?;
let embedding = embedder.embed_document(
    Some("Quantum Mechanics"),
    r"We study the Schrödinger equation $i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi$"
)?;
```

### 2. Equation Embeddings

Specialized embeddings for mathematical expressions:

```rust
let equation = r"\sum_{i=1}^{n} x_i^2 = \|x\|_2^2";
let embedding = embedder.embed(equation)?;

// Find similar equations
let similar = rag_index.query(&embedding, 10)?;
```

### 3. Command Embeddings

Pre-trained embeddings for LaTeX commands:

```rust
use libgrammstein::latex::CommandEmbeddings;

let cmd_embeddings = CommandEmbeddings::load("latex_commands.emb")?;

let alpha_emb = cmd_embeddings.get(r"\alpha")?;
let beta_emb = cmd_embeddings.get(r"\beta")?;

// Semantic similarity
let sim = ModernBertEmbedder::cosine_similarity(&alpha_emb, &beta_emb);
// Greek letters cluster together
```

## Embedding Configuration

```rust
pub struct EmbeddingConfig {
    /// Model configuration
    pub model_config: ModernBertConfig,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
    /// Normalize to unit length
    pub normalize: bool,
    /// Cache size (0 to disable)
    pub cache_size: usize,
    /// Batch size for parallel embedding
    pub batch_size: usize,
}
```

### Pooling Strategies

```rust
pub enum PoolingStrategy {
    /// [CLS] token embedding (default)
    Cls,
    /// Mean of all token embeddings
    MeanPooling,
    /// Max pooling across tokens
    MaxPooling,
}
```

| Strategy | Use Case | Quality | Speed |
|----------|----------|---------|-------|
| CLS | Classification | High | Fast |
| MeanPooling | Similarity | Higher | Medium |
| MaxPooling | Key features | Variable | Medium |

## Batch Embedding

Process multiple documents efficiently:

```rust
let documents = vec![
    "First equation: $E = mc^2$",
    "Second equation: $F = ma$",
    "Third equation: $V = IR$",
];

let embeddings = embedder.embed_batch(&documents)?;
```

## Document Embedding Workflow

```rust
use libgrammstein::neural::{BatchDocumentEmbedder, DocumentEmbedding};

let batch_embedder = BatchDocumentEmbedder::new(embedder);

// Embed with metadata
let documents = vec![
    ("doc1".to_string(), Some("Title 1".to_string()), "Content 1".to_string()),
    ("doc2".to_string(), Some("Title 2".to_string()), "Content 2".to_string()),
];

let doc_embeddings: Vec<DocumentEmbedding> = batch_embedder.embed_documents(&documents)?;

for emb in doc_embeddings {
    println!("Doc {}: {} dimensions", emb.document_id, emb.embedding.len());
}
```

## Similarity Computation

```rust
// Cosine similarity (recommended for normalized embeddings)
let sim = ModernBertEmbedder::cosine_similarity(&emb_a, &emb_b);

// Euclidean distance
let dist: f32 = emb_a.iter()
    .zip(emb_b.iter())
    .map(|(a, b)| (a - b).powi(2))
    .sum::<f32>()
    .sqrt();

// Dot product (for normalized embeddings, equals cosine)
let dot: f32 = emb_a.iter()
    .zip(emb_b.iter())
    .map(|(a, b)| a * b)
    .sum();
```

## Embedding Cache

The embedder caches computed embeddings:

```rust
let embedder = ModernBertEmbedder::new(EmbeddingConfig {
    cache_size: 10000,  // Cache up to 10k embeddings
    ..Default::default()
})?;

// First call: computes and caches
let emb1 = embedder.embed("equation")?;

// Second call: returns cached
let emb2 = embedder.embed("equation")?;

// Check cache stats
if let Some(cache_size) = embedder.cache_stats() {
    println!("Cached: {} embeddings", cache_size);
}

// Clear cache if needed
embedder.clear_cache();
```

## Training Command Embeddings

Pre-train command embeddings on LaTeX corpus:

```rust
use libgrammstein::latex::CommandEmbeddingTrainer;

let trainer = CommandEmbeddingTrainer::new(TrainerConfig {
    embedding_dim: 256,
    window_size: 5,
    min_count: 10,
    negative_samples: 5,
    epochs: 10,
    learning_rate: 0.025,
});

// Train on corpus
for paper in arxiv_corpus.iter() {
    let tokens = tokenizer.tokenize(&paper.content);
    let commands: Vec<_> = tokens.iter()
        .filter(|t| t.kind == LaTeXTokenKind::Command)
        .collect();
    trainer.add_document(&commands);
}

let embeddings = trainer.train()?;
embeddings.save("latex_commands.emb")?;
```

## Embedding Dimension

ModernBERT produces embeddings of dimension:

| Model | Hidden Size | Embedding Dim |
|-------|------------|---------------|
| ModernBERT-base | 768 | 768 |
| ModernBERT-large | 1024 | 1024 |

Access dimension:

```rust
let dim = embedder.embedding_dim();
println!("Embedding dimension: {}", dim);  // 768
```

## Normalization

Normalize embeddings to unit length:

```rust
// Automatic normalization during embedding
let config = EmbeddingConfig {
    normalize: true,  // Embeddings have L2 norm = 1
    ..Default::default()
};

// Manual normalization
let normalized = ModernBertEmbedder::normalize(&embedding);
```

## Query vs Document Embeddings

Some models benefit from different treatment:

```rust
// Document embedding (with title)
let doc_emb = embedder.embed_document(
    Some("Document Title"),
    "Document content here..."
)?;

// Query embedding (no prefix)
let query_emb = embedder.embed_query("search query")?;

// Use for retrieval
let results = index.query(&query_emb, 10)?;
```

## Serialization

Save and load embeddings:

```rust
// Save embeddings
embeddings.save("embeddings.bin")?;

// Load embeddings
let embeddings = CommandEmbeddings::load("embeddings.bin")?;

// Export to text format (for debugging)
embeddings.save_text("embeddings.txt")?;
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single embed | 5-10ms | GPU |
| Batch embed (32) | 20-30ms | GPU |
| Cache lookup | <0.1ms | Memory |
| Similarity | <0.01ms | 768-dim |

## GPU Acceleration

```rust
use libgrammstein::neural::Device;

let config = EmbeddingConfig {
    model_config: ModernBertConfig {
        device: Device::Cuda(0),  // First GPU
        ..Default::default()
    },
    ..Default::default()
};
```

## Related

- [Neural Rescorer](./rescorer.md): Using embeddings for rescoring
- [RAG](./rag.md): Embedding-based retrieval
- [Overview](./overview.md): Module architecture
