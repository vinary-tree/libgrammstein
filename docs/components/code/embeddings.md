# Code Embeddings

Neural code embeddings provide semantic representations of code using transformer models like UniXcoder and GraphCodeBERT.

## Overview

The embeddings module provides:

- **Code embedding**: Dense vector representations of code
- **Multiple models**: UniXcoder, GraphCodeBERT, CodeBERT
- **Caching**: Efficient storage of computed embeddings
- **Similarity scoring**: Semantic code comparison

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CodeEmbedder                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Embedding Models                          │ │
│  │                                                             │ │
│  │  UniXcoder ────► Unified cross-modal code understanding    │ │
│  │  GraphCodeBERT ► Data flow-aware embeddings                │ │
│  │  CodeBERT ─────► Original code BERT model                  │ │
│  │                                                             │ │
│  │  All models: 768-dimensional embeddings, 512 max length    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              ModernBertEmbedder Backend                     │ │
│  │                                                             │ │
│  │  • ONNX Runtime inference                                  │ │
│  │  • Batch processing                                        │ │
│  │  • Optional normalization                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Embedding Cache                           │ │
│  │                                                             │ │
│  │  DashMap<String, Vec<f32>> with automatic eviction         │ │
│  │  Configurable size (default: 10,000 embeddings)            │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## EmbeddingModel

Available code embedding models:

```rust
pub enum EmbeddingModel {
    /// UniXcoder - unified cross-modal model
    UniXcoder,
    /// GraphCodeBERT - data flow-aware embeddings
    GraphCodeBERT,
    /// CodeBERT - original code BERT
    CodeBERT,
    /// Custom/other model
    Custom,
}
```

### Model Characteristics

| Model | HuggingFace ID | Dimensions | Max Length | Best For |
|-------|----------------|------------|------------|----------|
| UniXcoder | `microsoft/unixcoder-base` | 768 | 512 | General code understanding |
| GraphCodeBERT | `microsoft/graphcodebert-base` | 768 | 512 | Data flow analysis |
| CodeBERT | `microsoft/codebert-base` | 768 | 512 | Basic code embeddings |

### Model Properties

```rust
use libgrammstein::code::EmbeddingModel;

let model = EmbeddingModel::UniXcoder;

// Get HuggingFace model ID
let model_id = model.hf_model_id();
println!("Model: {}", model_id);  // microsoft/unixcoder-base

// Get embedding dimension
let dim = model.embedding_dim();
println!("Dimensions: {}", dim);  // 768

// Get maximum sequence length
let max_len = model.max_length();
println!("Max length: {}", max_len);  // 512
```

## CodeEmbedderConfig

Configuration for the code embedder:

```rust
pub struct CodeEmbedderConfig {
    /// Which model to use
    pub model: EmbeddingModel,
    /// Device for inference (CPU, CUDA, etc.)
    pub device: Device,
    /// Whether to use caching
    pub use_cache: bool,
    /// Maximum cache size (number of embeddings)
    pub cache_size: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Batch size for bulk embedding
    pub batch_size: usize,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `UniXcoder` | Embedding model to use |
| `device` | `Cpu` | Inference device |
| `use_cache` | `true` | Enable embedding cache |
| `cache_size` | `10000` | Maximum cached embeddings |
| `normalize` | `true` | L2-normalize embeddings |
| `batch_size` | `32` | Batch size for bulk embedding |

### Creating Configuration

```rust
use libgrammstein::code::{CodeEmbedderConfig, EmbeddingModel};
use libgrammstein::neural::Device;

// Default configuration
let config = CodeEmbedderConfig::default();

// Custom configuration
let config = CodeEmbedderConfig {
    model: EmbeddingModel::GraphCodeBERT,
    device: Device::Cuda(0),  // Use GPU if available
    use_cache: true,
    cache_size: 50000,        // Larger cache
    normalize: true,
    batch_size: 64,           // Larger batches
};
```

## CodeEmbedder

Main interface for generating code embeddings:

```rust
pub struct CodeEmbedder {
    config: CodeEmbedderConfig,
    embedder: ModernBertEmbedder,
    cache: Option<DashMap<String, Vec<f32>>>,
}
```

### Creating an Embedder

```rust
use libgrammstein::code::{CodeEmbedder, CodeEmbedderConfig};

// With default configuration
let embedder = CodeEmbedder::new()?;

// With custom configuration
let config = CodeEmbedderConfig {
    model: EmbeddingModel::UniXcoder,
    cache_size: 20000,
    ..Default::default()
};
let embedder = CodeEmbedder::with_config(config)?;

// From local model path
let embedder = CodeEmbedder::from_path(
    "/path/to/model",
    CodeEmbedderConfig::default()
)?;
```

### Embedding Code

```rust
let embedder = CodeEmbedder::new()?;

// Embed a code snippet
let code = "def add(a, b): return a + b";
let embedding = embedder.embed(code)?;

println!("Embedding dimension: {}", embedding.len());  // 768
println!("First 5 values: {:?}", &embedding[..5]);
```

### Batch Embedding

```rust
let codes = vec![
    "def add(a, b): return a + b",
    "def sub(a, b): return a - b",
    "def mul(a, b): return a * b",
];

// Embed multiple snippets efficiently
let embeddings = embedder.embed_batch(&codes)?;

for (code, embedding) in codes.iter().zip(embeddings.iter()) {
    println!("Code: {} -> {} dims", &code[..20], embedding.len());
}
```

## Similarity Scoring

### Cosine Similarity

```rust
use libgrammstein::code::CodeEmbedder;

// Static method for comparing embeddings
let similarity = CodeEmbedder::cosine_similarity(&embedding_a, &embedding_b);
println!("Similarity: {:.3}", similarity);  // -1.0 to 1.0
```

### Scoring Code Similarity

```rust
let embedder = CodeEmbedder::new()?;

let code_a = "def add(x, y): return x + y";
let code_b = "def sum(a, b): return a + b";
let code_c = "class MyClass: pass";

// Score similarity between code snippets
let sim_ab = embedder.score_similarity(code_a, code_b)?;
let sim_ac = embedder.score_similarity(code_a, code_c)?;

println!("add vs sum: {:.3}", sim_ab);  // High similarity (~0.9)
println!("add vs class: {:.3}", sim_ac);  // Low similarity (~0.3)
```

### Scoring Completions

```rust
// Score how well a completion fits a context
let context = "def process_data(items):\n    result = []\n    for item in items:";
let candidate_a = "\n        result.append(item)";
let candidate_b = "\n        x = 42";

let score_a = embedder.score_completion(context, candidate_a)?;
let score_b = embedder.score_completion(context, candidate_b)?;

println!("Continuation score: {:.3}", score_a);  // Higher (more coherent)
println!("Unrelated score: {:.3}", score_b);     // Lower
```

## Caching

The embedder caches computed embeddings for efficiency:

```rust
let embedder = CodeEmbedder::new()?;

// First call computes embedding
let _ = embedder.embed("def foo(): pass")?;

// Second call uses cache
let _ = embedder.embed("def foo(): pass")?;  // Instant

// Check cache size
println!("Cached embeddings: {}", embedder.cache_size());

// Clear cache if needed
embedder.clear_cache();
```

### Cache Eviction

When the cache reaches capacity, ~10% of entries are evicted:

```rust
// With cache_size = 10000:
// At 10000 entries, evict ~1000 oldest entries
if cache.len() >= self.config.cache_size {
    let to_remove: Vec<String> = cache
        .iter()
        .take(self.config.cache_size / 10)
        .map(|e| e.key().clone())
        .collect();
    for key in to_remove {
        cache.remove(&key);
    }
}
```

## CodeEmbedderError

Error types for embedding operations:

```rust
pub enum CodeEmbedderError {
    /// Model loading failed
    ModelLoad(String),
    /// Embedding computation failed
    Embedding(String),
    /// Invalid input
    InvalidInput(String),
    /// Cache error
    Cache(String),
}
```

### Error Handling

```rust
use libgrammstein::code::{CodeEmbedder, CodeEmbedderError};

let embedder = CodeEmbedder::new()?;

match embedder.embed(code) {
    Ok(embedding) => {
        println!("Got embedding: {} dims", embedding.len());
    }
    Err(CodeEmbedderError::ModelLoad(msg)) => {
        eprintln!("Failed to load model: {}", msg);
    }
    Err(CodeEmbedderError::Embedding(msg)) => {
        eprintln!("Embedding failed: {}", msg);
    }
    Err(CodeEmbedderError::InvalidInput(msg)) => {
        eprintln!("Invalid input: {}", msg);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Integration Example

Complete example using code embeddings for code search:

```rust
use libgrammstein::code::{CodeEmbedder, CodeEmbedderConfig, EmbeddingModel};

struct CodeSearchIndex {
    embedder: CodeEmbedder,
    snippets: Vec<String>,
    embeddings: Vec<Vec<f32>>,
}

impl CodeSearchIndex {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = CodeEmbedderConfig {
            model: EmbeddingModel::UniXcoder,
            use_cache: true,
            normalize: true,
            ..Default::default()
        };

        Ok(Self {
            embedder: CodeEmbedder::with_config(config)?,
            snippets: Vec::new(),
            embeddings: Vec::new(),
        })
    }

    fn add_snippets(&mut self, snippets: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        let new_embeddings = self.embedder.embed_batch(snippets)?;

        for (snippet, embedding) in snippets.iter().zip(new_embeddings) {
            self.snippets.push(snippet.to_string());
            self.embeddings.push(embedding);
        }

        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> Result<Vec<(f32, &str)>, Box<dyn std::error::Error>> {
        let query_embedding = self.embedder.embed(query)?;

        let mut scores: Vec<(f32, usize)> = self.embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let sim = CodeEmbedder::cosine_similarity(&query_embedding, emb);
                (sim, i)
            })
            .collect();

        // Sort by similarity descending
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        Ok(scores
            .into_iter()
            .take(top_k)
            .map(|(score, idx)| (score, self.snippets[idx].as_str()))
            .collect())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = CodeSearchIndex::new()?;

    // Index some code snippets
    index.add_snippets(&[
        "def add(a, b): return a + b",
        "def subtract(a, b): return a - b",
        "def multiply(a, b): return a * b",
        "def divide(a, b): return a / b if b != 0 else None",
        "class Calculator: pass",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    ])?;

    // Search for similar code
    let query = "function to sum two numbers";
    let results = index.search(query, 3)?;

    println!("Query: {}", query);
    println!("Results:");
    for (score, snippet) in results {
        println!("  {:.3}: {}", score, snippet);
    }

    Ok(())
}
```

## Model Selection Guide

Choose the right model for your use case:

### UniXcoder (Recommended)

- Best overall performance
- Unified understanding across languages
- Good for code search and similarity

```rust
let config = CodeEmbedderConfig {
    model: EmbeddingModel::UniXcoder,
    ..Default::default()
};
```

### GraphCodeBERT

- Incorporates data flow information
- Better for semantic understanding
- Useful for bug detection

```rust
let config = CodeEmbedderConfig {
    model: EmbeddingModel::GraphCodeBERT,
    ..Default::default()
};
```

### CodeBERT

- Simpler model, faster inference
- Good baseline performance
- Lower resource requirements

```rust
let config = CodeEmbedderConfig {
    model: EmbeddingModel::CodeBERT,
    ..Default::default()
};
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Single embedding | O(n²) | Transformer attention |
| Batch embedding | O(b × n²) | b = batch size |
| Cosine similarity | O(d) | d = dimensions |
| Cache lookup | O(1) | DashMap |

### Optimization Tips

1. **Use batching**: Embed multiple snippets at once
2. **Enable caching**: Avoid recomputing embeddings
3. **Normalize embeddings**: Faster similarity computation
4. **Use GPU**: Enable CUDA for faster inference

## Thread Safety

`CodeEmbedder` is thread-safe for concurrent embedding:

```rust
use std::sync::Arc;
use rayon::prelude::*;

let embedder = Arc::new(CodeEmbedder::new()?);

let codes: Vec<&str> = vec![/* ... */];

let embeddings: Vec<_> = codes.par_iter()
    .map(|code| {
        embedder.embed(code).unwrap()
    })
    .collect();
```

The `DashMap` cache provides concurrent access without locks.

## Feature Flags

Code embeddings require the `code-neural` feature:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["code-neural"] }
```

## See Also

- [GNN](gnn.md) - Graph neural networks for code
- [Semantic Corrector](correctors/semantic.md) - Using embeddings for correction
- [Neural Module](../neural/overview.md) - Base neural infrastructure
- [Pipeline](pipeline.md) - End-to-end workflow
