# Code Embeddings Overview

libgrammstein provides neural code embeddings using state-of-the-art pre-trained models for semantic code understanding.

## What are Code Embeddings?

Code embeddings are dense vector representations of source code that capture semantic meaning. Similar code snippets produce similar vectors, enabling:

- **Code Search**: Find code by semantic similarity, not just text matching
- **Clone Detection**: Identify semantically similar code (even with different syntax)
- **Code Completion**: Rank completion candidates by semantic fit
- **Bug Detection**: Find code patterns similar to known buggy code

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Code Embedding Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Source Code                                                            │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Tokenization                                  │   │
│   │  "fn main() { println!(\"hello\"); }"                           │   │
│   │         ↓                                                        │   │
│   │  [CLS, fn, main, (, ), {, print, ln, !, ..., SEP]               │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                   Neural Model                                   │   │
│   │  ┌─────────────────────────────────────────────────────────┐    │   │
│   │  │        Transformer Encoder (ONNX Runtime)               │    │   │
│   │  │  • Self-attention over token sequence                   │    │   │
│   │  │  • Layer-by-layer representation learning               │    │   │
│   │  │  • Code structure awareness                             │    │   │
│   │  └─────────────────────────────────────────────────────────┘    │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                   Embedding Vector                               │   │
│   │  [0.12, -0.34, 0.56, ..., 0.78]  (256 or 768 dimensions)        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Supported Models

libgrammstein supports three pre-trained code embedding models:

| Model | Parameters | Embedding Dim | Max Length | Strengths |
|-------|------------|---------------|------------|-----------|
| [CodeT5+](codet5.md) | 110M | 256 | 512 | General code understanding |
| [UniXcoder](unixcoder.md) | 125M | 768 | 512 | Code-to-code similarity |
| [GraphCodeBERT](graphcodebert.md) | 125M | 768 | 512 | Data flow understanding |

All models use ONNX Runtime for efficient CPU inference.

## Quick Start

### Single Model Embedding

```rust
use libgrammstein::neural::code::{
    CodeT5Embedder, CodeT5Config,
    CodeEmbedder, CodeLanguage,
};

// Load model from directory
let config = CodeT5Config::codet5p_110m_embedding("/path/to/model");
let embedder = CodeT5Embedder::load(config)?;

// Embed code
let code = "fn calculate_sum(items: &[i32]) -> i32 { items.iter().sum() }";
let embedding = embedder.embed_code(code, CodeLanguage::Rust)?;

println!("Embedding dimension: {}", embedding.len());
println!("First 5 values: {:?}", &embedding[..5]);
```

### Ensemble Embedding

Combine multiple models for improved accuracy:

```rust
use libgrammstein::neural::code::{
    CodeT5Embedder, UniXcoderEmbedder, GraphCodeBertEmbedder,
    EnsembleCodeEmbedder, EnsembleStrategy,
    CodeEmbedder, CodeLanguage,
};
use std::sync::Arc;

// Load individual models
let codet5 = Arc::new(CodeT5Embedder::from_directory("/path/to/codet5p")?);
let unixcoder = Arc::new(UniXcoderEmbedder::from_directory("/path/to/unixcoder")?);
let graphcodebert = Arc::new(GraphCodeBertEmbedder::from_directory("/path/to/graphcodebert")?);

// Create ensemble with concatenation
let ensemble = EnsembleCodeEmbedder::new(vec![
    codet5.clone() as Arc<dyn CodeEmbedder>,
    unixcoder.clone() as Arc<dyn CodeEmbedder>,
    graphcodebert.clone() as Arc<dyn CodeEmbedder>,
]);

// Embed with ensemble
let embedding = ensemble.embed_code(code, CodeLanguage::Rust)?;
println!("Ensemble dimension: {}", embedding.len()); // 256 + 768 + 768 = 1792
```

### Code Similarity

```rust
use libgrammstein::neural::code::cosine_similarity;

// Embed two code snippets
let code1 = "def add(a, b): return a + b";
let code2 = "def sum(x, y): return x + y";

let emb1 = embedder.embed_code(code1, CodeLanguage::Python)?;
let emb2 = embedder.embed_code(code2, CodeLanguage::Python)?;

// Calculate similarity (0.0 to 1.0)
let similarity = cosine_similarity(&emb1, &emb2);
println!("Similarity: {:.2}", similarity); // ~0.95 (semantically similar)
```

## Supported Languages

All models support multiple programming languages:

```rust
pub enum CodeLanguage {
    Python,
    Java,
    JavaScript,
    TypeScript,
    Go,
    Ruby,
    Php,
    C,
    Cpp,
    CSharp,
    Rust,
    Kotlin,
    Scala,
    Swift,
    Haskell,
    OCaml,
    Elixir,
    Bash,
    Rholang,  // F1R3FLY.io process calculus
    MeTTa,    // F1R3FLY.io meta-language
    Unknown,
}
```

Parse language from file extension:

```rust
let lang = CodeLanguage::from_extension("rs");  // -> Rust
let lang = CodeLanguage::from_extension("rho"); // -> Rholang
let lang = CodeLanguage::from_extension("metta"); // -> MeTTa
```

## Core Types

### CodeEmbedder Trait

All embedders implement this trait:

```rust
pub trait CodeEmbedder: Send + Sync {
    /// Embed a single code snippet.
    fn embed_code(&self, code: &str, language: CodeLanguage) -> Result<Vec<f32>>;

    /// Embed multiple code snippets in a batch.
    fn embed_code_batch(
        &self,
        codes: &[&str],
        languages: &[CodeLanguage],
    ) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Get the model name.
    fn model_name(&self) -> &str;

    /// Get the maximum sequence length supported.
    fn max_sequence_length(&self) -> usize;

    /// Get supported languages.
    fn supported_languages(&self) -> &[CodeLanguage];
}
```

### CodeEmbeddingError

Error types for embedding operations:

```rust
pub enum CodeEmbeddingError {
    /// Model loading failed.
    ModelLoad(String),

    /// Tokenization failed.
    Tokenization(String),

    /// Inference failed.
    Inference(String),

    /// ONNX Runtime error.
    Onnx(String),

    /// Unsupported language.
    UnsupportedLanguage(String),

    /// I/O error.
    Io(std::io::Error),
}
```

## Use Cases

### Semantic Code Search

Build a code search engine:

```rust
use libgrammstein::neural::code::cosine_similarity;

struct CodeSearchEngine {
    embedder: Arc<dyn CodeEmbedder>,
    index: Vec<(String, Vec<f32>)>, // (code, embedding)
}

impl CodeSearchEngine {
    fn index_code(&mut self, code: &str, language: CodeLanguage) -> Result<()> {
        let embedding = self.embedder.embed_code(code, language)?;
        self.index.push((code.to_string(), embedding));
        Ok(())
    }

    fn search(&self, query: &str, language: CodeLanguage, top_k: usize) -> Result<Vec<&str>> {
        let query_embedding = self.embedder.embed_code(query, language)?;

        let mut scored: Vec<_> = self.index.iter()
            .map(|(code, emb)| (code.as_str(), cosine_similarity(&query_embedding, emb)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scored.into_iter()
            .take(top_k)
            .map(|(code, _)| code)
            .collect())
    }
}
```

### Clone Detection

Find semantically similar code:

```rust
fn detect_clones(
    embedder: &dyn CodeEmbedder,
    functions: &[(String, &str)],
    threshold: f32,
) -> Vec<(String, String, f32)> {
    let embeddings: Vec<_> = functions.iter()
        .map(|(name, code)| (name, embedder.embed_code(code, CodeLanguage::Unknown)))
        .collect();

    let mut clones = Vec::new();

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            if let (Ok(ref emb_i), Ok(ref emb_j)) = (&embeddings[i].1, &embeddings[j].1) {
                let sim = cosine_similarity(emb_i, emb_j);
                if sim >= threshold {
                    clones.push((
                        embeddings[i].0.clone(),
                        embeddings[j].0.clone(),
                        sim,
                    ));
                }
            }
        }
    }

    clones
}
```

### RAG Integration

Use with libgrammstein RAG for code retrieval:

```rust
use libgrammstein::rag::{RagIndex, RagConfig, Document};

// Create documents from code
let docs: Vec<Document> = code_files.iter()
    .map(|(path, code)| Document {
        id: path.to_string(),
        content: code.to_string(),
        metadata: Default::default(),
    })
    .collect();

// Build RAG index with code embeddings
let rag = RagIndex::build_with_embedder(
    docs,
    embedder,
    RagConfig::default(),
)?;

// Search for similar code
let results = rag.search("authentication handler", 10)?;
```

## Performance Considerations

### Memory Usage

| Model | ONNX Size | RAM (loaded) | Cache per 10K |
|-------|-----------|--------------|---------------|
| CodeT5+ | ~220MB | ~500MB | ~10MB |
| UniXcoder | ~500MB | ~1GB | ~30MB |
| GraphCodeBERT | ~500MB | ~1GB | ~30MB |

### Throughput

Approximate embeddings per second (CPU, single-threaded):

| Model | Single | Batched (32) |
|-------|--------|--------------|
| CodeT5+ | 50/sec | 200/sec |
| UniXcoder | 30/sec | 150/sec |
| GraphCodeBERT | 30/sec | 150/sec |

### Optimization Tips

1. **Use caching**: Enable the built-in cache for repeated embeddings
2. **Batch processing**: Use `embed_code_batch` for multiple snippets
3. **Thread count**: Tune `num_threads` for your CPU
4. **Optimization level**: Use level 3 for best inference speed

## Feature Flags

Enable code embedding features in `Cargo.toml`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["code-embedding"] }

# Or enable specific models
libgrammstein = { version = "0.1", features = [
    "code-embedding-codet5",
    "code-embedding-unixcoder",
    "code-embedding-graphcodebert"
]}

# Enable all models
libgrammstein = { version = "0.1", features = ["code-embedding-all"] }
```

## Related Components

- [CodeT5+](codet5.md) - CodeT5+ model documentation
- [UniXcoder](unixcoder.md) - UniXcoder model documentation
- [GraphCodeBERT](graphcodebert.md) - GraphCodeBERT model documentation
- [Ensemble](ensemble.md) - Multi-model ensemble strategies
- [Caching](caching.md) - Embedding cache configuration
