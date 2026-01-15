# CodeT5+ Embedder

CodeT5+ is Salesforce's open-source code large language model, optimized for code understanding and generation tasks.

## Model Overview

| Property | Value |
|----------|-------|
| **Organization** | Salesforce Research |
| **Model ID** | `Salesforce/codet5p-110m-embedding` |
| **Parameters** | 110M |
| **Embedding Dimension** | 256 |
| **Max Sequence Length** | 512 tokens |
| **Architecture** | Encoder-only (for embeddings) |

CodeT5+ was trained on a large corpus of code from 9 programming languages, making it effective for general code understanding tasks.

## Loading the Model

### From Directory

```rust
use libgrammstein::neural::code::{CodeT5Embedder, CodeT5Config, CodeLanguage};

// Load from a local directory containing model.onnx and tokenizer.json
let embedder = CodeT5Embedder::from_directory("/path/to/codet5p-110m-embedding")?;

// Embed code
let embedding = embedder.embed_code("fn main() { }", CodeLanguage::Rust)?;
```

### With Custom Configuration

```rust
let config = CodeT5Config {
    model_path: "/path/to/model.onnx".to_string(),
    tokenizer_path: "/path/to/tokenizer.json".to_string(),
    max_length: 512,
    use_language_prefix: false,
    num_threads: 4,
    optimization_level: 3,  // Maximum optimization
    cache_config: Some(CodeEmbeddingCacheConfig {
        max_entries: 10000,
        hash_keys: true,
    }),
    normalize: true,
    embedding_dim: Some(256),
};

let embedder = CodeT5Embedder::load(config)?;
```

## Configuration Options

### CodeT5Config

```rust
pub struct CodeT5Config {
    /// Path to ONNX model file.
    pub model_path: String,

    /// Path to tokenizer.json file.
    pub tokenizer_path: String,

    /// Maximum sequence length (default: 512).
    pub max_length: usize,

    /// Whether to use language prefix tokens.
    pub use_language_prefix: bool,

    /// Number of threads for inference.
    pub num_threads: usize,

    /// Graph optimization level (0-3).
    pub optimization_level: u8,

    /// Cache configuration (None to disable caching).
    pub cache_config: Option<CodeEmbeddingCacheConfig>,

    /// Whether to normalize embeddings.
    pub normalize: bool,

    /// Embedding dimension (detected from model or set explicitly).
    pub embedding_dim: Option<usize>,
}
```

### Configuration Presets

```rust
// Standard 110M embedding model
let config = CodeT5Config::codet5p_110m_embedding("/path/to/model");

// Custom settings
let config = CodeT5Config {
    model_path: "/custom/path/model.onnx".to_string(),
    tokenizer_path: "/custom/path/tokenizer.json".to_string(),
    max_length: 256,  // Shorter for faster inference
    num_threads: 8,   // More threads on larger machines
    ..Default::default()
};
```

## Optimization Levels

The `optimization_level` setting controls ONNX Runtime optimizations:

| Level | Description | Use Case |
|-------|-------------|----------|
| 0 | Disabled | Debugging |
| 1 | Basic | Compatibility |
| 2 | Extended | Balanced |
| 3 | Full (default) | Production |

## Supported Languages

CodeT5+ was trained on these languages:

- Python
- Java
- JavaScript
- Go
- Ruby
- PHP
- C
- C++
- C#

Other languages work but may have reduced accuracy.

## Examples

### Basic Embedding

```rust
let embedder = CodeT5Embedder::from_directory("/path/to/model")?;

// Python code
let python_code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)";
let embedding = embedder.embed_code(python_code, CodeLanguage::Python)?;

println!("Dimension: {}", embedding.len());  // 256
```

### Batch Embedding

```rust
let codes = vec![
    "def add(a, b): return a + b",
    "def sub(a, b): return a - b",
    "def mul(a, b): return a * b",
];
let languages = vec![CodeLanguage::Python; 3];

let embeddings = embedder.embed_code_batch(&codes.iter().map(|s| *s).collect::<Vec<_>>(), &languages)?;

for (code, emb) in codes.iter().zip(embeddings.iter()) {
    println!("{}: {} dimensions", &code[..20], emb.len());
}
```

### Similarity Computation

```rust
use libgrammstein::neural::code::cosine_similarity;

let code1 = "fn sum(nums: &[i32]) -> i32 { nums.iter().sum() }";
let code2 = "fn total(values: &[i32]) -> i32 { values.iter().fold(0, |a, b| a + b) }";
let code3 = "fn reverse(s: &str) -> String { s.chars().rev().collect() }";

let emb1 = embedder.embed_code(code1, CodeLanguage::Rust)?;
let emb2 = embedder.embed_code(code2, CodeLanguage::Rust)?;
let emb3 = embedder.embed_code(code3, CodeLanguage::Rust)?;

println!("sum vs total: {:.3}", cosine_similarity(&emb1, &emb2));    // ~0.85 (similar)
println!("sum vs reverse: {:.3}", cosine_similarity(&emb1, &emb3));  // ~0.20 (different)
```

### Cache Management

```rust
// Check cache size
if let Some(size) = embedder.cache_stats() {
    println!("Cached embeddings: {}", size);
}

// Clear cache
embedder.clear_cache();
```

## Model Acquisition

### Converting from HuggingFace

1. Download the PyTorch model:
```bash
git lfs install
git clone https://huggingface.co/Salesforce/codet5p-110m-embedding
```

2. Export to ONNX (Python):
```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("Salesforce/codet5p-110m-embedding")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-110m-embedding")

# Save tokenizer
tokenizer.save_pretrained("./codet5p-onnx")

# Export model
dummy_input = tokenizer("def foo(): pass", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "./codet5p-onnx/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
)
```

3. Load in Rust:
```rust
let embedder = CodeT5Embedder::from_directory("./codet5p-onnx")?;
```

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Single embedding | ~20ms |
| Batch (32) | ~150ms |
| Memory (loaded) | ~500MB |
| ONNX file size | ~220MB |

### Optimization Tips

1. **Increase threads** for larger machines:
```rust
let config = CodeT5Config {
    num_threads: num_cpus::get(),
    ..CodeT5Config::codet5p_110m_embedding("/path")
};
```

2. **Use caching** for repeated embeddings:
```rust
let config = CodeT5Config {
    cache_config: Some(CodeEmbeddingCacheConfig {
        max_entries: 50000,  // Cache more
        hash_keys: true,
    }),
    ..Default::default()
};
```

3. **Reduce sequence length** if codes are short:
```rust
let config = CodeT5Config {
    max_length: 256,  // Half the default
    ..Default::default()
};
```

## Thread Safety

The embedder is thread-safe (`Send + Sync`). The ONNX session is protected by a mutex:

```rust
use std::sync::Arc;
use rayon::prelude::*;

let embedder = Arc::new(CodeT5Embedder::from_directory("/path")?);

let embeddings: Vec<_> = code_snippets.par_iter()
    .map(|code| embedder.embed_code(code, CodeLanguage::Unknown))
    .collect::<Result<Vec<_>, _>>()?;
```

## See Also

- [Overview](overview.md) - Code embeddings introduction
- [UniXcoder](unixcoder.md) - Alternative model
- [GraphCodeBERT](graphcodebert.md) - Structure-aware model
- [Ensemble](ensemble.md) - Combining multiple models
