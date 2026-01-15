# UniXcoder Embedder

UniXcoder is Microsoft's unified cross-modal pre-trained model for programming languages, excelling at code-to-code similarity tasks.

## Model Overview

| Property | Value |
|----------|-------|
| **Organization** | Microsoft Research |
| **Model ID** | `microsoft/unixcoder-base` |
| **Parameters** | 125M |
| **Embedding Dimension** | 768 |
| **Max Sequence Length** | 512 tokens |
| **Architecture** | Unified encoder |

UniXcoder uses a unified architecture that handles:
- Code-to-code similarity
- Code-to-text matching
- Code clone detection
- Code search

## Loading the Model

### From Directory

```rust
use libgrammstein::neural::code::{UniXcoderEmbedder, UniXcoderConfig, CodeLanguage};

// Load from a local directory containing model.onnx and tokenizer.json
let embedder = UniXcoderEmbedder::from_directory("/path/to/unixcoder-base")?;

// Embed code
let embedding = embedder.embed_code("def hello(): print('world')", CodeLanguage::Python)?;
```

### With Custom Configuration

```rust
let config = UniXcoderConfig {
    model_path: "/path/to/model.onnx".to_string(),
    tokenizer_path: "/path/to/tokenizer.json".to_string(),
    max_length: 512,
    num_threads: 4,
    optimization_level: 3,
    cache_config: Some(CodeEmbeddingCacheConfig {
        max_entries: 10000,
        hash_keys: true,
    }),
    normalize: true,
    embedding_dim: 768,
};

let embedder = UniXcoderEmbedder::load(config)?;
```

## Configuration Options

### UniXcoderConfig

```rust
pub struct UniXcoderConfig {
    /// Path to ONNX model file.
    pub model_path: String,

    /// Path to tokenizer.json file.
    pub tokenizer_path: String,

    /// Maximum sequence length (default: 512).
    pub max_length: usize,

    /// Number of threads for inference.
    pub num_threads: usize,

    /// Graph optimization level (0-3).
    pub optimization_level: u8,

    /// Cache configuration (None to disable caching).
    pub cache_config: Option<CodeEmbeddingCacheConfig>,

    /// Whether to normalize embeddings.
    pub normalize: bool,

    /// Embedding dimension (768 for unixcoder-base).
    pub embedding_dim: usize,
}
```

## Supported Languages

UniXcoder was trained on CodeSearchNet (6 languages):

- Python
- Java
- JavaScript
- Go
- Ruby
- PHP

The model generalizes to other languages but with reduced accuracy.

## Embedding Strategy

UniXcoder uses CLS token pooling for embeddings:

```
┌──────────────────────────────────────────────────────────────────┐
│  Input: [CLS] def add(a, b): return a + b [SEP]                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  UniXcoder Encoder                           ││
│  │  • 12 transformer layers                                     ││
│  │  • 768 hidden dimension                                      ││
│  │  • Bidirectional attention                                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Output: [CLS_emb, t1_emb, t2_emb, ..., SEP_emb]               │
│             │                                                    │
│             ▼                                                    │
│  Embedding: CLS_emb (768 dimensions)                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Examples

### Basic Embedding

```rust
let embedder = UniXcoderEmbedder::from_directory("/path/to/model")?;

let code = r#"
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"#;

let embedding = embedder.embed_code(code, CodeLanguage::Python)?;
println!("Dimension: {}", embedding.len());  // 768
```

### Clone Detection

UniXcoder excels at detecting semantic clones:

```rust
use libgrammstein::neural::code::cosine_similarity;

// Type 1 Clone: Identical (same code)
let original = "def add(a, b): return a + b";
let clone1 = "def add(a, b): return a + b";

// Type 2 Clone: Renamed identifiers
let clone2 = "def sum(x, y): return x + y";

// Type 3 Clone: Modified structure
let clone3 = r#"
def addition(first, second):
    result = first + second
    return result
"#;

// Type 4 Clone: Semantic equivalence
let clone4 = "def add(a, b): return sum([a, b])";

// Not a clone
let different = "def multiply(a, b): return a * b";

let emb_orig = embedder.embed_code(original, CodeLanguage::Python)?;
let emb_c1 = embedder.embed_code(clone1, CodeLanguage::Python)?;
let emb_c2 = embedder.embed_code(clone2, CodeLanguage::Python)?;
let emb_c3 = embedder.embed_code(clone3, CodeLanguage::Python)?;
let emb_c4 = embedder.embed_code(clone4, CodeLanguage::Python)?;
let emb_diff = embedder.embed_code(different, CodeLanguage::Python)?;

println!("Type 1 (identical):   {:.3}", cosine_similarity(&emb_orig, &emb_c1));  // 1.000
println!("Type 2 (renamed):     {:.3}", cosine_similarity(&emb_orig, &emb_c2));  // ~0.95
println!("Type 3 (modified):    {:.3}", cosine_similarity(&emb_orig, &emb_c3));  // ~0.85
println!("Type 4 (semantic):    {:.3}", cosine_similarity(&emb_orig, &emb_c4));  // ~0.75
println!("Different function:   {:.3}", cosine_similarity(&emb_orig, &emb_diff)); // ~0.40
```

### Batch Processing

```rust
let codes = vec![
    "public int add(int a, int b) { return a + b; }",
    "public int subtract(int a, int b) { return a - b; }",
    "public String concat(String a, String b) { return a + b; }",
];
let languages = vec![CodeLanguage::Java; codes.len()];

let embeddings = embedder.embed_code_batch(
    &codes.iter().map(|s| *s).collect::<Vec<_>>(),
    &languages,
)?;

// Calculate pairwise similarities
for i in 0..embeddings.len() {
    for j in (i + 1)..embeddings.len() {
        let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
        println!("{} vs {}: {:.3}", i, j, sim);
    }
}
```

## Code Search

UniXcoder is particularly effective for code search:

```rust
struct CodeSearchIndex {
    embedder: UniXcoderEmbedder,
    documents: Vec<CodeDocument>,
    embeddings: Vec<Vec<f32>>,
}

struct CodeDocument {
    path: String,
    function_name: String,
    code: String,
    language: CodeLanguage,
}

impl CodeSearchIndex {
    fn search(&self, query_code: &str, top_k: usize) -> Vec<(&CodeDocument, f32)> {
        let query_emb = self.embedder
            .embed_code(query_code, CodeLanguage::Unknown)
            .expect("embedding failed");

        let mut results: Vec<_> = self.documents.iter()
            .zip(&self.embeddings)
            .map(|(doc, emb)| (doc, cosine_similarity(&query_emb, emb)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }
}
```

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Single embedding | ~30ms |
| Batch (32) | ~400ms |
| Memory (loaded) | ~1GB |
| ONNX file size | ~500MB |

### Comparison with CodeT5+

| Aspect | UniXcoder | CodeT5+ |
|--------|-----------|---------|
| Embedding dim | 768 | 256 |
| Speed | Slower | Faster |
| Clone detection | Better | Good |
| Memory | Higher | Lower |
| Languages | 6 | 9 |

## Model Acquisition

### Converting from HuggingFace

```bash
git lfs install
git clone https://huggingface.co/microsoft/unixcoder-base
```

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("microsoft/unixcoder-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")

tokenizer.save_pretrained("./unixcoder-onnx")

dummy_input = tokenizer("def foo(): pass", return_tensors="pt", padding=True)
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "./unixcoder-onnx/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
)
```

## Thread Safety

UniXcoder embedder is thread-safe:

```rust
use std::sync::Arc;
use std::thread;

let embedder = Arc::new(UniXcoderEmbedder::from_directory("/path")?);

let handles: Vec<_> = (0..4).map(|i| {
    let emb = Arc::clone(&embedder);
    thread::spawn(move || {
        let code = format!("def func{}(): pass", i);
        emb.embed_code(&code, CodeLanguage::Python)
    })
}).collect();

for handle in handles {
    let result = handle.join().unwrap()?;
    println!("Got embedding: {} dims", result.len());
}
```

## See Also

- [Overview](overview.md) - Code embeddings introduction
- [CodeT5+](codet5.md) - Smaller, faster model
- [GraphCodeBERT](graphcodebert.md) - Structure-aware model
- [Ensemble](ensemble.md) - Combining multiple models
