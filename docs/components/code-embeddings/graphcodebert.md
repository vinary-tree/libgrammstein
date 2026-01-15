# GraphCodeBERT Embedder

GraphCodeBERT is Microsoft's pre-trained model that incorporates code structure through data flow graphs, providing structure-aware code embeddings.

## Model Overview

| Property | Value |
|----------|-------|
| **Organization** | Microsoft Research |
| **Model ID** | `microsoft/graphcodebert-base` |
| **Parameters** | 125M |
| **Embedding Dimension** | 768 |
| **Max Sequence Length** | 512 tokens |
| **Architecture** | BERT with data flow |

GraphCodeBERT's key innovation is pre-training on data flow information, making it particularly effective for:
- Understanding variable relationships
- Detecting semantic code clones
- Code completion with context
- Variable misuse detection

## Data Flow Understanding

Unlike pure token-based models, GraphCodeBERT understands how data flows through code:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Code: x = a + b                                                         │
│        y = x * 2                                                         │
│        z = y - a                                                         │
│                                                                          │
│  Data Flow Graph:                                                        │
│                                                                          │
│      a ──────┬─────────────────────────────┐                            │
│              │                             │                            │
│              ▼                             ▼                            │
│      b ───► x ───► y ───► z ◄───────────────                           │
│                                                                          │
│  GraphCodeBERT sees:                                                     │
│  • x depends on a, b                                                     │
│  • y depends on x                                                        │
│  • z depends on y, a                                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Loading the Model

### From Directory

```rust
use libgrammstein::neural::code::{GraphCodeBertEmbedder, GraphCodeBertConfig, CodeLanguage};

// Load from a local directory containing model.onnx and tokenizer.json
let embedder = GraphCodeBertEmbedder::from_directory("/path/to/graphcodebert-base")?;

// Embed code
let embedding = embedder.embed_code("def compute(x): return x * 2", CodeLanguage::Python)?;
```

### With Custom Configuration

```rust
let config = GraphCodeBertConfig {
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
    use_data_flow: false,  // Most ONNX exports don't include DFG inputs
};

let embedder = GraphCodeBertEmbedder::load(config)?;
```

## Configuration Options

### GraphCodeBertConfig

```rust
pub struct GraphCodeBertConfig {
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

    /// Embedding dimension (768 for graphcodebert-base).
    pub embedding_dim: usize,

    /// Whether to use data flow input (if model supports it).
    pub use_data_flow: bool,
}
```

## Supported Languages

GraphCodeBERT was trained on CodeSearchNet (6 languages):

- Python
- Java
- JavaScript
- Go
- Ruby
- PHP

## Examples

### Basic Embedding

```rust
let embedder = GraphCodeBertEmbedder::from_directory("/path/to/model")?;

let code = r#"
def process_data(items):
    total = 0
    for item in items:
        value = item.get('amount', 0)
        total += value
    return total
"#;

let embedding = embedder.embed_code(code, CodeLanguage::Python)?;
println!("Dimension: {}", embedding.len());  // 768
```

### Variable Relationship Detection

GraphCodeBERT excels at understanding variable relationships:

```rust
use libgrammstein::neural::code::cosine_similarity;

// Similar data flow patterns
let code1 = r#"
def compute(a, b):
    x = a + b
    y = x * 2
    return y
"#;

let code2 = r#"
def calculate(m, n):
    sum_val = m + n
    result = sum_val * 2
    return result
"#;

// Different data flow (different computation)
let code3 = r#"
def compute(a, b):
    x = a * b
    y = x + 2
    return y
"#;

let emb1 = embedder.embed_code(code1, CodeLanguage::Python)?;
let emb2 = embedder.embed_code(code2, CodeLanguage::Python)?;
let emb3 = embedder.embed_code(code3, CodeLanguage::Python)?;

println!("Same flow (renamed): {:.3}", cosine_similarity(&emb1, &emb2));  // ~0.92
println!("Different flow:      {:.3}", cosine_similarity(&emb1, &emb3));  // ~0.75
```

### Code Search with Context

```rust
struct ContextAwareSearch {
    embedder: GraphCodeBertEmbedder,
    index: Vec<(String, Vec<f32>)>,
}

impl ContextAwareSearch {
    fn search_with_context(&self, query: &str, context: &str, top_k: usize) -> Vec<(&str, f32)> {
        // Combine query with usage context
        let full_query = format!("{}\n# Usage:\n{}", query, context);

        let query_emb = self.embedder
            .embed_code(&full_query, CodeLanguage::Python)
            .expect("embedding failed");

        let mut results: Vec<_> = self.index.iter()
            .map(|(code, emb)| (code.as_str(), cosine_similarity(&query_emb, emb)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }
}
```

### Semantic Clone Detection

```rust
fn detect_semantic_clones(
    embedder: &GraphCodeBertEmbedder,
    functions: &[(&str, &str)],  // (name, code)
    threshold: f32,
) -> Vec<(String, String, f32)> {
    // Embed all functions
    let embeddings: Vec<_> = functions.iter()
        .map(|(name, code)| {
            (name, embedder.embed_code(code, CodeLanguage::Unknown))
        })
        .collect();

    let mut clones = Vec::new();

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            if let (Ok(ref emb_i), Ok(ref emb_j)) = (&embeddings[i].1, &embeddings[j].1) {
                let similarity = cosine_similarity(emb_i, emb_j);
                if similarity >= threshold {
                    clones.push((
                        embeddings[i].0.to_string(),
                        embeddings[j].0.to_string(),
                        similarity,
                    ));
                }
            }
        }
    }

    clones.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    clones
}

// Usage
let functions = vec![
    ("process_v1", "def process(data): return sum(data) / len(data)"),
    ("process_v2", "def process(items): total = sum(items); count = len(items); return total / count"),
    ("transform", "def transform(x): return x.upper()"),
];

let clones = detect_semantic_clones(&embedder, &functions, 0.8);
for (f1, f2, sim) in clones {
    println!("Clone: {} <-> {} (similarity: {:.3})", f1, f2, sim);
}
```

## Position IDs

Some GraphCodeBERT ONNX exports require position_ids input:

```rust
// Check if model needs position_ids
if embedder.has_position_ids() {
    println!("This model uses explicit position IDs");
}
```

The embedder automatically handles position_ids when needed.

## Comparison with Other Models

| Aspect | GraphCodeBERT | UniXcoder | CodeT5+ |
|--------|---------------|-----------|---------|
| Data flow aware | Yes | No | No |
| Embedding dim | 768 | 768 | 256 |
| Best for | Structure | Clones | General |
| Memory | ~1GB | ~1GB | ~500MB |
| Speed | Moderate | Moderate | Fast |

### When to Use GraphCodeBERT

**Best for:**
- Code with complex variable relationships
- Detecting semantic equivalence despite different structure
- Understanding data dependencies
- Variable misuse detection

**Consider alternatives for:**
- Simple text matching
- When speed is critical
- Memory-constrained environments

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Single embedding | ~30ms |
| Batch (32) | ~400ms |
| Memory (loaded) | ~1GB |
| ONNX file size | ~500MB |

## Model Acquisition

### Converting from HuggingFace

```bash
git lfs install
git clone https://huggingface.co/microsoft/graphcodebert-base
```

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

tokenizer.save_pretrained("./graphcodebert-onnx")

dummy_input = tokenizer("def foo(): pass", return_tensors="pt", padding=True)
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "./graphcodebert-onnx/model.onnx",
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

The embedder is thread-safe:

```rust
use std::sync::Arc;
use rayon::prelude::*;

let embedder = Arc::new(GraphCodeBertEmbedder::from_directory("/path")?);

let embeddings: Vec<_> = code_snippets
    .par_iter()
    .map(|code| embedder.embed_code(code, CodeLanguage::Unknown))
    .collect::<Result<Vec<_>, _>>()?;
```

## See Also

- [Overview](overview.md) - Code embeddings introduction
- [CodeT5+](codet5.md) - Smaller, faster model
- [UniXcoder](unixcoder.md) - Clone detection specialist
- [Ensemble](ensemble.md) - Combining multiple models
