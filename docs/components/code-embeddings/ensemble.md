# Ensemble Code Embeddings

Ensemble embeddings combine multiple code models to achieve better representation quality than any single model.

## Why Ensembles?

Different models capture different aspects of code:

| Model | Strengths |
|-------|-----------|
| **CodeT5+** | General code understanding, efficiency |
| **UniXcoder** | Code-to-code similarity, cross-modal |
| **GraphCodeBERT** | Data flow, variable relationships |

By combining them, we get embeddings that capture:
- Syntactic patterns (all models)
- Semantic meaning (CodeT5+, UniXcoder)
- Structural relationships (GraphCodeBERT)

## Ensemble Strategies

libgrammstein supports four combination strategies:

```rust
pub enum EnsembleStrategy {
    /// Concatenate: [emb1 | emb2 | emb3]
    Concatenate,

    /// Weighted average: w1*emb1 + w2*emb2 + w3*emb3
    WeightedAverage,

    /// Element-wise maximum: max(emb1, emb2, emb3)
    MaxPooling,

    /// Simple average: (emb1 + emb2 + emb3) / 3
    MeanPooling,
}
```

### Strategy Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Ensemble Strategies                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Concatenate:                                                            │
│  ┌────────┬────────┬────────┐                                           │
│  │ CodeT5 │UniXcoder│GraphCB │  → 256 + 768 + 768 = 1792 dims           │
│  │  256d  │  768d  │  768d  │                                           │
│  └────────┴────────┴────────┘                                           │
│                                                                          │
│  Weighted Average (requires same dimensions):                            │
│  ┌────────┐   ┌────────┐   ┌────────┐                                   │
│  │UniXcoder│ × 0.4 + │GraphCB │ × 0.6 = │ Result │  → 768 dims          │
│  │  768d  │         │  768d  │         │  768d  │                       │
│  └────────┘         └────────┘         └────────┘                       │
│                                                                          │
│  Max Pooling (element-wise max):                                         │
│  [0.2, 0.5, 0.3]     [0.1, 0.8, 0.2]                                    │
│         ▼                   ▼                                            │
│         └───────────────────┘                                            │
│                   ▼                                                      │
│          [0.2, 0.8, 0.3]                                                │
│                                                                          │
│  Mean Pooling (simple average):                                          │
│  [0.2, 0.5, 0.3]     [0.1, 0.8, 0.2]                                    │
│         ▼                   ▼                                            │
│         └───────────────────┘                                            │
│                   ▼                                                      │
│          [0.15, 0.65, 0.25]                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Creating Ensembles

### Basic Ensemble (Concatenation)

```rust
use libgrammstein::neural::code::{
    CodeT5Embedder, UniXcoderEmbedder, GraphCodeBertEmbedder,
    EnsembleCodeEmbedder, EnsembleStrategy,
    CodeEmbedder, CodeLanguage,
};
use std::sync::Arc;

// Load individual models
let codet5 = Arc::new(CodeT5Embedder::from_directory("/path/to/codet5")?);
let unixcoder = Arc::new(UniXcoderEmbedder::from_directory("/path/to/unixcoder")?);
let graphcodebert = Arc::new(GraphCodeBertEmbedder::from_directory("/path/to/graphcodebert")?);

// Create ensemble with concatenation
let ensemble = EnsembleCodeEmbedder::new(vec![
    codet5.clone() as Arc<dyn CodeEmbedder>,
    unixcoder.clone() as Arc<dyn CodeEmbedder>,
    graphcodebert.clone() as Arc<dyn CodeEmbedder>,
]);

println!("Ensemble dimension: {}", ensemble.embedding_dim());
// 256 + 768 + 768 = 1792

let embedding = ensemble.embed_code("fn main() {}", CodeLanguage::Rust)?;
```

### Weighted Average Ensemble

For weighted average, all models must have the same embedding dimension:

```rust
// Use only models with same dimension (768)
let ensemble = EnsembleCodeEmbedder::with_strategy(
    vec![
        unixcoder.clone() as Arc<dyn CodeEmbedder>,
        graphcodebert.clone() as Arc<dyn CodeEmbedder>,
    ],
    EnsembleStrategy::WeightedAverage,
    Some(vec![0.4, 0.6]),  // Weight GraphCodeBERT higher
)?;

println!("Ensemble dimension: {}", ensemble.embedding_dim()); // 768
```

### Max Pooling Ensemble

```rust
let ensemble = EnsembleCodeEmbedder::with_strategy(
    vec![
        unixcoder.clone() as Arc<dyn CodeEmbedder>,
        graphcodebert.clone() as Arc<dyn CodeEmbedder>,
    ],
    EnsembleStrategy::MaxPooling,
    None,  // Weights ignored for max pooling
)?;
```

### Mean Pooling Ensemble

```rust
let ensemble = EnsembleCodeEmbedder::with_strategy(
    vec![
        unixcoder.clone() as Arc<dyn CodeEmbedder>,
        graphcodebert.clone() as Arc<dyn CodeEmbedder>,
    ],
    EnsembleStrategy::MeanPooling,
    None,
)?;
```

## EnsembleCodeEmbedder API

```rust
pub struct EnsembleCodeEmbedder {
    embedders: Vec<Arc<dyn CodeEmbedder>>,
    weights: Vec<f64>,
    strategy: EnsembleStrategy,
    embedding_dim: usize,
    normalize_final: bool,
}

impl EnsembleCodeEmbedder {
    /// Create with concatenation strategy.
    pub fn new(embedders: Vec<Arc<dyn CodeEmbedder>>) -> Self;

    /// Create with specified strategy and weights.
    pub fn with_strategy(
        embedders: Vec<Arc<dyn CodeEmbedder>>,
        strategy: EnsembleStrategy,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>;

    /// Set whether to normalize the final embedding.
    pub fn set_normalize_final(&mut self, normalize: bool);

    /// Get the ensemble strategy.
    pub fn strategy(&self) -> EnsembleStrategy;

    /// Get the weights.
    pub fn weights(&self) -> &[f64];

    /// Get the number of embedders.
    pub fn num_embedders(&self) -> usize;
}
```

## Choosing a Strategy

### Concatenation

**Best for:**
- Maximum information preservation
- When downstream models can handle larger dimensions
- When models have different embedding dimensions

**Trade-offs:**
- Larger embedding size (more memory/compute)
- May include redundant information

```rust
// Full information, largest dimension
let ensemble = EnsembleCodeEmbedder::new(vec![codet5, unixcoder, graphcodebert]);
// Dimension: 1792
```

### Weighted Average

**Best for:**
- Fixed-size embeddings
- When you know which model is better for your task
- Smooth blending of representations

**Trade-offs:**
- Requires same-dimension models
- May blur important distinctions

```rust
// Emphasize data flow understanding
let ensemble = EnsembleCodeEmbedder::with_strategy(
    vec![unixcoder, graphcodebert],
    EnsembleStrategy::WeightedAverage,
    Some(vec![0.3, 0.7]),  // Favor GraphCodeBERT
)?;
```

### Max Pooling

**Best for:**
- Capturing the "strongest signal" from each model
- When models are complementary
- Sparse-like representations

**Trade-offs:**
- Can amplify noise
- Loses averaging benefits

```rust
// Take strongest feature from each model
let ensemble = EnsembleCodeEmbedder::with_strategy(
    vec![unixcoder, graphcodebert],
    EnsembleStrategy::MaxPooling,
    None,
)?;
```

### Mean Pooling

**Best for:**
- Simple, balanced combination
- When models are roughly equal quality
- Noise reduction

**Trade-offs:**
- May dilute strong signals
- No model weighting

```rust
// Equal contribution from each model
let ensemble = EnsembleCodeEmbedder::with_strategy(
    vec![unixcoder, graphcodebert],
    EnsembleStrategy::MeanPooling,
    None,
)?;
```

## Examples

### Code Search with Ensemble

```rust
use libgrammstein::neural::code::cosine_similarity;

struct EnsembleCodeSearch {
    ensemble: EnsembleCodeEmbedder,
    index: Vec<(String, Vec<f32>)>,
}

impl EnsembleCodeSearch {
    fn add(&mut self, code: &str) -> Result<()> {
        let embedding = self.ensemble.embed_code(code, CodeLanguage::Unknown)?;
        self.index.push((code.to_string(), embedding));
        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> Result<Vec<(&str, f32)>> {
        let query_emb = self.ensemble.embed_code(query, CodeLanguage::Unknown)?;

        let mut results: Vec<_> = self.index.iter()
            .map(|(code, emb)| (code.as_str(), cosine_similarity(&query_emb, emb)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        Ok(results)
    }
}
```

### Batch Processing

```rust
let codes = vec![
    "def add(a, b): return a + b",
    "def multiply(a, b): return a * b",
    "def divide(a, b): return a / b",
];
let languages = vec![CodeLanguage::Python; codes.len()];

// Batch embedding with ensemble
let embeddings = ensemble.embed_code_batch(
    &codes.iter().map(|s| *s).collect::<Vec<_>>(),
    &languages,
)?;

for (code, emb) in codes.iter().zip(embeddings.iter()) {
    println!("{}: {} dimensions", &code[..15], emb.len());
}
```

### Comparison Experiment

```rust
use libgrammstein::neural::code::cosine_similarity;

fn compare_strategies(
    codes: &[(&str, &str)],  // (code1, code2) pairs
    embedders: &[Arc<dyn CodeEmbedder>],
) -> HashMap<String, Vec<f32>> {
    let mut results = HashMap::new();

    // Individual models
    for embedder in embedders {
        let similarities: Vec<f32> = codes.iter()
            .map(|(c1, c2)| {
                let e1 = embedder.embed_code(c1, CodeLanguage::Unknown).unwrap();
                let e2 = embedder.embed_code(c2, CodeLanguage::Unknown).unwrap();
                cosine_similarity(&e1, &e2)
            })
            .collect();
        results.insert(embedder.model_name().to_string(), similarities);
    }

    // Concatenation ensemble
    let concat = EnsembleCodeEmbedder::new(embedders.to_vec());
    let concat_sims: Vec<f32> = codes.iter()
        .map(|(c1, c2)| {
            let e1 = concat.embed_code(c1, CodeLanguage::Unknown).unwrap();
            let e2 = concat.embed_code(c2, CodeLanguage::Unknown).unwrap();
            cosine_similarity(&e1, &e2)
        })
        .collect();
    results.insert("Ensemble (concat)".to_string(), concat_sims);

    results
}
```

## Performance Considerations

### Memory

Each model adds to memory usage:

| Configuration | Approximate Memory |
|---------------|-------------------|
| CodeT5+ only | ~500MB |
| + UniXcoder | ~1.5GB |
| + GraphCodeBERT | ~2.5GB |

### Throughput

Ensemble throughput is limited by the slowest model:

| Configuration | Embeddings/sec |
|---------------|----------------|
| CodeT5+ only | ~50 |
| All three | ~15-20 |

### Optimization Tips

1. **Parallel model loading**:
```rust
use rayon::prelude::*;

let paths = vec![
    ("/path/to/codet5", "codet5"),
    ("/path/to/unixcoder", "unixcoder"),
];

let embedders: Vec<Arc<dyn CodeEmbedder>> = paths.par_iter()
    .map(|(path, name)| {
        match *name {
            "codet5" => Arc::new(CodeT5Embedder::from_directory(path).unwrap()) as Arc<dyn CodeEmbedder>,
            "unixcoder" => Arc::new(UniXcoderEmbedder::from_directory(path).unwrap()) as Arc<dyn CodeEmbedder>,
            _ => panic!("unknown model"),
        }
    })
    .collect();
```

2. **Use fewer models for production**:
```rust
// Often 2 models are sufficient
let ensemble = EnsembleCodeEmbedder::new(vec![codet5, graphcodebert]);
```

3. **Disable normalization for intermediate steps**:
```rust
let mut ensemble = EnsembleCodeEmbedder::new(embedders);
ensemble.set_normalize_final(false);  // Normalize only at final comparison
```

## See Also

- [Overview](overview.md) - Code embeddings introduction
- [CodeT5+](codet5.md) - Individual model documentation
- [UniXcoder](unixcoder.md) - Individual model documentation
- [GraphCodeBERT](graphcodebert.md) - Individual model documentation
- [Caching](caching.md) - Embedding cache configuration
