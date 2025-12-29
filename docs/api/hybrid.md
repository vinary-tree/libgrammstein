# HybridLanguageModel API Reference

The `HybridLanguageModel<D>` struct combines n-gram and embedding models for robust language modeling with OOV handling.

## Overview

Hybrid models leverage the strengths of both approaches:

- **N-gram models** provide accurate probabilities for seen n-grams
- **Embedding models** provide semantic similarity for OOV words
- **Configurable interpolation** balances the two components

## Type Parameters

| Parameter | Description |
|-----------|-------------|
| `D` | Dictionary backend implementing `MutableMappedDictionary<Value = NgramEntry> + Send + Sync` |

## Construction

### From Pre-trained Components

```rust
use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig};
use libgrammstein::ngram::NgramModel;
use libgrammstein::embedding::SubwordEmbedding;

let ngram_model = NgramModel::load("ngram.bin")?;
let embedding_model = SubwordEmbedding::load("embeddings.bin")?;

// With custom configuration
let config = HybridConfig::default();
let hybrid = HybridLanguageModel::new(ngram_model, embedding_model, config);

// With defaults
let hybrid = HybridLanguageModel::with_defaults(ngram_model, embedding_model);
```

### Loading from File

```rust
use libgrammstein::hybrid::HybridLanguageModel;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

// Binary format (requires serde-extras feature)
let model: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
    HybridLanguageModel::load("hybrid.bin")?;

// Portable format
let model = HybridLanguageModel::load_portable(
    "hybrid.portable.bin",
    || DynamicDawgChar::new()
)?;
```

## Configuration

### HybridConfig

```rust
use libgrammstein::hybrid::{HybridConfig, InterpolationStrategy};

let config = HybridConfig {
    strategy: InterpolationStrategy::Linear { alpha: 0.8 },
    cache_size: 50_000,
    embedding_smoothing: 1e-8,
    temperature: 1.0,
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `InterpolationStrategy` | `Linear { alpha: 0.8 }` | How to combine scores |
| `cache_size` | `usize` | `50_000` | LRU cache size for scores |
| `embedding_smoothing` | `f64` | `1e-8` | Smoothing for embedding probabilities |
| `temperature` | `f64` | `1.0` | Temperature for similarity-to-probability |

### Interpolation Strategies

#### Linear Interpolation

Combines probabilities: `P = α * P_ngram + (1-α) * P_embedding`

```rust
InterpolationStrategy::Linear { alpha: 0.8 }
```

#### Log-Linear Interpolation

Combines log probabilities: `log P = α * log P_ngram + (1-α) * log P_embedding`

```rust
InterpolationStrategy::LogLinear { alpha: 0.7 }
```

#### N-gram with Embedding Fallback

Uses n-gram for known words, embedding for OOV:

```rust
InterpolationStrategy::NgramWithEmbeddingFallback
```

#### Dynamic Weighting

Adjusts weight based on context length:

```rust
InterpolationStrategy::Dynamic {
    base_alpha: 0.5,       // Base weight for n-gram
    alpha_per_context: 0.1, // Additional weight per context word
    max_alpha: 0.9,        // Maximum n-gram weight
}
```

## Methods

### Scoring

#### `score(word, context) -> f64`

Score a word given context using the configured interpolation strategy.

```rust
let score = model.score("fox", &["the", "quick", "brown"]);
println!("log P(fox | the quick brown) = {:.4}", score);
```

**Returns:** Log probability of the word given context.

#### `sentence_log_prob(words) -> f64`

Compute total log probability of a sentence.

```rust
let log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
```

#### `perplexity(words) -> f64`

Compute perplexity of a sentence.

```rust
let ppl = model.perplexity(&["the", "quick", "brown", "fox"]);
println!("Perplexity: {:.2}", ppl);
```

Lower perplexity indicates better model fit.

### Prediction

#### `predict_next(context, candidates) -> Option<(String, f64)>`

Find the most likely next word from candidates.

```rust
let candidates = ["fox", "dog", "cat", "bird"];
if let Some((word, score)) = model.predict_next(&["the", "quick"], &candidates) {
    println!("Best: {} (score: {:.4})", word, score);
}
```

### Component Access

#### `ngram_model() -> &NgramModel<D>`

Get reference to the n-gram component.

```rust
let ngram = model.ngram_model();
println!("N-gram order: {}", ngram.order());
```

#### `embedding_model() -> &SubwordEmbedding`

Get reference to the embedding component.

```rust
let embedding = model.embedding_model();
println!("Embedding dim: {}", embedding.dim());
```

#### `config() -> &HybridConfig`

Get reference to the configuration.

```rust
let config = model.config();
println!("Cache size: {}", config.cache_size);
```

### Cache Management

#### `clear_cache()`

Clear the score cache.

```rust
model.clear_cache();
```

### Serialization (requires `serde-extras` feature)

#### `save(path) -> Result<()>`

Save model to binary file (requires D: Serialize).

```rust
model.save("hybrid.bin")?;
```

#### `load(path) -> Result<Self>`

Load model from binary file.

```rust
let model: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
    HybridLanguageModel::load("hybrid.bin")?;
```

#### `save_portable(path) -> Result<()>`

Save in portable format (works with any dictionary backend).

```rust
model.save_portable("hybrid.portable.bin")?;
```

#### `load_portable(path, factory) -> Result<Self>`

Load from portable format with dictionary factory.

```rust
let model = HybridLanguageModel::load_portable(
    "hybrid.portable.bin",
    || DynamicDawgChar::new()
)?;
```

## How Embedding Probabilities Work

The embedding model converts similarity to probability:

1. Compute context vector (average of context word embeddings)
2. Compute cosine similarity between word and context
3. Apply temperature scaling: `scaled_sim = similarity / temperature`
4. Convert to log probability: `log_prob = scaled_sim - 1.0`

For OOV words:
- Subword embeddings provide the word vector
- Even completely unseen words get reasonable probabilities

## Performance Considerations

1. **Cache Size**
   - Larger cache = better performance for repeated queries
   - Trade-off with memory usage

2. **Interpolation Strategy**
   - Linear: Simple, well-understood
   - LogLinear: Better for combining log-space models
   - Fallback: Fast for known words
   - Dynamic: Adapts to context availability

3. **Temperature**
   - Lower temperature = sharper probability distribution
   - Higher temperature = smoother distribution

## Example: Complete Workflow

```rust
use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig, InterpolationStrategy};
use libgrammstein::ngram::{NgramModel, TrainerBuilder, NgramEntry};
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

fn main() -> libgrammstein::Result<()> {
    // 1. Train components
    let reader = PlaintextReader::from_file("corpus.txt")?;

    let ngram_model = TrainerBuilder::new(DynamicDawgChar::new())
        .order(5)
        .train(&reader)?;

    let reader2 = PlaintextReader::from_file("corpus.txt")?;
    let embedding_model = EmbeddingTrainerBuilder::new()
        .dim(100)
        .epochs(5)
        .train(&reader2)?;

    // 2. Create hybrid model
    let config = HybridConfig {
        strategy: InterpolationStrategy::Linear { alpha: 0.7 },
        ..Default::default()
    };
    let hybrid = HybridLanguageModel::new(ngram_model, embedding_model, config);

    // 3. Score sentences
    let sentence = ["the", "quick", "brown", "fox"];
    let log_prob = hybrid.sentence_log_prob(&sentence);
    let ppl = hybrid.perplexity(&sentence);

    println!("Log probability: {:.4}", log_prob);
    println!("Perplexity: {:.2}", ppl);

    // 4. Handle OOV words
    let oov_score = hybrid.score("xyzzy", &["magic", "word"]);
    println!("OOV word score: {:.4}", oov_score);  // Still gets reasonable score

    // 5. Save model
    hybrid.save("hybrid.bin")?;

    Ok(())
}
```

## Use Cases

### Spell Correction Ranking

```rust
// Score correction candidates
let candidates = ["their", "there", "they're"];
let context = ["put", "it", "over"];

let mut scored: Vec<_> = candidates.iter()
    .map(|c| (c, model.score(c, &context)))
    .collect();

scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

println!("Best correction: {}", scored[0].0);
```

### Language Model Perplexity Evaluation

```rust
fn evaluate_corpus(model: &HybridLanguageModel<D>, sentences: &[Vec<&str>]) -> f64 {
    let total_log_prob: f64 = sentences.iter()
        .map(|s| model.sentence_log_prob(s))
        .sum();

    let total_words: usize = sentences.iter()
        .map(|s| s.len())
        .sum();

    (-total_log_prob / total_words as f64).exp()
}
```

## See Also

- [NgramModel](ngram.md) - N-gram component API
- [SubwordEmbedding](embedding.md) - Embedding component API
- [Training Guide](../training/hybrid.md) - Training workflow
- [Interpolation Strategies](../components/hybrid/interpolation.md) - Strategy details
