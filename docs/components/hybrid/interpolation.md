# Interpolation Strategies

This document describes the interpolation strategies available for combining n-gram and embedding scores in hybrid models.

## Overview

Hybrid models combine two probability sources:

- **N-gram model**: Statistical counts from training data
- **Embedding model**: Distributed semantic representations

Interpolation strategies determine how these are combined.

## Available Strategies

### Linear Interpolation

```rust
use libgrammstein::hybrid::{HybridConfig, InterpolationStrategy};

let config = HybridConfig {
    strategy: InterpolationStrategy::Linear { alpha: 0.7 },
    ..Default::default()
};
```

**Formula**:
```
P(w|c) = α × P_ngram(w|c) + (1-α) × P_embed(w|c)
```

**Characteristics**:
- Simple and predictable
- Direct probability combination
- Alpha = 1.0 → pure n-gram
- Alpha = 0.0 → pure embedding

**Best for**:
- General-purpose models
- When both components are reliable
- Default starting point

### Log-Linear Interpolation

```rust
let config = HybridConfig {
    strategy: InterpolationStrategy::LogLinear { alpha: 0.7 },
    ..Default::default()
};
```

**Formula**:
```
log P(w|c) = α × log P_ngram(w|c) + (1-α) × log P_embed(w|c)
```

**Characteristics**:
- Combines in log space
- Geometric mean behavior
- More stable numerically
- Sharper probability distinctions

**Best for**:
- Models with different scales
- When one model is very confident
- Preventing probability extremes

### N-gram with Embedding Fallback

```rust
let config = HybridConfig {
    strategy: InterpolationStrategy::NgramWithEmbeddingFallback,
    ..Default::default()
};
```

**Logic**:
```rust
if ngram.in_vocabulary(word) {
    P_ngram(w|c)
} else {
    P_embed(w|c)
}
```

**Characteristics**:
- Uses n-gram when data exists
- Falls back to embeddings for OOV
- No interpolation overhead for known words

**Best for**:
- Large n-gram training corpora
- When n-gram quality is high
- OOV-heavy inference scenarios

### Dynamic Interpolation

```rust
let config = HybridConfig {
    strategy: InterpolationStrategy::Dynamic {
        base_alpha: 0.7,
        oov_alpha: 0.3,
    },
    ..Default::default()
};
```

**Logic**:
```rust
let alpha = if ngram.in_vocabulary(word) {
    base_alpha
} else {
    oov_alpha  // More weight to embeddings for OOV
};
P(w|c) = α × P_ngram(w|c) + (1-α) × P_embed(w|c)
```

**Characteristics**:
- Adapts based on word knowledge
- Higher embedding weight for OOV
- Smooth transition between modes

**Best for**:
- Mixed vocabulary scenarios
- Domain adaptation
- Robustness across text types

## Choosing a Strategy

```
                    ┌─────────────────────────┐
                    │   OOV rate high?        │
                    └───────────┬─────────────┘
                                │
          ┌────────yes──────────┴──────────no─────────┐
          │                                            │
          ▼                                            ▼
┌─────────────────────┐                    ┌─────────────────────┐
│   Dynamic or        │                    │  N-gram data        │
│   NgramFallback     │                    │  quality high?      │
└─────────────────────┘                    └──────────┬──────────┘
                                                      │
                                     ┌─────yes────────┴──────no───────┐
                                     │                                 │
                                     ▼                                 ▼
                          ┌────────────────┐               ┌────────────────┐
                          │ Linear α=0.8   │               │ Linear α=0.5   │
                          │ or LogLinear   │               │ or Dynamic     │
                          └────────────────┘               └────────────────┘
```

## Alpha Tuning

### Grid Search

```rust
fn tune_alpha(
    ngram: &NgramModel<D>,
    embedding: &SubwordEmbedding,
    dev_set: &[Vec<String>],
) -> f64 {
    let mut best_alpha = 0.5;
    let mut best_perplexity = f64::INFINITY;

    for alpha in (0..=10).map(|i| i as f64 / 10.0) {
        let config = HybridConfig {
            strategy: InterpolationStrategy::Linear { alpha },
            ..Default::default()
        };
        let model = HybridLanguageModel::new(
            ngram.clone(),
            embedding.clone(),
            config,
        );

        let ppl = evaluate_perplexity(&model, dev_set);

        if ppl < best_perplexity {
            best_perplexity = ppl;
            best_alpha = alpha;
        }
    }

    best_alpha
}
```

### Recommended Values

| Scenario | Strategy | Alpha |
|----------|----------|-------|
| Large n-gram corpus | Linear | 0.8 - 0.9 |
| Small n-gram corpus | Linear | 0.5 - 0.7 |
| High OOV rate | Dynamic | base=0.7, oov=0.3 |
| Production (stable) | NgramFallback | N/A |
| Domain adaptation | LogLinear | 0.6 - 0.8 |

## Implementation Details

### Score Computation

```rust
impl<D> HybridLanguageModel<D> {
    pub fn score(&self, word: &str, context: &[&str]) -> f64 {
        match &self.config.strategy {
            Linear { alpha } => {
                let ngram_prob = self.ngram.log_prob(word, context).exp();
                let embed_prob = self.embedding_prob(word, context).exp();
                (alpha * ngram_prob + (1.0 - alpha) * embed_prob).ln()
            }

            LogLinear { alpha } => {
                let ngram_log = self.ngram.log_prob(word, context);
                let embed_log = self.embedding_log_prob(word, context);
                alpha * ngram_log + (1.0 - alpha) * embed_log
            }

            NgramWithEmbeddingFallback => {
                if self.ngram.in_vocabulary(word) {
                    self.ngram.log_prob(word, context)
                } else {
                    self.embedding_log_prob(word, context)
                }
            }

            Dynamic { base_alpha, oov_alpha } => {
                let alpha = if self.ngram.in_vocabulary(word) {
                    *base_alpha
                } else {
                    *oov_alpha
                };
                let ngram_prob = self.ngram.log_prob(word, context).exp();
                let embed_prob = self.embedding_prob(word, context).exp();
                (alpha * ngram_prob + (1.0 - alpha) * embed_prob).ln()
            }
        }
    }
}
```

### Embedding Probability

Converting embedding similarity to probability:

```rust
fn embedding_prob(&self, word: &str, context: &[&str]) -> f64 {
    if context.is_empty() {
        // Uniform over vocabulary
        return 1.0 / self.embedding.vocab_size() as f64;
    }

    // Compute context vector (average of context words)
    let context_vec = self.context_vector(context);

    // Similarity to word
    let word_vec = self.embedding.word_vector(word);
    let sim = cosine_similarity(&word_vec, &context_vec);

    // Convert to probability via softmax over vocabulary
    self.softmax_probability(word, &context_vec, sim)
}
```

## Caching

Interpolated scores are cached:

```rust
pub struct HybridLanguageModel<D> {
    // ...
    cache: Mutex<LruCache<CacheKey, f64>>,
}

impl<D> HybridLanguageModel<D> {
    pub fn score(&self, word: &str, context: &[&str]) -> f64 {
        let key = CacheKey::new(word, context);

        // Check cache
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(&score) = cache.get(&key) {
                return score;
            }
        }

        // Compute score
        let score = self.compute_score(word, context);

        // Update cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(key, score);
        }

        score
    }
}
```

## See Also

- [OOV Handling](oov-handling.md) - Out-of-vocabulary strategies
- [Hybrid API](../../api/hybrid.md) - Complete API reference
- [Hybrid Training](../../training/hybrid.md) - Training guide
