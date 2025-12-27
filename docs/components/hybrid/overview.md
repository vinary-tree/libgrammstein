# Hybrid Language Model

This document explains how libgrammstein combines N-gram models with subword embeddings into a unified hybrid language model.

## Motivation: Best of Both Worlds

N-gram models and embedding models have complementary strengths:

| Model Type | Strengths | Weaknesses |
|------------|-----------|------------|
| N-gram | Precise local context, fast lookup, well-understood | OOV problem, sparse for long contexts |
| Embeddings | Semantic similarity, handles OOV via subwords | Ignores word order, weaker at local patterns |

The **hybrid model** combines both to get:
- **Local precision** from N-grams
- **Semantic coverage** from embeddings
- **OOV handling** through subword representations

## How the Hybrid Model Works

Given a token and its context, the hybrid model computes a weighted combination of scores:

```
score(word | context) = λ₁ × ngram_score + λ₂ × embedding_score

Where:
- ngram_score = log P_MKN(word | context)
- embedding_score = cosine_similarity(word, context_embedding)
- λ₁ + λ₂ = 1 (interpolation weights)
```

### Scoring Flow

```
Input: ("fox", context=["the", "quick", "brown"])
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HybridLanguageModel                         │
│                                                                 │
│  ┌───────────────────────────┐   ┌───────────────────────────┐  │
│  │       N-gram Model        │   │   Embedding Model         │  │
│  │                           │   │                           │  │
│  │  Look up "the|quick|      │   │  Compute embeddings:      │  │
│  │          brown|fox"       │   │    v_fox = embed("fox")   │  │
│  │                           │   │    v_ctx = avg(embed(ctx))│  │
│  │  Apply MKN smoothing      │   │                           │  │
│  │  with backoff chain       │   │  Compute similarity:      │  │
│  │                           │   │    sim = v_fox · v_ctx    │  │
│  │  Result: -3.2 (log prob)  │   │  Result: 0.75             │  │
│  └─────────────┬─────────────┘   └─────────────┬─────────────┘  │
│                │                               │                 │
│                └──────────────┬────────────────┘                 │
│                               ▼                                  │
│                ┌───────────────────────────────┐                │
│                │     Interpolation Layer       │                │
│                │                               │                │
│                │  λ₁ = 0.8, λ₂ = 0.2           │                │
│                │  score = 0.8 × (-3.2)         │                │
│                │        + 0.2 × log(0.75)      │                │
│                │  score = -2.56 + (-0.058)     │                │
│                │  score = -2.618               │                │
│                └───────────────────────────────┘                │
│                               │                                  │
└───────────────────────────────┼──────────────────────────────────┘
                                ▼
                Output: log P(fox | the quick brown) = -2.618
```

## libgrammstein Implementation

### HybridLanguageModel Struct

```rust
pub struct HybridLanguageModel<D: MutableMappedDictionary<Value = NgramEntry>> {
    /// N-gram model with Modified Kneser-Ney smoothing
    ngram: NgramModel<D>,

    /// Subword embedding model
    embedding: SubwordEmbedding,

    /// Interpolation configuration
    config: HybridConfig,

    /// LRU cache for hot queries
    cache: Mutex<LruCache<CacheKey, f64>>,
}

#[derive(Clone, Debug)]
pub struct HybridConfig {
    /// Weight for N-gram score (default: 0.8)
    pub ngram_weight: f64,

    /// Weight for embedding score (default: 0.2)
    pub embedding_weight: f64,

    /// Cache size for frequently queried n-grams
    pub cache_size: usize,

    /// OOV handling strategy
    pub oov_strategy: OovStrategy,
}

#[derive(Clone, Debug)]
pub enum OovStrategy {
    /// Use only embedding score for OOV words
    EmbeddingOnly,

    /// Backoff to lower-order N-grams, supplement with embeddings
    BackoffWithEmbedding,

    /// Assign a fixed log probability to OOV
    FixedPenalty(f64),
}
```

### Core Scoring Methods

```rust
impl<D: MutableMappedDictionary<Value = NgramEntry>> HybridLanguageModel<D> {
    /// Score a word given its context
    pub fn score(&self, word: &str, context: &[&str]) -> f64 {
        // Check cache
        let cache_key = self.make_cache_key(word, context);
        if let Some(&cached) = self.cache.lock().unwrap().get(&cache_key) {
            return cached;
        }

        // Compute N-gram score
        let ngram_score = self.ngram.log_prob(word, context);

        // Compute embedding score
        let embedding_score = self.compute_embedding_score(word, context);

        // Interpolate
        let score = self.config.ngram_weight * ngram_score
                  + self.config.embedding_weight * embedding_score;

        // Cache and return
        self.cache.lock().unwrap().put(cache_key, score);
        score
    }

    /// Compute embedding-based score
    fn compute_embedding_score(&self, word: &str, context: &[&str]) -> f64 {
        if context.is_empty() {
            return 0.0;  // No context to compare against
        }

        // Get word embedding
        let word_emb = self.embedding.get_embedding(word);

        // Get context embedding (average of context word embeddings)
        let mut context_emb = Array1::zeros(self.embedding.dim());
        for ctx_word in context {
            context_emb += &self.embedding.get_embedding(ctx_word);
        }
        context_emb /= context.len() as f32;

        // Cosine similarity → log probability
        let similarity = word_emb.dot(&context_emb) as f64;

        // Convert similarity [-1, 1] to log probability
        // Using log(0.5 + 0.5 * similarity) to map to reasonable range
        (0.5 + 0.5 * similarity).ln()
    }

    /// Score a complete sentence
    pub fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        let order = self.ngram.order();
        let mut total = 0.0;

        for i in 0..tokens.len() {
            let context_start = i.saturating_sub(order - 1);
            let context = &tokens[context_start..i];
            let word = tokens[i];

            total += self.score(word, context);
        }

        total
    }
}
```

## OOV Handling Strategies

When a word is out-of-vocabulary for the N-gram model, the hybrid model can handle it several ways:

### Strategy 1: Embedding Only

For OOV words, rely entirely on embedding similarity:

```rust
OovStrategy::EmbeddingOnly

// When "splendiferous" is OOV:
// - N-gram model backs off to uniform distribution (low score)
// - Embedding captures semantic similarity to known words
// - Final score weighted toward embedding
```

### Strategy 2: Backoff with Embedding

Use N-gram backoff chain but boost with embedding similarity:

```rust
OovStrategy::BackoffWithEmbedding

// When "splendiferous" is OOV for 5-gram:
// 1. N-gram backs off: 5-gram → 4-gram → 3-gram → ...
// 2. Embedding provides semantic context
// 3. Both contribute to final score
```

### Strategy 3: Fixed Penalty

Assign a fixed log probability to OOV words:

```rust
OovStrategy::FixedPenalty(-10.0)

// When "splendiferous" is OOV:
// - ngram_score = -10.0 (fixed)
// - embedding_score computed normally
// - Interpolation applies
```

### OOV Detection

```rust
impl<D: MutableMappedDictionary<Value = NgramEntry>> HybridLanguageModel<D> {
    /// Check if a word is in the N-gram vocabulary
    pub fn is_known(&self, word: &str) -> bool {
        self.ngram.contains_unigram(word)
    }

    /// Get OOV rate for a token sequence
    pub fn oov_rate(&self, tokens: &[&str]) -> f64 {
        let oov_count = tokens.iter().filter(|w| !self.is_known(w)).count();
        oov_count as f64 / tokens.len() as f64
    }
}
```

## Interpolation Strategies

libgrammstein supports multiple interpolation strategies:

### Linear Interpolation (Default)

Simple weighted average:

```rust
InterpolationStrategy::Linear { ngram_weight: 0.8, embedding_weight: 0.2 }

score = 0.8 × ngram_score + 0.2 × embedding_score
```

### Log-Linear Interpolation

Multiply in probability space (add in log space):

```rust
InterpolationStrategy::LogLinear { ngram_weight: 0.8, embedding_weight: 0.2 }

score = 0.8 × ngram_score + 0.2 × embedding_score
// (Same formula, but embedding_score is already in log space)
```

### Adaptive Interpolation

Adjust weights based on N-gram count:

```rust
InterpolationStrategy::Adaptive { base_ngram_weight: 0.9 }

// If N-gram has high count → trust N-gram more
// If N-gram has low count → trust embedding more

let ngram_count = self.ngram.get_count(word, context);
let confidence = (ngram_count as f64 / 100.0).min(1.0);
let ngram_weight = 0.5 + 0.4 * confidence;  // 0.5 to 0.9
let embedding_weight = 1.0 - ngram_weight;

score = ngram_weight × ngram_score + embedding_weight × embedding_score
```

## Implementing the LanguageModel Trait

The hybrid model implements lling-llang's `LanguageModel` trait:

```rust
impl<D> LanguageModel for HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    fn score_sequence(&self, tokens: &[&str]) -> f64 {
        self.sentence_log_prob(tokens)
    }

    fn score_continuation(&self, prefix: &[&str], next: &str) -> f64 {
        self.score(next, prefix)
    }
}
```

This enables seamless integration with lling-llang's correction pipelines.

## Creating a Hybrid Model

### From Trained Components

```rust
use libgrammstein::prelude::*;

// Load or train N-gram model
let ngram: NgramModel<DynamicDawgChar<NgramEntry>> = NgramModel::load("ngram.bin")?;

// Load or train embedding model
let embedding = SubwordEmbedding::load("embedding.bin")?;

// Create hybrid model
let config = HybridConfig {
    ngram_weight: 0.8,
    embedding_weight: 0.2,
    cache_size: 10_000,
    oov_strategy: OovStrategy::BackoffWithEmbedding,
};

let hybrid = HybridLanguageModel::new(ngram, embedding, config);
```

### Training Both Components

```rust
use libgrammstein::prelude::*;

// Prepare corpus reader
let reader = PlaintextReader::from_directory("./corpus")?;

// Train N-gram model
let ngram = TrainerBuilder::new()
    .order(5)
    .min_count(2)
    .train(&reader)?;

// Train embedding model
let embedding = EmbeddingTrainer::new()
    .dimension(100)
    .epochs(20)
    .window(5)
    .train(&reader)?;

// Combine
let hybrid = HybridLanguageModel::new(ngram, embedding, HybridConfig::default());
hybrid.save("hybrid_model.bin")?;
```

## Thread Safety

The hybrid model is designed for concurrent access:

| Component | Thread Safety Mechanism |
|-----------|------------------------|
| `ngram` | `Arc<D>` where `D: Send + Sync` |
| `embedding` | Immutable embeddings + `Arc<DashMap>` cache |
| `config` | Plain data (Copy) |
| `cache` | `Mutex<LruCache>` for interior mutability |

All components satisfy `Send + Sync`, making `HybridLanguageModel` usable across threads.

## Performance Optimization

### Caching

The LRU cache stores recently computed scores:

```rust
cache_size: 10_000  // Default

// Cache hit: O(1) lookup
// Cache miss: Compute N-gram + embedding scores
```

### Batch Scoring

For efficiency, score multiple sequences in parallel:

```rust
use rayon::prelude::*;

let sequences: Vec<Vec<&str>> = ...;

let scores: Vec<f64> = sequences
    .par_iter()
    .map(|seq| hybrid.sentence_log_prob(seq))
    .collect();
```

### Pre-warming the Cache

For known high-frequency queries:

```rust
impl<D: MutableMappedDictionary<Value = NgramEntry>> HybridLanguageModel<D> {
    pub fn prewarm_cache(&self, common_contexts: &[(Vec<&str>, Vec<&str>)]) {
        for (context, words) in common_contexts {
            for word in words {
                self.score(word, context);
            }
        }
    }
}
```

## Memory Layout

```
HybridLanguageModel
├── ngram: NgramModel<D>
│   ├── dictionary: Arc<D>      # Trie with NgramEntry values
│   ├── smoothing: KneserNeySmoothing
│   └── vocab_size: usize
│
├── embedding: SubwordEmbedding
│   ├── word_embeddings: Array2<f32>     # [200K × 100] = 80MB
│   ├── subword_embeddings: Array2<f32>  # [2M × 100] = 800MB
│   ├── word_to_idx: HashMap
│   ├── idx_to_word: Vec<String>
│   └── cache: Arc<DashMap>
│
├── config: HybridConfig
│   ├── ngram_weight: f64
│   ├── embedding_weight: f64
│   ├── cache_size: usize
│   └── oov_strategy: OovStrategy
│
└── cache: Mutex<LruCache<CacheKey, f64>>
    └── Capacity: ~10,000 entries
```

## Comparison with Pure Models

| Metric | N-gram Only | Embedding Only | Hybrid |
|--------|-------------|----------------|--------|
| Perplexity | Lower for in-domain | Higher | Balanced |
| OOV handling | Poor | Excellent | Good |
| Query latency | ~100ns | ~1μs | ~1μs |
| Memory | 1-2GB | 1GB | 2-3GB |
| Training time | Hours | Days | Days |

## Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `ngram_weight` | 0.7-0.9 | Higher = more local context |
| `embedding_weight` | 0.1-0.3 | Higher = more semantic |
| `cache_size` | 10,000 | Larger = more memory, fewer recomputes |
| `oov_strategy` | BackoffWithEmbedding | How to handle unknown words |

## Next Steps

- [Interpolation](interpolation.md): Detailed interpolation strategies
- [OOV Handling](oov-handling.md): Out-of-vocabulary strategies
- [lling-llang Integration](../../integration/lling-llang/overview.md): Using in WFST pipelines
- [Training](../../training/hyperparameters.md): Tuning guide
