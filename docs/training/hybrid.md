# Hybrid Model Training Guide

This guide covers training and using hybrid language models that combine n-grams with embeddings.

## Overview

Hybrid models leverage the strengths of both approaches:

| Component | Strength | Weakness |
|-----------|----------|----------|
| N-gram | Accurate for seen n-grams | Poor OOV handling |
| Embedding | Semantic similarity, OOV handling | Less precise probabilities |
| **Hybrid** | Best of both | Slightly more complex |

## Training Workflow

### Step 1: Train N-gram Model

```rust
use libgrammstein::ngram::{TrainerBuilder, NgramEntry};
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

let reader = PlaintextReader::from_file("corpus.txt")?;

let ngram_model = TrainerBuilder::new(DynamicDawgChar::new())
    .order(5)
    .min_word_freq(5)
    .train(&reader)?;
```

### Step 2: Train Embedding Model

```rust
use libgrammstein::embedding::EmbeddingTrainerBuilder;

// Re-read corpus (readers are consumed)
let reader2 = PlaintextReader::from_file("corpus.txt")?;

let embedding_model = EmbeddingTrainerBuilder::new()
    .dim(100)
    .window_size(5)
    .min_count(5)
    .epochs(5)
    .train(&reader2)?;
```

### Step 3: Create Hybrid Model

```rust
use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig, InterpolationStrategy};

let config = HybridConfig {
    strategy: InterpolationStrategy::Linear { alpha: 0.7 },
    cache_size: 50_000,
    ..Default::default()
};

let hybrid = HybridLanguageModel::new(ngram_model, embedding_model, config);
```

## Interpolation Strategies

### Linear Interpolation

Combines probabilities in linear space:

```
P(w|c) = α × P_ngram(w|c) + (1-α) × P_embed(w|c)
```

```rust
InterpolationStrategy::Linear { alpha: 0.7 }
```

**Best for:** General use, balanced performance.

### Log-Linear Interpolation

Combines log probabilities:

```
log P(w|c) = α × log P_ngram(w|c) + (1-α) × log P_embed(w|c)
```

```rust
InterpolationStrategy::LogLinear { alpha: 0.7 }
```

**Best for:** When both models are well-calibrated.

### N-gram with Fallback

Uses n-gram for known words, embedding for OOV:

```rust
InterpolationStrategy::NgramWithEmbeddingFallback
```

**Best for:** When n-gram quality is high, OOV handling needed.

### Dynamic Weighting

Adjusts weight based on available context:

```rust
InterpolationStrategy::Dynamic {
    base_alpha: 0.5,        // Base n-gram weight
    alpha_per_context: 0.1, // +0.1 per context word
    max_alpha: 0.9,         // Maximum n-gram weight
}
```

With 3 context words: α = 0.5 + 0.1 × 3 = 0.8

**Best for:** Variable-length contexts where n-gram quality improves with more context.

## Choosing Alpha

The alpha parameter controls the balance:

| Alpha | N-gram Weight | Embedding Weight | Use Case |
|-------|---------------|------------------|----------|
| 0.9 | 90% | 10% | High-quality n-gram, rare OOV |
| 0.7 | 70% | 30% | Balanced (default) |
| 0.5 | 50% | 50% | Equal weighting |
| 0.3 | 30% | 70% | Small n-gram corpus |

### Tuning Alpha

Find optimal alpha on held-out data:

```rust
fn tune_alpha(
    ngram: &NgramModel<D>,
    embedding: &SubwordEmbedding,
    dev_corpus: &impl CorpusReader,
) -> f64 {
    let mut best_alpha = 0.5;
    let mut best_ppl = f64::INFINITY;

    for alpha_int in 1..=9 {
        let alpha = alpha_int as f64 / 10.0;

        let config = HybridConfig {
            strategy: InterpolationStrategy::Linear { alpha },
            ..Default::default()
        };
        let hybrid = HybridLanguageModel::new(
            ngram.clone(),
            embedding.clone(),
            config
        );

        let ppl = evaluate_perplexity(&hybrid, dev_corpus);

        if ppl < best_ppl {
            best_ppl = ppl;
            best_alpha = alpha;
        }

        println!("α={:.1}: perplexity={:.2}", alpha, ppl);
    }

    println!("Best: α={:.1} (ppl={:.2})", best_alpha, best_ppl);
    best_alpha
}
```

## Temperature Parameter

Controls sharpness of embedding probabilities:

```rust
let config = HybridConfig {
    temperature: 1.0,  // Default: neutral
    ..Default::default()
};
```

| Temperature | Effect |
|-------------|--------|
| < 1.0 | Sharper distribution, more confident |
| 1.0 | Neutral (default) |
| > 1.0 | Smoother distribution, less confident |

## Caching

The hybrid model caches computed scores:

```rust
let config = HybridConfig {
    cache_size: 50_000,  // Cache 50k (word, context) pairs
    ..Default::default()
};

// Clear cache when needed
hybrid.clear_cache();
```

## Evaluation

### Perplexity

```rust
fn evaluate_perplexity(
    model: &HybridLanguageModel<D>,
    test_corpus: &impl CorpusReader,
) -> f64 {
    let mut total_log_prob = 0.0;
    let mut total_words = 0usize;

    for sentence in test_corpus.sentences() {
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        total_log_prob += model.sentence_log_prob(&tokens);
        total_words += tokens.len();
    }

    (-total_log_prob / total_words as f64).exp()
}
```

### OOV Performance

```rust
fn evaluate_oov_handling(
    model: &HybridLanguageModel<D>,
    oov_sentences: &[Vec<&str>],
) {
    for sentence in oov_sentences {
        let score = model.sentence_log_prob(sentence);
        let ppl = model.perplexity(sentence);

        println!("Sentence: {:?}", sentence);
        println!("  Log prob: {:.4}", score);
        println!("  Perplexity: {:.2}", ppl);
    }
}

// Test with sentences containing OOV words
let oov_test = vec![
    vec!["the", "xyzzy", "jumped"],
    vec!["qwertyuiop", "is", "a", "word"],
];
evaluate_oov_handling(&hybrid, &oov_test);
```

## Serialization

### Binary Format

```rust
// Save (requires serde-extras feature and D: Serialize)
hybrid.save("hybrid.bin")?;

// Load
let loaded: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
    HybridLanguageModel::load("hybrid.bin")?;
```

### Portable Format

```rust
// Save portable (works with any D)
hybrid.save_portable("hybrid.portable.bin")?;

// Load with different backend
let loaded = HybridLanguageModel::load_portable(
    "hybrid.portable.bin",
    || DoubleArrayTrieChar::new()
)?;
```

## CLI Training

```bash
# Train hybrid model
grammstein train hybrid corpus.txt hybrid.bin \
    --ngram-order 5 \
    --embed-dim 100 \
    --lambda 0.7

# This trains both components and saves combined model
```

## Use Cases

### Spell Correction Ranking

```rust
fn rank_corrections(
    model: &HybridLanguageModel<D>,
    context: &[&str],
    candidates: &[&str],
) -> Vec<(&str, f64)> {
    let mut scored: Vec<_> = candidates.iter()
        .map(|&c| (c, model.score(c, context)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
}

let context = ["the", "quick", "brown"];
let candidates = ["fox", "fix", "fax", "fog"];
let ranked = rank_corrections(&hybrid, &context, &candidates);

println!("Best correction: {}", ranked[0].0);
```

### Language Detection Fallback

```rust
fn is_likely_target_language(
    model: &HybridLanguageModel<D>,
    sentence: &[&str],
    threshold: f64,
) -> bool {
    let ppl = model.perplexity(sentence);
    ppl < threshold  // Lower perplexity = more likely
}
```

### Sentence Generation

```rust
fn generate_next_word(
    model: &HybridLanguageModel<D>,
    context: &[&str],
    vocabulary: &[&str],
    temperature: f64,
) -> String {
    // Score all vocabulary words
    let mut scores: Vec<(String, f64)> = vocabulary.iter()
        .map(|&w| (w.to_string(), model.score(w, context)))
        .collect();

    // Apply temperature
    let max_score = scores.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
    let probs: Vec<f64> = scores.iter()
        .map(|(_, s)| ((s - max_score) / temperature).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    let probs: Vec<f64> = probs.iter().map(|p| p / sum).collect();

    // Sample from distribution
    sample_from_distribution(&scores, &probs)
}
```

## Complete Example

```rust
use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig, InterpolationStrategy};
use libgrammstein::ngram::{TrainerBuilder, NgramEntry};
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

fn main() -> libgrammstein::Result<()> {
    // Load corpus
    let corpus_path = "corpus.txt";

    // Train n-gram
    println!("Training n-gram model...");
    let reader1 = PlaintextReader::from_file(corpus_path)?;
    let ngram = TrainerBuilder::new(DynamicDawgChar::new())
        .order(5)
        .train(&reader1)?;

    // Train embeddings
    println!("Training embeddings...");
    let reader2 = PlaintextReader::from_file(corpus_path)?;
    let embedding = EmbeddingTrainerBuilder::new()
        .dim(100)
        .epochs(5)
        .train(&reader2)?;

    // Create hybrid
    let config = HybridConfig {
        strategy: InterpolationStrategy::Linear { alpha: 0.7 },
        ..Default::default()
    };
    let hybrid = HybridLanguageModel::new(ngram, embedding, config);

    // Test
    let test_sentence = ["the", "quick", "brown", "fox"];
    println!("\nTest sentence: {:?}", test_sentence);
    println!("Log probability: {:.4}", hybrid.sentence_log_prob(&test_sentence));
    println!("Perplexity: {:.2}", hybrid.perplexity(&test_sentence));

    // OOV test
    let oov_sentence = ["the", "xyzzy", "jumped"];
    println!("\nOOV sentence: {:?}", oov_sentence);
    println!("Log probability: {:.4}", hybrid.sentence_log_prob(&oov_sentence));
    println!("Perplexity: {:.2}", hybrid.perplexity(&oov_sentence));

    // Save
    hybrid.save("hybrid-model.bin")?;
    println!("\nModel saved");

    Ok(())
}
```

## See Also

- [HybridLanguageModel API](../api/hybrid.md) - Complete API reference
- [N-gram Training](ngram.md) - N-gram component training
- [Embedding Training](embedding.md) - Embedding component training
- [Hyperparameters](hyperparameters.md) - Tuning guide
