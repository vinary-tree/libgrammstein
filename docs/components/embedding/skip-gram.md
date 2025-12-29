# Skip-gram Training

This document describes the skip-gram with negative sampling algorithm used for training word embeddings in libgrammstein.

## Overview

Skip-gram learns word representations by predicting context words from center words:

```
Sentence: "the quick brown fox jumps"
          Window = 2

Center: "brown"
Context: ["the", "quick", "fox", "jumps"]

Objective: Maximize P(context | center)
```

## Algorithm

### Training Objective

Maximize log probability of observing context words:

```
L = Σ Σ log P(w_c | w_t)
    t c∈C(t)
```

Where:
- `w_t` = center word at position t
- `C(t)` = context words within window of t
- `P(w_c | w_t)` = softmax probability

### Negative Sampling

Full softmax is expensive. Negative sampling approximates:

```
log P(w_c | w_t) ≈ log σ(v_c · v_t) + Σ E[log σ(-v_n · v_t)]
                                      n∈N
```

Where:
- `σ` = sigmoid function
- `v_c` = context word vector
- `v_t` = target word vector
- `N` = set of negative samples

### Implementation

```rust
fn train_skip_gram(
    &mut self,
    center: usize,
    context: usize,
    negatives: &[usize],
    learning_rate: f32,
) {
    let center_vec = self.word_embeddings.row(center).to_owned();

    // Positive sample
    let context_vec = self.context_embeddings.row(context);
    let dot = center_vec.dot(&context_vec);
    let grad = (sigmoid(dot) - 1.0) * learning_rate;

    // Update context embedding
    self.context_embeddings.row_mut(context)
        .scaled_add(-grad, &center_vec);

    // Update center embedding
    let mut center_grad = context_vec.to_owned() * grad;

    // Negative samples
    for &neg in negatives {
        let neg_vec = self.context_embeddings.row(neg);
        let dot = center_vec.dot(&neg_vec);
        let grad = sigmoid(dot) * learning_rate;

        // Update negative embedding
        self.context_embeddings.row_mut(neg)
            .scaled_add(-grad, &center_vec);

        // Accumulate center gradient
        center_grad.scaled_add(grad, &neg_vec);
    }

    // Apply center update
    self.word_embeddings.row_mut(center) -= &center_grad;
}
```

## Configuration

### Training Parameters

```rust
let model = EmbeddingTrainerBuilder::new()
    .dim(100)           // Embedding dimension
    .window_size(5)     // Context window
    .negative_samples(5) // Negatives per positive
    .learning_rate(0.025)
    .min_learning_rate(0.0001)
    .epochs(5)
    .train(&corpus)?;
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 100 | Vector dimensionality |
| `window_size` | 5 | Context window (each side) |
| `negative_samples` | 5 | Negative samples per word |
| `learning_rate` | 0.025 | Initial learning rate |
| `min_learning_rate` | 0.0001 | Final learning rate |
| `epochs` | 5 | Training epochs |

## Negative Sampling Distribution

Words are sampled proportionally to frequency^0.75:

```rust
fn build_noise_distribution(word_counts: &[u64]) -> Vec<f64> {
    let total: f64 = word_counts.iter()
        .map(|&c| (c as f64).powf(0.75))
        .sum();

    word_counts.iter()
        .map(|&c| (c as f64).powf(0.75) / total)
        .collect()
}
```

This gives rare words slightly higher sampling probability than their frequency would suggest.

## Window Size

Context window determines which words are considered related:

```
Window = 2:
"the [quick] [brown] FOX [jumps] [over]"
       ←──────────────→

Window = 5:
"[the] [quick] [brown] FOX [jumps] [over] [the] [lazy] [dog]"
 ←────────────────────────────────────────────────────────→
```

### Dynamic Window

Training uses random window size for diversity:

```rust
fn sample_window(max_window: usize, rng: &mut impl Rng) -> usize {
    rng.gen_range(1..=max_window)
}
```

## Subword Integration

Skip-gram is augmented with subword embeddings:

```rust
fn train_with_subwords(
    &mut self,
    center_word: &str,
    context: usize,
    negatives: &[usize],
    lr: f32,
) {
    // Get center vector (word + subwords)
    let subwords = extract_subwords(center_word);
    let center_vec = self.compute_word_vector(center_word);

    // Train as usual
    let grad = self.compute_gradient(&center_vec, context, negatives);

    // Distribute gradient to word and subwords
    if let Some(word_idx) = self.word_to_idx.get(center_word) {
        self.word_embeddings.row_mut(*word_idx) -= &(grad.clone() * lr);
    }

    for subword in &subwords {
        let bucket = hash_subword(subword);
        self.subword_embeddings.row_mut(bucket) -= &(grad.clone() * lr);
    }
}
```

## Learning Rate Schedule

Learning rate decays linearly during training:

```rust
fn current_learning_rate(
    initial_lr: f32,
    min_lr: f32,
    progress: f64,
) -> f32 {
    let lr = initial_lr * (1.0 - progress) + min_lr * progress;
    lr.max(min_lr)
}
```

Progress is computed as `words_processed / total_words`.

## Training Pipeline

```rust
pub fn train(&mut self, corpus: &impl CorpusReader) -> Result<()> {
    // 1. Build vocabulary
    self.build_vocabulary(corpus)?;

    // 2. Build noise distribution
    self.build_noise_table();

    // 3. Initialize embeddings
    self.initialize_embeddings();

    // 4. Train
    for epoch in 0..self.epochs {
        for sentence in corpus.sentences() {
            let tokens = self.tokenize(&sentence);

            for i in 0..tokens.len() {
                let window = sample_window(self.window_size);

                for j in i.saturating_sub(window)..=(i + window).min(tokens.len() - 1) {
                    if i != j {
                        let negatives = self.sample_negatives();
                        self.train_skip_gram(tokens[i], tokens[j], &negatives);
                    }
                }
            }
        }
    }

    Ok(())
}
```

## Optimization Tips

### Memory Efficiency

```rust
// Use half precision for large models
.precision(Precision::Half)

// Reduce negative samples
.negative_samples(3)  // Instead of 5
```

### Training Speed

```rust
// Increase batch size (internal)
.batch_size(256)

// Use more threads
.num_threads(8)
```

### Quality

```rust
// More epochs for small corpora
.epochs(10)

// Larger window for semantic similarity
.window_size(10)

// Smaller window for syntactic similarity
.window_size(2)
```

## Convergence

Monitor training loss:

```rust
trainer.on_epoch(|epoch, loss| {
    println!("Epoch {}: loss = {:.4}", epoch, loss);
});

// Expected: Decreasing loss per epoch
// Epoch 1: loss = 3.4567
// Epoch 2: loss = 2.8901
// Epoch 3: loss = 2.5678
// ...
```

## See Also

- [BPE Tokenization](bpe.md) - Subword tokenization
- [Similarity Search](similarity.md) - Using trained embeddings
- [Hyperparameters](../../training/hyperparameters.md) - Tuning guide
