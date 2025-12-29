# Embedding Training Guide

This guide covers training FastText-style subword embeddings with libgrammstein.

## Overview

Subword embeddings learn distributed representations of words:

1. **Vocabulary building** - Count word frequencies, filter rare words
2. **Skip-gram training** - Predict context words from center word
3. **Negative sampling** - Efficient approximation of softmax
4. **Subword integration** - Update character n-gram embeddings

## Quick Start

```rust
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::corpus::PlaintextReader;

// Load corpus
let reader = PlaintextReader::from_file("corpus.txt")?;

// Train embeddings
let model = EmbeddingTrainerBuilder::new()
    .dim(100)
    .window_size(5)
    .min_count(5)
    .epochs(5)
    .train(&reader)?;

// Save model
model.save("embeddings.bin")?;
```

## Training Configuration

### Embedding Dimension

Controls the size of word vectors:

| Dimension | Quality | Memory | Speed |
|-----------|---------|--------|-------|
| 50 | Low | Small | Fast |
| 100 | Medium | Medium | Medium |
| 300 | High | Large | Slow |

**Recommendation:** 100 for small corpora, 300 for large corpora.

```rust
.dim(100)  // 100-dimensional vectors
```

### Context Window

Words considered as context:

```rust
.window_size(5)  // 5 words on each side
```

Larger windows capture broader semantic relationships.

### Minimum Word Count

Filter rare words from vocabulary:

```rust
.min_count(5)  // Words appearing < 5 times are ignored
```

Higher values reduce vocabulary size and training time.

### Training Epochs

Number of passes over the corpus:

```rust
.epochs(5)  // 5 passes
```

More epochs generally improve quality but increase training time.

### Negative Sampling

Number of negative samples per positive sample:

```rust
.neg_samples(5)  // 5 negative samples
```

More samples improve quality but slow training.

### Learning Rate

Initial learning rate (decays linearly):

```rust
.learning_rate(0.05)  // Default: 0.05
```

Higher rates train faster but may be unstable.

## Complete Configuration

```rust
let model = EmbeddingTrainerBuilder::new()
    .dim(100)              // Embedding dimension
    .window_size(5)        // Context window size
    .min_count(5)          // Minimum word frequency
    .neg_samples(5)        // Negative samples
    .epochs(5)             // Training epochs
    .learning_rate(0.05)   // Initial learning rate
    .batch_size(10000)     // Parallel batch size
    .train(&reader)?;
```

## How Skip-gram Works

For each word in the corpus:

1. **Select center word** with position `t`
2. **Sample context window** randomly from `[1, window_size]`
3. **For each context word** at position `t ± offset`:
   - Compute dot product with center word
   - Apply sigmoid to get probability
   - Compute gradient for positive sample
4. **Sample negative words** not in context
   - Compute gradients for negative samples
5. **Update embeddings** for center, context, and subwords

## Subword Embeddings

Subwords are character n-grams:

```
"hello" → ["<he", "hel", "ell", "llo", "lo>", "<hel", "hell", ...]
```

Configuration:

```rust
// In EmbeddingConfig
min_subword_len: 3,  // Minimum n-gram length
max_subword_len: 6,  // Maximum n-gram length
bucket_count: 2_000_000,  // Hash buckets
```

### OOV Word Handling

For out-of-vocabulary words, the model:

1. Extracts character n-grams
2. Hashes each to a bucket
3. Averages the subword embeddings

This provides reasonable vectors for unseen words.

## Progress Monitoring

```rust
use crossbeam_channel::bounded;
use std::thread;

let (tx, rx) = bounded(100);

thread::spawn(move || {
    while let Ok(progress) = rx.recv() {
        println!(
            "Epoch {}/{} | Words: {}/{} | LR: {:.6}",
            progress.epoch,
            total_epochs,
            progress.words_processed,
            progress.total_words,
            progress.learning_rate
        );
    }
});

let trainer = EmbeddingTrainer::new(config);
let model = trainer.train_with_progress(&reader, tx)?;
```

## Evaluating Embeddings

### Word Similarity

```rust
// Check similar words
let similar = model.most_similar("king", 10);
for (word, score) in similar {
    println!("{}: {:.4}", word, score);
}

// Expected output for well-trained model:
// queen: 0.8234
// prince: 0.7891
// monarch: 0.7654
```

### Word Analogies

```rust
// Test: king - man + woman ≈ queen
let results = model.analogy("man", "king", "woman", 5);
for (word, score) in results {
    println!("{}: {:.4}", word, score);
}
```

### Intrinsic Evaluation

Use standard benchmarks:

```rust
fn evaluate_similarity(
    model: &SubwordEmbedding,
    word_pairs: &[(String, String, f32)],  // (word1, word2, human_score)
) -> f64 {
    let mut predicted = Vec::new();
    let mut actual = Vec::new();

    for (w1, w2, score) in word_pairs {
        if model.contains(w1) && model.contains(w2) {
            predicted.push(model.similarity(w1, w2) as f64);
            actual.push(*score as f64);
        }
    }

    // Compute Spearman correlation
    spearman_correlation(&predicted, &actual)
}
```

## Memory Optimization

### Memory Usage

| Component | Size Formula |
|-----------|--------------|
| Word embeddings | vocab_size × dim × 4 bytes |
| Subword embeddings | bucket_count × dim × 4 bytes |
| Vocabulary | ~vocab_size × 20 bytes |

Example for dim=100, vocab=100k, buckets=2M:
- Word: 100k × 100 × 4 = 40 MB
- Subword: 2M × 100 × 4 = 800 MB
- Total: ~850 MB

### Reducing Memory

1. **Lower dimension**:
   ```rust
   .dim(50)  // Half the memory
   ```

2. **Fewer buckets**:
   ```rust
   // In EmbeddingConfig
   bucket_count: 500_000,  // 25% of default
   ```

3. **Higher min_count**:
   ```rust
   .min_count(10)  // Smaller vocabulary
   ```

## Training Tips

### Corpus Size Guidelines

| Corpus Size | Dimension | Epochs |
|-------------|-----------|--------|
| < 1M words | 50-100 | 10-20 |
| 1-10M words | 100 | 5-10 |
| 10-100M words | 100-200 | 3-5 |
| > 100M words | 200-300 | 1-3 |

### Quality Indicators

- Similar words should have high cosine similarity
- Analogies should work (king - man + woman ≈ queen)
- OOV words should have reasonable neighbors

### Common Issues

1. **Poor quality embeddings**
   - Increase epochs
   - Increase corpus size
   - Lower learning rate

2. **Training too slow**
   - Decrease epochs
   - Increase batch_size
   - Reduce dim

3. **Out of memory**
   - Reduce bucket_count
   - Reduce dim
   - Increase min_count

## Using Pre-trained Embeddings

Load and extend pre-trained models:

```rust
// Load pre-trained
let mut model = SubwordEmbedding::load("pretrained.bin")?;

// Use for downstream tasks
let vec = model.word_vector("hello");

// Find similar words
let similar = model.most_similar("computer", 10);
```

## CLI Training

```bash
# Train embeddings
grammstein train embedding corpus.txt embeddings.bin \
    --dim 100 \
    --window 5 \
    --min-count 5 \
    --epochs 5

# With checkpoints
grammstein train embedding large-corpus.txt embeddings.bin \
    --dim 300 \
    --epochs 10 \
    --checkpoint ./checkpoints
```

## Complete Example

```rust
use libgrammstein::embedding::{SubwordEmbedding, EmbeddingTrainerBuilder};
use libgrammstein::corpus::{WikipediaReader, WikipediaConfig};

fn main() -> libgrammstein::Result<()> {
    // Load Wikipedia
    let config = WikipediaConfig {
        max_articles: Some(100_000),
        ..Default::default()
    };
    let reader = WikipediaReader::from_dump_with_config("enwiki.xml.bz2", config)?;

    // Train
    println!("Training embeddings...");
    let model = EmbeddingTrainerBuilder::new()
        .dim(100)
        .window_size(5)
        .min_count(5)
        .epochs(5)
        .train(&reader)?;

    println!("Vocabulary: {} words", model.vocab_size());
    println!("Dimension: {}", model.dim());

    // Evaluate
    println!("\nSimilar to 'king':");
    for (word, score) in model.most_similar("king", 5) {
        println!("  {}: {:.4}", word, score);
    }

    println!("\nAnalogy: man:king :: woman:?");
    for (word, score) in model.analogy("man", "king", "woman", 3) {
        println!("  {}: {:.4}", word, score);
    }

    // Save
    model.save("wikipedia-embeddings.bin")?;
    println!("\nModel saved");

    Ok(())
}
```

## See Also

- [SubwordEmbedding API](../api/embedding.md) - Complete API reference
- [Hybrid Models](../training/hybrid.md) - Combining with n-grams
- [Hyperparameters](hyperparameters.md) - Tuning guide
