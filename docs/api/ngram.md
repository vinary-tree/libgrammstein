# NgramModel API Reference

The `NgramModel<D>` struct provides n-gram language model functionality with Modified Kneser-Ney smoothing.

## Overview

N-gram models estimate the probability of a word given its preceding context. The model uses:

- **Modified Kneser-Ney smoothing** for robust probability estimation
- **Trie-based storage** via `liblevenshtein` dictionary backends
- **Parallel training** with Rayon for efficient corpus processing

## Type Parameters

| Parameter | Description |
|-----------|-------------|
| `D` | Dictionary backend implementing `MutableMappedDictionary<Value = NgramEntry>` |

Common backends:
- `DynamicDawgChar<NgramEntry>` - General purpose, supports runtime updates
- `PathMapDictionary<NgramEntry>` - Simple hash-based storage
- `DoubleArrayTrieChar<NgramEntry>` - Fast read-only lookups

## Construction

### Training from Corpus

```rust
use libgrammstein::ngram::TrainerBuilder;
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

let reader = PlaintextReader::from_file("corpus.txt")?;
let dictionary = DynamicDawgChar::new();

let model = TrainerBuilder::new(dictionary)
    .order(5)           // 5-gram model
    .batch_size(10000)  // Parallel batch size
    .train(&reader)?;
```

### Loading from File

```rust
use libgrammstein::ngram::NgramModel;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

// Binary format (requires serde-extras feature)
let model: NgramModel<DynamicDawgChar<NgramEntry>> = NgramModel::load("model.bin")?;

// Portable format (works with any dictionary backend)
let model = NgramModel::load_portable("model.portable.bin", DynamicDawgChar::new)?;
```

## Methods

### Query Methods

#### `log_prob(word, context) -> f64`

Compute the log probability of a word given context.

```rust
// P(fox | quick brown) using up to (order-1) context words
let log_prob = model.log_prob("fox", &["quick", "brown"]);

// P(the) unigram probability
let unigram_prob = model.log_prob("the", &[]);
```

**Returns:** Log probability (base e). More negative = less likely.

#### `sentence_log_prob(tokens) -> f64`

Compute the total log probability of a sentence.

```rust
let tokens = ["the", "quick", "brown", "fox"];
let log_prob = model.sentence_log_prob(&tokens);
```

**Returns:** Sum of log probabilities for each word given its context.

#### `count(tokens) -> u64`

Get the raw count for an n-gram.

```rust
let bigram_count = model.count(&["quick", "brown"]);
let trigram_count = model.count(&["the", "quick", "brown"]);
```

#### `in_vocabulary(word) -> bool`

Check if a word was seen during training.

```rust
if model.in_vocabulary("fox") {
    println!("Known word");
}
```

### Model Properties

| Method | Return Type | Description |
|--------|-------------|-------------|
| `order()` | `usize` | Maximum n-gram order |
| `vocab_size()` | `usize` | Number of unique unigrams |
| `total_count()` | `u64` | Total token count in training corpus |
| `ngram_count()` | `usize` | Number of n-grams stored |
| `oov_log_prob()` | `f64` | Log probability for OOV words |

### Serialization (requires `serde-extras` feature)

#### `save(path) -> Result<()>`

Save model to binary file.

```rust
model.save("model.bin")?;
```

#### `load(path) -> Result<Self>`

Load model from binary file.

```rust
let model: NgramModel<DynamicDawgChar<NgramEntry>> = NgramModel::load("model.bin")?;
```

#### `save_portable(path) -> Result<()>`

Save in portable format (works with any dictionary backend).

```rust
model.save_portable("model.portable.bin")?;
```

#### `load_portable(path, factory) -> Result<Self>`

Load from portable format with dictionary factory.

```rust
let model = NgramModel::load_portable(
    "model.portable.bin",
    || DynamicDawgChar::new()
)?;
```

## Training Configuration

The `TrainerBuilder` provides a fluent API for configuring training:

```rust
let model = TrainerBuilder::new(dictionary)
    .order(5)              // N-gram order (default: 5)
    .batch_size(10000)     // Parallel batch size (default: 10000)
    .min_word_freq(1)      // Minimum word frequency (default: 1)
    .train(&reader)?;
```

### Training with Progress

```rust
use crossbeam_channel::bounded;

let (tx, rx) = bounded(100);

// Spawn progress monitor
std::thread::spawn(move || {
    while let Ok(progress) = rx.recv() {
        println!(
            "Sentences: {}, N-grams: {}, Time: {:.1}s",
            progress.sentences_processed,
            progress.ngrams_counted,
            progress.elapsed_secs
        );
    }
});

// Train with progress reporting
let trainer = NgramTrainer::new(dictionary, TrainingConfig::new(5));
let model = trainer.train_with_progress(&reader, tx)?;
```

## Smoothing

The model uses Modified Kneser-Ney smoothing with:

- **Absolute discounting** with order-specific discount values (D1, D2, D3+)
- **Interpolated backoff** to lower-order models
- **Continuation counts** for probability estimation

Default discount values:
- D1 = 0.5 (n-grams occurring once)
- D2 = 0.75 (n-grams occurring twice)
- D3+ = 0.9 (n-grams occurring 3+ times)

## Performance Considerations

1. **Dictionary Backend Selection**
   - Use `DynamicDawgChar` for general purpose with good compression
   - Use `PathMapDictionary` for simple cases without compression
   - Use `DoubleArrayTrieChar` for read-only production models

2. **Memory Usage**
   - Higher order models require more memory
   - Use `min_word_freq` to filter rare words
   - Portable format is smaller than direct serialization

3. **Training Speed**
   - Increase `batch_size` for better parallelization
   - Use streaming corpus readers for large files

## Example: Complete Workflow

```rust
use libgrammstein::ngram::{NgramModel, TrainerBuilder, NgramEntry};
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

fn main() -> libgrammstein::Result<()> {
    // 1. Load corpus
    let reader = PlaintextReader::from_file("corpus.txt")?;

    // 2. Train model
    let dictionary = DynamicDawgChar::new();
    let model = TrainerBuilder::new(dictionary)
        .order(5)
        .train(&reader)?;

    // 3. Query probabilities
    let log_prob = model.log_prob("world", &["hello"]);
    println!("log P(world|hello) = {:.4}", log_prob);

    // 4. Score sentences
    let sentence = ["the", "quick", "brown", "fox"];
    let sentence_prob = model.sentence_log_prob(&sentence);
    let perplexity = (-sentence_prob / sentence.len() as f64).exp();
    println!("Perplexity: {:.2}", perplexity);

    // 5. Save model
    model.save("model.bin")?;

    Ok(())
}
```

## See Also

- [Training Guide](../training/ngram.md) - Detailed training workflow
- [Hybrid Model](hybrid.md) - Combining n-grams with embeddings
- [CorpusReader Trait](traits.md) - Corpus reading interfaces
