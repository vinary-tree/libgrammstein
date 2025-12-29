# N-gram Training Guide

This guide covers the complete workflow for training n-gram language models with libgrammstein.

## Overview

N-gram models estimate word probabilities based on preceding context. Training involves:

1. **Corpus preparation** - Loading and preprocessing text
2. **N-gram counting** - Counting word sequences in parallel
3. **Smoothing** - Computing Modified Kneser-Ney parameters
4. **Serialization** - Saving the trained model

## Quick Start

```rust
use libgrammstein::ngram::{TrainerBuilder, NgramEntry};
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

// Load corpus
let reader = PlaintextReader::from_file("corpus.txt")?;

// Train 5-gram model
let dictionary = DynamicDawgChar::new();
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .train(&reader)?;

// Save model
model.save("model.bin")?;
```

## Corpus Preparation

### Supported Formats

| Format | Reader | Best For |
|--------|--------|----------|
| Plain text | `PlaintextReader` | Simple text files |
| Wikipedia | `WikipediaReader` | Large-scale training |
| Gutenberg | `GutenbergReader` | Book corpora |

### Plain Text

```rust
use libgrammstein::corpus::PlaintextReader;

// Single file
let reader = PlaintextReader::from_file("corpus.txt")?;

// Directory of files
let reader = PlaintextReader::from_directory("corpus/")?;

// In-memory string
let text = "The quick brown fox. The lazy dog.";
let reader = PlaintextReader::from_string(text);
```

### Wikipedia

```rust
use libgrammstein::corpus::{WikipediaReader, WikipediaConfig};

// Basic usage
let reader = WikipediaReader::from_dump("enwiki-latest.xml.bz2")?;

// With configuration
let config = WikipediaConfig {
    max_articles: Some(100_000),  // Limit articles
    skip_redirects: true,
    skip_disambiguation: true,
    ..Default::default()
};
let reader = WikipediaReader::from_dump_with_config("enwiki.xml.bz2", config)?;

// HTTP streaming (large dumps without downloading)
let reader = WikipediaReader::from_url(
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
)?;
```

### Preprocessing Pipeline

Apply quality filtering and normalization:

```rust
use libgrammstein::corpus::{
    PlaintextReader, QualityFilterBuilder, DeduplicatorBuilder,
    TextPreprocessorBuilder, PreprocessingPipelineBuilder
};

// Build pipeline
let filter = QualityFilterBuilder::new()
    .min_words(5)
    .max_word_repetition(0.3)
    .build();

let dedup = DeduplicatorBuilder::new()
    .mode(DeduplicationMode::Exact)
    .build();

let preprocessor = TextPreprocessorBuilder::new()
    .normalize_numbers(true)
    .normalize_urls(true)
    .build();

// Apply to corpus (use filtered sentences for training)
let reader = PlaintextReader::from_file("corpus.txt")?;
let filtered: Vec<String> = reader.sentences()
    .filter(|s| filter.accept(s))
    .filter(|s| dedup.is_unique(s))
    .map(|s| preprocessor.process(&s))
    .collect();
```

## Training Configuration

### N-gram Order

The order determines the maximum context length:

| Order | Name | Context | Example |
|-------|------|---------|---------|
| 1 | Unigram | 0 words | P(fox) |
| 2 | Bigram | 1 word | P(fox\|brown) |
| 3 | Trigram | 2 words | P(fox\|quick brown) |
| 5 | 5-gram | 4 words | P(fox\|the quick brown) |

**Recommendation:** Order 5 is a good default. Higher orders need more data.

### Minimum Word Frequency

Filter rare words to reduce model size:

```rust
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .min_word_freq(5)  // Ignore words appearing < 5 times
    .train(&reader)?;
```

### Batch Size

Control parallel processing granularity:

```rust
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .batch_size(10000)  // Process 10k sentences per batch
    .train(&reader)?;
```

**Recommendation:** Larger batches (10k-100k) are more efficient.

## Dictionary Backend Selection

| Backend | Memory | Speed | Updates | Best For |
|---------|--------|-------|---------|----------|
| `DynamicDawgChar` | Low | Good | Yes | General use |
| `PathMapDictionary` | High | Fast | Yes | Small models |
| `DoubleArrayTrieChar` | Low | Fastest | No | Production |

### DynamicDawgChar (Recommended)

```rust
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

let dictionary = DynamicDawgChar::<NgramEntry>::new();
```

Good compression, supports incremental updates.

### PathMapDictionary

```rust
use liblevenshtein::dictionary::pathmap::PathMapDictionary;

let dictionary = PathMapDictionary::<NgramEntry>::new();
```

Simple hash-based storage. Good for debugging.

### DoubleArrayTrieChar

```rust
use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;

let dictionary = DoubleArrayTrieChar::<NgramEntry>::new();
```

Fastest lookups but no updates after construction.

## Progress Monitoring

### Console Progress

```rust
use crossbeam_channel::bounded;
use std::thread;

let (tx, rx) = bounded(100);

// Progress monitor thread
thread::spawn(move || {
    while let Ok(progress) = rx.recv() {
        println!(
            "\rSentences: {} | N-grams: {} | Time: {:.1}s",
            progress.sentences_processed,
            progress.ngrams_counted,
            progress.elapsed_secs
        );
    }
    println!();
});

// Train with progress
let trainer = NgramTrainer::new(dictionary, TrainingConfig::new(5));
let model = trainer.train_with_progress(&reader, tx)?;
```

### With Progress Bar (indicatif)

```rust
use indicatif::{ProgressBar, ProgressStyle};

let pb = ProgressBar::new(total_sentences as u64);
pb.set_style(ProgressStyle::default_bar()
    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
    .progress_chars("#>-"));

thread::spawn(move || {
    while let Ok(progress) = rx.recv() {
        pb.set_position(progress.sentences_processed);
    }
    pb.finish_with_message("Training complete");
});
```

## Checkpointing

For long training runs, save periodic checkpoints:

```rust
use std::time::{Duration, Instant};

let checkpoint_interval = Duration::from_secs(300);  // Every 5 minutes
let mut last_checkpoint = Instant::now();

// During training callback
if last_checkpoint.elapsed() > checkpoint_interval {
    model.save("checkpoint.bin")?;
    last_checkpoint = Instant::now();
    log::info!("Checkpoint saved");
}
```

## Model Evaluation

### Perplexity

```rust
fn evaluate_perplexity(model: &NgramModel<D>, test_corpus: &impl CorpusReader) -> f64 {
    let mut total_log_prob = 0.0;
    let mut total_words = 0usize;

    for sentence in test_corpus.sentences() {
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        total_log_prob += model.sentence_log_prob(&tokens);
        total_words += tokens.len();
    }

    (-total_log_prob / total_words as f64).exp()
}

let test_reader = PlaintextReader::from_file("test.txt")?;
let ppl = evaluate_perplexity(&model, &test_reader);
println!("Test perplexity: {:.2}", ppl);
```

### Coverage

```rust
fn vocabulary_coverage(model: &NgramModel<D>, test_corpus: &impl CorpusReader) -> f64 {
    let mut known = 0usize;
    let mut total = 0usize;

    for sentence in test_corpus.sentences() {
        for word in sentence.split_whitespace() {
            total += 1;
            if model.in_vocabulary(word) {
                known += 1;
            }
        }
    }

    known as f64 / total as f64
}
```

## Memory Optimization

### For Large Corpora

1. **Stream the corpus** instead of loading all into memory:
   ```rust
   // WikipediaReader streams by default
   let reader = WikipediaReader::from_dump("enwiki.xml.bz2")?;
   ```

2. **Use memory-efficient dictionary**:
   ```rust
   let dictionary = DynamicDawgChar::new();  // Best compression
   ```

3. **Filter rare words**:
   ```rust
   .min_word_freq(5)  // Removes rare n-grams
   ```

4. **Lower n-gram order**:
   ```rust
   .order(3)  // Trigrams use less memory than 5-grams
   ```

### Memory Estimates

| Corpus Size | Order | Approx. Memory |
|-------------|-------|----------------|
| 1M sentences | 3 | ~500 MB |
| 1M sentences | 5 | ~1.5 GB |
| 10M sentences | 5 | ~10 GB |
| 100M sentences | 5 | ~50+ GB |

## Serialization

### Binary Format

Fast, compact, requires same dictionary type:

```rust
// Save
model.save("model.bin")?;

// Load (must specify dictionary type)
let loaded: NgramModel<DynamicDawgChar<NgramEntry>> =
    NgramModel::load("model.bin")?;
```

### Portable Format

Works with any dictionary backend:

```rust
// Save portable
model.save_portable("model.portable.bin")?;

// Load with different backend
let loaded = NgramModel::load_portable(
    "model.portable.bin",
    || DoubleArrayTrieChar::new()  // Different backend!
)?;
```

## CLI Training

Use the grammstein CLI for quick training:

```bash
# Train 5-gram model
grammstein train ngram corpus.txt model.bin --order 5

# With checkpoints
grammstein train ngram large-corpus.txt model.bin \
    --order 5 \
    --checkpoint ./checkpoints \
    --checkpoint-interval 100000

# Resume from checkpoint
grammstein train ngram large-corpus.txt model.bin \
    --resume ./checkpoints/latest.ckpt

# From Wikipedia dump
grammstein train ngram enwiki.xml.bz2 model.bin --order 5
```

## Complete Example

```rust
use libgrammstein::ngram::{NgramModel, TrainerBuilder, NgramEntry};
use libgrammstein::corpus::{WikipediaReader, WikipediaConfig};
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

fn main() -> libgrammstein::Result<()> {
    // Configure Wikipedia reader
    let config = WikipediaConfig {
        max_articles: Some(100_000),
        skip_redirects: true,
        ..Default::default()
    };
    let reader = WikipediaReader::from_dump_with_config("enwiki.xml.bz2", config)?;

    // Train with progress
    let dictionary = DynamicDawgChar::new();
    let model = TrainerBuilder::new(dictionary)
        .order(5)
        .min_word_freq(5)
        .batch_size(50_000)
        .train(&reader)?;

    println!("Vocabulary size: {}", model.vocab_size());
    println!("N-gram count: {}", model.ngram_count());

    // Evaluate
    let test = ["the", "quick", "brown", "fox"];
    let ppl = (-model.sentence_log_prob(&test) / test.len() as f64).exp();
    println!("Test perplexity: {:.2}", ppl);

    // Save
    model.save("wikipedia-5gram.bin")?;
    println!("Model saved");

    Ok(())
}
```

## See Also

- [NgramModel API](../api/ngram.md) - Complete API reference
- [Hyperparameter Tuning](hyperparameters.md) - Tuning guide
- [Large Corpora](large-corpora.md) - Memory optimization
