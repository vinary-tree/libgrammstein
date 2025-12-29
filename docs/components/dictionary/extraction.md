# Word Extraction

This document describes the word extraction process for building spelling dictionaries in libgrammstein.

## Overview

Word extraction counts unique words and their frequencies from a corpus:

```rust
use libgrammstein::dictionary::WordExtractor;
use liblevenshtein::dictionary::persistent_ar_trie_char::PersistentARTrieChar;

let extractor = WordExtractor::new(PersistentARTrieChar::new());

for sentence in corpus.sentences() {
    extractor.process_sentence(&sentence);
}

let stats = extractor.stats();
println!("Vocabulary: {} words", stats.vocabulary_size);
println!("Total tokens: {}", stats.total_tokens);
```

## WordExtractor

### Construction

```rust
use libgrammstein::dictionary::WordExtractor;

// With PersistentARTrieChar (recommended for extraction)
let extractor = WordExtractor::new(PersistentARTrieChar::new());

// With configuration
let extractor = WordExtractor::builder()
    .trie(PersistentARTrieChar::new())
    .lowercase(true)
    .normalize_unicode(true)
    .min_word_length(2)
    .max_word_length(50)
    .build();
```

### Processing

```rust
// Process sentence
extractor.process_sentence("The quick brown fox.");

// Process individual word
extractor.insert("hello");

// Increment existing word
extractor.insert_or_increment("hello");
```

### Concurrent Processing

PersistentARTrieChar supports atomic updates:

```rust
use rayon::prelude::*;

sentences.par_iter().for_each(|sentence| {
    // Safe concurrent access
    extractor.process_sentence(sentence);
});
```

## Word Entry

Each word stores extraction metadata:

```rust
pub struct WordEntry {
    pub frequency: u64,         // Occurrence count
    pub document_count: u32,    // Documents containing word
    pub first_seen: u64,        // Token position of first occurrence
    pub last_seen: u64,         // Token position of last occurrence
}

impl WordEntry {
    pub fn increment(&mut self) {
        self.frequency += 1;
    }
}
```

## Preprocessing

### Tokenization

```rust
pub fn tokenize(sentence: &str) -> impl Iterator<Item = &str> {
    sentence.split_whitespace()
        .flat_map(|word| {
            // Strip punctuation
            word.trim_matches(|c: char| !c.is_alphanumeric())
        })
        .filter(|w| !w.is_empty())
}
```

### Normalization

```rust
pub fn normalize_word(word: &str) -> String {
    let mut normalized = word.to_lowercase();

    // Unicode normalization (NFC)
    normalized = unicode_normalization::UnicodeNormalization::nfc(&normalized)
        .collect();

    // Handle special characters
    normalized = normalized
        .replace("'", "'")
        .replace("'", "'");

    normalized
}
```

### Filtering

```rust
pub fn is_valid_word(word: &str) -> bool {
    let len = word.chars().count();

    // Length check
    if len < 2 || len > 50 {
        return false;
    }

    // Must contain alphabetic character
    if !word.chars().any(|c| c.is_alphabetic()) {
        return false;
    }

    // Not purely numeric
    if word.chars().all(|c| c.is_numeric()) {
        return false;
    }

    true
}
```

## Statistics

### ExtractionStats

```rust
pub struct ExtractionStats {
    pub vocabulary_size: usize,
    pub total_tokens: u64,
    pub hapax_legomena: usize,    // Words occurring once
    pub dis_legomena: usize,      // Words occurring twice
    pub max_frequency: u64,
    pub mean_frequency: f64,
    pub median_frequency: u64,
}
```

### Computing Statistics

```rust
let stats = extractor.compute_stats();

println!("Vocabulary size: {}", stats.vocabulary_size);
println!("Total tokens: {}", stats.total_tokens);
println!("Hapax legomena: {} ({:.1}%)",
    stats.hapax_legomena,
    stats.hapax_legomena as f64 / stats.vocabulary_size as f64 * 100.0
);
```

## Frequency Analysis

### Zipf's Law

Word frequencies follow Zipf's law:

```
Frequency(rank r) ≈ C / r^α
```

Typical values:
- α ≈ 1.0 for natural language
- Top 100 words account for ~50% of tokens
- Hapax legomena are ~50% of vocabulary

### Frequency Distribution

```rust
fn frequency_distribution(extractor: &WordExtractor) -> HashMap<u64, u64> {
    let mut dist = HashMap::new();

    for (_, entry) in extractor.iter() {
        *dist.entry(entry.frequency).or_insert(0) += 1;
    }

    dist
}

// Example output:
// freq=1: 50000 words (hapax)
// freq=2: 12000 words
// freq=3: 6000 words
// ...
```

## Memory Optimization

### Streaming Extraction

For large corpora:

```rust
let extractor = WordExtractor::builder()
    .trie(PersistentARTrieChar::new())
    .streaming(true)  // Enable streaming mode
    .build();

for sentence in corpus.sentences() {
    extractor.process_sentence(&sentence);

    // Periodically checkpoint
    if extractor.total_tokens() % 10_000_000 == 0 {
        extractor.checkpoint("checkpoint.bin")?;
    }
}
```

### Incremental Updates

```rust
// Resume from checkpoint
let extractor = WordExtractor::load("checkpoint.bin")?;

// Continue processing
for sentence in remaining_corpus.sentences() {
    extractor.process_sentence(&sentence);
}
```

## Parallel Extraction

### Sharded Processing

```rust
use rayon::prelude::*;

// Create thread-local extractors
let extractors: Vec<_> = (0..num_threads)
    .map(|_| WordExtractor::new(PersistentARTrieChar::new()))
    .collect();

// Process in parallel
sentences.par_chunks(1000).enumerate().for_each(|(i, chunk)| {
    let extractor = &extractors[i % num_threads];
    for sentence in chunk {
        extractor.process_sentence(sentence);
    }
});

// Merge extractors
let merged = WordExtractor::merge(&extractors);
```

### Atomic Updates

PersistentARTrieChar uses atomic operations:

```rust
// Safe concurrent increment
pub fn insert_or_increment(&self, word: &str) {
    self.trie.modify(word, |entry| {
        match entry {
            Some(e) => e.increment(),
            None => WordEntry::new(),
        }
    });
}
```

## Export Formats

### Text Format

```rust
extractor.export_text("words.txt", |word, entry| {
    format!("{}\t{}", word, entry.frequency)
})?;

// Output:
// the	1234567
// of	987654
// and	876543
```

### JSON Format

```rust
extractor.export_json("words.json")?;

// Output:
// {"word": "the", "frequency": 1234567, "documents": 50000}
// {"word": "of", "frequency": 987654, "documents": 48000}
```

### Binary Format

```rust
extractor.save("words.bin")?;

// Compact binary with trie structure preserved
```

## Best Practices

1. **Use PersistentARTrieChar**: Best for concurrent extraction

2. **Normalize consistently**: Same rules during extraction and lookup

3. **Filter during extraction**: Reduces memory usage

4. **Checkpoint for large corpora**: Resume on failure

5. **Track document counts**: Useful for TF-IDF

6. **Compute statistics**: Helps set filtering thresholds

## See Also

- [Dictionary Overview](overview.md) - Module overview
- [Building Details](building.md) - Dictionary construction
- [Backend Selection](../../integration/liblevenshtein/backend-selection.md) - Trie choice
