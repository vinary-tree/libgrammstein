# Traits API Reference

This document covers the core traits used throughout libgrammstein.

## CorpusReader

The `CorpusReader` trait defines the interface for reading text corpora.

### Definition

```rust
pub trait CorpusReader: Send + Sync {
    /// Returns an iterator over sentences in the corpus.
    fn sentences(&self) -> Box<dyn Iterator<Item = String> + '_>;
}
```

### Implementations

#### PlaintextReader

Reads plain text files or directories.

```rust
use libgrammstein::corpus::PlaintextReader;

// From file
let reader = PlaintextReader::from_file("corpus.txt")?;

// From directory (reads all .txt files)
let reader = PlaintextReader::from_directory("corpus/")?;

// From string
let reader = PlaintextReader::from_string("Hello world. This is a test.");

for sentence in reader.sentences() {
    println!("{}", sentence);
}
```

#### WikipediaReader

Reads Wikipedia XML dump files with optional streaming.

```rust
use libgrammstein::corpus::{WikipediaReader, WikipediaConfig};

// Basic usage
let reader = WikipediaReader::from_dump("enwiki-latest.xml.bz2")?;

// With configuration
let config = WikipediaConfig {
    max_articles: Some(10000),
    skip_redirects: true,
    skip_disambiguation: true,
    ..Default::default()
};
let reader = WikipediaReader::from_dump_with_config("enwiki.xml.bz2", config)?;

// HTTP streaming (requires http-corpus feature)
let reader = WikipediaReader::from_url(
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
)?;
```

#### GutenbergReader

Reads Project Gutenberg text files with boilerplate removal.

```rust
use libgrammstein::corpus::GutenbergReader;

// Automatically strips header/footer boilerplate
let reader = GutenbergReader::from_file("pg12345.txt")?;

for sentence in reader.sentences() {
    println!("{}", sentence);
}
```

### Implementing Custom Readers

```rust
use libgrammstein::corpus::CorpusReader;

struct MyReader {
    data: Vec<String>,
}

impl CorpusReader for MyReader {
    fn sentences(&self) -> Box<dyn Iterator<Item = String> + '_> {
        Box::new(self.data.iter().cloned())
    }
}
```

## Document

The `Document` struct represents a parsed document with metadata.

```rust
pub struct Document {
    /// Document title (if available)
    pub title: Option<String>,

    /// Document content as sentences
    pub sentences: Vec<String>,

    /// Source identifier
    pub source: Option<String>,
}
```

## Tokenizer

The `Tokenizer` struct handles word tokenization.

### Usage

```rust
use libgrammstein::corpus::Tokenizer;

let tokenizer = Tokenizer::new();

// Iterate over words
for word in tokenizer.words("Hello, world!") {
    println!("{}", word);  // "hello", "world"
}
```

### Configuration

```rust
let tokenizer = Tokenizer::new()
    .lowercase(true)           // Convert to lowercase (default: true)
    .strip_punctuation(true)   // Remove punctuation (default: true)
    .min_length(1);            // Minimum word length (default: 1)
```

## Normalizer

The `Normalizer` struct handles text normalization.

### Usage

```rust
use libgrammstein::corpus::Normalizer;

let normalizer = Normalizer::new();
let text = normalizer.normalize("  Hello   World!  ");
println!("{}", text);  // "hello world!"
```

### Configuration

```rust
let normalizer = Normalizer::new()
    .lowercase(true)           // Convert to lowercase
    .collapse_whitespace(true) // Normalize whitespace
    .strip_accents(false)      // Remove diacritics
    .nfkc(true);               // Apply NFKC normalization
```

## MutableMappedDictionary

From `liblevenshtein`, this trait defines the dictionary storage interface.

### Key Methods

```rust
pub trait MutableMappedDictionary: Send + Sync {
    type Value;

    /// Insert a key with value
    fn insert_with_value(&self, key: &str, value: Self::Value);

    /// Look up value by key
    fn get(&self, key: &str) -> Option<Self::Value>;

    /// Check if key exists
    fn contains_key(&self, key: &str) -> bool;

    /// Number of entries
    fn len(&self) -> usize;
}
```

### Implementations

| Type | Description | Best For |
|------|-------------|----------|
| `DynamicDawgChar<V>` | DAWG with character nodes | General purpose, runtime updates |
| `PathMapDictionary<V>` | Simple hash-based | Simple cases, debugging |
| `DoubleArrayTrieChar<V>` | Double-array trie | Read-only production |

### Usage with N-gram Models

```rust
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};

let dictionary = DynamicDawgChar::<NgramEntry>::new();
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .train(&reader)?;
```

## IterableDictionary

Trait for dictionaries that support iteration (required for portable serialization).

```rust
pub trait IterableDictionary {
    /// Iterate over all (key, value) pairs
    fn iter_entries(&self) -> impl Iterator<Item = (String, Self::Value)>;
}
```

## QualityFilter Trait

Defines the interface for text quality filtering.

```rust
use libgrammstein::corpus::{QualityFilter, QualityFilterBuilder};

let filter = QualityFilterBuilder::new()
    .min_words(5)
    .max_word_repetition(0.3)
    .min_alpha_ratio(0.7)
    .build();

if filter.accept("This is a good sentence.") {
    println!("Accepted!");
}
```

## Deduplicator Trait

Defines the interface for deduplication.

```rust
use libgrammstein::corpus::{Deduplicator, DeduplicatorBuilder, DeduplicationMode};

let mut dedup = DeduplicatorBuilder::new()
    .mode(DeduplicationMode::Exact)
    .build();

// Returns true if this is the first time seeing this text
if dedup.is_unique("This is a test.") {
    println!("First occurrence");
}
```

## TextPreprocessor Trait

Defines the interface for text preprocessing.

```rust
use libgrammstein::corpus::{TextPreprocessor, TextPreprocessorBuilder};

let preprocessor = TextPreprocessorBuilder::new()
    .normalize_numbers(true)
    .normalize_urls(true)
    .normalize_emails(true)
    .expand_contractions(true)
    .build();

let result = preprocessor.process("Visit https://example.com for $100 deals!");
// Result: "Visit <URL> for <NUM> deals!"
```

## Serde Traits

Models that support serialization implement:

- `serde::Serialize` - For saving models
- `serde::Deserialize` - For loading models

These are feature-gated behind `serde-extras`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["serde-extras"] }
```

## See Also

- [NgramModel API](ngram.md) - N-gram model methods
- [SubwordEmbedding API](embedding.md) - Embedding methods
- [Error Types](errors.md) - Error handling
