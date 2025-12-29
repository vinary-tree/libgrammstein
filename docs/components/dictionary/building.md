# Dictionary Building

This document describes the dictionary building process for creating optimized spelling dictionaries in libgrammstein.

## Overview

Dictionary building converts extracted word counts into optimized structures:

```rust
use libgrammstein::dictionary::{WordExtractor, DictionaryBuilder, SpellingDictionary};

// After extraction
let extractor = /* ... */;

// Build dictionary
let dictionary = DictionaryBuilder::new()
    .from_extractor(&extractor)
    .min_frequency(5)
    .build()?;

// Save
dictionary.save("spelling.dict")?;
```

## DictionaryBuilder

### Construction

```rust
use libgrammstein::dictionary::DictionaryBuilder;

let builder = DictionaryBuilder::new()
    .from_extractor(&extractor)
    .min_frequency(5)           // Filter words by count
    .max_vocabulary(100_000)    // Limit vocabulary size
    .include_frequency(true)    // Store frequency data
    .include_metadata(true);    // Store additional metadata
```

### Building Process

```rust
// 1. Filter words by frequency
let filtered = extractor.filter(|word, entry| {
    entry.frequency >= builder.min_frequency
});

// 2. Sort by frequency (for vocabulary limiting)
let sorted: Vec<_> = filtered.into_iter()
    .sorted_by(|a, b| b.1.frequency.cmp(&a.1.frequency))
    .take(builder.max_vocabulary)
    .collect();

// 3. Build optimized trie
let trie = DoubleArrayTrieChar::from_sorted_iter(
    sorted.iter().map(|(w, e)| (w.as_str(), e.clone()))
);

// 4. Build PathMap for O(1) lookups
let path_map = PathMapDictionary::from_iter(
    sorted.iter().enumerate().map(|(i, (w, _))| (w.as_str(), i))
);
```

## SpellingDictionary

### Structure

```rust
pub struct SpellingDictionary {
    /// Fast prefix/lookup trie
    trie: DoubleArrayTrieChar<WordEntry>,

    /// Word -> index mapping
    path_map: PathMapDictionary<usize>,

    /// Metadata
    metadata: DictionaryMetadata,
}

pub struct DictionaryMetadata {
    pub vocabulary_size: usize,
    pub total_tokens: u64,
    pub language: Option<String>,
    pub created: DateTime<Utc>,
    pub source: String,
}
```

### Query Methods

```rust
impl SpellingDictionary {
    /// Check if word exists
    pub fn contains(&self, word: &str) -> bool {
        self.trie.contains(word)
    }

    /// Get word frequency
    pub fn frequency(&self, word: &str) -> Option<u64> {
        self.trie.get(word).map(|e| e.frequency)
    }

    /// Get word entry
    pub fn get(&self, word: &str) -> Option<&WordEntry> {
        self.trie.get(word)
    }

    /// Prefix search
    pub fn prefix_search(&self, prefix: &str) -> Vec<String> {
        self.trie.prefix_search(prefix)
    }

    /// Access underlying trie for fuzzy search
    pub fn trie(&self) -> &DoubleArrayTrieChar<WordEntry> {
        &self.trie
    }
}
```

## Trie Selection

### DoubleArrayTrieChar

Primary storage for fast lookups:

```rust
use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;

// Build from sorted iterator (most efficient)
let trie = DoubleArrayTrieChar::from_sorted_iter(
    words.iter().map(|(w, e)| (w.as_str(), e.clone()))
);
```

**Characteristics**:
- O(k) lookup where k = key length
- Compact memory representation
- Immutable after construction
- UTF-8 support (4-byte nodes)

### PathMapDictionary

Secondary index for O(1) word → index:

```rust
use liblevenshtein::dictionary::path_map::PathMapDictionary;

let path_map: PathMapDictionary<usize> = words.iter()
    .enumerate()
    .map(|(i, (w, _))| (w.as_str(), i))
    .collect();
```

**Use cases**:
- Frequency ranking by index
- Fast existence check
- Vocabulary iteration

## Filtering Strategies

### Frequency Threshold

```rust
// Keep words occurring 5+ times
.min_frequency(5)

// Typical thresholds:
// - Web corpus: 10-50
// - Books: 5-10
// - Small corpus: 2-3
```

### Vocabulary Size

```rust
// Keep top 100,000 words by frequency
.max_vocabulary(100_000)

// Typical sizes:
// - Basic spell check: 50,000
// - Full dictionary: 100,000-200,000
// - Technical domain: 200,000+
```

### Custom Filter

```rust
.filter(|word, entry| {
    // Keep if:
    entry.frequency >= 5 &&           // Minimum frequency
    word.chars().count() >= 2 &&      // Minimum length
    word.chars().count() <= 30 &&     // Maximum length
    !word.chars().all(|c| c.is_numeric())  // Not all digits
})
```

## Serialization

### Binary Format

```rust
// Save
dictionary.save("dictionary.bin")?;

// Load
let dictionary = SpellingDictionary::load("dictionary.bin")?;
```

Format structure:
```
[Header]
  - Magic bytes: 4 bytes
  - Version: 4 bytes
  - Metadata length: 4 bytes

[Metadata]
  - JSON-encoded DictionaryMetadata

[Trie]
  - DoubleArrayTrieChar serialized data

[PathMap]
  - PathMapDictionary serialized data
```

### Portable Format

For cross-platform compatibility:

```rust
// Export as text
dictionary.export_text("words.txt")?;

// Import from text
let dictionary = SpellingDictionary::from_text("words.txt")?;
```

## Merging Dictionaries

### Frequency Merge

```rust
let merged = DictionaryBuilder::merge(&[dict1, dict2, dict3])
    .merge_strategy(MergeStrategy::SumFrequencies)
    .build()?;
```

### Union Merge

```rust
let merged = DictionaryBuilder::merge(&[dict1, dict2])
    .merge_strategy(MergeStrategy::Union)  // Keep all words
    .build()?;
```

### Intersection Merge

```rust
let merged = DictionaryBuilder::merge(&[dict1, dict2])
    .merge_strategy(MergeStrategy::Intersection)  // Only common words
    .build()?;
```

## Optimization

### Memory Layout

DoubleArrayTrieChar is optimized for:
- Cache-friendly traversal
- Minimal pointer chasing
- Compact node representation

### Build Time

For large vocabularies:

```rust
// Use parallel building
let dictionary = DictionaryBuilder::new()
    .from_extractor(&extractor)
    .parallel(true)  // Parallel sort and construction
    .build()?;
```

### Compression

Optional zstd compression:

```rust
dictionary.save_compressed("dictionary.dict.zst")?;

let dictionary = SpellingDictionary::load_compressed("dictionary.dict.zst")?;
```

## Quality Checks

### Coverage Test

```rust
fn test_coverage(dict: &SpellingDictionary, test_words: &[String]) -> f64 {
    let found = test_words.iter()
        .filter(|w| dict.contains(w))
        .count();

    found as f64 / test_words.len() as f64
}
```

### False Positive Test

```rust
fn test_false_positives(dict: &SpellingDictionary, misspellings: &[String]) -> f64 {
    let false_pos = misspellings.iter()
        .filter(|w| dict.contains(w))
        .count();

    false_pos as f64 / misspellings.len() as f64
}
```

## CLI Usage

```bash
# Build dictionary from extracted words
grammstein dictionary build words.bin dictionary.dict \
    --min-frequency 5 \
    --max-vocabulary 100000

# Merge dictionaries
grammstein dictionary merge output.dict dict1.dict dict2.dict

# Validate dictionary
grammstein dictionary validate dictionary.dict --test-file words.txt
```

## Best Practices

1. **Use sorted input for DoubleArrayTrie**: Much faster construction

2. **Balance size vs. coverage**: More words = more memory

3. **Include frequency data**: Essential for ranking corrections

4. **Test before deployment**: Validate coverage and false positives

5. **Version dictionaries**: Track changes over time

6. **Document sources**: Record where data came from

## See Also

- [Dictionary Overview](overview.md) - Module overview
- [Extraction Details](extraction.md) - Word counting
- [WFST Integration](../../integration/dictionary-wfst.md) - Error correction
- [Backend Selection](../../integration/liblevenshtein/backend-selection.md) - Trie choice
