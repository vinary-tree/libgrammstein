# Dictionary Extraction Overview

This document provides an overview of the dictionary extraction module in libgrammstein for building spelling dictionaries.

## Purpose

Dictionary extraction creates word lists from corpora for use in:

- **Spell checking**: Validating words against known vocabulary
- **WFST error correction**: Building weighted finite-state transducers
- **Fuzzy matching**: Levenshtein automata with liblevenshtein
- **Vocabulary analysis**: Frequency statistics and coverage

## Architecture

```
Corpus → Preprocessing → Word Extraction → Filtering → Dictionary
           │                   │               │            │
           └─ Quality filter   └─ Count freq   └─ Min freq  └─ DoubleArrayTrie
                               └─ Track stats  └─ Max vocab     + PathMap
```

## Components

### Word Extractor

Counts word frequencies from corpus:

```rust
use libgrammstein::dictionary::WordExtractor;
use liblevenshtein::dictionary::persistent_ar_trie_char::PersistentARTrieChar;

let extractor = WordExtractor::new(PersistentARTrieChar::new());
extractor.process_corpus(&corpus_reader)?;

println!("Unique words: {}", extractor.vocabulary_size());
println!("Total tokens: {}", extractor.total_tokens());
```

### Dictionary Builder

Converts extracted words to optimized format:

```rust
use libgrammstein::dictionary::DictionaryBuilder;

let dictionary = DictionaryBuilder::new()
    .from_extractor(&extractor)
    .min_frequency(5)         // Filter rare words
    .max_vocabulary(100_000)  // Limit size
    .build()?;
```

### Spelling Dictionary

Final structure for WFST integration:

```rust
use libgrammstein::dictionary::SpellingDictionary;

let dict = SpellingDictionary::load("dictionary.bin")?;

// Check if word exists
if dict.contains("hello") {
    println!("Valid word");
}

// Get frequency
if let Some(freq) = dict.frequency("hello") {
    println!("Frequency: {}", freq);
}
```

## Data Flow

### Extraction Phase

```rust
// 1. Create extractor with concurrent trie
let extractor = WordExtractor::new(PersistentARTrieChar::new());

// 2. Process corpus (parallel)
for sentence in corpus.sentences() {
    for word in tokenize(&sentence) {
        extractor.insert_or_increment(&word);
    }
}

// 3. Compute statistics
let stats = extractor.compute_stats();
```

### Building Phase

```rust
// 1. Filter by frequency
let filtered = extractor.filter(|word, freq| freq >= 5);

// 2. Build optimized trie
let trie = DoubleArrayTrieChar::from_iter(
    filtered.iter().map(|(w, e)| (w.as_str(), e))
);

// 3. Build PathMap for word → entry lookup
let path_map = PathMapDictionary::from_iter(
    filtered.iter().enumerate().map(|(i, (w, e))| (w.as_str(), i))
);
```

### Serialization

```rust
// Save dictionary
dictionary.save("dictionary.bin")?;

// Load dictionary
let dictionary = SpellingDictionary::load("dictionary.bin")?;
```

## CLI Commands

```bash
# Extract words from corpus
grammstein dictionary extract ./corpus.txt ./words.dict

# Show dictionary info
grammstein dictionary info ./words.dict

# List top words
grammstein dictionary list ./words.dict --top 100

# Merge dictionaries
grammstein dictionary merge ./merged.dict ./dict1.dict ./dict2.dict

# Look up word
grammstein dictionary lookup ./words.dict "hello"
```

## Use Cases

### Spell Checker Vocabulary

```rust
let dict = SpellingDictionary::load("en_US.dict")?;

fn is_valid_word(word: &str) -> bool {
    dict.contains(&word.to_lowercase())
}
```

### WFST Error Correction

```rust
use liblevenshtein::levenshtein::Levenshtein;

let dict = SpellingDictionary::load("dictionary.bin")?;
let lev = Levenshtein::new(dict.trie());

// Find similar words within edit distance 2
let candidates = lev.search("recieve", 2);
// ["receive", "relieve", "deceive", ...]
```

### Frequency-Based Ranking

```rust
fn rank_candidates(dict: &SpellingDictionary, candidates: &[String]) -> Vec<String> {
    let mut ranked: Vec<_> = candidates.iter()
        .map(|w| (w, dict.frequency(w).unwrap_or(0)))
        .collect();

    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    ranked.into_iter().map(|(w, _)| w.clone()).collect()
}
```

## Memory Considerations

| Dictionary Size | Memory (DoubleArrayTrie) | Memory (PathMap) |
|-----------------|--------------------------|------------------|
| 100K words | ~4 MB | ~8 MB |
| 500K words | ~20 MB | ~40 MB |
| 1M words | ~40 MB | ~80 MB |
| 5M words | ~200 MB | ~400 MB |

## Best Practices

1. **Use PersistentARTrieChar for extraction**: Supports concurrent updates

2. **Use DoubleArrayTrieChar for production**: Fastest lookups

3. **Filter appropriately**: Balance coverage vs. size

4. **Include frequency data**: Essential for ranking candidates

5. **Normalize during extraction**: Consistent casing, Unicode normalization

## See Also

- [Extraction Details](extraction.md) - Word counting implementation
- [Building Details](building.md) - Dictionary construction
- [WFST Integration](../../integration/dictionary-wfst.md) - Error correction
- [Backend Selection](../../integration/liblevenshtein/backend-selection.md) - Trie choice
