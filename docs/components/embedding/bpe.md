# BPE Subword Tokenization

This document describes the Byte-Pair Encoding (BPE) subword tokenization used in libgrammstein's embedding model.

## Overview

BPE decomposes words into subword units, enabling:

- **Open vocabulary**: Handle any word, including OOV
- **Morphological awareness**: Capture prefixes, suffixes, stems
- **Compact representation**: Efficient vocabulary size

## How BPE Works

### Training

1. Start with character-level vocabulary
2. Iteratively merge most frequent adjacent pairs
3. Stop when vocabulary reaches target size

```
Initial: ['t', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k']

Iteration 1: Merge 't' + 'h' → 'th'
            ['th', 'e', ' ', 'q', 'u', 'i', 'c', 'k']

Iteration 2: Merge 'th' + 'e' → 'the'
            ['the', ' ', 'q', 'u', 'i', 'c', 'k']

...continue until vocab_size reached
```

### Encoding

Words are broken into learned subword units:

```
"unhappiness" → ["un", "happ", "iness"]
"preprocessing" → ["pre", "process", "ing"]
"xyz123" → ["x", "y", "z", "1", "2", "3"]
```

## Configuration

### Default Settings

```rust
use libgrammstein::embedding::EmbeddingTrainerBuilder;

let model = EmbeddingTrainerBuilder::new()
    .min_subword_len(3)     // Minimum subword length
    .max_subword_len(6)     // Maximum subword length
    .subword_vocab_size(10000)  // Number of subword units
    .train(&corpus)?;
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_subword_len` | 3 | Shortest subword to consider |
| `max_subword_len` | 6 | Longest subword to extract |
| `subword_vocab_size` | 10,000 | Target vocabulary size |

## Subword Hashing

For efficiency, subwords are hashed to buckets:

```rust
fn hash_subword(subword: &str, num_buckets: usize) -> usize {
    let mut hash: u64 = 2166136261;  // FNV-1a
    for byte in subword.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(16777619);
    }
    (hash % num_buckets as u64) as usize
}
```

This bounds memory while allowing unlimited subwords.

## Word Vector Computation

Word vectors combine word and subword embeddings:

```rust
fn word_vector(&self, word: &str) -> Array1<f32> {
    let subwords = extract_subwords(word, self.min_n, self.max_n);

    // Start with word embedding if known
    let mut vector = if let Some(idx) = self.word_to_idx.get(word) {
        self.word_embeddings.row(*idx).to_owned()
    } else {
        Array1::zeros(self.dim)
    };

    // Add subword embeddings
    for subword in &subwords {
        let bucket = hash_subword(subword, self.num_buckets);
        vector += &self.subword_embeddings.row(bucket);
    }

    // Normalize
    vector /= (subwords.len() + 1) as f32;
    vector
}
```

## Subword Extraction

### Algorithm

```rust
fn extract_subwords(word: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let mut subwords = Vec::new();

    // Add boundary markers
    let marked = format!("<{}>", word);
    let chars: Vec<char> = marked.chars().collect();

    for n in min_n..=max_n {
        for i in 0..=(chars.len().saturating_sub(n)) {
            let subword: String = chars[i..i+n].iter().collect();
            subwords.push(subword);
        }
    }

    subwords
}
```

### Examples

Word: "hello" (with markers: `<hello>`)

| n | Subwords |
|---|----------|
| 3 | `<he`, `hel`, `ell`, `llo`, `lo>` |
| 4 | `<hel`, `hell`, `ello`, `llo>` |
| 5 | `<hell`, `hello`, `ello>` |
| 6 | `<hello`, `hello>` |

## OOV Handling

BPE enables robust OOV handling:

```rust
// Unknown word
let word = "supercalifragilistic";

// Even if not in vocabulary, subwords provide a vector
let vector = model.word_vector(word);

// Find similar known words
let similar = model.most_similar(word, 5);
```

### Fallback Chain

1. Check word vocabulary → use word embedding
2. Extract subwords → average subword embeddings
3. If no subwords match → return zero vector

## Memory Efficiency

### Comparison with Full Vocabulary

| Approach | Memory for 1M words |
|----------|---------------------|
| Full vocabulary | ~400 MB |
| BPE (50K subwords) | ~20 MB |

### Trade-offs

| Vocabulary Size | Quality | Memory |
|-----------------|---------|--------|
| 10,000 | Good | Low |
| 50,000 | Better | Medium |
| 100,000 | Best | High |

## Training Integration

BPE is trained alongside skip-gram:

```rust
// 1. Count word frequencies
let word_counts = count_words(&corpus);

// 2. Build BPE vocabulary from frequent words
let bpe_vocab = train_bpe(&word_counts, vocab_size);

// 3. Train skip-gram with subword augmentation
for sentence in sentences {
    for (center, context) in skip_gram_pairs(sentence, window) {
        // Update word embedding
        update_word(center, context);

        // Update subword embeddings
        for subword in extract_subwords(center) {
            update_subword(subword, context);
        }
    }
}
```

## Best Practices

### Choosing Subword Range

```rust
// For morphologically rich languages (German, Finnish)
.min_subword_len(2)
.max_subword_len(8)

// For English
.min_subword_len(3)
.max_subword_len(6)

// For CJK (character-based)
.min_subword_len(1)
.max_subword_len(4)
```

### Vocabulary Sizing

| Corpus Size | Recommended Vocab |
|-------------|-------------------|
| < 1M tokens | 5,000 - 10,000 |
| 1M - 100M tokens | 10,000 - 50,000 |
| > 100M tokens | 50,000 - 100,000 |

### Quality Checks

```rust
// Check subword coverage
let coverage = test_words.iter()
    .filter(|w| !model.word_vector(w).iter().all(|&x| x == 0.0))
    .count() as f64 / test_words.len() as f64;

println!("Coverage: {:.1}%", coverage * 100.0);
```

## See Also

- [Skip-gram Training](skip-gram.md) - Training algorithm
- [Similarity Search](similarity.md) - Finding similar words
- [Embedding API](../../api/embedding.md) - Complete API
