# Data Flow Architecture

This document describes how data flows through libgrammstein during training and inference.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         libgrammstein                            │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   Corpus    │───>│   Training   │───>│      Model        │  │
│  │   Readers   │    │   Pipeline   │    │   (Serialized)    │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│         │                  │                      │             │
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   Quality   │    │  Dictionary  │    │     Queries       │  │
│  │   Filters   │    │   Backend    │    │  (log_prob, etc)  │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Training Pipeline

### Phase 1: Corpus Reading

```
Raw Text ──> CorpusReader ──> Sentences ──> Quality Filter ──> Clean Sentences
```

1. **Input Sources**
   - Plain text files (`PlaintextReader`)
   - Wikipedia XML dumps (`WikipediaReader`)
   - Project Gutenberg texts (`GutenbergReader`)
   - HTTP streams (Wikipedia URLs)

2. **Processing**
   - Sentence segmentation
   - Unicode normalization
   - Quality filtering (optional)
   - Deduplication (optional)

```rust
// Example flow
let reader = WikipediaReader::from_dump("enwiki.xml.bz2")?;

// Sentences are produced lazily
for sentence in reader.sentences() {
    // Each sentence is a clean string
    process(sentence);
}
```

### Phase 2: N-gram Counting

```
Clean Sentences ──> Tokenizer ──> Tokens ──> N-gram Extractor ──> Dictionary
```

1. **Tokenization**
   - Split on whitespace
   - Lowercase (optional)
   - Remove punctuation (optional)

2. **N-gram Extraction**
   - Extract all n-grams up to order N
   - For sentence ["the", "quick", "fox"]:
     - Unigrams: "the", "quick", "fox"
     - Bigrams: "the quick", "quick fox"
     - Trigrams: "the quick fox"

3. **Counting**
   - Atomic increment in dictionary
   - Track continuation counts for MKN

```rust
// Internal flow
for n in 1..=order {
    for i in 0..=(tokens.len() - n) {
        let ngram = &tokens[i..i+n];
        let key = ngram.join(" ");
        dictionary.insert_or_increment(&key);
    }
}
```

### Phase 3: Smoothing

```
Raw Counts ──> Continuation Counts ──> Discount Parameters ──> Smoothed Model
```

1. **Continuation Counts**
   - N₁, N₂, N₃+: Count of n-grams occurring 1, 2, 3+ times
   - Used for Modified Kneser-Ney discount calculation

2. **Discount Parameters**
   - D₁ = 1 - 2Y × N₂/N₁
   - D₂ = 2 - 3Y × N₃/N₂
   - D₃₊ = 3 - 4Y × N₄/N₃
   - Where Y = N₁/(N₁ + 2N₂)

### Phase 4: Embedding Training (if hybrid)

```
Sentences ──> Vocabulary ──> Skip-gram ──> Embeddings
                               │
                               ▼
                        Negative Sampling
```

1. **Vocabulary Building**
   - Count word frequencies
   - Filter by min_count
   - Build word-to-index mapping

2. **Skip-gram Training**
   - For each word, predict context words
   - Update word and subword embeddings
   - Use negative sampling for efficiency

## Inference Pipeline

### N-gram Query

```
Query ──> Context Lookup ──> Backoff Chain ──> Smoothed Probability
```

```rust
// Query: P(fox | quick brown)
let query = ["quick", "brown", "fox"];

// 1. Try trigram "quick brown fox"
if let Some(count) = trie.get("quick brown fox") {
    // Compute smoothed probability
    return mkn_probability(count, context_count);
}

// 2. Backoff to bigram "brown fox"
if let Some(count) = trie.get("brown fox") {
    return backoff_weight * mkn_probability(count, context_count);
}

// 3. Backoff to unigram "fox"
return backoff_weight² * unigram_probability("fox");
```

### Embedding Query

```
Word ──> Known? ─yes─> Word Embedding + Subword Average
           │
           no
           │
           ▼
      Subword Extraction ──> Hash to Buckets ──> Average Subword Embeddings
```

```rust
// Query: vector("hello")
if let Some(idx) = word_to_idx.get("hello") {
    // Known word: combine word and subword embeddings
    let word_vec = word_embeddings.row(idx);
    let subword_vec = average_subword_vectors("hello");
    return (word_vec + subword_vec) / 2.0;
} else {
    // OOV: use only subword embeddings
    return average_subword_vectors("hello");
}
```

### Hybrid Query

```
                         ┌──────────────┐
                         │  N-gram      │──> log P_ngram
Word + Context ─────────>│  Model       │
                         └──────────────┘
                                │
                                ├─────────> Interpolate ──> Final Score
                                │
                         ┌──────────────┐
                         │  Embedding   │──> log P_embed
Word + Context ─────────>│  Model       │
                         └──────────────┘
```

```rust
// Interpolation strategies
match strategy {
    Linear { alpha } => {
        let p_ngram = ngram.log_prob(word, context).exp();
        let p_embed = embedding_prob(word, context).exp();
        (alpha * p_ngram + (1-alpha) * p_embed).ln()
    }
    LogLinear { alpha } => {
        let lp_ngram = ngram.log_prob(word, context);
        let lp_embed = embedding_prob(word, context);
        alpha * lp_ngram + (1-alpha) * lp_embed
    }
    NgramWithEmbeddingFallback => {
        if ngram.in_vocabulary(word) {
            ngram.log_prob(word, context)
        } else {
            embedding_prob(word, context)
        }
    }
}
```

## Dictionary Backend Flow

### DynamicDawgChar (Default)

```
Insert ──> Find Prefix ──> Add Nodes ──> Store Value
                │
                ▼
         DAWG Compression (shared suffixes)
```

Benefits:
- Good compression
- Supports incremental updates
- Thread-safe (atomic operations)

### DoubleArrayTrieChar (Production)

```
Build Phase:
All Keys ──> Sort ──> Build Double Array ──> Frozen Structure

Query Phase:
Key ──> Base/Check Navigation ──> Value Lookup
```

Benefits:
- Fastest queries
- Smallest memory footprint
- Not updatable after construction

## Serialization Flow

### Binary Format

```
Model ──> serde::Serialize ──> bincode ──> Bytes ──> File
```

### Portable Format

```
Model ──> to_portable() ──> PortableModel ──> bincode ──> File

File ──> bincode ──> PortableModel ──> load_portable(factory) ──> Model<D>
```

The portable format stores n-grams as (key, entry) pairs, allowing reconstruction with any dictionary backend.

## Parallel Processing Flow

### Training Parallelism

```
        ┌─────────────────┐
        │   Sentences     │
        │     (Vec)       │
        └────────┬────────┘
                 │
                 │ par_chunks(batch_size)
                 ▼
┌────────┬───────┬───────┬────────┐
│ Thread │Thread │Thread │ Thread │
│   1    │  2    │  3    │   4    │
└───┬────┴───┬───┴───┬───┴────┬───┘
    │        │       │        │
    ▼        ▼       ▼        ▼
  Count    Count   Count    Count
 N-grams  N-grams N-grams  N-grams
    │        │       │        │
    └────────┴───────┴────────┘
                 │
                 │ Atomic merge into dictionary
                 ▼
        ┌────────────────┐
        │   Dictionary   │
        │  (Thread-safe) │
        └────────────────┘
```

### Query Parallelism

Queries are embarrassingly parallel:

```rust
let sentences: Vec<Vec<&str>> = /* ... */;

let scores: Vec<f64> = sentences.par_iter()
    .map(|s| model.sentence_log_prob(s))
    .collect();
```

## Caching Flow

### Embedding Cache

```
word_vector(word)
       │
       ▼
  ┌──────────┐
  │  Cache   │──hit──> Return cached vector
  │ (DashMap)│
  └────┬─────┘
       │ miss
       ▼
  Compute vector
       │
       ▼
  Store in cache (if space)
       │
       ▼
  Return vector
```

### Hybrid Score Cache

```
score(word, context)
       │
       ▼
  ┌──────────┐
  │   LRU    │──hit──> Return cached score
  │  Cache   │
  └────┬─────┘
       │ miss
       ▼
  Compute score
       │
       ▼
  Store in cache
       │
       ▼
  Return score
```

## See Also

- [Threading Model](threading.md) - Concurrency details
- [NgramModel API](../api/ngram.md) - Query interface
- [HybridLanguageModel API](../api/hybrid.md) - Hybrid interface
