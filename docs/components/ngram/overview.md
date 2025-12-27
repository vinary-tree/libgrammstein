# N-gram Language Model

This document explains what N-gram language models are, how they work, and how libgrammstein implements them.

## What is an N-gram Model?

An **N-gram language model** predicts the probability of a word given its preceding context. The model learns these probabilities by counting word sequences in a training corpus.

### The Core Idea

Given the sentence "the quick brown fox", the model answers questions like:

- What is P(fox | the quick brown)?
- What is P(brown | the quick)?
- What is P(sequence)?

### N-gram Terminology

| Term | Definition | Example |
|------|------------|---------|
| Unigram (1-gram) | Single word | "fox" |
| Bigram (2-gram) | Two consecutive words | "brown fox" |
| Trigram (3-gram) | Three consecutive words | "quick brown fox" |
| 4-gram | Four consecutive words | "the quick brown fox" |
| 5-gram | Five consecutive words | "the quick brown fox jumps" |

### Probability Formula

For an N-gram model of order N, the probability of a word w given its history is:

```
P(wₙ | w₁, w₂, ..., wₙ₋₁) ≈ P(wₙ | wₙ₋ₙ₊₁, ..., wₙ₋₁)
```

For example, a trigram (N=3) model approximates:

```
P(fox | the quick brown) ≈ P(fox | quick brown)
```

The full sentence probability is the product of individual word probabilities:

```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|wₙ₋ₙ₊₁,...,wₙ₋₁)
```

## How N-gram Models Work

### Step 1: Count N-grams

Given a corpus, count occurrences of all N-grams:

```
Corpus: "the quick brown fox the quick red fox"

Unigrams:
  "the": 2, "quick": 2, "brown": 1, "fox": 2, "red": 1

Bigrams:
  "the quick": 2, "quick brown": 1, "brown fox": 1,
  "quick red": 1, "red fox": 1

Trigrams:
  "the quick brown": 1, "quick brown fox": 1,
  "the quick red": 1, "quick red fox": 1
```

### Step 2: Compute Maximum Likelihood Estimates

The simplest probability estimate is the Maximum Likelihood Estimate (MLE):

```
P_MLE(w | context) = count(context, w) / count(context)

Example:
P_MLE(brown | the quick) = count("the quick brown") / count("the quick")
                         = 1 / 2
                         = 0.5
```

### Step 3: Handle Unseen N-grams (Smoothing)

MLE fails when an N-gram wasn't seen in training (count = 0). Smoothing techniques address this by:

1. **Reserving probability mass** for unseen events
2. **Backing off** to lower-order N-grams when data is sparse

libgrammstein uses **Modified Kneser-Ney smoothing**, the state-of-the-art technique.

## Modified Kneser-Ney Smoothing

See [Modified Kneser-Ney](modified-kneser-ney.md) for the complete algorithm. Here's a summary:

### Key Innovations

1. **Absolute Discounting**: Subtract a fixed discount D from each count
2. **Context-Dependent Discounts**: Different D values for count=1, count=2, and count≥3
3. **Continuation Counts**: For lower orders, use "number of contexts word appears in" instead of raw frequency

### The Formula

```
P_MKN(w | context) = max(count(context, w) - D(count), 0) / count(context)
                   + γ(context) × P_MKN(w | shorter_context)
```

Where:
- D(count) is the discount (D₁, D₂, or D₃₊ depending on count)
- γ(context) is the backoff weight (probability mass reserved for unseen words)
- The recursion continues until unigrams, which back off to uniform distribution

## libgrammstein Implementation

### NgramEntry

Each N-gram is stored with its statistics:

```rust
#[derive(Clone, Debug, Default)]
pub struct NgramEntry {
    /// Raw corpus count: how many times this N-gram appeared
    pub count: u64,

    /// Continuation count: how many unique preceding contexts
    /// Used for Modified Kneser-Ney's lower-order probabilities
    pub continuation_count: u32,

    /// Unique continuations: how many unique words follow this context
    /// Used to compute backoff weights
    pub unique_continuations: u32,
}
```

### NgramModel

The main model struct:

```rust
pub struct NgramModel<D: MutableMappedDictionary<Value = NgramEntry>> {
    /// Dictionary storing N-grams as "w1|w2|w3" → NgramEntry
    dictionary: Arc<D>,

    /// Model order (e.g., 5 for 5-gram)
    order: usize,

    /// Modified Kneser-Ney smoothing parameters
    smoothing: KneserNeySmoothing,

    /// Vocabulary size for uniform backoff
    vocab_size: usize,
}
```

### Key Methods

```rust
impl<D: MutableMappedDictionary<Value = NgramEntry>> NgramModel<D> {
    /// Compute log P(word | context)
    pub fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        // Apply Modified Kneser-Ney smoothing with backoff
    }

    /// Compute log P(w₁, w₂, ..., wₙ)
    pub fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        tokens.windows(self.order)
            .map(|window| {
                let context = &window[..window.len() - 1];
                let word = window[window.len() - 1];
                self.log_prob(word, context)
            })
            .sum()
    }

    /// Get the probability of a complete N-gram
    pub fn ngram_prob(&self, ngram: &[&str]) -> f64 {
        let context = &ngram[..ngram.len() - 1];
        let word = ngram[ngram.len() - 1];
        self.log_prob(word, context).exp()
    }
}
```

## Storage in Trie Dictionaries

N-grams are stored in liblevenshtein trie dictionaries for efficient access.

### Key Encoding

N-grams are encoded as pipe-separated strings:

```
"the quick brown" → "the|quick|brown"
```

This encoding enables:
- **Prefix matching**: Find all N-grams starting with "the|quick"
- **Efficient traversal**: DictZipper descends character-by-character
- **Compact storage**: Shared prefixes stored once

### Storage Example

```
Dictionary contents:
├── "the" → NgramEntry { count: 1000000, ... }
├── "the|quick" → NgramEntry { count: 50000, ... }
├── "the|quick|brown" → NgramEntry { count: 5000, ... }
├── "the|quick|red" → NgramEntry { count: 3000, ... }
└── "the|slow" → NgramEntry { count: 20000, ... }
```

### Backend Selection

| Backend | Use Case | Trade-offs |
|---------|----------|------------|
| `DynamicDawgChar` | Training, mutable models | Supports updates, serializable |
| `PathMapDictionary` | lling-llang integration | Distributed, structural sharing |
| `DoubleArrayTrieChar` | Production inference | Fastest reads, static only |

## Training Workflow

### Count Collection

```rust
use libgrammstein::ngram::{TrainerBuilder, NgramEntry};
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::DynamicDawgChar;

// Create dictionary for storing counts
let dictionary = DynamicDawgChar::<NgramEntry>::new();

// Train from corpus
let model = TrainerBuilder::new(dictionary)
    .order(5)                    // 5-gram model
    .min_count(2)                // Prune N-grams with count < 2
    .train(&PlaintextReader::from_file("corpus.txt")?)?;
```

### What Training Does

1. **First Pass**: Stream corpus, count all N-grams up to order N
2. **Collect Continuation Counts**: Count unique contexts for each word
3. **Compute Discounts**: Calculate D₁, D₂, D₃₊ from count statistics
4. **Store in Trie**: Insert all N-grams with their entries

### Memory Efficiency

Training uses streaming to handle large corpora:

```
Corpus: 10GB text file
                │
                ▼
┌─────────────────────────────────────┐
│  PlaintextReader.sentences()        │
│  - Memory-mapped file               │
│  - Yields one sentence at a time    │
└─────────────────────┬───────────────┘
                      │
                      ▼
┌─────────────────────────────────────┐
│  Parallel N-gram extraction         │
│  - Rayon par_bridge()               │
│  - Batch updates to trie            │
└─────────────────────────────────────┘
```

## Query Example

Let's trace through a probability query:

```rust
let log_prob = model.log_prob("fox", &["the", "quick", "brown"]);
```

### Step-by-Step

1. **Build Key**: `"the|quick|brown|fox"`

2. **Look Up Count**:
   ```
   count("the|quick|brown|fox") = 100
   count("the|quick|brown") = 500
   ```

3. **Apply MKN**:
   ```
   D₃₊ = 0.8 (for count ≥ 3)
   adjusted_count = max(100 - 0.8, 0) = 99.2
   P_highest = 99.2 / 500 = 0.1984
   ```

4. **Compute Backoff Weight**:
   ```
   unique_continuations("the|quick|brown") = 50
   γ = 50 × D₃₊ / 500 = 0.08
   ```

5. **Recursively Compute Lower Order**:
   ```
   P_lower = P_MKN("fox" | "quick", "brown")  // Recurse
   ```

6. **Combine**:
   ```
   P_final = P_highest + γ × P_lower
   log_prob = log(P_final)
   ```

## Performance Considerations

### Query Latency

| Factor | Impact |
|--------|--------|
| N-gram order | O(N) - one trie lookup per order |
| Dictionary type | DynamicDawg ~100ns, DoubleArray ~30ns |
| Cache | Hot queries skip computation entirely |

### Memory Usage

| Corpus Size | 5-gram Model Size | Notes |
|-------------|-------------------|-------|
| 100MB | ~200MB | Includes all N-grams with count ≥ 1 |
| 1GB | ~1-2GB | With min_count=2 pruning |
| 10GB | ~5-10GB | Significant pruning needed |

### Pruning Strategies

```rust
TrainerBuilder::new(dictionary)
    .order(5)
    .min_count(3)       // Remove N-grams appearing < 3 times
    .max_vocab(100_000) // Keep top 100K words only
    .train(&reader)?;
```

## Next Steps

- [Modified Kneser-Ney](modified-kneser-ney.md): Detailed smoothing algorithm
- [Trie Storage](trie-storage.md): Dictionary backend details
- [Query API](query-api.md): Complete query interface
- [Hybrid Overview](../hybrid/overview.md): Combining with embeddings
