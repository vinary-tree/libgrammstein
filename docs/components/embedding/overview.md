# Subword Embeddings

This document explains what subword embeddings are, how they work, and how libgrammstein implements them in the FastText style.

## What are Word Embeddings?

**Word embeddings** are dense vector representations of words. Each word is mapped to a fixed-dimensional vector (typically 100-300 dimensions) where semantically similar words have similar vectors.

### The Core Idea

Instead of representing words as sparse one-hot vectors (vocabulary-sized, mostly zeros), embeddings use dense vectors:

```
One-hot (vocabulary = 10,000):
"cat"  → [0, 0, 0, ..., 1, ..., 0, 0, 0]  (10,000 dimensions)
"dog"  → [0, 0, 0, ..., 0, ..., 1, 0, 0]  (10,000 dimensions)

Embedding (dimension = 100):
"cat"  → [0.23, -0.15, 0.89, ..., 0.42]   (100 dimensions)
"dog"  → [0.25, -0.12, 0.85, ..., 0.39]   (100 dimensions)
```

Similar words have similar vectors (high cosine similarity).

### Why Embeddings?

| Problem | One-Hot | Embeddings |
|---------|---------|------------|
| Memory | O(vocabulary) per word | O(dimension) per word |
| Similarity | No notion of similarity | Semantic similarity captured |
| OOV words | Cannot represent | Can approximate via subwords |
| Context | No context awareness | Trained on context |

## The Out-of-Vocabulary (OOV) Problem

Standard word embeddings fail for words not seen during training:

```
Training vocabulary: ["cat", "dog", "running", "quickly"]

Query: "fastly"  → ??? (not in vocabulary)
Query: "doggo"   → ??? (not in vocabulary)
```

**Subword embeddings** solve this by learning representations for character sequences.

## Subword Enrichment (FastText-style)

libgrammstein uses FastText-style subword enrichment:

1. Each word is represented as the sum of:
   - Its own word embedding (if it exists)
   - The embeddings of its character n-grams (subwords)

2. Subwords are character sequences of length 3-6:

```
Word: "running"
Subwords (n=3-6):
  "<ru", "run", "unn", "nni", "nin", "ing", "ng>"     (3-grams)
  "<run", "runn", "unni", "nnin", "ning", "ing>"     (4-grams)
  "<runn", "runni", "unnin", "nning", "ning>"        (5-grams)
  "<runni", "runnin", "unning", "nning>"             (6-grams)

Where < and > are word boundary markers.
```

### Hashing Subwords

Storing embeddings for all possible subwords is impractical. Instead, subwords are hashed to a fixed number of buckets:

```
bucket_count = 2,000,000  (typical value)

hash("run") mod bucket_count → bucket 123456
hash("ing") mod bucket_count → bucket 789012

Each bucket has a learnable embedding vector.
```

### Computing Word Embeddings

```rust
pub fn get_embedding(&self, word: &str) -> Array1<f32> {
    let mut embedding = Array1::zeros(self.dim);

    // Add word embedding if known
    if let Some(&idx) = self.word_to_idx.get(word) {
        embedding += &self.word_embeddings.row(idx);
    }

    // Add subword embeddings
    for subword in self.extract_subwords(word) {
        let bucket = self.hash_subword(&subword) % self.bucket_count;
        embedding += &self.subword_embeddings.row(bucket);
    }

    // Normalize
    let norm = embedding.dot(&embedding).sqrt();
    if norm > 0.0 {
        embedding /= norm;
    }

    embedding
}
```

## libgrammstein Implementation

### SubwordEmbedding Struct

```rust
pub struct SubwordEmbedding {
    /// Word embeddings: [vocab_size × dimension]
    word_embeddings: Array2<f32>,

    /// Subword embeddings: [bucket_count × dimension]
    subword_embeddings: Array2<f32>,

    /// Word to index mapping
    word_to_idx: HashMap<String, usize>,

    /// Index to word mapping
    idx_to_word: Vec<String>,

    /// Embedding dimension (100-300 typical)
    dim: usize,

    /// Number of subword buckets
    bucket_count: usize,

    /// Minimum subword length (typically 3)
    min_subword_len: usize,

    /// Maximum subword length (typically 6)
    max_subword_len: usize,

    /// Optional BPE tokenizer
    tokenizer: Option<BpeTokenizer>,

    /// LRU cache for computed embeddings
    cache: Arc<DashMap<String, Array1<f32>>>,
}
```

### Key Methods

```rust
impl SubwordEmbedding {
    /// Get the embedding for a word (cached)
    pub fn get_embedding(&self, word: &str) -> Array1<f32> {
        if let Some(cached) = self.cache.get(word) {
            return cached.clone();
        }

        let embedding = self.compute_embedding(word);
        self.cache.insert(word.to_string(), embedding.clone());
        embedding
    }

    /// Compute cosine similarity between two words
    pub fn similarity(&self, word1: &str, word2: &str) -> f32 {
        let emb1 = self.get_embedding(word1);
        let emb2 = self.get_embedding(word2);
        emb1.dot(&emb2)  // Already normalized
    }

    /// Find k nearest neighbors to a word
    pub fn nearest_neighbors(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        let query_emb = self.get_embedding(word);

        let mut similarities: Vec<_> = self.idx_to_word
            .iter()
            .enumerate()
            .map(|(idx, w)| {
                let sim = query_emb.dot(&self.word_embeddings.row(idx));
                (w.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }
}
```

## Training: Skip-gram with Negative Sampling

libgrammstein uses the **skip-gram** training objective with **negative sampling**.

### Skip-gram Objective

Given a center word, predict its context words:

```
Sentence: "the quick brown fox jumps"
Window size: 2

Center: "brown"
Context: ["the", "quick", "fox", "jumps"]

Training samples:
  (brown, the)   → positive
  (brown, quick) → positive
  (brown, fox)   → positive
  (brown, jumps) → positive
```

### Negative Sampling

Instead of computing a full softmax over the vocabulary, sample a few "negative" examples:

```
Positive: (brown, fox)   → should be similar
Negatives: (brown, table), (brown, computer), ... → should be dissimilar
```

The loss function:

```
L = -log σ(v_fox · v_brown) - Σᵢ log σ(-v_negᵢ · v_brown)

Where σ(x) = 1 / (1 + e^(-x))
```

### Training Loop

```rust
pub fn train<R: CorpusReader>(
    reader: R,
    config: EmbeddingConfig,
) -> Result<SubwordEmbedding> {
    let mut model = SubwordEmbedding::new(config);

    for epoch in 0..config.epochs {
        reader.sentences()
            .par_bridge()  // Rayon parallelism
            .for_each(|sentence| {
                let tokens = tokenize(&sentence);

                for i in 0..tokens.len() {
                    let center = &tokens[i];

                    // Context window
                    for j in (i.saturating_sub(config.window))..=(i + config.window).min(tokens.len() - 1) {
                        if i == j { continue; }
                        let context = &tokens[j];

                        // Update embeddings
                        model.train_pair(center, context, true);   // Positive
                        for neg in model.sample_negatives(config.neg_samples) {
                            model.train_pair(center, &neg, false); // Negative
                        }
                    }
                }
            });

        // Decay learning rate
        model.learning_rate *= 0.95;
    }

    Ok(model)
}
```

## BPE Tokenization (Optional)

For more sophisticated subword segmentation, libgrammstein supports **Byte-Pair Encoding (BPE)**.

### What is BPE?

BPE learns a vocabulary of subword units by iteratively merging the most frequent character pairs:

```
Initial: ["l", "o", "w", "e", "r", "</w>", "n", "e", "w", "e", "s", "t", "</w>"]

Iteration 1: Merge ("e", "s") → "es"
Iteration 2: Merge ("es", "t") → "est"
Iteration 3: Merge ("l", "o") → "lo"
...

Final vocabulary: ["lo", "w", "er</w>", "new", "est</w>", ...]
```

### BPE Tokenizer

```rust
pub struct BpeTokenizer {
    /// BPE merges in priority order
    merges: Vec<(String, String)>,

    /// Vocabulary of subword tokens
    vocab: HashMap<String, usize>,

    /// End-of-word marker
    eow: String,
}

impl BpeTokenizer {
    /// Train BPE vocabulary from corpus
    pub fn train<R: CorpusReader>(
        reader: R,
        vocab_size: usize,
    ) -> Self {
        // Count word frequencies
        // Initialize with character vocabulary
        // Iteratively merge most frequent pairs
        // Stop when vocab_size reached
    }

    /// Tokenize a word into BPE tokens
    pub fn tokenize(&self, word: &str) -> Vec<String> {
        // Apply learned merges greedily
    }
}
```

### Using BPE with Embeddings

```rust
let tokenizer = BpeTokenizer::train(&reader, 50_000)?;
let config = EmbeddingConfig {
    tokenizer: Some(tokenizer),
    ..Default::default()
};
let embeddings = EmbeddingTrainer::train(&reader, config)?;
```

## Similarity Scoring for Language Modeling

Embeddings contribute to language model scoring via **context similarity**:

```rust
impl SubwordEmbedding {
    /// Score how well a word fits the context
    pub fn context_score(&self, word: &str, context: &[&str]) -> f64 {
        let word_emb = self.get_embedding(word);

        // Compute context embedding (average of context words)
        let mut context_emb = Array1::zeros(self.dim);
        for ctx_word in context {
            context_emb += &self.get_embedding(ctx_word);
        }
        if !context.is_empty() {
            context_emb /= context.len() as f32;
        }

        // Cosine similarity
        word_emb.dot(&context_emb) as f64
    }
}
```

## Thread Safety

`SubwordEmbedding` is designed for concurrent access:

| Component | Thread Safety |
|-----------|---------------|
| `word_embeddings` | Immutable after training |
| `subword_embeddings` | Immutable after training |
| `cache` | `Arc<DashMap>` for lock-free concurrent access |

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `get_embedding` (cached) | O(1) | DashMap lookup |
| `get_embedding` (uncached) | O(s × d) | s = subwords, d = dimension |
| `similarity` | O(d) | Dot product |
| `nearest_neighbors` | O(V × d) | V = vocabulary size |
| Training (per epoch) | O(C × w × d) | C = corpus tokens, w = window |

## Memory Layout

```
SubwordEmbedding
├── word_embeddings: Array2<f32>
│   └── [vocab_size × dim] contiguous memory
│       e.g., [200,000 × 100] = 80MB
│
├── subword_embeddings: Array2<f32>
│   └── [bucket_count × dim] contiguous memory
│       e.g., [2,000,000 × 100] = 800MB
│
├── word_to_idx: HashMap<String, usize>
│   └── ~200,000 entries
│
├── idx_to_word: Vec<String>
│   └── ~200,000 strings
│
└── cache: Arc<DashMap<String, Array1<f32>>>
    └── LRU-evicted, max ~10,000 entries
```

## Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `dim` | 100-300 | Higher = more expressive, more memory |
| `window` | 5 | Larger = more context, slower training |
| `min_count` | 5 | Filter rare words |
| `bucket_count` | 2,000,000 | More = fewer hash collisions |
| `min_subword_len` | 3 | Character n-gram minimum |
| `max_subword_len` | 6 | Character n-gram maximum |
| `neg_samples` | 5-10 | More = slower but better gradients |
| `epochs` | 5-20 | More = better quality, longer training |
| `learning_rate` | 0.025 | Initial learning rate |

## Next Steps

- [BPE Tokenizer](bpe-tokenizer.md): Detailed BPE algorithm
- [Skip-gram Training](skip-gram.md): Training with negative sampling
- [Similarity](similarity.md): Cosine similarity and nearest neighbors
- [Hybrid Overview](../hybrid/overview.md): Combining with N-gram models
