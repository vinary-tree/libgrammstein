# SubwordEmbedding API Reference

The `SubwordEmbedding` struct provides FastText-style word embeddings with subword (character n-gram) enrichment.

## Overview

Subword embeddings combine:

- **Word-level embeddings** for known vocabulary words
- **Subword embeddings** (character n-grams) for OOV word handling
- **Skip-gram training** with negative sampling

This approach provides robust word representations even for words not seen during training.

## Construction

### Training from Corpus

```rust
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::corpus::PlaintextReader;

let reader = PlaintextReader::from_file("corpus.txt")?;

let model = EmbeddingTrainerBuilder::new()
    .dim(100)           // Embedding dimension
    .window_size(5)     // Context window size
    .min_count(5)       // Minimum word frequency
    .epochs(5)          // Training epochs
    .neg_samples(5)     // Negative samples per positive
    .learning_rate(0.05)
    .train(&reader)?;
```

### Loading from File

```rust
use libgrammstein::embedding::SubwordEmbedding;

// Binary format (requires serde-extras feature)
let model = SubwordEmbedding::load("embeddings.bin")?;
```

### From Pre-computed Embeddings

```rust
use libgrammstein::embedding::SubwordEmbedding;
use ndarray::Array2;

let word_embeddings: Array2<f32> = /* ... */;
let subword_embeddings: Array2<f32> = /* ... */;
let vocab: Vec<String> = /* ... */;

let model = SubwordEmbedding::from_embeddings(
    word_embeddings,
    subword_embeddings,
    vocab
);
```

## Methods

### Word Vectors

#### `word_vector(word) -> Array1<f32>`

Get the embedding vector for a word.

```rust
let vec = model.word_vector("hello");
println!("Dimension: {}", vec.len());
```

For known words, returns the average of word embedding and subword embeddings.
For OOV words, returns only the averaged subword embeddings.

#### `word_vector_uncached(word) -> Array1<f32>`

Get word vector without using the cache.

```rust
let vec = model.word_vector_uncached("hello");
```

#### `sentence_vector(words) -> Array1<f32>`

Get a sentence embedding by averaging word vectors.

```rust
let vec = model.sentence_vector(&["the", "quick", "brown", "fox"]);
```

### Similarity

#### `similarity(word1, word2) -> f32`

Compute cosine similarity between two words.

```rust
let sim = model.similarity("king", "queen");
println!("Similarity: {:.4}", sim);  // e.g., 0.7234
```

**Returns:** Cosine similarity in range [-1, 1].

#### `most_similar(word, k) -> Vec<(String, f32)>`

Find the k most similar words to a query word.

```rust
let similar = model.most_similar("king", 10);
for (word, score) in similar {
    println!("{}: {:.4}", word, score);
}
```

**Returns:** Vector of (word, similarity) pairs, sorted by descending similarity.

#### `most_similar_to_vector(vector, k, exclude) -> Vec<(String, f32)>`

Find similar words to a given vector.

```rust
let query_vec = model.word_vector("king");
let similar = model.most_similar_to_vector(query_vec.view(), 10, Some("king"));
```

### Analogies

#### `analogy(a, b, c, k) -> Vec<(String, f32)>`

Perform word analogy: "a is to b as c is to ?"

Computes `b - a + c` and finds the most similar words.

```rust
// "king" - "man" + "woman" ≈ "queen"
let results = model.analogy("man", "king", "woman", 5);
for (word, score) in results {
    println!("{}: {:.4}", word, score);
}
```

### Vocabulary

#### `contains(word) -> bool`

Check if word is in vocabulary.

```rust
if model.contains("hello") {
    println!("Known word");
}
```

#### `word_index(word) -> Option<usize>`

Get the vocabulary index for a word.

```rust
if let Some(idx) = model.word_index("hello") {
    println!("Index: {}", idx);
}
```

#### `index_to_word(idx) -> Option<&str>`

Get the word at a vocabulary index.

```rust
if let Some(word) = model.index_to_word(0) {
    println!("First word: {}", word);
}
```

#### `embedding_by_index(idx) -> Option<ArrayView1<f32>>`

Get word embedding by index (without subword enrichment).

```rust
if let Some(emb) = model.embedding_by_index(0) {
    println!("Embedding: {:?}", emb);
}
```

### Model Properties

| Method | Return Type | Description |
|--------|-------------|-------------|
| `dim()` | `usize` | Embedding dimension |
| `vocab_size()` | `usize` | Vocabulary size |
| `bucket_count()` | `usize` | Number of subword hash buckets |

### Cache Management

#### `clear_cache()`

Clear the word vector cache.

```rust
model.clear_cache();
```

### Configuration

#### `with_subword_range(min, max) -> Self`

Set the subword (character n-gram) length range.

```rust
let model = model.with_subword_range(3, 6);  // 3-6 character n-grams
```

#### `with_cache_size(size) -> Self`

Set maximum cache size.

```rust
let model = model.with_cache_size(100_000);
```

### Serialization (requires `serde-extras` feature)

#### `save(path) -> Result<()>`

Save model to binary file.

```rust
model.save("embeddings.bin")?;
```

#### `load(path) -> Result<Self>`

Load model from binary file.

```rust
let model = SubwordEmbedding::load("embeddings.bin")?;
```

## Training Configuration

The `EmbeddingTrainerBuilder` provides a fluent API:

```rust
let model = EmbeddingTrainerBuilder::new()
    .dim(100)              // Embedding dimension (default: 100)
    .window_size(5)        // Context window (default: 5)
    .min_count(5)          // Min word frequency (default: 5)
    .neg_samples(5)        // Negative samples (default: 5)
    .epochs(5)             // Training epochs (default: 5)
    .learning_rate(0.05)   // Initial learning rate (default: 0.05)
    .batch_size(10000)     // Parallel batch size (default: 10000)
    .train(&reader)?;
```

### Training with Progress

```rust
use crossbeam_channel::bounded;

let (tx, rx) = bounded(100);

// Monitor progress
std::thread::spawn(move || {
    while let Ok(progress) = rx.recv() {
        println!(
            "Epoch {}/{}, Words: {}/{}, LR: {:.6}",
            progress.epoch,
            total_epochs,
            progress.words_processed,
            progress.total_words,
            progress.learning_rate
        );
    }
});

let trainer = EmbeddingTrainer::new(config);
let model = trainer.train_with_progress(&reader, tx)?;
```

## Subword Hashing

Subwords (character n-grams) are hashed to bucket indices:

```rust
use libgrammstein::embedding::{extract_subwords, hash_subword};

// Extract subwords for a word
let subwords = extract_subwords("hello", 3, 6);
// ["<he", "hel", "ell", "llo", "lo>", "<hel", "hell", "ello", "llo>", ...]

// Hash to bucket
let bucket = hash_subword("hel", 2_000_000);
```

Default configuration:
- Bucket count: 2,000,000
- Min subword length: 3
- Max subword length: 6

## Performance Considerations

1. **Dimension Selection**
   - 100 dimensions works well for small corpora
   - 300 dimensions for large corpora (Wikipedia-scale)
   - Higher dimensions = more memory, slower similarity search

2. **Vocabulary Filtering**
   - Use `min_count` to filter rare words
   - Default of 5 works well for medium corpora

3. **Training Speed**
   - More epochs = better quality, slower training
   - Decrease `neg_samples` for faster training
   - Enable parallel processing with larger batch sizes

4. **Caching**
   - Cache stores computed word vectors
   - Clear cache after modifying embeddings
   - Set appropriate cache size for memory constraints

## Example: Complete Workflow

```rust
use libgrammstein::embedding::{SubwordEmbedding, EmbeddingTrainerBuilder};
use libgrammstein::corpus::PlaintextReader;

fn main() -> libgrammstein::Result<()> {
    // 1. Load corpus
    let reader = PlaintextReader::from_file("corpus.txt")?;

    // 2. Train embeddings
    let model = EmbeddingTrainerBuilder::new()
        .dim(100)
        .window_size(5)
        .epochs(5)
        .train(&reader)?;

    // 3. Find similar words
    println!("Words similar to 'king':");
    for (word, score) in model.most_similar("king", 10) {
        println!("  {}: {:.4}", word, score);
    }

    // 4. Compute analogies
    println!("\nman:king :: woman:?");
    for (word, score) in model.analogy("man", "king", "woman", 5) {
        println!("  {}: {:.4}", word, score);
    }

    // 5. Test OOV handling
    let oov_vec = model.word_vector("untrainedword");
    println!("\nOOV vector dimension: {}", oov_vec.len());

    // 6. Save model
    model.save("embeddings.bin")?;

    Ok(())
}
```

## See Also

- [Training Guide](../training/embedding.md) - Detailed training workflow
- [Hybrid Model](hybrid.md) - Combining embeddings with n-grams
- [BPE Tokenization](../components/embedding/bpe.md) - Byte-pair encoding
