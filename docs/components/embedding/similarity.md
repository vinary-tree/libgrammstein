# Similarity Search

This document describes similarity operations available on word embeddings in libgrammstein.

## Overview

Word embeddings enable semantic similarity computation:

```rust
let model = SubwordEmbedding::load("embeddings.bin")?;

// Find similar words
let similar = model.most_similar("king", 5);
// [("queen", 0.85), ("prince", 0.78), ("monarch", 0.75), ...]

// Word analogies
let result = model.analogy("king", "man", "woman", 3);
// [("queen", 0.92), ...]
```

## Similarity Metrics

### Cosine Similarity

Default metric for comparing word vectors:

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}
```

**Range**: [-1.0, 1.0]
- 1.0 = identical direction
- 0.0 = orthogonal (unrelated)
- -1.0 = opposite direction

### Euclidean Distance

Alternative metric for nearest neighbor:

```rust
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
```

**Range**: [0.0, ∞)
- 0.0 = identical
- Higher = more different

## Core Operations

### most_similar

Find words most similar to a query.

```rust
fn most_similar(&self, word: &str, k: usize) -> Vec<(String, f32)>
```

**Example**:
```rust
let similar = model.most_similar("computer", 5);
for (word, score) in similar {
    println!("{}: {:.4}", word, score);
}
// Output:
// laptop: 0.8234
// software: 0.7891
// hardware: 0.7654
// machine: 0.7234
// system: 0.6891
```

### similarity

Compute similarity between two words.

```rust
fn similarity(&self, word1: &str, word2: &str) -> f32
```

**Example**:
```rust
let sim = model.similarity("dog", "cat");
println!("dog-cat similarity: {:.4}", sim);  // ~0.75

let sim = model.similarity("dog", "computer");
println!("dog-computer similarity: {:.4}", sim);  // ~0.15
```

### analogy

Solve word analogies: A is to B as C is to ?

```rust
fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f32)>
```

**Formula**: `result = vec(C) - vec(A) + vec(B)`

**Example**:
```rust
// king - man + woman ≈ queen
let results = model.analogy("king", "man", "woman", 3);
for (word, score) in results {
    println!("{}: {:.4}", word, score);
}
// Output:
// queen: 0.8567
// princess: 0.7234
// monarch: 0.6891
```

**Common Analogies**:
- king:man :: queen:woman (gender)
- Paris:France :: Tokyo:Japan (capital-country)
- big:bigger :: small:smaller (comparative)

### doesnt_match

Find the word that doesn't belong.

```rust
fn doesnt_match(&self, words: &[&str]) -> Option<String>
```

**Example**:
```rust
let odd = model.doesnt_match(&["dog", "cat", "car", "rabbit"]);
println!("Odd one out: {:?}", odd);  // Some("car")
```

## Vector Operations

### word_vector

Get the embedding vector for a word.

```rust
fn word_vector(&self, word: &str) -> Array1<f32>
```

**Example**:
```rust
let vec = model.word_vector("hello");
println!("Dimension: {}", vec.len());  // e.g., 100
println!("First 5 values: {:?}", &vec.as_slice().unwrap()[..5]);
```

### has_word

Check if word is in vocabulary.

```rust
fn has_word(&self, word: &str) -> bool
```

Note: With subword embeddings, all words have vectors (via subwords).

### normalize_vector

Normalize a vector to unit length.

```rust
fn normalize_vector(v: &mut Array1<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        *v /= norm;
    }
}
```

## Batch Operations

### batch_most_similar

Find similar words for multiple queries.

```rust
fn batch_most_similar(
    &self,
    words: &[&str],
    k: usize,
) -> Vec<Vec<(String, f32)>>
```

More efficient than individual calls.

### batch_similarity

Compute pairwise similarities.

```rust
fn batch_similarity(
    &self,
    pairs: &[(&str, &str)],
) -> Vec<f32>
```

**Example**:
```rust
let pairs = [("dog", "cat"), ("dog", "car"), ("cat", "kitten")];
let sims = model.batch_similarity(&pairs);
// [0.75, 0.15, 0.82]
```

## Performance

### Caching

Vectors are cached for repeated access:

```rust
// First access: compute from word + subwords
let v1 = model.word_vector("hello");

// Second access: retrieved from cache
let v2 = model.word_vector("hello");
```

### Approximate Nearest Neighbor

For large vocabularies, use approximate search:

```rust
// Build index for fast similarity search
let index = model.build_similarity_index();

// Fast approximate search
let similar = index.query("computer", 5);
```

### GPU Acceleration

For batch operations on large models:

```rust
#[cfg(feature = "gpu")]
{
    let gpu_model = model.to_gpu()?;
    let similarities = gpu_model.batch_similarity_matrix(&words)?;
}
```

## Quality Evaluation

### Similarity Benchmarks

```rust
// Word similarity test sets
let pairs = [
    ("dog", "cat", 0.80),      // Expected similarity
    ("car", "automobile", 0.95),
    ("happy", "sad", 0.30),
];

let mut correlation = 0.0;
for (w1, w2, expected) in pairs {
    let actual = model.similarity(w1, w2);
    correlation += (actual - expected).abs();
}
```

### Analogy Accuracy

```rust
// Test analogy solving
let tests = [
    ("king", "man", "woman", "queen"),
    ("Paris", "France", "Tokyo", "Japan"),
];

let mut correct = 0;
for (a, b, c, expected) in tests {
    let results = model.analogy(a, b, c, 1);
    if results[0].0 == expected {
        correct += 1;
    }
}
println!("Analogy accuracy: {:.1}%", correct as f64 / tests.len() as f64 * 100.0);
```

## Use Cases

### Spelling Correction

```rust
// Find similar correctly-spelled words
let candidates = model.most_similar("recieve", 10);
// [("receive", 0.89), ("received", 0.85), ...]
```

### Query Expansion

```rust
// Expand search query with synonyms
fn expand_query(model: &SubwordEmbedding, query: &str) -> Vec<String> {
    let mut expanded = vec![query.to_string()];

    for word in query.split_whitespace() {
        let similar = model.most_similar(word, 3);
        for (w, score) in similar {
            if score > 0.7 {
                expanded.push(w);
            }
        }
    }

    expanded
}
```

### Clustering

```rust
// Group words by semantic similarity
fn cluster_words(
    model: &SubwordEmbedding,
    words: &[&str],
    threshold: f32,
) -> Vec<Vec<String>> {
    // Simple greedy clustering
    let mut clusters: Vec<Vec<String>> = Vec::new();

    for word in words {
        let vec = model.word_vector(word);
        let mut added = false;

        for cluster in &mut clusters {
            let centroid = compute_centroid(model, cluster);
            if cosine_similarity(&vec, &centroid) > threshold {
                cluster.push(word.to_string());
                added = true;
                break;
            }
        }

        if !added {
            clusters.push(vec![word.to_string()]);
        }
    }

    clusters
}
```

## See Also

- [BPE Tokenization](bpe.md) - Subword handling
- [Skip-gram Training](skip-gram.md) - Training algorithm
- [Embedding API](../../api/embedding.md) - Complete API
