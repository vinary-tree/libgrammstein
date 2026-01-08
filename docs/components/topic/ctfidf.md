# c-TF-IDF Keyword Extraction

The c-TF-IDF (class-based TF-IDF) algorithm extracts representative keywords for each topic cluster.

## What is c-TF-IDF?

Standard TF-IDF scores terms by document frequency. c-TF-IDF adapts this for topic modeling by treating each cluster as a single document:

```
Standard TF-IDF:                    c-TF-IDF:
┌────────────────────┐              ┌────────────────────┐
│ Document 1         │              │ Topic 1            │
│   terms → scores   │              │   (all docs in     │
│ Document 2         │              │    cluster 1)      │
│   terms → scores   │              │   terms → scores   │
│ ...                │              │ Topic 2            │
└────────────────────┘              │   terms → scores   │
                                    │ ...                │
                                    └────────────────────┘
```

## Algorithm

### c-TF-IDF Formula

```
c-TF-IDF(t, c) = tf(t, c) × log(1 + A / freq(t))

where:
  t = term
  c = topic cluster
  tf(t, c) = normalized term frequency in cluster c
  A = average words per cluster
  freq(t) = number of clusters containing term t
```

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: Build Vocabulary                                                 │
│                                                                          │
│   All documents → Tokenize → Filter by min_df/max_df → Vocabulary       │
│                                                                          │
│   min_df: Minimum documents containing term (remove rare terms)         │
│   max_df: Maximum document frequency (remove common terms)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: Aggregate by Cluster                                             │
│                                                                          │
│   For each cluster c:                                                    │
│     Concatenate all documents in c                                       │
│     Count term frequencies → tf_raw(t, c)                               │
│                                                                          │
│   Compute: A = mean(total_words per cluster)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: Normalize TF                                                     │
│                                                                          │
│   For each cluster c:                                                    │
│     total_words_c = sum(tf_raw(t, c)) for all t                         │
│     tf(t, c) = tf_raw(t, c) / total_words_c                             │
│                                                                          │
│   Normalization ensures comparable scores across clusters               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: Compute IDF Component                                            │
│                                                                          │
│   For each term t:                                                       │
│     freq(t) = number of clusters containing t                           │
│     idf(t) = log(1 + A / freq(t))                                       │
│                                                                          │
│   Rare terms (low freq) get higher IDF                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: Compute c-TF-IDF Scores                                          │
│                                                                          │
│   For each (term t, cluster c):                                          │
│     score(t, c) = tf(t, c) × idf(t)                                     │
│                                                                          │
│   Select top_k terms per cluster as keywords                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Keyword Extraction

```rust
use libgrammstein::topic::CtfIdf;

// Create extractor with document frequency filters
let ctfidf = CtfIdf::new(
    2,     // min_df: term must appear in >= 2 documents
    0.95,  // max_df: term must appear in <= 95% of documents
);

// Extract keywords
let keywords = ctfidf.extract_keywords(
    &texts,    // Document texts
    &labels,   // Cluster assignments
    10,        // Top keywords per cluster
)?;

// keywords: Vec<Vec<(String, f32)>> indexed by cluster
for (cluster, words) in keywords.iter().enumerate() {
    println!("Cluster {}:", cluster);
    for (word, score) in words {
        println!("  {}: {:.4}", word, score);
    }
}
```

### With Configuration

```rust
use libgrammstein::topic::CtfIdfConfig;

let config = CtfIdfConfig {
    min_df: 3,       // Minimum document frequency
    max_df: 0.80,    // Maximum document frequency (proportion)
    lowercase: true, // Convert to lowercase
    max_vocab: None, // No vocabulary limit
};

let ctfidf = CtfIdf::with_config(config);
let keywords = ctfidf.extract_keywords(&texts, &labels, 10)?;
```

## Vocabulary Building

### Tokenization

Simple whitespace and punctuation tokenization:

```rust
// "Hello, World!" → ["hello", "world"]
// "machine-learning" → ["machine", "learning"]
```

### Document Frequency Filtering

```rust
// min_df: Remove terms appearing in fewer than N documents
// Removes: typos, rare technical terms, noise

// max_df: Remove terms appearing in more than P% of documents
// Removes: stopwords, common words ("the", "is", "and")

// Example for 1000 documents:
let config = CtfIdfConfig {
    min_df: 5,     // Must appear in >= 5 documents
    max_df: 0.70,  // Must appear in <= 700 documents
    ..Default::default()
};
```

### Vocabulary Limits

```rust
// Limit vocabulary size for memory/speed
let config = CtfIdfConfig {
    max_vocab: Some(10000),  // Keep top 10k terms by document frequency
    ..Default::default()
};
```

## Comparison with TF-IDF

| Aspect | TF-IDF | c-TF-IDF |
|--------|--------|----------|
| Unit | Document | Cluster (topic) |
| TF computation | Per document | Per cluster (aggregated) |
| IDF computation | Across documents | Across clusters |
| Purpose | Document similarity | Topic keywords |
| Output | Document vectors | Per-topic keyword lists |

## Example

```
Documents:
  Doc 1: "machine learning algorithms"      → Cluster 0
  Doc 2: "deep learning neural networks"    → Cluster 0
  Doc 3: "image recognition vision"         → Cluster 1
  Doc 4: "computer vision detection"        → Cluster 1

Cluster 0 (ML):
  Combined text: "machine learning algorithms deep learning neural networks"

  Term frequencies:
    learning: 2
    machine: 1
    algorithms: 1
    deep: 1
    neural: 1
    networks: 1

  tf(learning, 0) = 2/8 = 0.25
  idf(learning) = log(1 + 4/2) = 1.10  (appears in 2 clusters)

  c-TF-IDF(learning, 0) = 0.25 × 1.10 = 0.275

Cluster 0 keywords: learning, neural, networks, machine, ...
Cluster 1 keywords: vision, image, recognition, computer, ...
```

## Performance

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Tokenization | O(total_words) |
| Vocabulary building | O(total_words) |
| TF computation | O(total_words) |
| IDF computation | O(vocab_size × num_clusters) |
| Score computation | O(vocab_size × num_clusters) |

### Memory

```
Memory ≈ vocab_size × (num_clusters + 1) × 4 bytes

For 50k vocabulary, 20 clusters:
Memory ≈ 50,000 × 21 × 4 = 4.2 MB
```

## Best Practices

### 1. Tune Document Frequency Filters

```rust
// For small datasets (< 1000 docs)
let config = CtfIdfConfig {
    min_df: 2,
    max_df: 0.90,
    ..Default::default()
};

// For large datasets (> 100k docs)
let config = CtfIdfConfig {
    min_df: 10,
    max_df: 0.50,
    ..Default::default()
};
```

### 2. Use Lowercase for Consistency

```rust
let config = CtfIdfConfig {
    lowercase: true,  // "Machine" = "machine"
    ..Default::default()
};
```

### 3. Limit Vocabulary for Large Corpora

```rust
let config = CtfIdfConfig {
    max_vocab: Some(50000),
    ..Default::default()
};
```

### 4. Consider Stopword Removal

While max_df removes many stopwords, explicit removal can help:

```rust
// Pre-filter texts
let filtered_texts: Vec<String> = texts
    .iter()
    .map(|t| remove_stopwords(t))
    .collect();
```

## Keyword Quality

Good keywords should be:

1. **Discriminative**: High score in one cluster, low in others
2. **Frequent**: Appear multiple times in the cluster
3. **Meaningful**: Not stopwords or common terms

```rust
// Check keyword quality
for (word, score) in &keywords[cluster] {
    // High score indicates discriminative keyword
    if *score > 0.1 {
        println!("Strong keyword: {} ({:.4})", word, score);
    }
}
```

## See Also

- [Overview](overview.md) - Topic module introduction
- [Clustering](clustering.md) - Cluster computation
- [Dendrogram](dendrogram.md) - Hierarchy for topic count
