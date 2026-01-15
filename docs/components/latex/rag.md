# Equation RAG (Retrieval-Augmented Generation)

The RAG module enables retrieval of similar equations from a pre-indexed corpus, supporting correction validation and equation completion.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Equation RAG System                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query: Partial or malformed equation                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Query Embedder  │ ← ModernBERT                           │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Retrieval       │ ← Similarity search                    │
│  │ Backend         │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│  ┌────────┴────────────────────────┐                        │
│  │                                 │                        │
│  ▼                                 ▼                        │
│  ┌─────────────────┐   ┌─────────────────┐                 │
│  │ Exact Backend   │   │ HNSW Backend    │                 │
│  │ (small index)   │   │ (large index)   │                 │
│  └────────┬────────┘   └────────┬────────┘                 │
│           │                     │                           │
│           └──────────┬──────────┘                           │
│                      │                                      │
│                      ▼                                      │
│  ┌─────────────────────────────────────────┐               │
│  │ Top-K Similar Equations                  │               │
│  │ with synopses and metadata               │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Use Cases

1. **Correction Validation**: Verify corrections against known equations
2. **Equation Completion**: Suggest completions for partial equations
3. **Error Detection**: Find similar correct equations for malformed input
4. **Documentation Lookup**: Find explanations for equations

## Basic Usage

```rust
use libgrammstein::rag::{RagIndex, ExactCosineBackend, Retriever, RetrievalConfig};
use libgrammstein::neural::{ModernBertEmbedder, EmbeddingConfig};

// Create embedder
let embedder = ModernBertEmbedder::new(EmbeddingConfig::default())?;

// Load index
let index = RagIndex::<ExactCosineBackend>::load("equation_index.bin")?;

// Create retriever
let config = RetrievalConfig {
    top_k: 10,
    min_similarity: 0.5,
    ..Default::default()
};

let mut retriever = Retriever::new(Arc::new(index), embedder, config);
```

## Querying

```rust
// Query with equation
let query = r"\sum_{i=1}^{n} x_i";
let results = retriever.query(query)?;

for result in &results {
    println!("{}. [{:.2}] {}", result.rank, result.score, result.display_title());
    println!("   Synopsis: {}", result.synopsis);
}
```

Output:

```
1. [0.95] Arithmetic Sum
   Synopsis: Sum of n terms
2. [0.87] Series Definition
   Synopsis: Definition of finite series
3. [0.82] Mean Formula
   Synopsis: Arithmetic mean as sum divided by n
```

## Index Building

### Creating Documents

```rust
use libgrammstein::rag::{Document, DocumentBuilder, DocumentMeta};

let doc = DocumentBuilder::new("eq:sum_formula")
    .title("Sum Formula")
    .content(r"\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n")
    .explicit_synopsis("The sum of a sequence of n terms")
    .language("en")
    .build()?;
```

### Building Index

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

let config = IndexBuilderConfig {
    embedding_config: EmbeddingConfig::default(),
    generate_synopsis: true,
    batch_size: 100,
};

let mut builder = IndexBuilder::<ExactCosineBackend>::new(config)?;

// Add documents
for equation in equation_corpus {
    let doc = DocumentBuilder::new(&equation.id)
        .title(&equation.name)
        .content(&equation.latex)
        .build()?;
    builder.add_document(doc)?;
}

// Build and save
let index = builder.build()?;
index.save("equation_index.bin")?;
```

## Retrieval Backends

### Exact Cosine Backend

Best for indices up to ~100K documents:

```rust
use libgrammstein::rag::ExactCosineBackend;

let backend = ExactCosineBackend::new(768);  // embedding dimension
let index = RagIndex::new(backend);
```

Features:
- BLAS-accelerated dot product
- Pre-normalized embeddings
- Exact (not approximate) results

### HNSW Backend

For large indices (100K+ documents):

```rust
#[cfg(feature = "rag-hnsw")]
use libgrammstein::rag::HnswBackend;

let backend = HnswBackend::new(HnswConfig {
    embedding_dim: 768,
    ef_construction: 200,
    m: 16,
    ef_search: 100,
});

let index = RagIndex::new(backend);
```

Features:
- Approximate nearest neighbor
- Sublinear query time
- Higher memory usage

## Retrieval Configuration

```rust
pub struct RetrievalConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Minimum similarity threshold
    pub min_similarity: f32,
    /// Include explicit synopses
    pub include_explicit_synopsis: bool,
    /// Include generated synopses
    pub include_generated_synopsis: bool,
}
```

## Document Structure

```rust
pub struct Document {
    /// Unique identifier
    pub id: DocumentId,
    /// Document title
    pub title: Option<String>,
    /// Full content (LaTeX)
    pub content: String,
    /// Synopsis (summary)
    pub synopsis: Option<String>,
    /// Language tag
    pub language: LanguageTag,
    /// Additional metadata
    pub metadata: DocumentMetadata,
}
```

## Retrieval Results

```rust
pub struct RetrievalResult {
    /// Document URI
    pub uri: String,
    /// Document title
    pub title: Option<String>,
    /// Synopsis
    pub synopsis: String,
    /// Is synopsis explicit (vs generated)
    pub synopsis_is_explicit: bool,
    /// Similarity score (0.0 to 1.0)
    pub score: f32,
    /// Rank in results
    pub rank: usize,
}
```

## Batch Retrieval

```rust
use libgrammstein::rag::BatchRetriever;

let batch_retriever = BatchRetriever::new(retriever);

let queries = vec![
    r"\sum_{i=1}^{n} x_i",
    r"\int_0^1 f(x) dx",
    r"\frac{d}{dx} e^x",
];

let results = batch_retriever.query_batch(&queries)?;

for (query, query_results) in queries.iter().zip(results.iter()) {
    println!("Query: {}", query);
    for result in query_results {
        println!("  {}: {}", result.rank, result.display_title());
    }
}
```

## Synopsis Generation

Automatic synopsis generation using extractive summarization:

```rust
use libgrammstein::neural::{Summarizer, SummarizerConfig};

let summarizer = Summarizer::new(SummarizerConfig {
    max_synopsis_sentences: 2,
    min_sentence_length: 10,
    model_config: ModernBertConfig::default(),
})?;

let synopsis = summarizer.summarize(&document_content)?;
```

## Query Pre-Embedding

For repeated queries:

```rust
// Pre-compute embedding
let embedding = embedder.embed_query(r"\sum_{i=1}^{n} x_i")?;

// Use cached embedding
let results = retriever.query_with_embedding(&embedding)?;
```

## Formatting Results

```rust
use libgrammstein::rag::format_results;

let results = retriever.query(query)?;
let formatted = format_results(&results);
println!("{}", formatted);
```

Output:

```
1. [0.95] Sum Formula
   URI: eq:sum_formula
   Synopsis (explicit): The sum of a sequence of n terms

2. [0.87] Series Definition
   URI: eq:series_def
   Synopsis (generated): Definition and properties of finite series
```

## Index Persistence

```rust
// Save index
index.save("equation_index.bin")?;

// Load index
let index = RagIndex::<ExactCosineBackend>::load("equation_index.bin")?;
```

## Performance

| Index Size | Backend | Query Time | Memory |
|------------|---------|------------|--------|
| 10K | Exact | 2ms | 100MB |
| 100K | Exact | 20ms | 1GB |
| 100K | HNSW | 5ms | 1.5GB |
| 1M | HNSW | 10ms | 15GB |

## Integration with Correction

```rust
// During correction, validate against known equations
fn validate_correction(original: &str, corrected: &str, retriever: &mut Retriever) -> bool {
    let original_results = retriever.query(original).unwrap_or_default();
    let corrected_results = retriever.query(corrected).unwrap_or_default();

    // If corrected version matches better, it's likely valid
    let original_best = original_results.first().map(|r| r.score).unwrap_or(0.0);
    let corrected_best = corrected_results.first().map(|r| r.score).unwrap_or(0.0);

    corrected_best > original_best
}
```

## Related

- [Embeddings](./embedding.md): Embedding generation
- [Neural Rescorer](./rescorer.md): Neural scoring
- [Overview](./overview.md): Module architecture
