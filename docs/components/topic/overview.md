# Topic Extraction Overview

The topic module provides BERTopic-style topic modeling for document clustering and keyword extraction.

## What is Topic Modeling?

Topic modeling discovers latent themes in a document collection:

```
Documents                          Topics
┌───────────────┐                 ┌─────────────────────────────────────┐
│ Doc 1: ML     │                 │ Topic 0: machine, learning, model   │
│ Doc 2: NLP    │    ───────►     │ Topic 1: natural, language, text    │
│ Doc 3: CV     │                 │ Topic 2: image, vision, recognition │
│ Doc 4: RL     │                 │ Topic 3: agent, reward, policy      │
│ ...           │                 └─────────────────────────────────────┘
└───────────────┘
        │
        ▼
Document → Topic assignments:
  Doc 1 → Topic 0 (ML)
  Doc 2 → Topic 1 (NLP)
  Doc 3 → Topic 2 (CV)
  ...
```

## BERTopic-Style Algorithm

The topic module implements a BERTopic-inspired pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: Document Embeddings (from RAG module)                           │
│                                                                          │
│   Documents → ModernBERT → 768-dim embeddings                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: Hierarchical Agglomerative Clustering                           │
│                                                                          │
│   Embeddings → Distance Matrix → HAC → Dendrogram → Cluster Labels     │
│                                                                          │
│   Linkage methods: Ward, Complete, Average, Single                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: c-TF-IDF Keyword Extraction                                     │
│                                                                          │
│   Clusters + Documents → c-TF-IDF scores → Top keywords per topic      │
│                                                                          │
│   c-TF-IDF = tf × log(1 + avg_words / freq_t)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: Topic Model                                                     │
│                                                                          │
│   TopicId → Topic { keywords, description, document_count }            │
│   DocumentId → Vec<TopicId>                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Topic Module                                     │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    TopicExtractor                                 │  │
│  │                                                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │  │
│  │  │ Hierarchical    │  │     CtfIdf      │  │  Description    │   │  │
│  │  │ Clustering      │  │ (keywords)      │  │  Generator      │   │  │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │  │
│  │           │                    │                    │            │  │
│  │           └────────────────────┼────────────────────┘            │  │
│  │                                │                                  │  │
│  │                                ▼                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │                    TopicModel                               │ │  │
│  │  │                                                             │ │  │
│  │  │  Topics: HashMap<TopicId, Topic>                           │ │  │
│  │  │  Assignments: HashMap<DocumentId, Vec<TopicId>>            │ │  │
│  │  │  Dendrogram: Linkage matrix                                │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Topic                                          │  │
│  │                                                                   │  │
│  │  id: TopicId                                                      │  │
│  │  keywords: Vec<(String, f32)>  // word, c-TF-IDF score           │  │
│  │  description: String           // generated or keyword-based     │  │
│  │  document_count: usize                                           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Extract Topics from RAG Index

```rust
use libgrammstein::topic::TopicConfig;

// Load RAG index
let mut index = RagIndex::load("./index")?;

// Get embeddings and document texts
let embeddings = index.get_all_embeddings();
let texts: Vec<_> = index.iter()
    .map(|(_, meta)| meta.synopsis.clone())
    .collect();

// Extract topics
let config = TopicConfig::default();
index.extract_topics(&config, &embeddings, &texts)?;

// Access topics
if let Some(model) = index.topic_model() {
    for topic in model.topics() {
        println!("Topic {}: {}", topic.id.0, topic.keyword_summary(5));
    }
}
```

### Standalone Topic Extraction

```rust
use libgrammstein::topic::{TopicExtractor, TopicConfig};

let config = TopicConfig {
    num_topics: Some(10),  // Target 10 topics
    min_topic_size: 5,     // Minimum 5 documents per topic
    top_keywords: 10,      // Extract 10 keywords per topic
    ..Default::default()
};

let extractor = TopicExtractor::new(config);
let model = extractor.fit(&embeddings, &texts)?;

for topic in model.topics() {
    println!("Topic {}: {}", topic.id.0, topic.description);
    for (word, score) in &topic.keywords[..5] {
        println!("  {}: {:.4}", word, score);
    }
}
```

## Configuration

```rust
use libgrammstein::topic::TopicConfig;

let config = TopicConfig {
    // Target number of topics (None = auto-determined from dendrogram)
    num_topics: Some(20),

    // Minimum documents per topic
    min_topic_size: 3,

    // Number of keywords to extract per topic
    top_keywords: 10,

    // Clustering linkage method
    linkage: LinkageMethod::Ward,

    // c-TF-IDF configuration
    min_df: 2,       // Minimum document frequency
    max_df: 0.95,    // Maximum document frequency (proportion)
};
```

## Key Concepts

### TopicId

32-bit topic identifier:

```rust
use libgrammstein::topic::TopicId;

let id = TopicId::new(0);
println!("Topic {}", id.0);
```

### Topic

Individual topic with keywords and metadata:

```rust
pub struct Topic {
    pub id: TopicId,
    pub keywords: Vec<(String, f32)>,  // (word, c-TF-IDF score)
    pub description: String,
    pub document_count: usize,
}

// Get keyword summary
let summary = topic.keyword_summary(5);  // "word1, word2, word3, word4, word5"
```

### TopicModel

Container for extracted topics:

```rust
// Iterate topics
for topic in model.topics() {
    println!("{}: {}", topic.id.0, topic.description);
}

// Get specific topic
if let Some(topic) = model.get(TopicId::new(0)) {
    println!("Found: {}", topic.description);
}

// Get document topics
if let Some(topic_ids) = model.document_topics(doc_id) {
    println!("Document belongs to {} topics", topic_ids.len());
}

// Get dendrogram
let dendrogram = model.dendrogram();
```

## Components

### Hierarchical Clustering

Groups documents using agglomerative clustering:

```rust
use libgrammstein::topic::{HierarchicalClustering, LinkageMethod};

let clustering = HierarchicalClustering::new(LinkageMethod::Ward);
let (labels, dendrogram) = clustering.fit(&embeddings, num_clusters)?;
```

See [Clustering](clustering.md) for details.

### c-TF-IDF

Extracts representative keywords per topic:

```rust
use libgrammstein::topic::CtfIdf;

let ctfidf = CtfIdf::new(min_df, max_df);
let keywords = ctfidf.extract_keywords(&texts, &labels, top_k)?;
```

See [c-TF-IDF](ctfidf.md) for details.

### Dendrogram

Represents clustering hierarchy:

```rust
// Cut dendrogram at k clusters
let labels = dendrogram.cut_tree(k);

// Cut at distance threshold
let labels = dendrogram.cut_by_distance(threshold);

// Get number of merges
let n_merges = dendrogram.len();
```

See [Dendrogram](dendrogram.md) for details.

## Integration with RAG

### Store Topics in Index

```rust
// Extract and store topics
index.extract_topics(&config, &embeddings, &texts)?;

// Topics are automatically saved with index
index.save("./index")?;

// Load index with topics
let index = RagIndex::load("./index")?;
assert!(index.topic_model().is_some());
```

### Display Topics in Query Results

```rust
for (meta, score) in index.query(&embedding, 10) {
    println!("{}: {}", meta.title.unwrap_or_default(), meta.synopsis);

    // Show document topics
    if !meta.topic_ids.is_empty() {
        if let Some(model) = index.topic_model() {
            let topic_names: Vec<_> = meta.topic_ids.iter()
                .filter_map(|id| model.get(*id))
                .map(|t| t.keyword_summary(3))
                .collect();
            println!("  Topics: {}", topic_names.join(", "));
        }
    }
}
```

## Thread Safety

The topic extractor uses parallel algorithms:

- Distance matrix computation with rayon
- Lock-free cluster assignments with atomics
- Parallel c-TF-IDF computation

```rust
// Safe to use in parallel contexts
use rayon::prelude::*;

let models: Vec<_> = datasets.par_iter()
    .map(|(embs, texts)| {
        let extractor = TopicExtractor::new(config.clone());
        extractor.fit(embs, texts)
    })
    .collect();
```

## Feature Flags

Enable the topic module with the `rag` feature:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["rag"] }
```

## See Also

- [Clustering](clustering.md) - Hierarchical agglomerative clustering
- [c-TF-IDF](ctfidf.md) - Keyword extraction algorithm
- [Dendrogram](dendrogram.md) - Topic hierarchy navigation
- [RAG Overview](../rag/overview.md) - RAG integration
- [RAG Index](../rag/index.md) - Topic storage
