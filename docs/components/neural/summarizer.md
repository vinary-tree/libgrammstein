# Extractive Summarizer

The `Summarizer` extracts representative sentences from documents using embedding-based similarity and Maximal Marginal Relevance (MMR).

## What is Extractive Summarization?

Extractive summarization selects the most important sentences from a document without generating new text:

```
Original Document (10 sentences):
┌─────────────────────────────────────────────────────────────────────────┐
│ S1: Machine learning is a subset of artificial intelligence.           │
│ S2: It enables computers to learn from data.                           │
│ S3: The field was founded in the 1950s.                               │
│ S4: Early work focused on symbolic AI.                                 │
│ S5: Neural networks emerged as a key approach.                         │
│ S6: Deep learning revolutionized the field in 2012.                   │
│ S7: Applications include image recognition and NLP.                    │
│ S8: Models require large amounts of training data.                     │
│ S9: GPUs accelerated neural network training.                          │
│ S10: The technology continues to advance rapidly.                      │
└─────────────────────────────────────────────────────────────────────────┘

Extractive Summary (3 sentences):
┌─────────────────────────────────────────────────────────────────────────┐
│ S1: Machine learning is a subset of artificial intelligence.           │
│ S6: Deep learning revolutionized the field in 2012.                   │
│ S7: Applications include image recognition and NLP.                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Algorithm

The summarizer uses a three-step process:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: Sentence Splitting & Filtering                                  │
│                                                                          │
│ Document → Split on .!? → Filter by length → Valid sentences            │
│                                                                          │
│ Handles abbreviations: Dr., Mr., Mrs., etc., e.g., i.e., vs.           │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: Embedding & Centroid Computation                                │
│                                                                          │
│ Sentences → ModernBERT Embedder → Embeddings (768-dim)                  │
│                                                                          │
│ Centroid = normalize(mean(embeddings))                                  │
│                                                                          │
│ Relevance[i] = cosine_similarity(embedding[i], centroid)                │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: MMR Selection                                                   │
│                                                                          │
│ For k sentences:                                                         │
│   1. Select highest relevance sentence                                  │
│   2. For remaining selections:                                          │
│      MMR(s) = λ × relevance(s) - (1-λ) × max_sim(s, selected)          │
│      Select sentence with highest MMR                                   │
│                                                                          │
│ λ = 1.0 - diversity_threshold                                           │
└────────────────────────────────────────────────────────────────────────┘
```

### Maximal Marginal Relevance (MMR)

MMR balances relevance and diversity:

```
MMR(s) = λ × Relevance(s) - (1-λ) × max(Similarity(s, s_selected))

where:
  λ = relevance weight (higher = more relevant, less diverse)
  Relevance(s) = cosine_similarity(s, document_centroid)
  Similarity(s, s') = cosine_similarity(embedding(s), embedding(s'))
```

## Configuration

```rust
use libgrammstein::neural::SummarizerConfig;

let config = SummarizerConfig {
    // Number of sentences to extract
    num_sentences: 3,

    // Minimum sentence length (characters)
    min_sentence_length: 20,

    // Maximum sentence length (characters)
    max_sentence_length: 500,

    // Preserve original document order in output
    preserve_order: true,

    // Diversity threshold (0.0 = all relevance, 1.0 = all diversity)
    diversity_threshold: 0.3,
};
```

### Diversity Threshold

| Value | Effect |
|-------|--------|
| 0.0 | Pure relevance (may select redundant sentences) |
| 0.3 | Balanced (default) |
| 0.5 | Equal relevance and diversity |
| 0.7 | Diversity-focused |
| 1.0 | Maximum diversity (may miss key content) |

## Creating a Summarizer

### From Configuration

```rust
use libgrammstein::neural::{Summarizer, SummarizerConfig};

let config = SummarizerConfig::default();
let summarizer = Summarizer::new(config)?;
```

### From Existing Model

```rust
use std::sync::Arc;
use libgrammstein::neural::{ModernBertEmbedder, Summarizer, SummarizerConfig};

let embedder = ModernBertEmbedder::new(embedder_config)?;
let summarizer = Summarizer::from_embedder(embedder, SummarizerConfig::default());
```

## Extractive Summarization

### Basic Usage

```rust
let document = r#"
Machine learning is a subset of artificial intelligence that enables
computers to learn from data. The field was founded in the 1950s by
pioneers like Alan Turing. Deep learning, a subfield using neural
networks with many layers, has revolutionized applications like
image recognition and natural language processing.
"#;

let sentences = summarizer.extractive(document)?;

for (i, sentence) in sentences.iter().enumerate() {
    println!("{}. {}", i + 1, sentence.text);
    println!("   Score: {:.4}", sentence.similarity_score);
}
```

### ScoredSentence Structure

```rust
pub struct ScoredSentence {
    /// The sentence text
    pub text: String,

    /// Original position in document (0-indexed)
    pub original_index: usize,

    /// Similarity to document centroid
    pub similarity_score: f32,
}
```

## Creating Synopses

For RAG integration, create a `Synopsis` with source tracking:

```rust
use libgrammstein::neural::Synopsis;

let document = "Long document text here...";

// Check for explicit synopsis in metadata
let explicit_synopsis: Option<&str> = metadata.get("synopsis");

let synopsis = summarizer.create_synopsis(document, explicit_synopsis)?;

match synopsis.source {
    SynopsisSource::Explicit => {
        println!("Using author-provided synopsis");
    }
    SynopsisSource::Generated => {
        println!("Generated synopsis: {}", synopsis.text);
    }
}
```

### Synopsis Structure

```rust
pub struct Synopsis {
    /// The synopsis text
    pub text: String,

    /// Whether explicit (author-provided) or generated
    pub source: SynopsisSource,
}

pub enum SynopsisSource {
    /// Synopsis provided explicitly (e.g., from metadata)
    Explicit,

    /// Synopsis generated by summarizer
    Generated,
}
```

### Creating Synopses Directly

```rust
// From explicit text
let synopsis = Synopsis::explicit("This document covers machine learning basics.");

// Mark as generated
let synopsis = Synopsis::generated("Machine learning is a subset of AI...");

// Check source
if synopsis.is_explicit() {
    println!("Author provided this synopsis");
}
```

## Sentence Splitting

The summarizer handles common abbreviations:

```rust
// These are NOT split:
// "Dr. Smith went to the store."  → 1 sentence
// "I.e., this is an example."     → 1 sentence
// "The U.S. is large."            → 1 sentence

// These ARE split:
// "Hello. World."                 → 2 sentences
// "What? Really!"                 → 2 sentences
```

### Supported Abbreviations

- Titles: Dr., Mr., Mrs., Ms., Prof., Jr., Sr.
- Latin: etc., e.g., i.e., vs., viz.
- Academic: Ph.D., B.A., M.A.
- Common: No., Fig., St.

## Length Filtering

Sentences outside the length bounds are excluded:

```rust
let config = SummarizerConfig {
    min_sentence_length: 20,   // Skip very short sentences
    max_sentence_length: 500,  // Skip very long sentences
    ..Default::default()
};
```

This helps exclude:
- Fragments ("Yes.")
- Headers ("Chapter 1")
- Overly complex run-on sentences

## Order Preservation

Control whether output matches document order:

```rust
// Preserve original order (default)
let config = SummarizerConfig {
    preserve_order: true,
    ..Default::default()
};
// Output: [S1, S5, S8]  (as they appear in document)

// Sort by relevance
let config = SummarizerConfig {
    preserve_order: false,
    ..Default::default()
};
// Output: [S5, S1, S8]  (highest relevance first)
```

## Integration with RAG

The summarizer integrates with the RAG pipeline:

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

// IndexBuilder uses Summarizer for auto-synopsis
let builder_config = IndexBuilderConfig {
    auto_synopsis: true,  // Generate synopses for documents without explicit ones
    ..Default::default()
};

let builder = IndexBuilder::new(builder_config)?;
let index = builder.build_from_directory("./docs", None)?;

// Documents now have synopses for display in search results
for (meta, score) in index.query(&query_embedding, 5) {
    println!("{}: {}", meta.title.unwrap_or_default(), meta.synopsis);
}
```

## Error Handling

```rust
use libgrammstein::neural::NeuralError;

match summarizer.extractive(document) {
    Ok(sentences) => {
        for s in sentences {
            println!("{}", s.text);
        }
    }
    Err(NeuralError::Inference(msg)) => {
        eprintln!("Embedding failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Performance Tips

### 1. Batch Similar-Length Documents

The embedder is most efficient when processing similar-length texts.

### 2. Adjust Sentence Count

More sentences = more embedding computation:

```rust
let config = SummarizerConfig {
    num_sentences: 2,  // Fewer for speed
    ..Default::default()
};
```

### 3. Pre-filter Documents

Skip very short documents that don't need summarization:

```rust
if document.len() < 500 {
    // Document is short enough to use as-is
    Synopsis::explicit(document)
} else {
    summarizer.create_synopsis(document, None)?
}
```

## See Also

- [Overview](overview.md) - Neural module introduction
- [Embedder](embedder.md) - Embedding generation
- [RAG Builder](../rag/builder.md) - Synopsis in RAG pipeline
- [Document](../rag/document.md) - Document metadata with synopsis
