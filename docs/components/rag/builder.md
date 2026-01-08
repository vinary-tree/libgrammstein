# RAG Index Builder

The `IndexBuilder` constructs RAG indices from document collections with automatic embedding and synopsis generation.

## Builder Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         IndexBuilder                                     │
│                                                                          │
│  Document Sources                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  DocumentBuilder    →    Files (txt, md, html)                  │    │
│  │  (manual)               (directory scan)                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Processing Pipeline                          │    │
│  │                                                                 │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │    │
│  │  │ Read        │ →  │ Summarize   │ →  │ Embed               │ │    │
│  │  │ Content     │    │ (optional)  │    │ (ModernBERT)        │ │    │
│  │  └─────────────┘    └─────────────┘    └─────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Document                                     │    │
│  │                                                                 │    │
│  │  uri, title, synopsis, language, embedding, metadata            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    RagIndex                                     │    │
│  │                                                                 │    │
│  │  Backend (embeddings) + Metadata (HashMap)                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

```rust
use libgrammstein::rag::IndexBuilderConfig;
use libgrammstein::neural::{EmbeddingConfig, SummarizerConfig};

let config = IndexBuilderConfig {
    // Embedding configuration
    embedding_config: EmbeddingConfig::default(),

    // Summarizer configuration (for auto-synopsis)
    summarizer_config: SummarizerConfig::default(),

    // Batch size for parallel processing
    batch_size: 32,

    // Automatically generate synopses for documents without explicit ones
    auto_synopsis: true,
};
```

## Creating a Builder

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

let config = IndexBuilderConfig::default();
let builder = IndexBuilder::new(config)?;
```

## Building from Directory

### Basic Usage

```rust
let index = builder.build_from_directory("./documents", None)?;
println!("Indexed {} documents", index.len());
```

### With Progress Callback

```rust
let index = builder.build_from_directory("./documents", Some(&|current, total| {
    let percent = 100 * current / total;
    println!("[{:3}%] Processing {}/{}", percent, current, total);
}))?;
```

### Supported File Types

| Extension | Content Type |
|-----------|--------------|
| `.txt` | Plain text |
| `.md` | Markdown |
| `.html` | HTML (text extracted) |

## Building from DocumentBuilders

### Manual Document Construction

```rust
use libgrammstein::rag::DocumentBuilder;

let builders = vec![
    DocumentBuilder::new("file:///doc1.md")
        .title("Introduction")
        .content("Machine learning is...")
        .synopsis("Overview of ML concepts"),  // Explicit synopsis

    DocumentBuilder::new("file:///doc2.md")
        .title("Guide")
        .content("This guide covers..."),
        // No explicit synopsis - will be generated

    DocumentBuilder::new("https://example.com/api")
        .title("API Reference")
        .content("API documentation...")
        .metadata_source("website"),
];

// Build index from builders
let index = builder.build_from_builders(builders, None)?;
```

### With Progress

```rust
let index = builder.build_from_builders(builders, Some(&|current, total| {
    println!("Processing {}/{}", current, total);
}))?;
```

## Extending Existing Index

```rust
use libgrammstein::rag::{IndexBuilder, RagIndex, DocumentBuilder};

// Load existing index
let mut index: RagIndex<_> = RagIndex::load("./index")?;
let initial_count = index.len();

// Create new document builders
let new_builders = vec![
    DocumentBuilder::new("file:///new_doc1.md")
        .title("New Document 1")
        .content("New content..."),
    DocumentBuilder::new("file:///new_doc2.md")
        .title("New Document 2")
        .content("More content..."),
];

// Extend index
let added = builder.extend_index(&mut index, new_builders, Some(&|current, total| {
    println!("Adding {}/{}", current, total);
}))?;

println!("Extended: {} → {} documents", initial_count, index.len());

// Save updated index
index.save("./index")?;
```

## Processing Pipeline

### Step 1: Content Reading

```rust
// From directory scan
let files: Vec<_> = std::fs::read_dir("./docs")?
    .filter_map(|e| e.ok())
    .filter(|e| {
        let path = e.path();
        path.is_file() && matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("txt") | Some("md") | Some("html")
        )
    })
    .collect();
```

### Step 2: Synopsis Generation

```rust
// If auto_synopsis is enabled and no explicit synopsis:
// 1. Pass content to Summarizer
// 2. Extract top sentences using MMR
// 3. Join into synopsis string
```

### Step 3: Embedding Generation

```rust
// For each document:
// 1. Encode content with ModernBertEmbedder
// 2. Pool to single vector (default: mean pooling)
// 3. Normalize to unit length
```

### Step 4: Index Storage

```rust
// Add to index:
// 1. Embedding → Backend
// 2. Metadata → HashMap
```

## Parallel Index Builder

For large document collections:

```rust
use libgrammstein::rag::ParallelIndexBuilder;

let parallel_builder = ParallelIndexBuilder::new(config)?;

// Processes documents in parallel using rayon
let index = parallel_builder.build_from_directory("./large_corpus", Some(&|cur, tot| {
    println!("Progress: {}/{}", cur, tot);
}))?;
```

### Thread Safety

The parallel builder uses:
- Shared `Arc<ModernBertModel>` for embedding
- Thread-safe `&self` API for embedder and summarizer
- Parallel iteration with rayon

## Synopsis Handling

### Explicit Synopsis (Preferred)

```rust
let builder = DocumentBuilder::new("file:///doc.md")
    .content("Full document content...")
    .synopsis("Author-provided summary");  // Will be used as-is
```

### Auto-Generated Synopsis

```rust
let builder = DocumentBuilder::new("file:///doc.md")
    .content("Full document content...");
    // No synopsis - builder will generate using Summarizer
```

### Disable Auto-Synopsis

```rust
let config = IndexBuilderConfig {
    auto_synopsis: false,  // Don't generate synopses
    ..Default::default()
};
```

## Metadata from Files

### Title from Filename

```rust
let path = Path::new("./docs/introduction-to-ml.md");

let builder = DocumentBuilder::new(format!("file://{}", path.display()))
    .title(path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default())  // "introduction-to-ml"
    .content(std::fs::read_to_string(path)?);
```

### Extract from Markdown Front Matter

```rust
// If document has YAML front matter:
// ---
// title: "My Document"
// author: "Jane Doe"
// ---
// Content here...

fn parse_frontmatter(content: &str) -> Option<(String, String, String)> {
    if content.starts_with("---") {
        // Parse YAML front matter
        // Return (title, author, remaining_content)
    }
    None
}
```

## Error Handling

```rust
use libgrammstein::rag::RagError;

match builder.build_from_directory("./docs", None) {
    Ok(index) => {
        println!("Built index with {} documents", index.len());
    }
    Err(RagError::Io(e)) => {
        eprintln!("File error: {}", e);
    }
    Err(RagError::EmbeddingError(msg)) => {
        eprintln!("Embedding failed: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Best Practices

### 1. Provide Explicit Synopses When Available

```rust
// Better search results with author synopses
let builder = DocumentBuilder::new(uri)
    .content(content)
    .synopsis(metadata.get("abstract").unwrap_or(&generated_synopsis));
```

### 2. Use Progress Callbacks for Large Collections

```rust
let index = builder.build_from_directory("./large_corpus", Some(&|cur, tot| {
    eprint!("\rProgress: {}/{} ({:.1}%)", cur, tot, 100.0 * cur as f32 / tot as f32);
}))?;
eprintln!();  // New line after progress
```

### 3. Save Checkpoints for Very Large Indices

```rust
const CHECKPOINT_INTERVAL: usize = 10_000;

let builders = collect_document_builders("./huge_corpus")?;
let chunks: Vec<_> = builders.chunks(CHECKPOINT_INTERVAL).collect();

let mut index = RagIndex::new(config);
for (i, chunk) in chunks.iter().enumerate() {
    builder.extend_index(&mut index, chunk.to_vec(), None)?;
    index.save(&format!("./checkpoints/index_{}", i))?;
    println!("Checkpoint {} saved", i);
}
```

### 4. Configure Summarizer for Document Type

```rust
// For academic papers (longer abstracts)
let config = IndexBuilderConfig {
    summarizer_config: SummarizerConfig {
        num_sentences: 5,
        min_sentence_length: 30,
        ..Default::default()
    },
    ..Default::default()
};

// For short articles
let config = IndexBuilderConfig {
    summarizer_config: SummarizerConfig {
        num_sentences: 2,
        ..Default::default()
    },
    ..Default::default()
};
```

## See Also

- [Overview](overview.md) - RAG module introduction
- [Document](document.md) - Document structures
- [Index](index.md) - RagIndex operations
- [Summarizer](../neural/summarizer.md) - Synopsis generation
- [Embedder](../neural/embedder.md) - Document embedding
