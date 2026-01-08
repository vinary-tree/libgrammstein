# RAG Documents

The RAG module provides structures for representing documents with rich metadata.

## Document Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Document Hierarchy                               │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Document (full)                                │  │
│  │                                                                   │  │
│  │  uri: String                                                      │  │
│  │  title: Option<String>                                            │  │
│  │  synopsis: Synopsis                                               │  │
│  │  language: LanguageTag                                            │  │
│  │  embedding: Vec<f32>  ◄─── 768-dim ModernBERT embedding          │  │
│  │  metadata: DocumentMetadata                                       │  │
│  │  topic_ids: Vec<TopicId>                                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                 │                                        │
│                    (drop embedding)                                      │
│                                 ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    DocumentMeta (lightweight)                     │  │
│  │                                                                   │  │
│  │  uri: String                                                      │  │
│  │  title: Option<String>                                            │  │
│  │  synopsis: String                                                 │  │
│  │  synopsis_source: SynopsisSource                                  │  │
│  │  language: LanguageTag                                            │  │
│  │  metadata: DocumentMetadata                                       │  │
│  │  topic_ids: Vec<TopicId>                                         │  │
│  │                                                                   │  │
│  │  (No embedding - stored separately in backend)                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## DocumentId

Documents are identified by a 32-bit integer:

```rust
use libgrammstein::rag::DocumentId;

let id = DocumentId::new(42);
println!("ID: {}", id.0);  // 42
```

IDs are assigned sequentially when documents are added to an index.

## Document

The full document representation including embedding:

```rust
use libgrammstein::rag::Document;

pub struct Document {
    /// Unique identifier
    pub id: DocumentId,

    /// Document URI (file path, URL, etc.)
    pub uri: String,

    /// Optional human-readable title
    pub title: Option<String>,

    /// Document synopsis (explicit or generated)
    pub synopsis: Synopsis,

    /// Document language
    pub language: LanguageTag,

    /// 768-dimensional embedding vector
    pub embedding: Vec<f32>,

    /// Rich metadata
    pub metadata: DocumentMetadata,

    /// Associated topic IDs (from topic extraction)
    pub topic_ids: Vec<TopicId>,
}
```

## DocumentMeta

Lightweight metadata for storage and display (without embedding):

```rust
use libgrammstein::rag::DocumentMeta;

pub struct DocumentMeta {
    pub uri: String,
    pub title: Option<String>,
    pub synopsis: String,
    pub synopsis_source: SynopsisSource,
    pub language: LanguageTag,
    pub metadata: DocumentMetadata,
    pub topic_ids: Vec<TopicId>,
}
```

This is what's stored in the index and returned from queries.

## DocumentBuilder

Fluent API for constructing documents:

```rust
use libgrammstein::rag::{DocumentBuilder, LanguageTag};

let builder = DocumentBuilder::new("file:///docs/intro.md")
    .title("Introduction to Machine Learning")
    .content("Machine learning is a subset of artificial intelligence...")
    .synopsis("Overview of ML concepts and applications")  // Explicit synopsis
    .language(LanguageTag::english_us())
    .metadata_author("John Doe")
    .metadata_source("textbook")
    .metadata_extra("chapter", "1");
```

### Builder Methods

| Method | Description |
|--------|-------------|
| `new(uri)` | Create builder with URI |
| `title(str)` | Set document title |
| `content(str)` | Set document content |
| `synopsis(str)` | Set explicit synopsis |
| `language(tag)` | Set language tag |
| `metadata_author(str)` | Add author |
| `metadata_source(str)` | Set source corpus |
| `metadata_content_type(str)` | Set MIME type |
| `metadata_date(str)` | Set publication date |
| `metadata_extra(key, val)` | Add custom metadata |

## LanguageTag

ISO 639-1 language codes with optional dialect:

```rust
use libgrammstein::rag::LanguageTag;

// Using helpers
let en_us = LanguageTag::english_us();     // "en-US"
let en_uk = LanguageTag::english_uk();     // "en-GB"
let de = LanguageTag::german();            // "de"
let es = LanguageTag::spanish();           // "es"
let fr = LanguageTag::french();            // "fr"

// Custom language
let custom = LanguageTag::new("ja", Some("JP"));  // "ja-JP"

// Parse from string
let parsed = LanguageTag::parse("en-US")?;

// Format to string
let formatted = en_us.to_string();  // "en-US"
```

### LanguageTag Structure

```rust
pub struct LanguageTag {
    /// ISO 639-1 language code (e.g., "en")
    pub language: String,

    /// Optional dialect/region (e.g., "US")
    pub dialect: Option<String>,
}
```

## DocumentMetadata

Rich metadata with builder pattern:

```rust
use libgrammstein::rag::DocumentMetadata;

let metadata = DocumentMetadata::default()
    .with_content_type("text/markdown")
    .with_source("wikipedia")
    .with_date("2024-01-15")
    .with_author("Jane Smith")
    .with_author("John Doe")  // Multiple authors
    .with_extra("category", "science")
    .with_extra("version", "1.0");
```

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `content_type` | `Option<String>` | MIME type (e.g., "text/plain") |
| `source` | `Option<String>` | Source corpus identifier |
| `date` | `Option<String>` | Publication date |
| `authors` | `Vec<String>` | List of authors |
| `extras` | `HashMap<String, String>` | Custom key-value pairs |

### Accessing Metadata

```rust
let meta = DocumentMetadata::default()
    .with_source("arxiv")
    .with_extra("doi", "10.1234/example");

// Access fields
if let Some(source) = &meta.source {
    println!("Source: {}", source);
}

// Access extras
if let Some(doi) = meta.extras.get("doi") {
    println!("DOI: {}", doi);
}
```

## Synopsis and SynopsisSource

Track whether synopsis is author-provided or generated:

```rust
use libgrammstein::neural::{Synopsis, SynopsisSource};

// Explicit synopsis (from metadata)
let explicit = Synopsis::explicit("This document covers ML basics.");

// Generated synopsis (from summarizer)
let generated = Synopsis::generated("Machine learning is a branch of AI...");

// Check source
match synopsis.source {
    SynopsisSource::Explicit => println!("Author provided"),
    SynopsisSource::Generated => println!("Auto-generated"),
}

// Boolean check
if synopsis.is_explicit() {
    println!("Using author's synopsis");
}
```

## Creating Documents from Files

### Single File

```rust
use libgrammstein::rag::DocumentBuilder;

let content = std::fs::read_to_string("./doc.txt")?;
let path = std::path::Path::new("./doc.txt");

let builder = DocumentBuilder::new(format!("file://{}", path.display()))
    .title(path.file_stem().map(|s| s.to_string_lossy().to_string()))
    .content(content);
```

### Directory Scan

The `IndexBuilder` handles this automatically:

```rust
use libgrammstein::rag::{IndexBuilder, IndexBuilderConfig};

let builder = IndexBuilder::new(IndexBuilderConfig::default())?;
let index = builder.build_from_directory("./docs", None)?;
```

Supported file types: `.txt`, `.md`, `.html`

## Document Conversion

### Document to DocumentMeta

```rust
let doc: Document = /* ... */;

// Create metadata (drops embedding)
let meta = DocumentMeta {
    uri: doc.uri.clone(),
    title: doc.title.clone(),
    synopsis: doc.synopsis.text.clone(),
    synopsis_source: doc.synopsis.source,
    language: doc.language.clone(),
    metadata: doc.metadata.clone(),
    topic_ids: doc.topic_ids.clone(),
};
```

## Serialization

`DocumentMeta` and `DocumentMetadata` are serializable:

```rust
use serde_json;

let meta = DocumentMeta { /* ... */ };

// Serialize
let json = serde_json::to_string(&meta)?;

// Deserialize
let loaded: DocumentMeta = serde_json::from_str(&json)?;
```

## Display Helpers

```rust
let meta = DocumentMeta {
    uri: "file:///docs/intro.md".to_string(),
    title: Some("Introduction".to_string()),
    // ...
};

// Display title (falls back to URI if no title)
let display = meta.title.as_deref().unwrap_or(&meta.uri);
println!("{}", display);  // "Introduction"
```

## Best Practices

### 1. Use Meaningful URIs

```rust
// Good: informative URIs
let doc = DocumentBuilder::new("https://example.com/articles/ml-intro")
    .build()?;

// Avoid: opaque URIs
let doc = DocumentBuilder::new("doc-12345")
    .build()?;
```

### 2. Provide Explicit Synopses When Available

```rust
// Check for existing metadata
if let Some(abstract_text) = metadata.get("abstract") {
    builder = builder.synopsis(abstract_text);
}
// Otherwise, summarizer will generate one
```

### 3. Include Language Information

```rust
// Enables language-specific processing
let doc = DocumentBuilder::new(uri)
    .language(LanguageTag::german())
    .build()?;
```

### 4. Use Extras for Custom Metadata

```rust
let doc = DocumentBuilder::new(uri)
    .metadata_extra("department", "engineering")
    .metadata_extra("classification", "internal")
    .build()?;
```

## See Also

- [Overview](overview.md) - RAG module introduction
- [Builder](builder.md) - Index construction with documents
- [Index](index.md) - Document storage and retrieval
- [Summarizer](../neural/summarizer.md) - Synopsis generation
