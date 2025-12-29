# Corpus Formats

This document describes the corpus formats supported by libgrammstein for training language models.

## Overview

libgrammstein supports multiple input formats through the `CorpusReader` trait:

| Format | Reader | Use Case |
|--------|--------|----------|
| Plain text | `PlaintextReader` | Simple text files |
| Wikipedia | `WikipediaReader` | XML dumps |
| Gutenberg | `GutenbergReader` | Project Gutenberg texts |
| HTTP streams | `HttpCorpusReader` | Remote resources |

## Plain Text

### Format

Simple UTF-8 text files with one sentence or paragraph per line:

```text
The quick brown fox jumps over the lazy dog.
Natural language processing enables computers to understand text.
Machine learning models can process language effectively.
```

### Usage

```rust
use libgrammstein::corpus::PlaintextReader;

// From file
let reader = PlaintextReader::from_file("corpus.txt")?;

// From string
let text = "The quick brown fox. Another sentence.";
let reader = PlaintextReader::from_string(text);

// Iterate sentences
for sentence in reader.sentences() {
    println!("{}", sentence);
}
```

### Sentence Segmentation

PlaintextReader performs automatic sentence segmentation:

- Splits on `.`, `!`, `?` followed by whitespace
- Handles abbreviations (Mr., Dr., etc.)
- Respects quotation marks

```rust
let text = "Dr. Smith said \"Hello!\" How are you?";
let reader = PlaintextReader::from_string(text);
// Yields: "Dr. Smith said \"Hello!\"", "How are you?"
```

### Configuration

```rust
let reader = PlaintextReader::builder()
    .file("corpus.txt")?
    .lowercase(true)          // Convert to lowercase
    .normalize_whitespace(true)
    .min_sentence_length(5)   // Skip short sentences
    .max_sentence_length(100) // Skip long sentences
    .build()?;
```

## Wikipedia Dumps

### Format

MediaWiki XML dump format (compressed or uncompressed):

```xml
<mediawiki>
  <page>
    <title>Article Title</title>
    <revision>
      <text>Article content with [[links]] and {{templates}}</text>
    </revision>
  </page>
</mediawiki>
```

### Usage

```rust
use libgrammstein::corpus::WikipediaReader;

// From local dump
let reader = WikipediaReader::from_dump("enwiki-latest-pages-articles.xml.bz2")?;

// From HTTP stream
let reader = WikipediaReader::from_url(
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
)?;

for sentence in reader.sentences() {
    process(&sentence);
}
```

### Processing

WikipediaReader handles:

- Decompression (bzip2, gzip)
- XML parsing
- MediaWiki markup stripping
- Template removal
- Link text extraction

```
Input:  "[[Albert Einstein]] developed the {{theory|special relativity}}"
Output: "Albert Einstein developed the special relativity"
```

### Configuration

```rust
let reader = WikipediaReader::builder()
    .dump("enwiki.xml.bz2")
    .skip_redirects(true)      // Skip redirect pages
    .skip_disambig(true)       // Skip disambiguation pages
    .skip_stubs(true)          // Skip stub articles
    .min_article_length(100)   // Skip short articles
    .namespace_filter(vec![0]) // Only main namespace
    .build()?;
```

## Project Gutenberg

### Format

Plain text books with headers/footers:

```text
*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***

It is a truth universally acknowledged, that a single man in
possession of a good fortune, must be in want of a wife.

*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***
```

### Usage

```rust
use libgrammstein::corpus::GutenbergReader;

// From file
let reader = GutenbergReader::from_file("pg1342.txt")?;

// From URL
let reader = GutenbergReader::from_url(
    "https://www.gutenberg.org/files/1342/1342-0.txt"
)?;

for sentence in reader.sentences() {
    process(&sentence);
}
```

### Processing

GutenbergReader handles:

- Header/footer removal
- Chapter heading detection
- Paragraph preservation
- Encoding normalization (Latin-1, UTF-8)

### Configuration

```rust
let reader = GutenbergReader::builder()
    .file("pg1342.txt")
    .skip_front_matter(true)  // Skip title, TOC
    .skip_back_matter(true)   // Skip license
    .preserve_paragraphs(true)
    .build()?;
```

## HTTP Streaming

### Usage

Stream corpora over HTTP without full download:

```rust
use libgrammstein::corpus::HttpCorpusReader;

let reader = HttpCorpusReader::new(
    "https://example.com/large-corpus.txt.gz",
    PlaintextReader::new(),
)?;

// Streams and decompresses on-the-fly
for sentence in reader.sentences() {
    process(&sentence);
}
```

### Features

- Streaming (no full download required)
- Automatic decompression (gzip, bzip2, xz)
- Connection retry/resume
- Rate limiting support

### Configuration

```rust
let reader = HttpCorpusReader::builder()
    .url("https://example.com/corpus.txt.gz")
    .timeout(Duration::from_secs(30))
    .retry_count(3)
    .rate_limit(1000)  // bytes/sec
    .build()?;
```

## Custom Formats

### Implementing CorpusReader

```rust
use libgrammstein::corpus::CorpusReader;

pub struct JsonCorpusReader {
    path: PathBuf,
}

impl CorpusReader for JsonCorpusReader {
    fn sentences(&self) -> Box<dyn Iterator<Item = String> + '_> {
        let file = File::open(&self.path).unwrap();
        let reader = BufReader::new(file);

        Box::new(reader.lines().filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(String::from)
        }))
    }
}
```

### Chaining Readers

```rust
use libgrammstein::corpus::ChainReader;

let combined = ChainReader::new(vec![
    Box::new(PlaintextReader::from_file("corpus1.txt")?),
    Box::new(WikipediaReader::from_dump("wiki.xml.bz2")?),
    Box::new(GutenbergReader::from_file("book.txt")?),
]);

// Iterates through all sources
for sentence in combined.sentences() {
    process(&sentence);
}
```

### Filtering

```rust
use libgrammstein::corpus::FilteredReader;

let reader = PlaintextReader::from_file("corpus.txt")?;
let filtered = FilteredReader::new(reader)
    .min_words(5)
    .max_words(50)
    .language("en")
    .quality_threshold(0.8);
```

## Format Detection

Automatic format detection from extension:

```rust
use libgrammstein::corpus::auto_reader;

let reader = auto_reader("data/corpus.txt")?;      // PlaintextReader
let reader = auto_reader("data/wiki.xml.bz2")?;    // WikipediaReader
let reader = auto_reader("data/pg1342.txt")?;      // GutenbergReader (by content)
```

## Best Practices

1. **Use streaming for large files**: Avoid loading entire corpus into memory

2. **Filter early**: Apply quality filters during reading, not after

3. **Parallelize reading**: Use multiple readers for independent files

4. **Cache preprocessed**: Save normalized text for retraining

5. **Monitor memory**: Track memory usage for large corpora

## See Also

- [Streaming Implementation](streaming.md) - Streaming details
- [Quality Filtering](../../training/large-corpora.md) - Preprocessing
- [CorpusReader Trait](../../api/traits.md) - API reference
