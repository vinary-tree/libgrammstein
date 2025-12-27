# Corpus Processing

This document explains how libgrammstein processes training corpora efficiently, handling large datasets through streaming and parallel processing.

## The Challenge: Large Corpora

Training language models requires large amounts of text data:

| Corpus | Size | Notes |
|--------|------|-------|
| Wikipedia (English) | ~15-20GB text | Diverse topics, professionally edited |
| Project Gutenberg | ~10GB | Public domain literature |
| Common Crawl | 100GB+ | Web-scale data |

Loading these entirely into memory is impractical. libgrammstein uses **streaming** to process data incrementally.

## Streaming Architecture

```
Corpus Files                          Training Output
(10GB+ on disk)                       (Models in memory)
      │                                      ▲
      ▼                                      │
┌─────────────────────────────────────────────────────────────────────┐
│                        Corpus Reader                                 │
│                                                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │
│  │ WikipediaReader │   │ GutenbergReader │   │ PlaintextReader │   │
│  │                 │   │                 │   │                 │   │
│  │ - XML streaming │   │ - Directory     │   │ - Single file   │   │
│  │ - bzip2 decomp  │   │   traversal     │   │ - Directory     │   │
│  │ - Article filter│   │ - Book splitting│   │   glob pattern  │   │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘   │
│           │                     │                     │             │
│           └─────────────────────┼─────────────────────┘             │
│                                 ▼                                   │
│                    ┌────────────────────────┐                       │
│                    │  impl Iterator<Item =  │                       │
│                    │       Document>        │                       │
│                    └───────────┬────────────┘                       │
│                                │                                    │
│                                ▼                                    │
│                    ┌────────────────────────┐                       │
│                    │     Tokenizer          │                       │
│                    │  - Sentence splitting  │                       │
│                    │  - Word tokenization   │                       │
│                    │  - Normalization       │                       │
│                    └───────────┬────────────┘                       │
│                                │                                    │
│                                ▼                                    │
│                    ┌────────────────────────┐                       │
│                    │  impl Iterator<Item =  │                       │
│                    │       String>          │◄── sentences()        │
│                    └───────────┬────────────┘                       │
│                                │                                    │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Rayon par_bridge()    │
                    │  - Parallel processing │
                    │  - Work stealing       │
                    └────────────────────────┘
```

## CorpusReader Trait

The core abstraction for corpus access:

```rust
/// Trait for streaming corpus access
pub trait CorpusReader: Send {
    /// Iterate over documents (articles, chapters, books)
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + '_>;

    /// Iterate over sentences across all documents
    fn sentences(&self) -> Box<dyn Iterator<Item = String> + '_>;

    /// Estimate total tokens (for progress tracking)
    fn estimated_tokens(&self) -> Option<usize>;

    /// Get corpus metadata
    fn metadata(&self) -> CorpusMetadata;
}

/// A single document from the corpus
pub struct Document {
    /// Document identifier (filename, article title, etc.)
    pub id: String,

    /// Document content
    pub content: String,

    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Corpus metadata
pub struct CorpusMetadata {
    /// Human-readable name
    pub name: String,

    /// Approximate size in bytes
    pub size_bytes: Option<u64>,

    /// Approximate document count
    pub document_count: Option<usize>,
}
```

## Corpus Formats

### PlaintextReader

For simple text files or directories:

```rust
use libgrammstein::corpus::PlaintextReader;

// Single file
let reader = PlaintextReader::from_file("corpus.txt")?;

// Directory of text files
let reader = PlaintextReader::from_directory("./corpus/")?;

// With glob pattern
let reader = PlaintextReader::from_glob("./corpus/**/*.txt")?;

// Iterate sentences
for sentence in reader.sentences() {
    println!("{}", sentence);
}
```

### WikipediaReader

For Wikipedia XML dumps (compressed):

```rust
use libgrammstein::corpus::WikipediaReader;

// From compressed dump
let reader = WikipediaReader::from_dump("enwiki-latest-pages-articles.xml.bz2")?;

// Filter to main namespace (articles only)
let reader = reader.with_namespace_filter(&[0]);

// Skip redirect pages
let reader = reader.skip_redirects();

// Process articles
for doc in reader.documents() {
    println!("Article: {}", doc.id);
    for sentence in doc.sentences() {
        // Process sentence...
    }
}
```

Implementation details:

```rust
pub struct WikipediaReader {
    path: PathBuf,
    namespace_filter: Option<HashSet<u32>>,
    skip_redirects: bool,
}

impl WikipediaReader {
    pub fn documents(&self) -> impl Iterator<Item = Document> + '_ {
        // Use quick-xml for streaming XML parsing
        // Decompress bzip2 on-the-fly
        // Never load full dump into memory

        let file = File::open(&self.path)?;
        let decoder = bzip2::read::BzDecoder::new(file);
        let reader = quick_xml::Reader::from_reader(BufReader::new(decoder));

        WikipediaDocumentIterator {
            reader,
            namespace_filter: &self.namespace_filter,
            skip_redirects: self.skip_redirects,
            current_title: String::new(),
            current_text: String::new(),
            in_page: false,
        }
    }
}
```

### GutenbergReader

For Project Gutenberg plaintext files:

```rust
use libgrammstein::corpus::GutenbergReader;

// From directory of .txt files
let reader = GutenbergReader::from_directory("./gutenberg/")?;

// Skip header/footer boilerplate
let reader = reader.skip_boilerplate();

// Process books
for doc in reader.documents() {
    println!("Book: {}", doc.id);
}
```

## Tokenization

### Sentence Splitting

```rust
pub struct SentenceTokenizer {
    /// Regex for sentence boundaries
    boundary_pattern: Regex,

    /// Abbreviations that don't end sentences
    abbreviations: HashSet<String>,

    /// Minimum sentence length (characters)
    min_length: usize,
}

impl SentenceTokenizer {
    pub fn split(&self, text: &str) -> impl Iterator<Item = &str> {
        // Split on .!? followed by whitespace and capital letter
        // Handle abbreviations (Dr., Mr., etc.)
        // Handle ellipsis (...)
        // Handle quotes and parentheses
    }
}
```

### Word Tokenization

```rust
pub struct WordTokenizer {
    /// Whether to lowercase all tokens
    lowercase: bool,

    /// Whether to strip punctuation
    strip_punctuation: bool,

    /// Minimum word length
    min_length: usize,

    /// Maximum word length (filter garbage)
    max_length: usize,
}

impl WordTokenizer {
    pub fn tokenize<'a>(&self, sentence: &'a str) -> impl Iterator<Item = &'a str> {
        sentence.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() >= self.min_length && w.len() <= self.max_length)
            .map(|w| if self.lowercase { w.to_lowercase() } else { w.to_string() })
    }
}
```

### Text Normalization

```rust
use unicode_normalization::UnicodeNormalization;

pub struct TextNormalizer {
    /// Unicode normalization form
    normalization: NormalizationForm,

    /// Whether to strip accents
    strip_accents: bool,

    /// Character replacements
    replacements: HashMap<char, char>,
}

impl TextNormalizer {
    pub fn normalize(&self, text: &str) -> String {
        let normalized = match self.normalization {
            NormalizationForm::NFC => text.nfc().collect(),
            NormalizationForm::NFD => text.nfd().collect(),
            NormalizationForm::NFKC => text.nfkc().collect(),
            NormalizationForm::NFKD => text.nfkd().collect(),
        };

        if self.strip_accents {
            self.remove_accents(&normalized)
        } else {
            normalized
        }
    }
}
```

## Parallel Processing

### Rayon Integration

libgrammstein uses Rayon for parallel corpus processing:

```rust
use rayon::prelude::*;

impl<R: CorpusReader> NgramTrainer<R> {
    pub fn train(&self, reader: R) -> Result<NgramModel> {
        // Create thread-safe dictionary
        let dictionary = Arc::new(RwLock::new(DynamicDawgChar::new()));

        // Process sentences in parallel
        reader.sentences()
            .par_bridge()  // Convert iterator to parallel iterator
            .chunks(10_000)  // Process in batches
            .for_each(|batch| {
                // Count N-grams in batch
                let local_counts = self.count_ngrams(&batch);

                // Merge into global dictionary
                let mut dict = dictionary.write().unwrap();
                for (key, count) in local_counts {
                    dict.update_or_insert(&key, NgramEntry::default(), |e| {
                        e.count += count;
                    });
                }
            });

        // Build model from dictionary
        Ok(NgramModel::from_dictionary(dictionary.into_inner().unwrap()))
    }
}
```

### Batch Processing

Process sentences in batches to reduce lock contention:

```rust
/// Batch size for parallel processing
const BATCH_SIZE: usize = 10_000;

pub fn count_ngrams_parallel<R: CorpusReader>(
    reader: R,
    order: usize,
) -> HashMap<String, u64> {
    let global_counts = Arc::new(DashMap::new());

    reader.sentences()
        .par_bridge()
        .chunks(BATCH_SIZE)
        .for_each(|batch| {
            // Count locally first
            let mut local_counts = HashMap::new();

            for sentence in batch {
                let tokens: Vec<_> = sentence.split_whitespace().collect();
                for n in 1..=order {
                    for window in tokens.windows(n) {
                        let key = window.join("|");
                        *local_counts.entry(key).or_insert(0) += 1;
                    }
                }
            }

            // Merge into global (DashMap is lock-free per key)
            for (key, count) in local_counts {
                global_counts.entry(key).or_insert(0).fetch_add(count, Ordering::SeqCst);
            }
        });

    Arc::try_unwrap(global_counts).unwrap().into_iter().collect()
}
```

## Memory-Efficient Processing

### Streaming Without Full Load

The key to handling large corpora is never loading them fully:

```rust
// BAD: Loads entire corpus into memory
let corpus: String = std::fs::read_to_string("corpus.txt")?;
let sentences: Vec<&str> = corpus.split('.').collect();

// GOOD: Streams line by line
let file = File::open("corpus.txt")?;
let reader = BufReader::new(file);
for line in reader.lines() {
    process_sentence(&line?);
}

// BETTER: libgrammstein's streaming
let reader = PlaintextReader::from_file("corpus.txt")?;
for sentence in reader.sentences() {
    process_sentence(&sentence);
}
```

### Pre-allocated Buffers

Reuse buffers to reduce allocations:

```rust
pub struct TrainingBuffers {
    /// Reusable sentence buffer
    sentence_buffer: String,

    /// Reusable token buffer
    tokens: Vec<String>,

    /// Reusable N-gram key buffer
    key_buffer: String,
}

impl TrainingBuffers {
    pub fn with_capacity(max_sentence_len: usize, max_tokens: usize) -> Self {
        Self {
            sentence_buffer: String::with_capacity(max_sentence_len),
            tokens: Vec::with_capacity(max_tokens),
            key_buffer: String::with_capacity(256),
        }
    }

    pub fn build_ngram_key(&mut self, tokens: &[&str]) -> &str {
        self.key_buffer.clear();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                self.key_buffer.push('|');
            }
            self.key_buffer.push_str(token);
        }
        &self.key_buffer
    }
}
```

## Progress Reporting

Track training progress with channels:

```rust
use crossbeam_channel::{bounded, Sender, Receiver};

/// Training progress update
pub struct TrainingProgress {
    pub sentences_processed: u64,
    pub ngrams_counted: u64,
    pub elapsed_secs: f64,
    pub estimated_remaining_secs: Option<f64>,
}

/// Train with progress callback
pub fn train_with_progress<R: CorpusReader>(
    reader: R,
    progress_tx: Sender<TrainingProgress>,
) -> Result<NgramModel> {
    let start = Instant::now();
    let total_estimate = reader.estimated_tokens();
    let mut processed = 0u64;

    reader.sentences()
        .enumerate()
        .par_bridge()
        .for_each(|(i, sentence)| {
            // Process sentence...

            // Report progress every 10K sentences
            if i % 10_000 == 0 {
                let _ = progress_tx.try_send(TrainingProgress {
                    sentences_processed: i as u64,
                    ngrams_counted: processed,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                    estimated_remaining_secs: total_estimate.map(|t| {
                        let rate = i as f64 / start.elapsed().as_secs_f64();
                        (t - i) as f64 / rate
                    }),
                });
            }
        });

    Ok(model)
}
```

## Example: Complete Training Pipeline

```rust
use libgrammstein::prelude::*;
use libgrammstein::corpus::{PlaintextReader, WikipediaReader};

fn main() -> Result<()> {
    // Create combined reader from multiple sources
    let wikipedia = WikipediaReader::from_dump("enwiki.xml.bz2")?
        .with_namespace_filter(&[0])
        .skip_redirects();

    let gutenberg = PlaintextReader::from_directory("./gutenberg/")?;

    // Chain readers
    let combined = ChainedReader::new(vec![
        Box::new(wikipedia),
        Box::new(gutenberg),
    ]);

    // Setup progress reporting
    let (progress_tx, progress_rx) = crossbeam_channel::bounded(100);

    // Spawn progress display thread
    std::thread::spawn(move || {
        while let Ok(progress) = progress_rx.recv() {
            println!(
                "Processed {} sentences, {} N-grams, {:.1}s elapsed",
                progress.sentences_processed,
                progress.ngrams_counted,
                progress.elapsed_secs
            );
        }
    });

    // Train model
    let model = TrainerBuilder::new()
        .order(5)
        .min_count(2)
        .progress(progress_tx)
        .train(&combined)?;

    model.save("language_model.bin")?;

    Ok(())
}
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Sentence iteration | O(1) per sentence | Streaming, no full load |
| Parallel processing | O(C/P) | C = corpus size, P = cores |
| Dictionary merge | O(B × log N) | B = batch size, N = dict size |

### Memory Usage

| Corpus Size | Peak Memory | Notes |
|-------------|-------------|-------|
| 1GB | ~2GB | Dictionary + buffers |
| 10GB | ~10GB | Depends on vocabulary |
| 100GB | ~20GB | With aggressive pruning |

## Next Steps

- [Streaming](streaming.md): Detailed streaming implementation
- [Formats](formats.md): Format-specific details
- [N-gram Training](../../training/ngram-training.md): Training workflow
- [Architecture Overview](../../architecture/overview.md): System design
