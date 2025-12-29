# Streaming Implementation

This document describes the streaming architecture used in libgrammstein for processing large corpora efficiently.

## Overview

Streaming enables processing corpora larger than available memory:

```rust
// Process 100GB corpus with ~100MB memory
let reader = WikipediaReader::from_dump("enwiki.xml.bz2")?;

for sentence in reader.sentences() {
    // Each sentence is yielded and released
    trainer.process(&sentence);
}
```

## Architecture

### Iterator-Based Design

All corpus readers implement lazy iteration:

```rust
pub trait CorpusReader {
    /// Returns an iterator over sentences
    fn sentences(&self) -> Box<dyn Iterator<Item = String> + '_>;
}
```

**Key properties**:
- No materialization of full corpus
- Back-pressure from consumer
- Deterministic memory usage

### Pipeline Stages

```
File/HTTP → Decompress → Parse → Filter → Tokenize → Consumer
     │           │          │        │         │
     └─ Buffer   └─ Buffer  └─ Item  └─ Item   └─ Batch
       64KB        64KB       by       by        configurable
                              item     item
```

## HTTP Streaming

### Chunked Transfer

```rust
use libgrammstein::corpus::HttpCorpusReader;

let reader = HttpCorpusReader::new(url)?;

// Streams data in chunks
for sentence in reader.sentences() {
    process(&sentence);
}
```

### Implementation

```rust
pub struct HttpCorpusReader {
    response: Response,
    decoder: Option<Decoder>,
    buffer: Vec<u8>,
}

impl HttpCorpusReader {
    fn read_chunk(&mut self) -> io::Result<Option<Vec<u8>>> {
        // Read from HTTP response
        let mut chunk = vec![0u8; CHUNK_SIZE];
        let n = self.response.read(&mut chunk)?;

        if n == 0 {
            return Ok(None);
        }

        // Decompress if needed
        if let Some(decoder) = &mut self.decoder {
            chunk = decoder.decode(&chunk[..n])?;
        }

        Ok(Some(chunk))
    }
}
```

### Decompression Support

| Format | Extension | Auto-detected |
|--------|-----------|---------------|
| gzip | .gz | Yes |
| bzip2 | .bz2 | Yes |
| xz | .xz | Yes |
| zstd | .zst | Yes |
| raw | none | Yes |

```rust
// Automatic decompression
let reader = HttpCorpusReader::from_url(
    "https://example.com/corpus.txt.gz"
)?;
```

## XML Streaming

### SAX-like Parsing

Wikipedia dumps are parsed without loading full DOM:

```rust
use quick_xml::Reader;

pub struct WikipediaReader {
    reader: Reader<BufReader<File>>,
    current_page: Option<PageBuilder>,
}

impl Iterator for WikipediaSentences<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            match self.reader.read_event(&mut self.buffer) {
                Ok(Event::Start(e)) if e.name() == b"text" => {
                    self.in_text = true;
                }
                Ok(Event::Text(e)) if self.in_text => {
                    let text = e.unescape_and_decode(&self.reader).ok()?;
                    return Some(self.extract_sentence(&text)?);
                }
                Ok(Event::Eof) => return None,
                _ => continue,
            }
        }
    }
}
```

### Memory Profile

For a 20GB compressed Wikipedia dump:

| Stage | Memory Usage |
|-------|--------------|
| HTTP buffer | 64 KB |
| Decompression | 1 MB |
| XML parser | 10 MB |
| Current article | 100 KB |
| **Total** | ~12 MB |

## Parallel Streaming

### Rayon Integration

```rust
use rayon::prelude::*;

// Collect into batches for parallel processing
let batch_size = 10_000;
let mut batch = Vec::with_capacity(batch_size);

for sentence in reader.sentences() {
    batch.push(sentence);

    if batch.len() >= batch_size {
        // Process batch in parallel
        batch.par_iter().for_each(|s| {
            trainer.process(s);
        });
        batch.clear();
    }
}
```

### Sharded Reading

For multiple files:

```rust
use rayon::prelude::*;

let files: Vec<PathBuf> = glob("corpus/*.txt")?.collect();

files.par_iter().for_each(|path| {
    let reader = PlaintextReader::from_file(path).unwrap();
    let trainer = trainer.clone();

    for sentence in reader.sentences() {
        trainer.process(&sentence);
    }
});
```

## Buffering Strategies

### Ring Buffer

For producer-consumer pattern:

```rust
pub struct BufferedReader {
    inner: Box<dyn CorpusReader>,
    buffer: VecDeque<String>,
    buffer_size: usize,
}

impl BufferedReader {
    fn refill_buffer(&mut self) {
        while self.buffer.len() < self.buffer_size {
            match self.inner.sentences().next() {
                Some(s) => self.buffer.push_back(s),
                None => break,
            }
        }
    }
}
```

### Prefetch

```rust
use crossbeam_channel::{bounded, Receiver, Sender};

pub struct PrefetchReader {
    receiver: Receiver<String>,
    _handle: JoinHandle<()>,
}

impl PrefetchReader {
    pub fn new(reader: impl CorpusReader + Send + 'static, buffer: usize) -> Self {
        let (tx, rx) = bounded(buffer);

        let handle = thread::spawn(move || {
            for sentence in reader.sentences() {
                if tx.send(sentence).is_err() {
                    break;
                }
            }
        });

        Self { receiver: rx, _handle: handle }
    }
}
```

## Error Handling

### Graceful Degradation

```rust
impl Iterator for StreamingSentences<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            match self.try_next() {
                Ok(Some(s)) => return Some(s),
                Ok(None) => return None,
                Err(e) => {
                    // Log error and continue
                    eprintln!("Warning: {}", e);
                    self.errors += 1;

                    if self.errors > self.max_errors {
                        return None;
                    }
                    continue;
                }
            }
        }
    }
}
```

### Retry Logic

```rust
impl HttpCorpusReader {
    fn read_with_retry(&mut self) -> io::Result<Option<Vec<u8>>> {
        for attempt in 0..self.max_retries {
            match self.read_chunk() {
                Ok(chunk) => return Ok(chunk),
                Err(e) if e.kind() == ErrorKind::TimedOut => {
                    thread::sleep(Duration::from_secs(1 << attempt));
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(ErrorKind::TimedOut, "Max retries exceeded"))
    }
}
```

## Progress Tracking

### Byte-Based Progress

```rust
pub struct ProgressReader<R> {
    inner: R,
    bytes_read: Arc<AtomicU64>,
    total_bytes: u64,
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

// Check progress
let progress = bytes_read.load(Ordering::Relaxed) as f64 / total_bytes as f64;
println!("Progress: {:.1}%", progress * 100.0);
```

### Sentence Count

```rust
pub struct CountingReader {
    inner: Box<dyn CorpusReader>,
    count: Arc<AtomicU64>,
}

impl Iterator for CountingSentences<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        let result = self.inner.next()?;
        self.count.fetch_add(1, Ordering::Relaxed);
        Some(result)
    }
}
```

## Best Practices

1. **Use streaming for files > 1GB**: Avoid memory issues

2. **Set appropriate buffer sizes**: Balance memory vs. I/O efficiency

3. **Handle errors gracefully**: Don't fail on single bad records

4. **Track progress**: Provide feedback for long operations

5. **Parallelize when possible**: Use batch processing with Rayon

6. **Monitor memory**: Use tools like `heaptrack` for large jobs

## See Also

- [Corpus Formats](formats.md) - Supported formats
- [Large Corpora](../../training/large-corpora.md) - Training at scale
- [Threading Model](../../architecture/threading.md) - Concurrency
