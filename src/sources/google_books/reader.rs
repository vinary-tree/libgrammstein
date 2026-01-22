//! N-gram readers for Google Books data.
//!
//! Provides both file-based and HTTP streaming readers for Google Books n-gram files.
//! Both readers decompress gzip on-the-fly and yield parsed `NgramRecord`s.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;

use super::aggregator::AggregatedNgram;
use super::parser::{parse_ngram_line, NgramRecord, ParseError};
#[cfg(feature = "google-books")]
use super::task_manager::RetryAfter;

/// Trait for n-gram readers.
pub trait NgramReader: Iterator<Item = Result<NgramRecord, ReaderError>> {
    /// Get the current byte offset (for checkpointing).
    fn byte_offset(&self) -> u64;

    /// Total bytes (if known).
    fn total_bytes(&self) -> Option<u64>;
}

/// Errors that can occur during n-gram reading.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse error.
    #[error("Parse error at line {line}: {error}")]
    Parse {
        line: u64,
        #[source]
        error: ParseError,
    },

    /// HTTP error.
    #[error("HTTP error: {0}")]
    Http(String),

    /// Decompression error.
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// Rate limited (HTTP 429).
    ///
    /// This error variant captures the URL and optional Retry-After header value
    /// from HTTP 429 responses. The TaskManager can use this to schedule retries
    /// with proper backoff timing.
    #[cfg(feature = "google-books")]
    #[error("Rate limited (HTTP 429) for {url}")]
    RateLimited {
        /// URL that was rate limited.
        url: String,
        /// Parsed Retry-After header value (if present).
        retry_after: Option<RetryAfter>,
    },
}

/// Saves a failed HTTP response (headers + body) to disk for post-mortem analysis.
/// Returns the path where the response was saved.
fn save_failed_response(
    url: &str,
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &[u8],
) -> Option<std::path::PathBuf> {
    use std::io::Write;

    // Determine output directory
    let dir = std::env::temp_dir().join("grammstein-failed-responses");

    if std::fs::create_dir_all(&dir).is_err() {
        tracing::warn!("Failed to create directory for failed responses: {}", dir.display());
        return None;
    }

    // Generate unique filename from URL
    let filename: String = url
        .replace("://", "_")
        .replace('/', "_")
        .replace('?', "_")
        .chars()
        .take(100)
        .collect();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("{}_{}.response", filename, timestamp));

    // Write headers and body
    let file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("Failed to create response dump file: {}", e);
            return None;
        }
    };
    let mut writer = std::io::BufWriter::new(file);

    // Write URL and HTTP status line
    writeln!(writer, "URL: {}", url).ok();
    writeln!(writer, "HTTP/1.1 {}", status).ok();

    // Write headers
    for (name, value) in headers.iter() {
        if let Ok(v) = value.to_str() {
            writeln!(writer, "{}: {}", name, v).ok();
        } else {
            writeln!(writer, "{}: <binary>", name).ok();
        }
    }
    writeln!(writer).ok();

    // Write body
    if let Err(e) = writer.write_all(body) {
        tracing::warn!("Failed to write response body: {}", e);
        return None;
    }

    tracing::info!("Saved failed response to: {}", path.display());
    Some(path)
}

/// File-based n-gram reader for local gzip files.
///
/// Uses `flate2` for gzip decompression with buffered I/O.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::sources::google_books::FileNgramReader;
///
/// let reader = FileNgramReader::open("googlebooks-eng-all-1gram-20200217-a.gz")?;
/// for result in reader {
///     let record = result?;
///     println!("{}: {}", record.ngram, record.match_count);
/// }
/// ```
pub struct FileNgramReader {
    /// Buffered reader over gzip decoder.
    reader: BufReader<GzDecoder<File>>,

    /// Line buffer for reuse.
    line_buffer: String,

    /// Current line number (1-indexed).
    current_line: u64,

    /// Compressed bytes read (for progress tracking).
    compressed_bytes_read: u64,

    /// Total compressed size (if known).
    total_compressed_size: Option<u64>,

    /// Source file path.
    path: PathBuf,

    /// Whether to skip POS-tagged n-grams.
    skip_pos_tags: bool,

    /// Minimum count filter.
    min_count: u64,
}

impl FileNgramReader {
    /// Open a gzip-compressed n-gram file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ReaderError> {
        Self::open_with_options(path, false, 0)
    }

    /// Open with filtering options.
    pub fn open_with_options<P: AsRef<Path>>(
        path: P,
        skip_pos_tags: bool,
        min_count: u64,
    ) -> Result<Self, ReaderError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let total_compressed_size = file.metadata().map(|m| m.len()).ok();

        let decoder = GzDecoder::new(file);
        let reader = BufReader::with_capacity(64 * 1024, decoder);

        Ok(Self {
            reader,
            line_buffer: String::with_capacity(256),
            current_line: 0,
            compressed_bytes_read: 0,
            total_compressed_size,
            path: path.to_path_buf(),
            skip_pos_tags,
            min_count,
        })
    }

    /// Open multiple files and chain them together.
    pub fn open_all<P: AsRef<Path>>(
        paths: &[P],
        skip_pos_tags: bool,
        min_count: u64,
    ) -> Result<MultiFileReader, ReaderError> {
        MultiFileReader::new(paths, skip_pos_tags, min_count)
    }

    /// Get the source file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read next valid record, skipping filtered lines.
    fn next_valid_record(&mut self) -> Option<Result<NgramRecord, ReaderError>> {
        loop {
            self.line_buffer.clear();

            match self.reader.read_line(&mut self.line_buffer) {
                Ok(0) => return None, // EOF
                Ok(_) => {
                    self.current_line += 1;

                    // Trim trailing newline
                    let line = self.line_buffer.trim_end();

                    // Skip empty lines
                    if line.is_empty() {
                        continue;
                    }

                    // Parse the line
                    match parse_ngram_line(line) {
                        Ok(record) => {
                            // Apply filters
                            if record.match_count < self.min_count {
                                continue;
                            }

                            if self.skip_pos_tags && super::parser::contains_pos_tag(&record.ngram)
                            {
                                continue;
                            }

                            return Some(Ok(record));
                        }
                        Err(e) => {
                            return Some(Err(ReaderError::Parse {
                                line: self.current_line,
                                error: e,
                            }));
                        }
                    }
                }
                Err(e) => return Some(Err(ReaderError::Io(e))),
            }
        }
    }
}

impl Iterator for FileNgramReader {
    type Item = Result<NgramRecord, ReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_valid_record()
    }
}

impl NgramReader for FileNgramReader {
    fn byte_offset(&self) -> u64 {
        // For gzip streams, we track line number instead of byte offset
        // since seeking within gzip is not practical
        self.current_line
    }

    fn total_bytes(&self) -> Option<u64> {
        self.total_compressed_size
    }
}

/// Multi-file reader that chains multiple n-gram files.
pub struct MultiFileReader {
    /// Paths to read.
    paths: Vec<PathBuf>,

    /// Current file index.
    current_index: usize,

    /// Current reader.
    current_reader: Option<FileNgramReader>,

    /// Filtering options.
    skip_pos_tags: bool,
    min_count: u64,

    /// Total lines read across all files.
    total_lines: u64,
}

impl MultiFileReader {
    /// Create a new multi-file reader.
    pub fn new<P: AsRef<Path>>(
        paths: &[P],
        skip_pos_tags: bool,
        min_count: u64,
    ) -> Result<Self, ReaderError> {
        let paths: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();

        let mut reader = Self {
            paths,
            current_index: 0,
            current_reader: None,
            skip_pos_tags,
            min_count,
            total_lines: 0,
        };

        // Open first file
        reader.open_next_file()?;

        Ok(reader)
    }

    /// Open the next file in the sequence.
    fn open_next_file(&mut self) -> Result<bool, ReaderError> {
        if self.current_index >= self.paths.len() {
            self.current_reader = None;
            return Ok(false);
        }

        let path = &self.paths[self.current_index];
        self.current_reader = Some(FileNgramReader::open_with_options(
            path,
            self.skip_pos_tags,
            self.min_count,
        )?);
        self.current_index += 1;

        Ok(true)
    }

    /// Get the current file being read.
    pub fn current_file(&self) -> Option<&Path> {
        self.current_reader.as_ref().map(|r| r.path())
    }

    /// Get the number of files remaining.
    pub fn files_remaining(&self) -> usize {
        self.paths.len().saturating_sub(self.current_index)
    }
}

impl Iterator for MultiFileReader {
    type Item = Result<NgramRecord, ReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut reader) = self.current_reader {
                match reader.next() {
                    Some(result) => {
                        self.total_lines += 1;
                        return Some(result);
                    }
                    None => {
                        // Current file exhausted, try next
                        match self.open_next_file() {
                            Ok(true) => continue,
                            Ok(false) => return None,
                            Err(e) => return Some(Err(e)),
                        }
                    }
                }
            } else {
                return None;
            }
        }
    }
}

impl NgramReader for MultiFileReader {
    fn byte_offset(&self) -> u64 {
        self.total_lines
    }

    fn total_bytes(&self) -> Option<u64> {
        None // Not meaningful for multi-file
    }
}

/// HTTP streaming n-gram reader.
///
/// Streams n-grams directly from Google's servers with async gzip decompression.
/// Uses `reqwest` for HTTP and `async_compression` for streaming decompression.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::sources::google_books::{HttpNgramReader, languages::get_file_url};
///
/// let url = get_file_url("en", 1, "a").unwrap();
/// let mut reader = HttpNgramReader::new(&url).await?;
///
/// while let Some(result) = reader.next().await {
///     let record = result?;
///     println!("{}: {}", record.ngram, record.match_count);
/// }
/// ```
pub struct HttpNgramReader {
    /// Source URL.
    url: String,

    /// Lines read.
    lines_read: u64,

    /// Content length (if known).
    content_length: Option<u64>,

    /// Skip POS tags.
    skip_pos_tags: bool,

    /// Minimum count.
    min_count: u64,

    /// Internal state for async iteration.
    _state: HttpReaderState,
}

/// Internal state for HTTP reader (placeholder for async implementation).
enum HttpReaderState {
    /// Not yet started.
    Pending,
    /// Currently reading.
    Reading,
    /// Finished.
    Done,
}

impl HttpNgramReader {
    /// Create a new HTTP reader for the given URL.
    ///
    /// This creates the reader but does not start the HTTP request.
    /// Call `read_all()` or iterate to begin streaming.
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            lines_read: 0,
            content_length: None,
            skip_pos_tags: false,
            min_count: 0,
            _state: HttpReaderState::Pending,
        }
    }

    /// Create with filtering options.
    pub fn with_options(url: &str, skip_pos_tags: bool, min_count: u64) -> Self {
        Self {
            url: url.to_string(),
            lines_read: 0,
            content_length: None,
            skip_pos_tags,
            min_count,
            _state: HttpReaderState::Pending,
        }
    }

    /// Get the source URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Stream all records asynchronously.
    ///
    /// Returns a stream of `NgramRecord` results that can be processed
    /// with async iterators or collected.
    #[cfg(feature = "google-books")]
    pub async fn stream_records(
        &mut self,
    ) -> Result<impl tokio_stream::Stream<Item = Result<NgramRecord, ReaderError>> + '_, ReaderError>
    {
        use async_compression::tokio::bufread::GzipDecoder;
        use std::time::Duration;
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio_stream::StreamExt;
        use tokio_util::io::StreamReader;

        // Create HTTP client with timeouts for long-running imports
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute total timeout
            .connect_timeout(Duration::from_secs(30)) // 30 second connection timeout
            .build()
            .map_err(|e| ReaderError::Http(format!("Failed to build HTTP client: {}", e)))?;

        // Make HTTP request
        let response = client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| ReaderError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ReaderError::Http(format!(
                "HTTP {} for {}",
                response.status(),
                self.url
            )));
        }

        // Get content length if available
        self.content_length = response.content_length();

        // Convert response body to stream
        let byte_stream = response.bytes_stream();

        // Map stream items to io::Result for StreamReader compatibility
        let mapped_stream = byte_stream.map(|result| {
            result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        });

        // Create async reader from stream
        let stream_reader = StreamReader::new(mapped_stream);

        // Wrap in gzip decoder
        let decoder = GzipDecoder::new(BufReader::new(stream_reader));
        let buf_reader = BufReader::new(decoder);

        // Create line stream
        let lines = tokio_stream::wrappers::LinesStream::new(buf_reader.lines());

        // Parse lines into records
        let skip_pos = self.skip_pos_tags;
        let min_count = self.min_count;
        let mut line_num = 0u64;

        let record_stream = lines.filter_map(move |line_result| {
            line_num += 1;

            match line_result {
                Ok(line) => {
                    if line.is_empty() {
                        return None;
                    }

                    match parse_ngram_line(&line) {
                        Ok(record) => {
                            // Apply filters
                            if record.match_count < min_count {
                                return None;
                            }
                            if skip_pos && super::parser::contains_pos_tag(&record.ngram) {
                                return None;
                            }
                            Some(Ok(record))
                        }
                        Err(e) => Some(Err(ReaderError::Parse {
                            line: line_num,
                            error: e,
                        })),
                    }
                }
                Err(e) => Some(Err(ReaderError::Io(e))),
            }
        });

        // Note: State is tracked implicitly by the stream's lifecycle
        Ok(record_stream)
    }

    /// Read all records into a vector (blocking, for simpler use cases).
    ///
    /// **DEPRECATED**: This method buffers the entire file into memory.
    /// For large 2-gram files (50-100M n-grams), this can consume 6-8 GB per file.
    /// Use [`stream_records()`](Self::stream_records) or [`stream_aggregated()`](Self::stream_aggregated)
    /// for memory-efficient processing.
    ///
    /// # Memory Safety
    ///
    /// This method enforces a maximum of 10 million records to prevent OOM.
    /// For larger files, use streaming methods instead.
    #[cfg(feature = "google-books")]
    #[deprecated(
        since = "0.2.0",
        note = "This method can cause OOM for large files. Use stream_records() or stream_aggregated() instead."
    )]
    pub async fn read_all(&mut self) -> Result<Vec<NgramRecord>, ReaderError> {
        use tokio_stream::StreamExt;

        const MAX_RECORDS: usize = 10_000_000;

        // Clone URL before creating stream to avoid borrow conflict
        let url = self.url.clone();
        let stream = self.stream_records().await?;
        tokio::pin!(stream);

        let mut records = Vec::new();
        while let Some(result) = stream.next().await {
            records.push(result?);
            if records.len() >= MAX_RECORDS {
                log::warn!(
                    "read_all() hit {} record limit for {}. Use stream_records() for larger files.",
                    MAX_RECORDS,
                    url
                );
                break;
            }
        }

        Ok(records)
    }

    /// Read and aggregate records by year.
    ///
    /// **DEPRECATED**: This method buffers the entire file into memory. For large
    /// 2-gram files (50-100M n-grams), this can consume 6-8 GB per file.
    /// Use [`stream_aggregated()`](Self::stream_aggregated) for memory-efficient processing.
    ///
    /// # Memory Safety
    ///
    /// This method enforces a maximum of 10 million aggregated records to prevent OOM.
    #[cfg(feature = "google-books")]
    #[deprecated(
        since = "0.2.0",
        note = "This method can cause OOM for large files. Use stream_aggregated() instead."
    )]
    pub async fn read_aggregated(
        &mut self,
        year_range: Option<(u16, u16)>,
    ) -> Result<Vec<AggregatedNgram>, ReaderError> {
        use super::aggregator::YearAggregator;
        use tokio_stream::StreamExt;

        const MAX_AGGREGATED: usize = 10_000_000;

        // Clone URL before creating stream to avoid borrow conflict
        let url = self.url.clone();
        let stream = self.stream_records().await?;
        tokio::pin!(stream);

        let mut aggregator = YearAggregator::new(year_range);
        let mut results = Vec::new();

        while let Some(result) = stream.next().await {
            let record = result?;
            if let Some(aggregated) = aggregator.push(record) {
                results.push(aggregated);
                if results.len() >= MAX_AGGREGATED {
                    log::warn!(
                        "read_aggregated() hit {} record limit for {}. Use stream_aggregated() for larger files.",
                        MAX_AGGREGATED,
                        url
                    );
                    return Ok(results);
                }
            }
        }

        // Flush final n-gram
        if let Some(aggregated) = aggregator.flush() {
            results.push(aggregated);
        }

        Ok(results)
    }

    /// Stream aggregated records by year (memory-efficient for large files).
    ///
    /// Unlike `read_aggregated()`, this yields n-grams one at a time instead
    /// of buffering the entire file. Essential for large 2-gram files that can
    /// contain 50-100M aggregated n-grams (6-8GB in memory).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tokio_stream::StreamExt;
    ///
    /// let mut reader = HttpNgramReader::new(&url);
    /// let stream = reader.stream_aggregated(Some((2000, 2020)));
    /// tokio::pin!(stream);
    ///
    /// while let Some(result) = stream.next().await {
    ///     let aggregated = result?;
    ///     println!("{}: {}", aggregated.ngram, aggregated.total_count);
    /// }
    /// ```
    #[cfg(feature = "google-books")]
    pub fn stream_aggregated(
        &mut self,
        year_range: Option<(u16, u16)>,
    ) -> impl tokio_stream::Stream<Item = Result<AggregatedNgram, ReaderError>> + '_ {
        use super::aggregator::YearAggregator;

        let skip_pos = self.skip_pos_tags;
        let min_count = self.min_count;
        let url = self.url.clone();

        async_stream::try_stream! {
            use async_compression::tokio::bufread::GzipDecoder;
            use std::time::Duration;
            use tokio::io::{AsyncBufReadExt, BufReader};
            use tokio_stream::StreamExt;
            use tokio_util::io::StreamReader;

            // Create HTTP client with timeouts for long-running imports
            // - timeout: Total request timeout (5 min for large files)
            // - connect_timeout: Time to establish connection (30s)
            // - read_timeout: Time allowed without receiving data (60s)
            //   Prevents hanging on slow/stalled connections that trickle data
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(60))
                .user_agent("Mozilla/5.0 (compatible; libgrammstein/0.1; +https://github.com/vinary-tree/libgrammstein)")
                .build()
                .map_err(|e| ReaderError::Http(format!("Failed to build HTTP client: {}", e)))?;

            // Make HTTP request
            let response = client
                .get(&url)
                .send()
                .await
                .map_err(|e| ReaderError::Http(e.to_string()))?;

            let status = response.status();
            if !status.is_success() {
                // Check for rate limiting
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    // Parse Retry-After header if present
                    let retry_after = response.headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(RetryAfter::parse);

                    tracing::warn!(
                        "Rate limited (429) for {}, Retry-After: {:?}",
                        url,
                        retry_after
                    );

                    Err(ReaderError::RateLimited {
                        url: url.clone(),
                        retry_after,
                    })?;
                }
                if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                    Err(ReaderError::Http(format!(
                        "Service unavailable (HTTP 503) for {} - Google may be throttling",
                        url
                    )))?;
                }
                Err(ReaderError::Http(format!("HTTP {} for {}", status, url)))?;
            }

            // Log Content-Length for debugging large file issues
            if let Some(content_length) = response.content_length() {
                tracing::debug!("Downloading {} ({} bytes compressed)", url, content_length);
            }

            // Extract response metadata before consuming body
            let status = response.status();
            let headers_clone = response.headers().clone();
            let content_length = response.content_length();

            let content_type = headers_clone
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_lowercase());
            let content_encoding = headers_clone
                .get("content-encoding")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_lowercase());

            tracing::debug!(
                "Response for {}: Content-Type={:?}, Content-Encoding={:?}, Length={:?}",
                url, content_type, content_encoding, content_length
            );

            // Determine if this is an error response that should be saved
            let is_html = content_type.as_ref().map(|t| t.contains("text/html")).unwrap_or(false);
            let is_gzip = content_encoding.as_ref().map(|e| e.contains("gzip")).unwrap_or(false)
                || url.ends_with(".gz");
            let is_plain_text = !is_gzip && content_type.as_ref().map(|t| t.starts_with("text/")).unwrap_or(false);

            // Handle error responses: consume body, save to disk, fail with clear message
            if is_html || is_plain_text {
                let bytes = response.bytes().await.map_err(|e| ReaderError::Http(e.to_string()))?;

                // Save response for post-mortem analysis
                let saved_path = save_failed_response(&url, status, &headers_clone, &bytes);

                let preview = String::from_utf8_lossy(&bytes[..bytes.len().min(500)]);
                let preview_short = &preview[..preview.len().min(200)];

                if is_html {
                    tracing::error!(
                        "HTML error page received for {}. Saved to: {:?}. Preview: {}",
                        url, saved_path, preview_short
                    );
                    Err(ReaderError::Http(format!(
                        "Server returned HTML error page for {} (saved to {:?})",
                        url, saved_path
                    )))?;
                } else {
                    tracing::error!(
                        "Plain text received for {}. Saved to: {:?}. Preview: {}",
                        url, saved_path, preview_short
                    );
                    Err(ReaderError::Decompression(format!(
                        "Server returned plain text for {} (saved to {:?})",
                        url, saved_path
                    )))?;
                }
            } else {
                // Normal case: stream the gzip-compressed body
                let byte_stream = response.bytes_stream();

                // Map stream items to io::Result for StreamReader compatibility
                let mapped_stream = byte_stream.map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                });

                // Create async reader from stream
                let stream_reader = StreamReader::new(mapped_stream);

                // Wrap in gzip decoder
                let decoder = GzipDecoder::new(BufReader::new(stream_reader));
                let buf_reader = BufReader::new(decoder);

                // Create line stream
                let lines = tokio_stream::wrappers::LinesStream::new(buf_reader.lines());
                tokio::pin!(lines);

                let mut aggregator = YearAggregator::new(year_range);
                let mut line_num = 0u64;

                while let Some(line_result) = lines.next().await {
                    line_num += 1;
                    let line = line_result?;

                    if line.is_empty() {
                        continue;
                    }

                    match super::parser::parse_ngram_line(&line) {
                        Ok(record) => {
                        // Apply filters
                        if record.match_count < min_count {
                            continue;
                        }
                        if skip_pos && super::parser::contains_pos_tag(&record.ngram) {
                            continue;
                        }
                        if let Some(aggregated) = aggregator.push(record) {
                            yield aggregated;
                        }
                    }
                    Err(e) => {
                        Err(ReaderError::Parse { line: line_num, error: e })?;
                    }
                }
            }

                // Flush final n-gram
                if let Some(aggregated) = aggregator.flush() {
                    yield aggregated;
                }
            }
        }
    }
}

/// Builder for creating n-gram readers with common options.
pub struct ReaderBuilder {
    skip_pos_tags: bool,
    min_count: u64,
    year_range: Option<(u16, u16)>,
}

impl ReaderBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self {
            skip_pos_tags: false,
            min_count: 0,
            year_range: None,
        }
    }

    /// Skip n-grams containing POS tags (e.g., "word_NOUN").
    pub fn skip_pos_tags(mut self, skip: bool) -> Self {
        self.skip_pos_tags = skip;
        self
    }

    /// Filter n-grams below minimum count.
    pub fn min_count(mut self, count: u64) -> Self {
        self.min_count = count;
        self
    }

    /// Filter by year range (inclusive).
    pub fn year_range(mut self, start: u16, end: u16) -> Self {
        self.year_range = Some((start, end));
        self
    }

    /// Open a local file.
    pub fn open_file<P: AsRef<Path>>(self, path: P) -> Result<FileNgramReader, ReaderError> {
        FileNgramReader::open_with_options(path, self.skip_pos_tags, self.min_count)
    }

    /// Open multiple local files.
    pub fn open_files<P: AsRef<Path>>(self, paths: &[P]) -> Result<MultiFileReader, ReaderError> {
        MultiFileReader::new(paths, self.skip_pos_tags, self.min_count)
    }

    /// Create an HTTP reader.
    pub fn http_reader(self, url: &str) -> HttpNgramReader {
        HttpNgramReader::with_options(url, self.skip_pos_tags, self.min_count)
    }
}

impl Default for ReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for wrapping readers with year aggregation.
pub trait AggregateReaderExt: Iterator<Item = Result<NgramRecord, ReaderError>> + Sized {
    /// Aggregate n-gram records by summing across years.
    ///
    /// This filters out errors and aggregates valid records.
    fn aggregate(self, year_range: Option<(u16, u16)>) -> AggregatingReaderIterator<Self> {
        AggregatingReaderIterator::new(self, year_range)
    }
}

impl<I: Iterator<Item = Result<NgramRecord, ReaderError>>> AggregateReaderExt for I {}

/// Iterator adapter that aggregates results from a fallible reader.
pub struct AggregatingReaderIterator<I> {
    inner: I,
    aggregator: super::aggregator::YearAggregator,
    flushed: bool,
    errors: Vec<ReaderError>,
}

impl<I> AggregatingReaderIterator<I>
where
    I: Iterator<Item = Result<NgramRecord, ReaderError>>,
{
    fn new(inner: I, year_range: Option<(u16, u16)>) -> Self {
        Self {
            inner,
            aggregator: super::aggregator::YearAggregator::new(year_range),
            flushed: false,
            errors: Vec::new(),
        }
    }

    /// Get any errors that occurred during aggregation.
    pub fn errors(&self) -> &[ReaderError] {
        &self.errors
    }

    /// Take ownership of collected errors.
    pub fn take_errors(&mut self) -> Vec<ReaderError> {
        std::mem::take(&mut self.errors)
    }
}

impl<I> Iterator for AggregatingReaderIterator<I>
where
    I: Iterator<Item = Result<NgramRecord, ReaderError>>,
{
    type Item = AggregatedNgram;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(Ok(record)) => {
                    if let Some(aggregated) = self.aggregator.push(record) {
                        return Some(aggregated);
                    }
                    // Continue to next record
                }
                Some(Err(e)) => {
                    // Collect error and continue
                    self.errors.push(e);
                }
                None => {
                    // Stream exhausted, flush remaining
                    if !self.flushed {
                        self.flushed = true;
                        return self.aggregator.flush();
                    }
                    return None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_gzip_file(content: &str) -> NamedTempFile {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let file = NamedTempFile::new().expect("Failed to create temp file");
        let path = file.path().to_owned();

        {
            let file = File::create(&path).expect("Failed to create file");
            let mut encoder = GzEncoder::new(file, Compression::default());
            encoder
                .write_all(content.as_bytes())
                .expect("Failed to write");
            encoder.finish().expect("Failed to finish compression");
        }

        // Reopen so the tempfile handle points to the written file
        NamedTempFile::new().expect("Failed to create temp file")
    }

    #[test]
    fn test_reader_builder() {
        let builder = ReaderBuilder::new()
            .skip_pos_tags(true)
            .min_count(100)
            .year_range(2000, 2020);

        assert!(builder.skip_pos_tags);
        assert_eq!(builder.min_count, 100);
        assert_eq!(builder.year_range, Some((2000, 2020)));
    }

    #[test]
    fn test_http_reader_creation() {
        let reader = HttpNgramReader::with_options(
            "https://storage.googleapis.com/books/ngrams/books/test.gz",
            true,
            40,
        );

        assert!(reader.url().contains("storage.googleapis.com"));
        assert!(reader.skip_pos_tags);
        assert_eq!(reader.min_count, 40);
    }
}
