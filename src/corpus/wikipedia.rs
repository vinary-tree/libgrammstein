//! Wikipedia XML dump reader with streaming bz2 decompression.
//!
//! This module provides a streaming reader for Wikipedia XML dumps, supporting
//! both local files and HTTP streaming with automatic disk-space-aware switching.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use bzip2::read::BzDecoder;
use quick_xml::events::Event;
use quick_xml::Reader;

use super::{CorpusReader, Document, Normalizer, Tokenizer};

lazy_static::lazy_static! {
    // Wiki markup patterns for stripping
    static ref TEMPLATE: regex::Regex = regex::Regex::new(r"\{\{[^}]*\}\}").expect("valid regex");
    static ref LINK: regex::Regex = regex::Regex::new(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]").expect("valid regex");
    static ref EXTERNAL_LINK: regex::Regex = regex::Regex::new(r"\[https?://[^\s\]]+\s*([^\]]*)\]").expect("valid regex");
    static ref REF: regex::Regex = regex::Regex::new(r"<ref[^>]*>.*?</ref>|<ref[^/]*/>").expect("valid regex");
    static ref TAG: regex::Regex = regex::Regex::new(r"<[^>]+>").expect("valid regex");
    static ref HEADING: regex::Regex = regex::Regex::new(r"={2,}[^=]+={2,}").expect("valid regex");
    static ref BOLD_ITALIC: regex::Regex = regex::Regex::new(r"'{2,5}").expect("valid regex");
    static ref CATEGORY: regex::Regex = regex::Regex::new(r"\[\[Category:[^\]]+\]\]").expect("valid regex");
    static ref FILE: regex::Regex = regex::Regex::new(r"\[\[(?:File|Image):[^\]]+\]\]").expect("valid regex");
}

/// Configuration for Wikipedia corpus reading.
#[derive(Clone, Debug)]
pub struct WikipediaConfig {
    /// Namespaces to include (0 = main articles).
    pub namespace_filter: Vec<i32>,
    /// Skip redirect pages.
    pub skip_redirects: bool,
    /// Maximum articles to read (None = unlimited).
    pub max_articles: Option<usize>,
    /// Minimum article text length (in characters).
    pub min_text_length: usize,
}

impl Default for WikipediaConfig {
    fn default() -> Self {
        Self {
            namespace_filter: vec![0], // Main namespace only
            skip_redirects: true,
            max_articles: None,
            min_text_length: 100, // Skip very short articles
        }
    }
}

/// Loading strategy for remote URLs.
#[cfg(feature = "http-corpus")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStrategy {
    /// Stream directly from HTTP (slower but uses less disk).
    Stream,
    /// Download first, then process (faster but requires disk space).
    Download,
    /// Auto-select based on available disk space.
    Auto,
}

#[cfg(feature = "http-corpus")]
impl Default for LoadStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Streaming Wikipedia XML dump reader with bz2 decompression.
///
/// Supports both local files and HTTP URLs. For HTTP URLs, automatically
/// chooses between streaming and downloading based on available disk space.
pub struct WikipediaReader {
    path: PathBuf,
    config: WikipediaConfig,
    normalizer: Normalizer,
    tokenizer: Tokenizer,
    #[cfg(feature = "http-corpus")]
    load_strategy: LoadStrategy,
}

impl WikipediaReader {
    /// Create a new Wikipedia reader for a local file.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::with_config(path, WikipediaConfig::default())
    }

    /// Create a Wikipedia reader with custom configuration.
    pub fn with_config(path: impl AsRef<Path>, config: WikipediaConfig) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Wikipedia dump not found: {}", path.display()),
            ));
        }
        Ok(Self {
            path,
            config,
            normalizer: Normalizer::default(),
            tokenizer: Tokenizer::new(),
            #[cfg(feature = "http-corpus")]
            load_strategy: LoadStrategy::Auto,
        })
    }

    /// Create a Wikipedia reader from a URL (HTTP streaming).
    #[cfg(feature = "http-corpus")]
    pub fn from_url(url: &str, config: WikipediaConfig) -> std::io::Result<Self> {
        Self::from_url_with_strategy(url, config, LoadStrategy::Auto)
    }

    /// Create a Wikipedia reader from a URL with explicit loading strategy.
    #[cfg(feature = "http-corpus")]
    pub fn from_url_with_strategy(
        url: &str,
        config: WikipediaConfig,
        strategy: LoadStrategy,
    ) -> std::io::Result<Self> {
        // For HTTP URLs, we store the URL in the path and handle it specially
        Ok(Self {
            path: PathBuf::from(url),
            config,
            normalizer: Normalizer::default(),
            tokenizer: Tokenizer::new(),
            load_strategy: strategy,
        })
    }

    /// Check if this reader is for an HTTP URL.
    #[cfg(feature = "http-corpus")]
    fn is_http_url(&self) -> bool {
        let path_str = self.path.to_string_lossy();
        path_str.starts_with("http://") || path_str.starts_with("https://")
    }

    /// Determine the best loading strategy for a URL.
    #[cfg(feature = "http-corpus")]
    fn determine_strategy(&self) -> LoadStrategy {
        if self.load_strategy != LoadStrategy::Auto {
            return self.load_strategy;
        }

        // Try to get remote file size and available disk space
        let url = self.path.to_string_lossy();

        // HEAD request to get Content-Length
        let remote_size = match ureq::head(&url).call() {
            Ok(resp) => resp
                .header("Content-Length")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0),
            Err(_) => 0,
        };

        if remote_size == 0 {
            // Can't determine size, default to streaming
            log::info!("Cannot determine remote file size, using HTTP streaming");
            return LoadStrategy::Stream;
        }

        // Get available disk space
        let available = match fs2::available_space(std::env::temp_dir()) {
            Ok(space) => space,
            Err(_) => {
                log::warn!("Cannot determine available disk space, using HTTP streaming");
                return LoadStrategy::Stream;
            }
        };

        // Require 2x margin for download + extraction overhead
        let required = remote_size * 2;

        if required > (available as f64 * 0.9) as u64 {
            log::info!(
                "Insufficient disk space ({} available, {} required), using HTTP streaming",
                humansize::format_size(available, humansize::BINARY),
                humansize::format_size(required, humansize::BINARY)
            );
            LoadStrategy::Stream
        } else {
            log::info!(
                "Sufficient disk space ({} available), downloading for best performance",
                humansize::format_size(available, humansize::BINARY)
            );
            LoadStrategy::Download
        }
    }

    /// Create a reader for the underlying data source.
    fn create_reader(&self) -> std::io::Result<Box<dyn BufRead + Send>> {
        #[cfg(feature = "http-corpus")]
        if self.is_http_url() {
            return self.create_http_reader();
        }

        // Local file
        let file = File::open(&self.path)?;
        let is_bz2 = self.path.extension().map_or(false, |ext| ext == "bz2");

        if is_bz2 {
            let decoder = BzDecoder::new(file);
            Ok(Box::new(BufReader::with_capacity(64 * 1024, decoder)))
        } else {
            Ok(Box::new(BufReader::with_capacity(64 * 1024, file)))
        }
    }

    /// Create a reader for HTTP URLs.
    #[cfg(feature = "http-corpus")]
    fn create_http_reader(&self) -> std::io::Result<Box<dyn BufRead + Send>> {
        let url = self.path.to_string_lossy();
        let strategy = self.determine_strategy();

        match strategy {
            LoadStrategy::Stream => {
                // Stream directly from HTTP
                let resp = ureq::get(&url)
                    .call()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                let reader = resp.into_reader();
                let is_bz2 = url.ends_with(".bz2");

                if is_bz2 {
                    let decoder = BzDecoder::new(reader);
                    Ok(Box::new(BufReader::with_capacity(64 * 1024, decoder)))
                } else {
                    Ok(Box::new(BufReader::with_capacity(64 * 1024, reader)))
                }
            }
            LoadStrategy::Download | LoadStrategy::Auto => {
                // Download to temp file first
                let resp = ureq::get(&url)
                    .call()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                let mut temp_file = tempfile::NamedTempFile::new()?;
                let mut reader = resp.into_reader();
                std::io::copy(&mut reader, &mut temp_file)?;

                let file = temp_file.reopen()?;
                let is_bz2 = url.ends_with(".bz2");

                if is_bz2 {
                    let decoder = BzDecoder::new(file);
                    Ok(Box::new(BufReader::with_capacity(64 * 1024, decoder)))
                } else {
                    Ok(Box::new(BufReader::with_capacity(64 * 1024, file)))
                }
            }
        }
    }
}

impl CorpusReader for WikipediaReader {
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + Send + '_> {
        match self.create_reader() {
            Ok(reader) => Box::new(WikipediaIterator::new(
                reader,
                self.config.clone(),
                self.normalizer.clone(),
            )),
            Err(e) => {
                log::error!("Failed to open Wikipedia dump: {}", e);
                Box::new(std::iter::empty())
            }
        }
    }

    fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
        let tokenizer = self.tokenizer.clone();
        Box::new(self.documents().flat_map(move |doc| {
            tokenizer.sentences(&doc.content).collect::<Vec<_>>()
        }))
    }
}

/// Iterator over Wikipedia articles from an XML dump.
struct WikipediaIterator<R: BufRead> {
    reader: Reader<R>,
    buf: Vec<u8>,
    config: WikipediaConfig,
    normalizer: Normalizer,
    // State machine fields
    in_page: bool,
    in_title: bool,
    in_text: bool,
    in_ns: bool,
    current_title: String,
    current_text: String,
    current_ns: i32,
    articles_read: usize,
}

impl<R: BufRead + Send> WikipediaIterator<R> {
    fn new(reader: R, config: WikipediaConfig, normalizer: Normalizer) -> Self {
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.config_mut().trim_text(true);

        Self {
            reader: xml_reader,
            buf: Vec::with_capacity(8192),
            config,
            normalizer,
            in_page: false,
            in_title: false,
            in_text: false,
            in_ns: false,
            current_title: String::new(),
            current_text: String::new(),
            current_ns: 0,
            articles_read: 0,
        }
    }
}

impl<R: BufRead + Send> Iterator for WikipediaIterator<R> {
    type Item = Document;

    fn next(&mut self) -> Option<Self::Item> {
        // Check max articles limit
        if let Some(max) = self.config.max_articles {
            if self.articles_read >= max {
                return None;
            }
        }

        loop {
            self.buf.clear();
            match self.reader.read_event_into(&mut self.buf) {
                Ok(Event::Start(e)) => match e.name().as_ref() {
                    b"page" => {
                        self.in_page = true;
                        self.current_title.clear();
                        self.current_text.clear();
                        self.current_ns = 0;
                    }
                    b"title" if self.in_page => self.in_title = true,
                    b"text" if self.in_page => self.in_text = true,
                    b"ns" if self.in_page => self.in_ns = true,
                    _ => {}
                },
                Ok(Event::End(e)) => match e.name().as_ref() {
                    b"page" => {
                        self.in_page = false;

                        // Check namespace filter
                        if !self.config.namespace_filter.contains(&self.current_ns) {
                            continue;
                        }

                        // Skip redirects
                        if self.config.skip_redirects
                            && (self.current_text.starts_with("#REDIRECT")
                                || self.current_text.starts_with("#redirect"))
                        {
                            continue;
                        }

                        // Skip short articles
                        if self.current_text.len() < self.config.min_text_length {
                            continue;
                        }

                        // Strip wiki markup
                        let cleaned = strip_wiki_markup(&self.current_text);
                        let normalized = self.normalizer.normalize(&cleaned);

                        if normalized.len() < self.config.min_text_length {
                            continue;
                        }

                        self.articles_read += 1;

                        return Some(Document {
                            id: Some(self.current_title.clone()),
                            title: Some(self.current_title.clone()),
                            content: normalized,
                            source: None,
                        });
                    }
                    b"title" => self.in_title = false,
                    b"text" => self.in_text = false,
                    b"ns" => self.in_ns = false,
                    _ => {}
                },
                Ok(Event::Text(e)) => {
                    if self.in_title {
                        if let Ok(text) = e.unescape() {
                            self.current_title.push_str(&text);
                        }
                    } else if self.in_text {
                        if let Ok(text) = e.unescape() {
                            self.current_text.push_str(&text);
                        }
                    } else if self.in_ns {
                        if let Ok(text) = e.unescape() {
                            self.current_ns = text.parse().unwrap_or(0);
                        }
                    }
                }
                Ok(Event::CData(e)) => {
                    // Some Wikipedia dumps use CDATA for text content
                    if self.in_text {
                        if let Ok(text) = std::str::from_utf8(&e) {
                            self.current_text.push_str(text);
                        }
                    }
                }
                Ok(Event::Eof) => return None,
                Err(e) => {
                    log::warn!("XML parse error at position {}: {}", self.reader.buffer_position(), e);
                    continue;
                }
                _ => {}
            }
        }
    }
}

/// Strip Wikipedia markup from text.
fn strip_wiki_markup(text: &str) -> String {
    // Remove templates like {{...}}
    let text = TEMPLATE.replace_all(text, "");

    // Extract link text: [[Page|Display]] -> Display, [[Page]] -> Page
    let text = LINK.replace_all(&text, "$1");

    // Extract external link text: [http://... Display] -> Display
    let text = EXTERNAL_LINK.replace_all(&text, "$1");

    // Remove references
    let text = REF.replace_all(&text, "");

    // Remove HTML tags
    let text = TAG.replace_all(&text, "");

    // Remove headings (== Heading ==)
    let text = HEADING.replace_all(&text, " ");

    // Remove bold/italic markers
    let text = BOLD_ITALIC.replace_all(&text, "");

    // Remove category links
    let text = CATEGORY.replace_all(&text, "");

    // Remove file/image links
    let text = FILE.replace_all(&text, "");

    text.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_wiki_markup() {
        let input = "This is '''bold''' and ''italic'' text.";
        let output = strip_wiki_markup(input);
        assert_eq!(output, "This is bold and italic text.");

        let input = "See [[Wikipedia|the free encyclopedia]] for more.";
        let output = strip_wiki_markup(input);
        assert_eq!(output, "See the free encyclopedia for more.");

        let input = "A simple [[link]] here.";
        let output = strip_wiki_markup(input);
        assert_eq!(output, "A simple link here.");
    }

    #[test]
    fn test_config_defaults() {
        let config = WikipediaConfig::default();
        assert_eq!(config.namespace_filter, vec![0]);
        assert!(config.skip_redirects);
        assert_eq!(config.max_articles, None);
        assert_eq!(config.min_text_length, 100);
    }
}
