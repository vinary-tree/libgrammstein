//! Corpus utility command implementations.

use std::collections::HashMap;
use std::fs;
use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use console::style;
use rand::prelude::{Rng, SeedableRng, StdRng};
use serde::{Deserialize, Serialize};

use crate::cli::args::{
    CorpusCleanArgs, CorpusCommands, CorpusDetectArgs, CorpusDownloadArgs, CorpusFormat,
    CorpusListArgs, CorpusSampleArgs, CorpusSource, CorpusStatsArgs, OutputFormat,
};
use crate::cli::error::{CliError, CliResult};
use crate::cli::output;
use crate::corpus::{CorpusReader, GutenbergReader, PlaintextReader, Tokenizer, WikipediaReader};
use crate::language::wikipedia_dump_url;

const SAMPLE_DOWNLOAD_BYTES: u64 = 100 * 1024 * 1024;

/// Run the corpus command.
pub fn run(cmd: CorpusCommands, verbose: bool) -> CliResult<()> {
    match cmd {
        CorpusCommands::Stats(args) => corpus_stats(args, verbose),
        CorpusCommands::Sample(args) => corpus_sample(args, verbose),
        CorpusCommands::Download(args) => corpus_download(args, verbose),
        CorpusCommands::Detect(args) => corpus_detect(args, verbose),
        CorpusCommands::List(args) => corpus_list(args, verbose),
        CorpusCommands::Clean(args) => corpus_clean(args, verbose),
    }
}

/// Create a corpus reader based on path and format.
fn create_corpus_reader(path: &str, format: CorpusFormat) -> CliResult<Box<dyn CorpusReader>> {
    let path_obj = Path::new(path);

    match format {
        CorpusFormat::Plaintext => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    PlaintextReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    PlaintextReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Wikipedia => {
            // Check if it's an HTTP URL
            #[cfg(feature = "http-corpus")]
            if path.starts_with("http://") || path.starts_with("https://") {
                return Ok(Box::new(
                    WikipediaReader::from_url(path, crate::corpus::WikipediaConfig::default())
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ));
            }

            // Local file
            if path_obj.exists() {
                Ok(Box::new(
                    WikipediaReader::new(path_obj).map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Gutenberg => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    GutenbergReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    GutenbergReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
    }
}

/// Show corpus statistics.
fn corpus_stats(args: CorpusStatsArgs, verbose: bool) -> CliResult<()> {
    let path = Path::new(&args.corpus);
    let format_str = format!("{:?}", args.format);

    // Determine if directory or file
    let corpus_type = if path.is_dir() {
        format!("{} (directory)", format_str)
    } else if path.exists() {
        format!("{} (file)", format_str)
    } else {
        return Err(CliError::file_not_found(path));
    };

    if verbose {
        eprintln!("Analyzing corpus: {}", args.corpus);
        eprintln!("  Format: {}", corpus_type);
    }

    eprintln!("Loading corpus...");

    // Create corpus reader
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    // Get document count
    let doc_count = reader.document_count();

    eprintln!("Analyzing sentences and tokens...");

    // Count statistics
    let tokenizer = Tokenizer::new();
    let mut total_sentences = 0u64;
    let mut total_tokens = 0u64;
    let mut word_counts: HashMap<String, u64> = HashMap::new();
    let mut tokens_per_doc: Vec<usize> = Vec::new();

    for doc in reader.documents() {
        let mut doc_tokens = 0usize;

        for sentence in tokenizer.sentences(&doc.content) {
            total_sentences += 1;
            for word in tokenizer.words(&sentence) {
                total_tokens += 1;
                doc_tokens += 1;
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        tokens_per_doc.push(doc_tokens);
    }

    let unique_words = word_counts.len();

    // Get top 10 words
    let mut word_vec: Vec<(String, u64)> = word_counts.into_iter().collect();
    word_vec.sort_by_key(|w| std::cmp::Reverse(w.1));
    let top_words: Vec<(String, u64)> = word_vec.into_iter().take(10).collect();

    // Calculate document statistics
    let num_docs = doc_count.unwrap_or(tokens_per_doc.len());
    let (min_tokens, max_tokens, avg_tokens) = if !tokens_per_doc.is_empty() {
        let min = *tokens_per_doc.iter().min().unwrap_or(&0);
        let max = *tokens_per_doc.iter().max().unwrap_or(&0);
        let sum: usize = tokens_per_doc.iter().sum();
        let avg = sum as f64 / tokens_per_doc.len() as f64;
        (min, max, avg)
    } else {
        (0, 0, 0.0)
    };

    // Print statistics
    output::print_corpus_stats(
        &args.corpus,
        &corpus_type,
        num_docs as u64,
        total_sentences,
        total_tokens,
        unique_words as u64,
        &top_words,
    );

    // Print additional statistics
    println!();
    println!("{}", style("Token distribution:").bold());
    println!("  Min tokens/doc:  {}", min_tokens);
    println!("  Max tokens/doc:  {}", max_tokens);
    println!("  Avg tokens/doc:  {:.1}", avg_tokens);

    Ok(())
}

/// Sample sentences from corpus.
fn corpus_sample(args: CorpusSampleArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Sampling from corpus: {}", args.corpus);
        eprintln!("  Count: {}", args.count);
        if let Some(seed) = args.seed {
            eprintln!("  Seed: {}", seed);
        }
    }

    // Create corpus reader
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    // Initialize RNG - use StdRng for seeding capability
    let mut rng = if let Some(seed) = args.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    // Collect all sentences (for small corpora) or use reservoir sampling
    eprintln!("Reading corpus...");

    // Use reservoir sampling to efficiently sample from the stream
    let mut reservoir: Vec<String> = Vec::with_capacity(args.count);
    let mut total_seen = 0u64;

    for sentence in reader.sentences() {
        total_seen += 1;

        if reservoir.len() < args.count {
            // Fill reservoir
            reservoir.push(sentence);
        } else {
            // Reservoir sampling: replace with probability count/total_seen
            let j = rng.gen_range(0..total_seen);
            if (j as usize) < args.count {
                reservoir[j as usize] = sentence;
            }
        }
    }

    if reservoir.is_empty() {
        eprintln!(
            "{}: Corpus contains no sentences",
            style("warning").yellow()
        );
        return Ok(());
    }

    // Print samples
    println!("{}", style("Sample sentences:").bold());
    for (i, sample) in reservoir.iter().enumerate() {
        println!("  {}. {}", i + 1, sample);
    }

    println!();
    println!(
        "{} {} sentences sampled from {} total",
        style("info:").cyan(),
        reservoir.len(),
        total_seen
    );

    Ok(())
}

/// Download corpus for language.
fn corpus_download(args: CorpusDownloadArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Downloading corpus for language: {}", args.language);
        eprintln!("  Source: {:?}", args.source);
        if let Some(ref output) = args.output {
            eprintln!("  Output: {}", output.display());
        }
    }

    let url = corpus_download_url(args.source, &args.language)?;
    let output_dir = args
        .output
        .unwrap_or_else(|| CorpusCache::cache_dir().join("corpora"));
    fs::create_dir_all(&output_dir)
        .map_err(|e| CliError::io(format!("{}: {}", output_dir.display(), e)))?;

    let output_path = corpus_download_path(&output_dir, &url, args.sample)?;
    let bytes_downloaded = download_http_file(&url, &output_path, args.resume, args.sample)?;

    let mut cache = CorpusCache::load()?;
    cache.register(CacheEntry::new(
        args.source,
        output_path.clone(),
        bytes_downloaded,
        Some(args.language.clone()),
    ));
    cache.save()?;

    println!("{}", style("Corpus Download").bold().underlined());
    println!();
    println!("Language: {}", style(&args.language).cyan());
    println!("Source:   {:?}", args.source);
    println!("URL:      {}", style(&url).green());
    println!("Output:   {}", output_path.display());
    println!("Size:     {}", format_bytes(bytes_downloaded));
    println!();
    println!(
        "{} {}",
        style("success:").green().bold(),
        if args.sample {
            "sample corpus cached"
        } else {
            "corpus cached"
        }
    );

    Ok(())
}

fn corpus_download_url(source: CorpusSource, language: &str) -> CliResult<String> {
    match source {
        CorpusSource::Wikipedia => Ok(wikipedia_dump_url(language)),
        CorpusSource::Gutenberg => Err(CliError::unsupported(
            "Project Gutenberg does not publish a stable language-specific bulk text archive; download text files from https://www.gutenberg.org/ and use --format gutenberg",
        )),
        CorpusSource::Oscar => Err(CliError::unsupported(
            "OSCAR distributions are published through dataset hosting workflows; download the desired language split from https://oscar-project.github.io/documentation/ and use --format plaintext",
        )),
    }
}

fn corpus_download_path(output_dir: &Path, url: &str, sample: bool) -> CliResult<PathBuf> {
    let filename = url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| CliError::invalid_argument(format!("URL has no filename: {}", url)))?;

    let filename = if sample {
        format!("{}.sample", filename)
    } else {
        filename.to_string()
    };

    Ok(output_dir.join(filename))
}

fn download_http_file(url: &str, output_path: &Path, resume: bool, sample: bool) -> CliResult<u64> {
    if output_path.exists() {
        let size = fs::metadata(output_path)
            .map_err(|e| CliError::io(format!("{}: {}", output_path.display(), e)))?
            .len();
        return Ok(size);
    }

    let part_path = partial_download_path(output_path);
    let mut start = if resume && part_path.exists() {
        fs::metadata(&part_path)
            .map_err(|e| CliError::io(format!("{}: {}", part_path.display(), e)))?
            .len()
    } else {
        0
    };

    if sample && start >= SAMPLE_DOWNLOAD_BYTES {
        fs::rename(&part_path, output_path).map_err(|e| {
            CliError::io(format!(
                "rename {} -> {}: {}",
                part_path.display(),
                output_path.display(),
                e
            ))
        })?;
        return Ok(SAMPLE_DOWNLOAD_BYTES);
    }

    let response = match request_download(url, start, sample) {
        Ok(response) => response,
        Err(ureq::Error::Status(416, _)) if start > 0 => {
            fs::remove_file(&part_path)
                .map_err(|e| CliError::io(format!("{}: {}", part_path.display(), e)))?;
            start = 0;
            request_download(url, start, sample)
                .map_err(|e| CliError::io(format!("download {}: {}", url, e)))?
        }
        Err(e) => return Err(CliError::io(format!("download {}: {}", url, e))),
    };

    let append = start > 0 && response.status() == 206;
    if start > 0 && !append {
        start = 0;
    }

    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&part_path)
        .map_err(|e| CliError::io(format!("{}: {}", part_path.display(), e)))?;

    let mut reader = response.into_reader();
    let copied = if sample {
        let remaining = SAMPLE_DOWNLOAD_BYTES.saturating_sub(start);
        copy_limited(&mut reader, &mut output, remaining)?
    } else {
        io::copy(&mut reader, &mut output)
            .map_err(|e| CliError::io(format!("write {}: {}", part_path.display(), e)))?
    };
    output
        .flush()
        .map_err(|e| CliError::io(format!("flush {}: {}", part_path.display(), e)))?;

    fs::rename(&part_path, output_path).map_err(|e| {
        CliError::io(format!(
            "rename {} -> {}: {}",
            part_path.display(),
            output_path.display(),
            e
        ))
    })?;

    Ok(start + copied)
}

// `ureq::Error` is a foreign type matched on by variant at the call site (the
// 416-range retry); boxing it would obscure that match to shrink a cold
// error-path return, so the large-Err variant is accepted here.
#[allow(clippy::result_large_err)]
fn request_download(url: &str, start: u64, sample: bool) -> Result<ureq::Response, ureq::Error> {
    let mut request = ureq::get(url);

    if sample {
        let end = SAMPLE_DOWNLOAD_BYTES.saturating_sub(1);
        request = request.set("Range", &format!("bytes={}-{}", start, end));
    } else if start > 0 {
        request = request.set("Range", &format!("bytes={}-", start));
    }

    request.call()
}

fn copy_limited<R: Read, W: Write>(reader: &mut R, writer: &mut W, limit: u64) -> CliResult<u64> {
    let mut limited = reader.take(limit);
    io::copy(&mut limited, writer).map_err(|e| CliError::io(format!("copy response body: {}", e)))
}

fn partial_download_path(path: &Path) -> PathBuf {
    let filename = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "download".into());
    path.with_file_name(format!("{}.part", filename))
}

/// Detect corpus language.
fn corpus_detect(args: CorpusDetectArgs, verbose: bool) -> CliResult<()> {
    use whatlang::detect;

    if verbose {
        eprintln!("Detecting language of corpus: {}", args.corpus);
    }

    // Create corpus reader
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    eprintln!("Sampling text for language detection...");

    // Sample sentences to build detection corpus
    let mut sample_text = String::new();
    let sample_limit = 10000; // Characters to sample
    let mut sentence_count = 0u64;

    for sentence in reader.sentences() {
        sample_text.push_str(&sentence);
        sample_text.push(' ');
        sentence_count += 1;

        if sample_text.len() >= sample_limit {
            break;
        }
    }

    if sample_text.is_empty() {
        return Err(CliError::corpus(
            "Corpus contains no text for language detection".to_string(),
        ));
    }

    // Detect language
    let detection = detect(&sample_text);

    match detection {
        Some(info) => {
            let lang_code = lang_to_code(info.lang());
            let confidence = info.confidence() * 100.0;
            let reliable = info.is_reliable();

            println!(
                "{}",
                style("Language Detection Results").bold().underlined()
            );
            println!();
            println!(
                "Detected language: {} ({})",
                style(lang_code).cyan().bold(),
                lang_name(info.lang())
            );
            println!("Confidence:        {:.1}%", confidence);
            println!(
                "Reliable:          {}",
                if reliable {
                    style("yes").green()
                } else {
                    style("no").yellow()
                }
            );
            println!();
            println!(
                "Sample size:       {} sentences ({} characters)",
                sentence_count,
                sample_text.len()
            );

            if !reliable {
                println!();
                println!(
                    "{}: Detection confidence is low. Consider providing more text.",
                    style("note").yellow()
                );
            }
        }
        None => {
            println!(
                "{}: Could not detect language. Text may be too short or contain mixed languages.",
                style("warning").yellow()
            );
        }
    }

    Ok(())
}

/// Convert whatlang Lang to ISO 639-1 code.
#[allow(unreachable_patterns)]
fn lang_to_code(lang: whatlang::Lang) -> &'static str {
    use whatlang::Lang::*;
    match lang {
        Eng => "en",
        Spa => "es",
        Deu => "de",
        Fra => "fr",
        Por => "pt",
        Ita => "it",
        Nld => "nl",
        Rus => "ru",
        Cmn => "zh",
        Jpn => "ja",
        Kor => "ko",
        Ara => "ar",
        Hin => "hi",
        Pol => "pl",
        Tur => "tr",
        Vie => "vi",
        Ind => "id",
        Tha => "th",
        Swe => "sv",
        Ces => "cs",
        Dan => "da",
        Fin => "fi",
        Ell => "el",
        Heb => "he",
        Hun => "hu",
        Nob => "nb",
        Ron => "ro",
        Slk => "sk",
        Ukr => "uk",
        Bul => "bg",
        Cat => "ca",
        Hrv => "hr",
        Est => "et",
        Lav => "lv",
        Lit => "lt",
        Slv => "sl",
        Epo => "eo",
        Lat => "la",
        _ => "unknown",
    }
}

/// Get human-readable language name.
#[allow(unreachable_patterns)]
fn lang_name(lang: whatlang::Lang) -> &'static str {
    use whatlang::Lang::*;
    match lang {
        Eng => "English",
        Spa => "Spanish",
        Deu => "German",
        Fra => "French",
        Por => "Portuguese",
        Ita => "Italian",
        Nld => "Dutch",
        Rus => "Russian",
        Cmn => "Chinese",
        Jpn => "Japanese",
        Kor => "Korean",
        Ara => "Arabic",
        Hin => "Hindi",
        Pol => "Polish",
        Tur => "Turkish",
        Vie => "Vietnamese",
        Ind => "Indonesian",
        Tha => "Thai",
        Swe => "Swedish",
        Ces => "Czech",
        Dan => "Danish",
        Fin => "Finnish",
        Ell => "Greek",
        Heb => "Hebrew",
        Hun => "Hungarian",
        Nob => "Norwegian Bokmål",
        Ron => "Romanian",
        Slk => "Slovak",
        Ukr => "Ukrainian",
        Bul => "Bulgarian",
        Cat => "Catalan",
        Hrv => "Croatian",
        Est => "Estonian",
        Lav => "Latvian",
        Lit => "Lithuanian",
        Slv => "Slovenian",
        Epo => "Esperanto",
        Lat => "Latin",
        _ => "Unknown",
    }
}

// =============================================================================
// Corpus Cache Management
// =============================================================================

/// Metadata for a cached corpus file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Source type of the corpus.
    pub source: CorpusSource,
    /// Path where the corpus is stored.
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// When the corpus was downloaded.
    pub downloaded_at: DateTime<Utc>,
    /// Whether the corpus was streamed (not stored locally).
    pub was_streamed: bool,
    /// Language code (if known).
    pub language: Option<String>,
    /// Model names trained on this corpus.
    pub used_by_models: Vec<String>,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(
        source: CorpusSource,
        path: PathBuf,
        size_bytes: u64,
        language: Option<String>,
    ) -> Self {
        Self {
            source,
            path,
            size_bytes,
            downloaded_at: Utc::now(),
            was_streamed: false,
            language,
            used_by_models: Vec::new(),
        }
    }

    /// Format size as human-readable string.
    pub fn format_size(&self) -> String {
        format_bytes(self.size_bytes)
    }

    /// Get age in days.
    pub fn age_days(&self) -> i64 {
        (Utc::now() - self.downloaded_at).num_days()
    }
}

/// Cache metadata for downloaded corpus files.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusCache {
    /// Cached corpus entries.
    pub entries: Vec<CacheEntry>,
}

impl CorpusCache {
    /// Get the default cache directory.
    pub fn cache_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("grammstein")
    }

    /// Get the cache metadata file path.
    pub fn cache_file() -> PathBuf {
        Self::cache_dir().join("corpus_cache.json")
    }

    /// Load cache from disk.
    pub fn load() -> CliResult<Self> {
        let path = Self::cache_file();
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)
            .map_err(|e| CliError::io(format!("{}: {}", path.display(), e)))?;

        serde_json::from_str(&content)
            .map_err(|e| CliError::corpus(format!("Failed to parse cache file: {}", e)))
    }

    /// Save cache to disk.
    pub fn save(&self) -> CliResult<()> {
        let path = Self::cache_file();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| CliError::io(format!("{}: {}", parent.display(), e)))?;
        }

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| CliError::corpus(format!("Failed to serialize cache: {}", e)))?;

        fs::write(&path, content)
            .map_err(|e| CliError::io(format!("{}: {}", path.display(), e)))?;

        Ok(())
    }

    /// Register a new cache entry.
    pub fn register(&mut self, entry: CacheEntry) {
        // Remove any existing entry for the same path
        self.entries.retain(|e| e.path != entry.path);
        self.entries.push(entry);
    }

    /// Remove entries matching a filter.
    pub fn remove_matching<F>(&mut self, predicate: F) -> Vec<CacheEntry>
    where
        F: Fn(&CacheEntry) -> bool,
    {
        let mut removed = Vec::new();
        self.entries.retain(|e| {
            if predicate(e) {
                removed.push(e.clone());
                false
            } else {
                true
            }
        });
        removed
    }

    /// Total cached size in bytes.
    pub fn total_size(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Number of cached entries.
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// List cached corpus files.
fn corpus_list(args: CorpusListArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Loading corpus cache...");
    }

    let cache = CorpusCache::load()?;

    if cache.entries.is_empty() {
        println!("{}", style("No cached corpora found.").dim());
        println!();
        println!("Download corpora with:");
        println!("  grammstein corpus download <language>");
        return Ok(());
    }

    match args.format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&cache.entries)
                .map_err(|e| CliError::corpus(format!("Failed to serialize: {}", e)))?;
            println!("{}", json);
        }
        OutputFormat::Table => {
            println!("{}", style("Cached Corpora").bold().underlined());
            println!();

            if args.verbose {
                // Detailed table
                println!(
                    "{:<12} {:<10} {:<12} {:<8} {}",
                    style("Source").bold(),
                    style("Language").bold(),
                    style("Size").bold(),
                    style("Age").bold(),
                    style("Path").bold(),
                );
                println!("{}", "-".repeat(70));

                for entry in &cache.entries {
                    let lang = entry.language.as_deref().unwrap_or("-");
                    let age = format!("{} days", entry.age_days());
                    println!(
                        "{:<12} {:<10} {:<12} {:<8} {}",
                        format!("{:?}", entry.source),
                        lang,
                        entry.format_size(),
                        age,
                        entry.path.display(),
                    );
                }
            } else {
                // Compact table
                println!(
                    "{:<12} {:<10} {:<12} {}",
                    style("Source").bold(),
                    style("Language").bold(),
                    style("Size").bold(),
                    style("Downloaded").bold(),
                );
                println!("{}", "-".repeat(50));

                for entry in &cache.entries {
                    let lang = entry.language.as_deref().unwrap_or("-");
                    let date = entry.downloaded_at.format("%Y-%m-%d").to_string();
                    println!(
                        "{:<12} {:<10} {:<12} {}",
                        format!("{:?}", entry.source),
                        lang,
                        entry.format_size(),
                        date,
                    );
                }
            }

            println!();
            println!(
                "Total: {} in {} {}",
                style(format_bytes(cache.total_size())).cyan().bold(),
                cache.count(),
                if cache.count() == 1 {
                    "corpus"
                } else {
                    "corpora"
                }
            );
        }
    }

    Ok(())
}

/// Clean cached corpus files.
fn corpus_clean(args: CorpusCleanArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Loading corpus cache...");
    }

    let mut cache = CorpusCache::load()?;

    if cache.entries.is_empty() {
        println!("{}", style("No cached corpora to clean.").dim());
        return Ok(());
    }

    // Build filter predicate
    let filter = |entry: &CacheEntry| -> bool {
        // Never clean streamed entries
        if entry.was_streamed {
            return false;
        }

        // Filter by source if specified
        if let Some(ref source) = args.source {
            if &entry.source != source {
                return false;
            }
        }

        // Filter by age if specified
        if let Some(older_than) = args.older_than {
            if entry.age_days() < older_than as i64 {
                return false;
            }
        }

        // If --all specified, clean everything (that passes other filters)
        if args.all {
            return true;
        }

        // If no specific filters, need --all
        args.source.is_some() || args.older_than.is_some()
    };

    // Find entries to remove
    let to_remove: Vec<_> = cache
        .entries
        .iter()
        .filter(|e| filter(e))
        .cloned()
        .collect();

    if to_remove.is_empty() {
        println!("{}", style("No corpora match the cleanup criteria.").dim());
        if !args.all && args.source.is_none() && args.older_than.is_none() {
            println!();
            println!("Use one of:");
            println!("  --all          Clean all cached corpora");
            println!("  --source TYPE  Clean specific source type");
            println!("  --older-than N Clean corpora older than N days");
        }
        return Ok(());
    }

    // Calculate total size to free
    let total_to_free: u64 = to_remove.iter().map(|e| e.size_bytes).sum();

    // Show what would be deleted
    println!("{}", style("Corpora to clean:").bold());
    for entry in &to_remove {
        let lang = entry.language.as_deref().unwrap_or("-");
        println!("  {:?} ({}) - {}", entry.source, lang, entry.format_size());
    }
    println!();
    println!(
        "Total: {} to be freed",
        style(format_bytes(total_to_free)).cyan().bold()
    );

    // Dry run stops here
    if args.dry_run {
        println!();
        println!("{}", style("Dry run - no files deleted.").yellow());
        return Ok(());
    }

    // Confirm unless --force
    if !args.force {
        println!();
        print!("Delete these corpora? [y/N] ");
        use std::io::{self, Write};
        io::stdout()
            .flush()
            .map_err(|e| CliError::io(format!("stdout: {}", e)))?;

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| CliError::io(format!("stdin: {}", e)))?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("{}", style("Cancelled.").dim());
            return Ok(());
        }
    }

    // Delete files
    let mut deleted_count = 0;
    let mut deleted_size = 0u64;
    let mut failed = Vec::new();

    for entry in &to_remove {
        if entry.path.exists() {
            match fs::remove_file(&entry.path) {
                Ok(()) => {
                    deleted_count += 1;
                    deleted_size += entry.size_bytes;
                    if verbose {
                        eprintln!("Deleted: {}", entry.path.display());
                    }
                }
                Err(e) => {
                    failed.push((entry.path.clone(), e.to_string()));
                }
            }
        } else {
            // File already gone, just remove from cache
            deleted_count += 1;
        }
    }

    // Update cache
    cache.remove_matching(|e| to_remove.iter().any(|r| r.path == e.path));
    cache.save()?;

    // Report results
    println!();
    println!(
        "{} Deleted {} {}, freed {}",
        style("✓").green().bold(),
        deleted_count,
        if deleted_count == 1 {
            "corpus"
        } else {
            "corpora"
        },
        style(format_bytes(deleted_size)).cyan().bold()
    );

    if !failed.is_empty() {
        println!();
        println!("{}", style("Failed to delete:").red().bold());
        for (path, err) in failed {
            println!("  {}: {}", path.display(), err);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wikipedia_download_url_uses_language_dump() {
        let url = corpus_download_url(CorpusSource::Wikipedia, "de").expect("dump url");
        assert_eq!(
            url,
            "https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2"
        );
    }

    #[test]
    fn corpus_download_path_preserves_remote_filename() {
        let dir = Path::new("/tmp/cache");
        let path = corpus_download_path(
            dir,
            "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
            false,
        )
        .expect("download path");

        assert_eq!(path, dir.join("enwiki-latest-pages-articles.xml.bz2"));
    }

    #[test]
    fn corpus_sample_download_path_is_distinct() {
        let dir = Path::new("/tmp/cache");
        let path = corpus_download_path(dir, "https://example.test/corpus.xml.bz2", true)
            .expect("sample path");

        assert_eq!(path, dir.join("corpus.xml.bz2.sample"));
    }

    #[test]
    fn partial_download_path_keeps_final_path_separate() {
        let path = Path::new("/tmp/cache/corpus.xml.bz2");
        assert_eq!(
            partial_download_path(path),
            PathBuf::from("/tmp/cache/corpus.xml.bz2.part")
        );
    }

    #[test]
    fn copy_limited_stops_at_limit() {
        let mut input = std::io::Cursor::new(b"abcdef".as_slice());
        let mut output = Vec::new();

        let copied = copy_limited(&mut input, &mut output, 3).expect("copy within limit");

        assert_eq!(copied, 3);
        assert_eq!(output, b"abc");
    }
}
