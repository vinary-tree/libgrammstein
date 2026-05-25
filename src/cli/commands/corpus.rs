//! Corpus utility command implementations.

use std::collections::HashMap;
use std::fs;
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
    word_vec.sort_by(|a, b| b.1.cmp(&a.1));
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

    // Get the download URL based on source and language
    let url = match args.source {
        CorpusSource::Wikipedia => wikipedia_dump_url(&args.language),
        CorpusSource::Gutenberg => {
            return Err(CliError::unsupported(
                "Gutenberg download not yet implemented. Visit https://www.gutenberg.org/",
            ));
        }
        CorpusSource::Oscar => {
            return Err(CliError::unsupported(
                "OSCAR download not yet implemented. Visit https://oscar-project.github.io/documentation/",
            ));
        }
    };

    // For now, provide instructions for manual download
    println!("{}", style("Corpus Download").bold().underlined());
    println!();
    println!("Language: {}", style(&args.language).cyan());
    println!("Source:   {:?}", args.source);
    println!();
    println!("{}", style("Download URL:").bold());
    println!("  {}", style(&url).green());
    println!();
    println!("{}", style("Manual download instructions:").bold());
    println!("  1. Download the file using wget or curl:");
    println!("     wget -c \"{}\"", url);
    println!();
    println!("  2. The file is bz2-compressed XML. You can use it directly with:");
    println!(
        "     grammstein train ngram {} model.bin --format wikipedia",
        url.split('/').last().unwrap_or("dump.xml.bz2")
    );
    println!();

    if args.sample {
        println!(
            "{}: Sample download (--sample) is not yet implemented.",
            style("note").yellow()
        );
    }

    if args.resume {
        println!(
            "{}: Resume download (--resume) is not yet implemented.",
            style("note").yellow()
        );
    }

    // Return Ok since we provided useful information
    Ok(())
}

/// Get Wikipedia dump URL for a language.
fn wikipedia_dump_url(lang: &str) -> String {
    format!(
        "https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2",
        lang, lang
    )
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
        Zho => "zh",
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
        Zho => "Chinese",
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
