//! CLI argument definitions using clap.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

/// grammstein - Language model training and experimentation CLI.
///
/// A unified CLI for training, evaluating, and querying N-gram and hybrid
/// language models that integrate with lling-llang WFST text correction.
#[derive(Parser, Debug)]
#[command(name = "grammstein")]
#[command(author = "Dylon Edwards")]
#[command(version)]
#[command(about = "Language model training and experimentation CLI")]
#[command(
    long_about = "A unified CLI for training, evaluating, and querying N-gram and hybrid \
    language models that integrate with lling-llang WFST text correction."
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress progress bars and status messages.
    #[arg(short, long, global = true)]
    pub quiet: bool,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train language models from corpus.
    #[command(subcommand)]
    Train(TrainCommands),

    /// Evaluate language models.
    #[command(subcommand)]
    Eval(EvalCommands),

    /// Query language models.
    #[command(subcommand)]
    Query(QueryCommands),

    /// Manage installed models.
    #[command(subcommand)]
    Models(ModelsCommands),

    /// Corpus utilities.
    #[command(subcommand)]
    Corpus(CorpusCommands),

    /// Convert between model formats.
    #[command(subcommand)]
    Convert(ConvertCommands),

    /// Interactive exploration session.
    Repl(ReplArgs),
}

// =============================================================================
// Train Commands
// =============================================================================

/// Training subcommands.
#[derive(Subcommand, Debug)]
pub enum TrainCommands {
    /// Train N-gram language model.
    Ngram(TrainNgramArgs),

    /// Train subword embeddings.
    Embedding(TrainEmbeddingArgs),

    /// Combine N-gram and embedding models into hybrid.
    Hybrid(TrainHybridArgs),

    /// Import N-gram model from Google Books N-grams.
    #[cfg(feature = "google-books")]
    ImportGoogleBooks(ImportGoogleBooksArgs),
}

/// Arguments for training N-gram models.
#[derive(Args, Debug)]
pub struct TrainNgramArgs {
    /// Corpus path (file, directory, glob, or URL).
    #[arg(value_name = "CORPUS")]
    pub corpus: String,

    /// Output model path (.bin).
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// N-gram order.
    #[arg(short, long, default_value = "5")]
    pub order: usize,

    /// Minimum N-gram frequency.
    #[arg(short, long, default_value = "2")]
    pub min_count: u64,

    /// Parallel batch size.
    #[arg(short, long, default_value = "10000")]
    pub batch_size: usize,

    /// Corpus format.
    #[arg(short, long, value_enum, default_value = "plaintext")]
    pub format: CorpusFormat,

    /// Lowercase all tokens.
    #[arg(long)]
    pub lowercase: bool,

    /// Language tag (BCP 47): en, en-US, de-DE, zh-Hans, etc.
    #[arg(short = 'L', long)]
    pub language: Option<String>,

    /// Auto-detect language from corpus sample.
    #[arg(long)]
    pub detect_language: bool,

    /// Checkpoint options.
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    /// Resource management options.
    #[command(flatten)]
    pub resources: ResourceArgs,

    /// Delete downloaded corpus after successful training.
    /// Only applies to auto-downloaded corpora, not local files or streamed data.
    #[arg(long)]
    pub auto_clean: bool,
}

/// Arguments for training embedding models.
#[derive(Args, Debug)]
pub struct TrainEmbeddingArgs {
    /// Corpus path (file, directory, glob, or URL).
    #[arg(value_name = "CORPUS")]
    pub corpus: String,

    /// Output model path (.bin).
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Embedding dimension.
    #[arg(short, long, default_value = "100")]
    pub dim: usize,

    /// Context window size.
    #[arg(short, long, default_value = "5")]
    pub window: usize,

    /// Minimum word frequency.
    #[arg(short, long, default_value = "5")]
    pub min_count: u64,

    /// Negative samples per word.
    #[arg(short, long, default_value = "5")]
    pub neg_samples: usize,

    /// Training epochs.
    #[arg(short, long, default_value = "5")]
    pub epochs: u32,

    /// Learning rate.
    #[arg(short, long, default_value = "0.025")]
    pub learning_rate: f64,

    /// Corpus format.
    #[arg(short, long, value_enum, default_value = "plaintext")]
    pub format: CorpusFormat,

    /// Language tag (BCP 47): en, en-US, de-DE, zh-Hans, etc.
    #[arg(short = 'L', long)]
    pub language: Option<String>,

    /// Auto-detect language from corpus sample.
    #[arg(long)]
    pub detect_language: bool,

    /// Build vocabulary in first pass (uses less peak memory).
    #[arg(long)]
    pub vocab_first: bool,

    /// Checkpoint options.
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    /// Resource management options.
    #[command(flatten)]
    pub resources: ResourceArgs,

    /// Delete downloaded corpus after successful training.
    /// Only applies to auto-downloaded corpora, not local files or streamed data.
    #[arg(long)]
    pub auto_clean: bool,
}

/// Arguments for creating hybrid models.
#[derive(Args, Debug)]
pub struct TrainHybridArgs {
    /// Path to trained N-gram model.
    #[arg(value_name = "NGRAM_MODEL")]
    pub ngram_model: PathBuf,

    /// Path to trained embedding model.
    #[arg(value_name = "EMBEDDING_MODEL")]
    pub embedding_model: PathBuf,

    /// Output hybrid model path.
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Interpolation strategy.
    #[arg(short, long, value_enum, default_value = "linear")]
    pub strategy: InterpolationStrategy,

    /// Interpolation weight (0=embedding, 1=ngram).
    #[arg(short, long, default_value = "0.8")]
    pub alpha: f64,

    /// Score cache size.
    #[arg(long, default_value = "50000")]
    pub cache_size: usize,
}

/// Arguments for importing Google Books N-grams.
#[cfg(feature = "google-books")]
#[derive(Args, Debug)]
pub struct ImportGoogleBooksArgs {
    /// Output model path (PersistentARTrie format).
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Language code (en, de, fr, es, it, ru, he, zh).
    #[arg(short = 'L', long, default_value = "en")]
    pub language: String,

    /// Minimum n-gram order to import.
    #[arg(long, default_value = "1")]
    pub min_order: u8,

    /// Maximum n-gram order to import (1-5).
    #[arg(long, default_value = "5")]
    pub max_order: u8,

    /// Minimum frequency threshold.
    #[arg(short, long, default_value = "40")]
    pub min_count: u64,

    /// Minimum year (filter older publications).
    #[arg(long)]
    pub min_year: Option<u16>,

    /// Maximum year (filter newer publications).
    #[arg(long)]
    pub max_year: Option<u16>,

    /// Import from local gzip files instead of HTTP.
    #[arg(long, value_name = "DIR")]
    pub local_files: Option<PathBuf>,

    /// Number of parallel download streams.
    #[arg(long, default_value = "4")]
    pub parallel: usize,

    /// Skip n-grams with POS tags (e.g., _NOUN_).
    #[arg(long)]
    pub skip_pos_tags: bool,

    /// Force fresh import (ignore existing checkpoint).
    #[arg(long)]
    pub no_resume: bool,

    /// Keep shard files after merge (default: delete to save disk space).
    ///
    /// When using sharded storage mode, temporary shard files are created during
    /// import and merged at the end. By default, these are deleted after merge
    /// to save disk space. Use this flag to preserve them for debugging or to
    /// allow incremental updates in the future.
    #[arg(long)]
    pub keep_shards: bool,

    /// Download n-gram files to local cache before importing.
    ///
    /// Each worker downloads the .gz file to a local temporary file first,
    /// then imports from the local file. Improves reliability on unstable
    /// connections. Cached files are stored in `{output_dir}/grammstein-cache/`
    /// and deleted after successful import or when all retries are exhausted.
    #[arg(long)]
    pub cache_files: bool,

    /// Sharding mode for storage.
    ///
    /// - enabled: Use sharding to reduce thread contention (default)
    /// - disabled: Use single trie (for debugging or constrained environments)
    #[arg(long, value_enum, default_value = "enabled")]
    pub sharding: ShardingModeArg,

    /// Import only this prefix (e.g., "j" for 1-grams, "th" for 2-5 grams).
    ///
    /// Valid prefixes for 1-grams: a-z, other
    /// Valid prefixes for 2-5 grams: aa-zz, other, punctuation
    #[arg(long)]
    pub prefix: Option<String>,

    /// Lock-free overlay flush threshold (entries per shard).
    ///
    /// Controls memory usage during parallel imports. Lower values use less
    /// memory but flush more frequently. Default: auto-scaled based on
    /// --parallel value (50K for >=8 workers, 100K otherwise).
    #[arg(long, value_name = "ENTRIES")]
    pub lockfree_flush_threshold: Option<u64>,

    /// Transaction chunk size for prefix imports (entries per chunk).
    ///
    /// Controls how many n-grams are buffered in a single transaction before
    /// committing a chunk. Lower values reduce memory usage (critical for
    /// 2-gram files with 50-100M entries), but increase WAL write frequency.
    /// Set to 0 to disable chunking (buffer entire file in one transaction).
    #[arg(long, default_value = "500000", value_name = "ENTRIES")]
    pub tx_chunk_size: u64,

    /// Resource management options.
    #[command(flatten)]
    pub resources: ResourceArgs,
}

// =============================================================================
// Eval Commands
// =============================================================================

/// Evaluation subcommands.
#[derive(Subcommand, Debug)]
pub enum EvalCommands {
    /// Evaluate model perplexity on test corpus.
    Perplexity(EvalPerplexityArgs),

    /// Compare multiple models side-by-side.
    Compare(EvalCompareArgs),
}

/// Arguments for perplexity evaluation.
#[derive(Args, Debug)]
pub struct EvalPerplexityArgs {
    /// Path to trained model (.bin).
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Test corpus path.
    #[arg(value_name = "TEST_CORPUS")]
    pub test_corpus: String,

    /// Corpus format.
    #[arg(short, long, value_enum, default_value = "plaintext")]
    pub format: CorpusFormat,

    /// Show per-sentence perplexity.
    #[arg(long)]
    pub per_sentence: bool,

    /// Write results to file (JSON).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for model comparison.
#[derive(Args, Debug)]
pub struct EvalCompareArgs {
    /// Test corpus path.
    #[arg(value_name = "TEST_CORPUS")]
    pub test_corpus: String,

    /// Models to compare (2 or more).
    #[arg(value_name = "MODEL", required = true, num_args = 2..)]
    pub models: Vec<PathBuf>,

    /// Corpus format.
    #[arg(short, long, value_enum, default_value = "plaintext")]
    pub format: CorpusFormat,

    /// Write comparison to file (JSON).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

// =============================================================================
// Query Commands
// =============================================================================

/// Query subcommands.
#[derive(Subcommand, Debug)]
pub enum QueryCommands {
    /// Score a sentence or continuation.
    Score(QueryScoreArgs),

    /// Find similar words (embedding).
    Similar(QuerySimilarArgs),

    /// Get top completions for context.
    Completions(QueryCompletionsArgs),
}

/// Arguments for scoring.
#[derive(Args, Debug)]
pub struct QueryScoreArgs {
    /// Path to trained model.
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Tokens to score (or read from stdin if omitted).
    #[arg(value_name = "TOKENS")]
    pub tokens: Vec<String>,

    /// Score as complete sentence.
    #[arg(long)]
    pub sentence: bool,

    /// Score last token given preceding context.
    #[arg(long)]
    pub continuation: bool,

    /// Output as JSON.
    #[arg(short, long)]
    pub json: bool,
}

/// Arguments for finding similar words.
#[derive(Args, Debug)]
pub struct QuerySimilarArgs {
    /// Path to embedding or hybrid model.
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Query word.
    #[arg(value_name = "WORD")]
    pub word: String,

    /// Number of similar words to return.
    #[arg(short = 'n', long, default_value = "10")]
    pub top: usize,

    /// Output as JSON.
    #[arg(short, long)]
    pub json: bool,
}

/// Arguments for getting completions.
#[derive(Args, Debug)]
pub struct QueryCompletionsArgs {
    /// Path to trained model.
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Context tokens.
    #[arg(value_name = "CONTEXT", required = true)]
    pub context: Vec<String>,

    /// Number of completions to return.
    #[arg(short = 'n', long, default_value = "10")]
    pub top: usize,

    /// Output as JSON.
    #[arg(short, long)]
    pub json: bool,
}

// =============================================================================
// Models Commands
// =============================================================================

/// Model management subcommands.
#[derive(Subcommand, Debug)]
pub enum ModelsCommands {
    /// List installed models.
    List(ModelsListArgs),

    /// Display model information.
    Info(ModelsInfoArgs),
}

/// Arguments for listing models.
#[derive(Args, Debug)]
pub struct ModelsListArgs {
    /// Filter by language (BCP 47 tag).
    #[arg(short = 'L', long)]
    pub language: Option<String>,

    /// Output format.
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,

    /// Models directory.
    #[arg(long, default_value = "./models")]
    pub models_dir: PathBuf,
}

/// Arguments for model info.
#[derive(Args, Debug)]
pub struct ModelsInfoArgs {
    /// Path to model file.
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Output as JSON.
    #[arg(short, long)]
    pub json: bool,
}

// =============================================================================
// Corpus Commands
// =============================================================================

/// Corpus utility subcommands.
#[derive(Subcommand, Debug)]
pub enum CorpusCommands {
    /// Show corpus statistics.
    Stats(CorpusStatsArgs),

    /// Sample sentences from corpus.
    Sample(CorpusSampleArgs),

    /// Download corpus for language.
    Download(CorpusDownloadArgs),

    /// Detect corpus language.
    Detect(CorpusDetectArgs),

    /// List cached corpus files.
    List(CorpusListArgs),

    /// Remove cached corpus files.
    Clean(CorpusCleanArgs),
}

/// Arguments for corpus statistics.
#[derive(Args, Debug)]
pub struct CorpusStatsArgs {
    /// Corpus path.
    #[arg(value_name = "CORPUS")]
    pub corpus: String,

    /// Corpus format.
    #[arg(short, long, value_enum, default_value = "plaintext")]
    pub format: CorpusFormat,
}

/// Arguments for corpus sampling.
#[derive(Args, Debug)]
pub struct CorpusSampleArgs {
    /// Corpus path.
    #[arg(value_name = "CORPUS")]
    pub corpus: String,

    /// Number of sentences to sample.
    #[arg(short = 'n', long, default_value = "10")]
    pub count: usize,

    /// Corpus format.
    #[arg(short, long, value_enum, default_value = "plaintext")]
    pub format: CorpusFormat,

    /// Random seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Arguments for corpus download.
#[derive(Args, Debug)]
pub struct CorpusDownloadArgs {
    /// Language code (en, de, fr, es, zh, ja, etc.).
    #[arg(value_name = "LANGUAGE")]
    pub language: String,

    /// Corpus source.
    #[arg(short, long, value_enum, default_value = "wikipedia")]
    pub source: CorpusSource,

    /// Output directory.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Download only sample (first 100MB).
    #[arg(long)]
    pub sample: bool,

    /// Resume interrupted download.
    #[arg(long)]
    pub resume: bool,
}

/// Arguments for language detection.
#[derive(Args, Debug)]
pub struct CorpusDetectArgs {
    /// Corpus path.
    #[arg(value_name = "CORPUS")]
    pub corpus: String,

    /// Corpus format.
    #[arg(short, long, value_enum, default_value = "plaintext")]
    pub format: CorpusFormat,
}

/// Arguments for listing cached corpora.
#[derive(Args, Debug)]
pub struct CorpusListArgs {
    /// Show detailed information including file sizes and usage.
    #[arg(short, long)]
    pub verbose: bool,

    /// Output format.
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

/// Arguments for cleaning cached corpora.
#[derive(Args, Debug)]
pub struct CorpusCleanArgs {
    /// Only show what would be deleted (dry run).
    #[arg(long)]
    pub dry_run: bool,

    /// Clean specific corpus by source type.
    #[arg(short, long, value_enum)]
    pub source: Option<CorpusSource>,

    /// Clean corpora older than N days.
    #[arg(long)]
    pub older_than: Option<u32>,

    /// Force cleanup without confirmation.
    #[arg(long, short)]
    pub force: bool,

    /// Clean all cached corpora.
    #[arg(long)]
    pub all: bool,
}

// =============================================================================
// Convert Commands
// =============================================================================

/// Conversion subcommands.
#[derive(Subcommand, Debug)]
pub enum ConvertCommands {
    /// Convert to static DoubleArrayTrie (fast inference).
    ToStatic(ConvertToStaticArgs),

    /// Translate trained model to PathMap for production deployment.
    #[cfg(feature = "google-books")]
    ToPathmap(ConvertToPathmapArgs),

    /// Extract dictionary from n-gram model's 1-grams.
    #[cfg(feature = "google-books")]
    ExtractDict(ExtractDictArgs),

    /// Export model info.
    Info(ConvertInfoArgs),
}

/// Arguments for static conversion.
#[derive(Args, Debug)]
pub struct ConvertToStaticArgs {
    /// Input model path.
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output model path.
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,
}

/// Arguments for model info export.
#[derive(Args, Debug)]
pub struct ConvertInfoArgs {
    /// Model path.
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,
}

/// Arguments for PathMap translation.
#[cfg(feature = "google-books")]
#[derive(Args, Debug)]
pub struct ConvertToPathmapArgs {
    /// Input model path (PersistentARTrie format).
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output PathMap path.
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Verify translation integrity after completion.
    #[arg(long)]
    pub verify: bool,
}

/// Arguments for dictionary extraction.
#[cfg(feature = "google-books")]
#[derive(Args, Debug)]
pub struct ExtractDictArgs {
    /// Input n-gram model path.
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Output dictionary path (DoubleArrayTrieChar format).
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Minimum frequency threshold for vocabulary.
    #[arg(short, long, default_value = "100")]
    pub min_count: u64,

    /// Only extract unigrams (1-grams) for vocabulary.
    #[arg(long)]
    pub unigrams_only: bool,
}

// =============================================================================
// REPL Command
// =============================================================================

/// Arguments for the REPL.
#[derive(Args, Debug)]
pub struct ReplArgs {
    /// Optional model to load at startup.
    #[arg(value_name = "MODEL")]
    pub model: Option<PathBuf>,

    /// History file path.
    #[arg(long, default_value = "~/.grammstein_history")]
    pub history: PathBuf,
}

// =============================================================================
// Shared Argument Groups
// =============================================================================

/// Checkpoint-related arguments.
#[derive(Args, Debug)]
pub struct CheckpointArgs {
    /// Save checkpoints to directory.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,

    /// Resume from checkpoint (path or "latest").
    #[arg(long)]
    pub resume: Option<String>,

    /// Sentences/epochs between checkpoints.
    #[arg(long, default_value = "1000000")]
    pub checkpoint_interval: u64,

    /// Maximum checkpoints to keep.
    #[arg(long, default_value = "5")]
    pub keep_checkpoints: usize,
}

/// Resource management arguments.
#[derive(Args, Debug)]
pub struct ResourceArgs {
    /// Number of parallel threads.
    #[arg(long)]
    pub threads: Option<usize>,

    /// Memory limit (e.g., "8G", "16G").
    #[arg(long)]
    pub max_memory: Option<String>,

    /// Disable progress bar.
    #[arg(long)]
    pub no_progress: bool,
}

// =============================================================================
// Value Enums
// =============================================================================

/// Corpus format.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CorpusFormat {
    /// Plain text files (one sentence per line or paragraph-based).
    #[default]
    Plaintext,
    /// Wikipedia XML dump (optionally bz2 compressed).
    Wikipedia,
    /// Project Gutenberg plain text format.
    Gutenberg,
}

/// Interpolation strategy for hybrid models.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InterpolationStrategy {
    /// Linear interpolation: α * ngram + (1-α) * embedding.
    #[default]
    Linear,
    /// Log-linear interpolation.
    LogLinear,
    /// Use N-gram with embedding fallback for OOV.
    NgramFallback,
    /// Dynamic weighting based on context.
    Dynamic,
}

/// Output format for listings.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Formatted table.
    #[default]
    Table,
    /// JSON output.
    Json,
}

/// Corpus download source.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorpusSource {
    /// Wikipedia dump.
    #[default]
    Wikipedia,
    /// Project Gutenberg.
    Gutenberg,
    /// OSCAR corpus.
    Oscar,
}

/// CLI argument for sharding mode.
#[cfg(feature = "google-books")]
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShardingModeArg {
    /// Use sharding to reduce thread contention (default).
    #[default]
    Enabled,
    /// Use single trie (for debugging or constrained environments).
    Disabled,
}
