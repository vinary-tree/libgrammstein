//! N-gram model training with parallel corpus processing.
//!
//! This module provides the training pipeline for byte-native n-gram language
//! models:
//! - Streaming corpus reading
//! - Parallel n-gram counting with Rayon over a lock-free [`TermIdStore`]
//! - Continuation-count collection for Modified Kneser-Ney
//! - Progress reporting
//!
//! The trainer counts n-grams as **LEB128 term-id byte keys** (via
//! [`MutableNgramStore::bump`]) — the same self-delimiting encoding the
//! Google-Books importer uses — so a token that literally contains the historic
//! `'|'` separator is stored losslessly and the delimiter-collision class cannot
//! occur.

use super::model::NgramModel;
use super::smoothing::KneserNeySmoothing;
use super::store::{MutableByteMappedDictionary, MutableNgramStore, TermIdStore};
use super::vocabulary::{create_vocabulary, open_or_create_vocabulary, SharedVocabARTrie};
use crate::corpus::{CorpusReader, PrefetchConfig, PrefetchingReader, Tokenizer};
use crate::Result;

use crossbeam_channel::Sender;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Training progress information.
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    /// Number of sentences processed.
    pub sentences_processed: u64,

    /// Number of n-grams counted.
    pub ngrams_counted: u64,

    /// Elapsed time in seconds.
    pub elapsed_secs: f64,
}

/// Vocabulary source for n-gram training.
///
/// Every trained model is term-id-keyed, so a vocabulary is always required.
/// This selects where it comes from.
#[derive(Debug, Clone, Default)]
pub enum VocabularyMode {
    /// Create an ephemeral vocabulary in a fresh temporary directory (the
    /// default).
    ///
    /// The vocabulary lives inside the trained model (an `Arc`); its backing
    /// file is created under the system temp directory and is not auto-deleted
    /// (the file is tiny). Callers who need a managed, persisted vocabulary
    /// should use [`VocabularyMode::Create`] or [`VocabularyMode::Shared`].
    #[default]
    Ephemeral,

    /// Create (or open) a persistent vocabulary at the given path.
    Create(PathBuf),

    /// Use an existing shared vocabulary.
    ///
    /// Useful when training multiple models with a consistent vocabulary, or
    /// when integrating with the Google Books import pipeline.
    Shared(SharedVocabARTrie),
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Maximum n-gram order (e.g., 5 for 5-grams).
    pub order: usize,

    /// Batch size for parallel processing.
    pub batch_size: usize,

    /// Minimum word frequency to include in vocabulary.
    pub min_word_freq: u64,

    /// Vocabulary source.
    pub vocabulary_mode: VocabularyMode,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            order: 5,
            batch_size: 10_000,
            min_word_freq: 1,
            vocabulary_mode: VocabularyMode::default(),
        }
    }
}

impl TrainingConfig {
    /// Create a new training configuration.
    pub fn new(order: usize) -> Self {
        Self {
            order,
            batch_size: 10_000,
            min_word_freq: 1,
            vocabulary_mode: VocabularyMode::default(),
        }
    }

    /// Set the batch size for parallel processing.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set minimum word frequency.
    pub fn with_min_word_freq(mut self, min_freq: u64) -> Self {
        self.min_word_freq = min_freq;
        self
    }
}

/// Create an ephemeral vocabulary in a unique temporary directory.
///
/// The returned vocabulary is self-contained (its state lives in the process
/// overlay and inside the trained model's `Arc`); the backing directory is not
/// removed on drop.
fn create_ephemeral_vocabulary() -> SharedVocabARTrie {
    use std::sync::atomic::AtomicU64 as SeqCounter;
    static SEQ: SeqCounter = SeqCounter::new(0);

    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "libgrammstein-train-vocab-{}-{}",
        std::process::id(),
        seq
    ));
    std::fs::create_dir_all(&dir).expect("failed to create ephemeral vocabulary directory");
    create_vocabulary(&dir.join("vocabulary")).expect("failed to create ephemeral vocabulary")
}

/// N-gram trainer with parallel corpus processing over a byte-native store.
///
/// Uses Rayon for CPU-bound parallel processing and libdictenstein's lock-free
/// `&self` byte read-modify-write for n-gram counting. The trained model is a
/// [`NgramModel<TermIdStore<B>>`] — term-id-keyed, so training is
/// delimiter-collision-free.
pub struct NgramTrainer<B>
where
    B: MutableByteMappedDictionary + Send + Sync,
{
    /// The byte-native n-gram store being built.
    store: TermIdStore<B>,

    /// Training configuration.
    config: TrainingConfig,

    /// Training statistics.
    stats: TrainingStats,

    /// Word tokenizer.
    tokenizer: Tokenizer,
}

/// Training statistics with atomic counters for thread safety.
#[derive(Default)]
pub struct TrainingStats {
    sentences_processed: AtomicU64,
    ngrams_counted: AtomicU64,
    tokens_processed: AtomicU64,
}

impl TrainingStats {
    /// Get the number of sentences processed.
    pub fn sentences_processed(&self) -> u64 {
        self.sentences_processed.load(Ordering::Relaxed)
    }

    /// Get the number of n-grams counted.
    pub fn ngrams_counted(&self) -> u64 {
        self.ngrams_counted.load(Ordering::Relaxed)
    }

    /// Get the number of tokens processed.
    pub fn tokens_processed(&self) -> u64 {
        self.tokens_processed.load(Ordering::Relaxed)
    }

    /// Increment sentence count.
    pub fn inc_sentences(&self) {
        self.sentences_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment n-gram count.
    pub fn inc_ngrams(&self, count: u64) {
        self.ngrams_counted.fetch_add(count, Ordering::Relaxed);
    }

    /// Increment token count.
    pub fn inc_tokens(&self, count: u64) {
        self.tokens_processed.fetch_add(count, Ordering::Relaxed);
    }
}

impl<B> NgramTrainer<B>
where
    B: MutableByteMappedDictionary + Send + Sync + 'static,
{
    /// Create a new trainer with the given byte backend and configuration.
    ///
    /// The vocabulary is resolved from the configuration's `vocabulary_mode`:
    /// - [`VocabularyMode::Ephemeral`]: a fresh temp-dir vocabulary
    /// - [`VocabularyMode::Create`]: opens or creates a vocabulary at the path
    /// - [`VocabularyMode::Shared`]: uses the provided shared vocabulary
    pub fn new(backend: B, config: TrainingConfig) -> Self {
        let order = config.order;

        let vocabulary = match &config.vocabulary_mode {
            VocabularyMode::Ephemeral => create_ephemeral_vocabulary(),
            VocabularyMode::Create(path) => {
                open_or_create_vocabulary(path).expect("Failed to create vocabulary")
            }
            VocabularyMode::Shared(vocab) => vocab.clone(),
        };

        Self {
            store: TermIdStore::new(backend, vocabulary, order),
            config,
            stats: TrainingStats::default(),
            tokenizer: Tokenizer::new(),
        }
    }

    /// Get a reference to the vocabulary used by this trainer.
    pub fn vocabulary(&self) -> &SharedVocabARTrie {
        self.store.vocabulary()
    }

    /// Set a custom tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Train the n-gram model from a corpus reader.
    ///
    /// This is the main training entry point that:
    /// 1. Counts n-grams in parallel using prefetched batches
    /// 2. Collects continuation counts
    /// 3. Computes smoothing parameters
    ///
    /// # Arguments
    ///
    /// * `reader` - Corpus reader providing sentences (takes ownership)
    ///
    /// # Returns
    ///
    /// The trained [`NgramModel<TermIdStore<B>>`] or an error.
    pub fn train<R: CorpusReader + 'static>(self, reader: R) -> Result<NgramModel<TermIdStore<B>>> {
        let start = std::time::Instant::now();

        // Phase 1: Count n-grams with prefetched streaming
        self.count_ngrams(reader)?;

        // Phase 2: Collect continuation counts (for MKN smoothing)
        self.collect_continuation_counts();

        // Phase 3: Compute smoothing parameters
        let smoothing = self.compute_smoothing_params();

        let elapsed = start.elapsed().as_secs_f64();
        log::info!(
            "Training complete: {} sentences, {} n-grams in {:.2}s",
            self.stats.sentences_processed(),
            self.stats.ngrams_counted(),
            elapsed
        );

        // Compute vocabulary size (unique unigrams)
        let vocab_size = self.count_unigrams();
        let total_count = self.stats.tokens_processed();

        Ok(NgramModel::new(
            self.store,
            smoothing,
            vocab_size,
            total_count,
        ))
    }

    /// Train with progress reporting via channel.
    pub fn train_with_progress<R: CorpusReader + 'static>(
        self,
        reader: R,
        progress_tx: Sender<TrainingProgress>,
    ) -> Result<NgramModel<TermIdStore<B>>> {
        let start = std::time::Instant::now();

        // Phase 1: Count n-grams with progress using prefetched streaming
        self.count_ngrams_with_progress(reader, &progress_tx, &start)?;

        // Phase 2: Collect continuation counts
        self.collect_continuation_counts();

        // Phase 3: Compute smoothing parameters
        let smoothing = self.compute_smoothing_params();

        // Final progress
        let _ = progress_tx.try_send(TrainingProgress {
            sentences_processed: self.stats.sentences_processed(),
            ngrams_counted: self.stats.ngrams_counted(),
            elapsed_secs: start.elapsed().as_secs_f64(),
        });

        let vocab_size = self.count_unigrams();
        let total_count = self.stats.tokens_processed();

        Ok(NgramModel::new(
            self.store,
            smoothing,
            vocab_size,
            total_count,
        ))
    }

    /// Count n-grams from corpus in parallel using prefetched streaming.
    ///
    /// Every n-gram of order `1..=order` in each sentence is counted uniformly
    /// through [`MutableNgramStore::bump`] (a lock-free term-id byte RMW); the
    /// vocabulary grows on demand.
    fn count_ngrams<R: CorpusReader + 'static>(&self, reader: R) -> Result<()> {
        let order = self.config.order;
        let store = &self.store;
        let stats = &self.stats;
        let tokenizer = &self.tokenizer;

        // Configure prefetch for this training run
        let config = PrefetchConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_ram_fraction(0.10);

        let prefetch = PrefetchingReader::with_config(reader, config);
        let mut received_any = false;

        // Process prefetched batches in parallel
        for batch in prefetch.batches() {
            received_any = true;

            batch.par_iter().for_each(|sentence| {
                // Tokenize into owned strings, then work with references
                let token_strings: Vec<String> = tokenizer.words(sentence).collect();

                if token_strings.is_empty() {
                    return;
                }

                let tokens: Vec<&str> = token_strings.iter().map(String::as_str).collect();

                stats.inc_tokens(tokens.len() as u64);
                stats.inc_sentences();

                let mut ngram_count = 0u64;

                // Extract and count n-grams of all orders up to max.
                for n in 1..=order.min(tokens.len()) {
                    for i in 0..=(tokens.len() - n) {
                        store.bump(&tokens[i..i + n], 1);
                        ngram_count += 1;
                    }
                }

                stats.inc_ngrams(ngram_count);
            });
        }

        if !received_any {
            return Err(crate::Error::EmptyCorpus);
        }

        Ok(())
    }

    /// Count n-grams with progress reporting using prefetched streaming.
    fn count_ngrams_with_progress<R: CorpusReader + 'static>(
        &self,
        reader: R,
        progress_tx: &Sender<TrainingProgress>,
        start: &std::time::Instant,
    ) -> Result<()> {
        let order = self.config.order;
        let store = &self.store;
        let stats = &self.stats;
        let tokenizer = &self.tokenizer;

        // Configure prefetch for this training run
        let config = PrefetchConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_ram_fraction(0.10);

        let prefetch = PrefetchingReader::with_config(reader, config);
        let mut received_any = false;

        // Send progress every 10,000 sentences
        let progress_interval = 10_000usize;

        for batch in prefetch.batches() {
            received_any = true;

            batch.par_iter().for_each(|sentence| {
                let token_strings: Vec<String> = tokenizer.words(sentence).collect();

                if token_strings.is_empty() {
                    return;
                }

                let tokens: Vec<&str> = token_strings.iter().map(String::as_str).collect();

                stats.inc_tokens(tokens.len() as u64);
                stats.inc_sentences();

                let mut ngram_count = 0u64;

                for n in 1..=order.min(tokens.len()) {
                    for i in 0..=(tokens.len() - n) {
                        store.bump(&tokens[i..i + n], 1);
                        ngram_count += 1;
                    }
                }

                stats.inc_ngrams(ngram_count);

                // Send progress periodically
                let processed = stats.sentences_processed();
                if processed as usize % progress_interval == 0 {
                    let _ = progress_tx.try_send(TrainingProgress {
                        sentences_processed: processed,
                        ngrams_counted: stats.ngrams_counted(),
                        elapsed_secs: start.elapsed().as_secs_f64(),
                    });
                }
            });
        }

        if !received_any {
            return Err(crate::Error::EmptyCorpus);
        }

        Ok(())
    }

    /// Collect continuation counts for Modified Kneser-Ney smoothing.
    ///
    /// A single byte-native pass over [`iter_term_ids`](MutableNgramStore::iter_term_ids)
    /// tracks, for every stored n-gram `w₁…wₙ` (`n ≥ 2`):
    /// - **predecessors** of the suffix `w₂…wₙ` — the unique `w₁` preceding it —
    ///   written as the suffix's `continuation_count` (`N₁₊(•, w₂…wₙ)`); at the
    ///   unigram level this is the Kneser-Ney continuation count `N₁₊(•, w)`; and
    /// - **successors** of the prefix `w₁…wₙ₋₁` — the unique `wₙ` following it —
    ///   written as the prefix's `unique_continuations` (`N₁₊(w₁…wₙ₋₁, •)`).
    ///
    /// This mirrors the Google-Books importer's proven statistics pass
    /// (`sources/google_books/sharding/mkn.rs`).
    ///
    /// # Memory
    ///
    /// The predecessor/successor sets are held in memory during the pass. For very
    /// large corpora (10M+ n-grams) this can use several GB; the importer's sharded
    /// pipeline is the scale path, this in-memory trainer targets small/medium
    /// corpora and tests.
    fn collect_continuation_counts(&self) {
        log::debug!("Collecting continuation counts (term-id) for MKN smoothing");

        let entry_count = self.stats.ngrams_counted();
        if entry_count > 5_000_000 {
            log::warn!(
                "Collecting continuation counts for {} n-grams may use significant memory (2-5GB). \
                 Consider the sharded Google-Books importer for large corpora.",
                entry_count
            );
        }

        // suffix (w₂…wₙ) -> set of unique predecessors w₁
        let mut predecessor_sets: HashMap<Vec<u64>, HashSet<u64>> = HashMap::new();
        // prefix (w₁…wₙ₋₁) -> set of unique successors wₙ
        let mut successor_sets: HashMap<Vec<u64>, HashSet<u64>> = HashMap::new();

        for (ids, _entry) in self.store.iter_term_ids() {
            if ids.len() < 2 {
                continue;
            }

            let predecessor = ids[0];
            let suffix = ids[1..].to_vec();
            predecessor_sets
                .entry(suffix)
                .or_default()
                .insert(predecessor);

            let successor = ids[ids.len() - 1];
            let prefix = ids[..ids.len() - 1].to_vec();
            successor_sets.entry(prefix).or_default().insert(successor);
        }

        for (suffix, predecessors) in predecessor_sets {
            self.store
                .set_continuation_count_ids(&suffix, predecessors.len() as u32);
        }

        for (prefix, successors) in successor_sets {
            self.store
                .set_unique_continuations_ids(&prefix, successors.len() as u32);
        }

        log::debug!("Continuation count collection complete");
    }

    /// Count n-grams by frequency for MKN discount computation.
    ///
    /// Returns (n1, n2, n3, n4): the number of n-grams occurring exactly 1, 2, 3,
    /// and 4 times.
    fn count_ngram_frequencies(&self) -> (u64, u64, u64, u64) {
        let mut n1 = 0u64;
        let mut n2 = 0u64;
        let mut n3 = 0u64;
        let mut n4 = 0u64;

        for (_ids, entry) in self.store.iter_term_ids() {
            match entry.count() {
                1 => n1 += 1,
                2 => n2 += 1,
                3 => n3 += 1,
                4 => n4 += 1,
                _ => {}
            }
        }

        log::debug!(
            "N-gram frequency counts: n1={}, n2={}, n3={}, n4={}",
            n1,
            n2,
            n3,
            n4
        );

        (n1, n2, n3, n4)
    }

    /// Compute Modified Kneser-Ney smoothing parameters from corpus statistics.
    fn compute_smoothing_params(&self) -> KneserNeySmoothing {
        let (n1, n2, n3, n4) = self.count_ngram_frequencies();

        let smoothing = if n1 > 0 && n2 > 0 && n3 > 0 && n4 > 0 {
            log::info!("Computing optimal MKN discounts from corpus statistics");
            KneserNeySmoothing::from_counts(n1, n2, n3, n4)
        } else {
            log::debug!(
                "Insufficient count diversity (n1={}, n2={}, n3={}, n4={}), using default MKN discounts",
                n1, n2, n3, n4
            );
            KneserNeySmoothing::new(self.config.order)
        };

        // Attach N₁₊(•,•) (total distinct bigram types) — the correct denominator
        // for lower-order continuation probabilities. Continuation counts were
        // populated in Phase 2, so summing them over the unigram entries now
        // yields Σ_w N₁₊(•,w).
        smoothing.with_total_bigram_types(self.count_total_bigram_types())
    }

    /// Sum `N₁₊(•,w)` over all unigram entries to obtain `N₁₊(•,•)`, the total
    /// number of distinct bigram types (the lower-order continuation denominator).
    fn count_total_bigram_types(&self) -> u64 {
        let mut total: u64 = 0;
        for (ids, entry) in self.store.iter_term_ids() {
            if ids.len() == 1 {
                total += entry.continuation_count() as u64;
            }
        }
        total
    }

    /// Count unique unigrams (vocabulary size).
    fn count_unigrams(&self) -> usize {
        let mut count = 0;
        for (ids, _entry) in self.store.iter_term_ids() {
            if ids.len() == 1 {
                count += 1;
            }
        }
        count
    }
}

/// Builder for training with fluent API.
pub struct TrainerBuilder<B>
where
    B: MutableByteMappedDictionary + Send + Sync,
{
    backend: B,
    config: TrainingConfig,
    tokenizer: Option<Tokenizer>,
}

impl<B> TrainerBuilder<B>
where
    B: MutableByteMappedDictionary + Send + Sync + 'static,
{
    /// Create a new trainer builder over a byte backend (e.g. an in-memory
    /// `DynamicDawg<NgramEntry>` or a disk-backed `Arc<PersistentARTrie<NgramEntry>>`).
    ///
    /// Without an explicit vocabulary the trainer uses an ephemeral temp-dir
    /// vocabulary ([`VocabularyMode::Ephemeral`]).
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            config: TrainingConfig::default(),
            tokenizer: None,
        }
    }

    /// Set the n-gram order.
    pub fn order(mut self, order: usize) -> Self {
        self.config.order = order;
        self
    }

    /// Set the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set minimum word frequency.
    pub fn min_word_freq(mut self, freq: u64) -> Self {
        self.config.min_word_freq = freq;
        self
    }

    /// Set custom tokenizer.
    pub fn tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Create (or open) a persistent vocabulary at the given path during training.
    pub fn with_vocabulary_path(mut self, path: PathBuf) -> Self {
        self.config.vocabulary_mode = VocabularyMode::Create(path);
        self
    }

    /// Use an existing shared vocabulary for training.
    ///
    /// Useful when training multiple models with a consistent vocabulary, or when
    /// integrating with the Google Books import pipeline.
    pub fn with_vocabulary(mut self, vocab: SharedVocabARTrie) -> Self {
        self.config.vocabulary_mode = VocabularyMode::Shared(vocab);
        self
    }

    /// Build the trainer.
    pub fn build(self) -> NgramTrainer<B> {
        let mut trainer = NgramTrainer::new(self.backend, self.config);
        if let Some(tokenizer) = self.tokenizer {
            trainer = trainer.with_tokenizer(tokenizer);
        }
        trainer
    }

    /// Build and immediately train from corpus.
    pub fn train<R: CorpusReader + 'static>(self, reader: R) -> Result<NgramModel<TermIdStore<B>>> {
        self.build().train(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::super::entry::NgramEntry;
    use super::super::vocabulary::create_vocabulary;
    use super::*;
    use crate::corpus::PlaintextReader;
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_corpus(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");
        path
    }

    fn backend() -> DynamicDawg<NgramEntry> {
        DynamicDawg::<NgramEntry>::new()
    }

    #[test]
    fn test_train_simple_corpus() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_corpus(
            dir.path(),
            "The quick brown fox. The quick brown dog. The lazy fox.",
        );

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let model = TrainerBuilder::new(backend())
            .order(3)
            .train(reader)
            .expect("Training failed");

        assert!(model.vocab_size() > 0);
        assert!(model.ngram_count() > 0);
    }

    #[test]
    fn test_bigram_counts() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_corpus(dir.path(), "a b a b a b");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let model = TrainerBuilder::new(backend())
            .order(2)
            .train(reader)
            .expect("Training failed");

        // "a b" should appear 3 times
        assert!(model.count(&["a", "b"]) >= 2);
    }

    #[test]
    fn test_vocabulary_trainer_basic() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let corpus_path = create_test_corpus(dir.path(), "the quick brown fox the quick brown dog");

        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");

        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");

        let model = TrainerBuilder::new(backend())
            .order(3)
            .with_vocabulary(vocab.clone())
            .train(reader)
            .expect("Training with vocabulary failed");

        assert!(model.vocab_size() > 0, "Vocabulary should contain entries");
        assert!(model.ngram_count() > 0, "Model should contain n-grams");

        // The shared vocabulary was populated during training.
        assert!(
            vocab.as_ref().contains("the"),
            "Expected 'the' in vocabulary"
        );
        assert!(
            vocab.as_ref().contains("quick"),
            "Expected 'quick' in vocabulary"
        );
        assert!(
            vocab.as_ref().contains("brown"),
            "Expected 'brown' in vocabulary"
        );
        assert!(
            vocab.as_ref().contains("fox"),
            "Expected 'fox' in vocabulary"
        );

        // The bigram is retrievable by tokens (term-id keyed, no '|').
        assert!(
            model.count(&["the", "quick"]) >= 1,
            "Expected 'the quick' bigram"
        );
    }

    #[test]
    fn test_pipe_in_token_no_corruption() {
        // The delimiter-collision acceptance test: a token literally containing
        // '|' trains, scores, and is retrievable — impossible under the old
        // '|'-joined key scheme.
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");

        let corpus_path = create_test_corpus(dir.path(), "foo|bar baz foo|bar baz foo|bar baz");
        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");

        let model = TrainerBuilder::new(backend())
            .order(2)
            .with_vocabulary(vocab.clone())
            .train(reader)
            .expect("Training failed");

        assert!(
            vocab.as_ref().contains("foo|bar"),
            "Expected 'foo|bar' as a single token in vocabulary"
        );
        assert!(
            vocab.as_ref().contains("baz"),
            "Expected 'baz' in vocabulary"
        );

        // Exactly two unique words.
        assert_eq!(model.vocab_size(), 2, "Should have exactly 2 unique words");

        // The bigram ["foo|bar", "baz"] is stored and retrievable as a single
        // two-token n-gram (would corrupt to 3 tokens under '|'-joining).
        assert!(
            model.count(&["foo|bar", "baz"]) >= 3,
            "Expected 'foo|bar baz' bigram count >= 3, got {}",
            model.count(&["foo|bar", "baz"])
        );
        // A finite log-probability confirms scoring works through the term-id path.
        assert!(model.log_prob("baz", &["foo|bar"]).is_finite());
    }

    #[test]
    fn test_default_ephemeral_vocabulary() {
        // With no explicit vocabulary, training uses an ephemeral temp-dir vocab
        // and still produces a scoreable model.
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_corpus(dir.path(), "a b c a b c");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let config = TrainingConfig::new(2);
        assert!(
            matches!(config.vocabulary_mode, VocabularyMode::Ephemeral),
            "Default mode should be Ephemeral"
        );

        let model = TrainerBuilder::new(backend())
            .order(2)
            .train(reader)
            .expect("Training failed");

        assert!(model.vocab_size() > 0);
        assert!(model.ngram_count() > 0);
        assert!(model.log_prob("b", &["a"]).is_finite());
    }

    #[test]
    fn test_vocabulary_mode_shared() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("shared_vocab.artrie");

        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");

        // Pre-populate the vocabulary
        vocab.as_ref().insert("pre").expect("insert pre");
        vocab
            .as_ref()
            .insert("populated")
            .expect("insert populated");
        vocab.as_ref().insert("words").expect("insert words");

        let corpus_path = create_test_corpus(dir.path(), "pre populated words are here");
        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");

        let model = TrainerBuilder::new(backend())
            .order(2)
            .with_vocabulary(vocab.clone())
            .train(reader)
            .expect("Training with shared vocabulary failed");

        assert!(
            vocab.as_ref().len() > 3,
            "Vocabulary should have grown with new words"
        );
        assert!(model.vocab_size() > 0);
    }

    #[test]
    fn test_continuation_counts() {
        // MKN continuation counts are computed without panics and yield finite
        // log-probabilities.
        let dir = TempDir::new().expect("Failed to create temp dir");
        let corpus_path = create_test_corpus(dir.path(), "the quick fox the slow fox the big fox");

        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");

        let model = TrainerBuilder::new(backend())
            .order(2)
            .train(reader)
            .expect("Training failed");

        assert!(model.vocab_size() > 0);
        assert!(model.ngram_count() > 0);

        let log_prob = model.log_prob("fox", &["quick"]);
        assert!(log_prob.is_finite(), "Log probability should be finite");
    }

    #[test]
    fn test_total_bigram_types_populated() {
        // "the quick fox the slow fox the big fox" has 7 distinct bigram types:
        // (the,quick)(quick,fox)(fox,the)(the,slow)(slow,fox)(the,big)(big,fox),
        // so N₁₊(•,•) = Σ_w N₁₊(•,w) must be carried on the smoothing params.
        let dir = TempDir::new().expect("Failed to create temp dir");
        let corpus_path = create_test_corpus(dir.path(), "the quick fox the slow fox the big fox");

        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");

        let model = TrainerBuilder::new(backend())
            .order(2)
            .train(reader)
            .expect("Training failed");

        let total = model.smoothing().total_bigram_types();
        assert_eq!(total, 7, "N₁₊(•,•) must equal the 7 distinct bigram types");
        assert!(
            total <= model.ngram_count() as u64,
            "N₁₊(•,•) ({}) must not exceed entry count ({})",
            total,
            model.ngram_count()
        );
    }
}
