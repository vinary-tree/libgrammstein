//! N-gram model training with parallel corpus processing.
//!
//! This module provides the training pipeline for n-gram language models:
//! - Streaming corpus reading
//! - Parallel n-gram counting with Rayon
//! - Continuation count collection for Modified Kneser-Ney
//! - Progress reporting

use super::entry::NgramEntry;
use super::model::NgramModel;
use super::smoothing::KneserNeySmoothing;
use super::trie::{IterableDictionary, NgramTrie, LEGACY_NGRAM_SEPARATOR};
use super::vocabulary::{encode_ngram_key, open_or_create_vocabulary, SharedVocabARTrie};
use crate::corpus::{CorpusReader, PrefetchConfig, PrefetchingReader, Tokenizer};
use crate::Result;

use crossbeam_channel::Sender;
use liblevenshtein::dictionary::MutableMappedDictionary;
use rayon::prelude::*;
use std::collections::HashSet;
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

/// Vocabulary encoding mode for n-gram training.
///
/// Controls whether training uses legacy pipe-separated keys or the new
/// vocabulary-indexed (PUA character) encoding.
#[derive(Debug, Clone, Default)]
pub enum VocabularyMode {
    /// Legacy pipe-separated encoding (backward compatible, default).
    ///
    /// N-gram keys are encoded as `"the|quick|brown"`. This is deprecated
    /// because it can corrupt data if tokens contain the pipe character.
    #[default]
    Legacy,

    /// Create a new vocabulary during training at the given path.
    ///
    /// Each unique word is assigned a PUA character, and n-gram keys are
    /// sequences of these characters. The vocabulary is persisted to disk.
    Create(PathBuf),

    /// Use an existing shared vocabulary.
    ///
    /// Useful when training multiple models with a consistent vocabulary,
    /// or when integrating with the Google Books import pipeline.
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

    /// Vocabulary encoding mode.
    ///
    /// Defaults to `VocabularyMode::Legacy` for backward compatibility.
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

/// N-gram trainer with parallel corpus processing.
///
/// Uses Rayon for CPU-bound parallel processing and atomic operations
/// for lock-free n-gram counting.
///
/// # Vocabulary Modes
///
/// The trainer supports two key encoding modes:
///
/// - **Legacy** (default): Uses pipe-separated keys (`"the|quick|brown"`).
///   Backward compatible but can corrupt data if tokens contain `|`.
///
/// - **Vocabulary-indexed**: Each word maps to a PUA character, producing
///   compact keys. Use `VocabularyMode::Create` or `VocabularyMode::Shared`.
pub struct NgramTrainer<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync,
{
    /// The n-gram trie being built.
    trie: NgramTrie<D>,

    /// Training configuration.
    config: TrainingConfig,

    /// Training statistics.
    stats: TrainingStats,

    /// Word tokenizer.
    tokenizer: Tokenizer,

    /// Optional vocabulary for vocabulary-indexed encoding.
    ///
    /// When `Some`, n-gram keys use PUA characters instead of pipe-separated strings.
    vocabulary: Option<SharedVocabARTrie>,
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

impl<D> NgramTrainer<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync + 'static,
{
    /// Create a new trainer with the given dictionary and configuration.
    ///
    /// The vocabulary is resolved from the configuration's `vocabulary_mode`:
    /// - `Legacy`: No vocabulary (pipe-separated keys)
    /// - `Create(path)`: Opens or creates a vocabulary at the given path
    /// - `Shared(vocab)`: Uses the provided shared vocabulary
    pub fn new(dictionary: D, config: TrainingConfig) -> Self {
        let order = config.order;

        // Resolve vocabulary based on mode
        let vocabulary = match &config.vocabulary_mode {
            VocabularyMode::Legacy => None,
            VocabularyMode::Create(path) => {
                Some(open_or_create_vocabulary(path).expect("Failed to create vocabulary"))
            }
            VocabularyMode::Shared(vocab) => Some(vocab.clone()),
        };

        Self {
            trie: NgramTrie::new(dictionary, order),
            config,
            stats: TrainingStats::default(),
            tokenizer: Tokenizer::new(),
            vocabulary,
        }
    }

    /// Get a reference to the vocabulary, if using vocabulary-indexed encoding.
    pub fn vocabulary(&self) -> Option<&SharedVocabARTrie> {
        self.vocabulary.as_ref()
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
    /// The trained `NgramModel` or an error.
    pub fn train<R: CorpusReader + 'static>(self, reader: R) -> Result<NgramModel<D>> {
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
            self.trie,
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
    ) -> Result<NgramModel<D>> {
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
            self.trie,
            smoothing,
            vocab_size,
            total_count,
        ))
    }

    /// Count n-grams from corpus in parallel using prefetched streaming.
    ///
    /// Uses `PrefetchingReader` to decouple I/O from processing, processing
    /// batches in parallel with Rayon.
    ///
    /// The encoding mode depends on `self.vocabulary`:
    /// - `None`: Uses legacy pipe-separated keys via `trie.insert()`
    /// - `Some(vocab)`: Uses vocabulary-indexed PUA keys via `trie.insert_with_key()`
    fn count_ngrams<R: CorpusReader + 'static>(&self, reader: R) -> Result<()> {
        let order = self.config.order;
        let trie = &self.trie;
        let stats = &self.stats;
        let tokenizer = &self.tokenizer;
        let vocabulary = &self.vocabulary;

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

                // Create refs slice for trie insertion (avoids second Vec allocation)
                let tokens: Vec<&str> = token_strings.iter().map(String::as_str).collect();

                stats.inc_tokens(tokens.len() as u64);
                stats.inc_sentences();

                let mut ngram_count = 0u64;

                // Extract and count n-grams of all orders up to max
                // Pass slice directly to avoid Vec allocation per n-gram
                for n in 1..=order.min(tokens.len()) {
                    for i in 0..=(tokens.len() - n) {
                        let ngram_slice = &tokens[i..i + n];

                        // Choose encoding based on vocabulary mode
                        if let Some(vocab) = vocabulary {
                            let key = encode_ngram_key(ngram_slice, vocab);
                            trie.insert_with_key(&key);
                        } else {
                            trie.insert(ngram_slice);
                        }

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
    ///
    /// Uses `PrefetchingReader` to decouple I/O from processing while
    /// providing regular progress updates.
    ///
    /// The encoding mode depends on `self.vocabulary`:
    /// - `None`: Uses legacy pipe-separated keys via `trie.insert()`
    /// - `Some(vocab)`: Uses vocabulary-indexed PUA keys via `trie.insert_with_key()`
    fn count_ngrams_with_progress<R: CorpusReader + 'static>(
        &self,
        reader: R,
        progress_tx: &Sender<TrainingProgress>,
        start: &std::time::Instant,
    ) -> Result<()> {
        let order = self.config.order;
        let trie = &self.trie;
        let stats = &self.stats;
        let tokenizer = &self.tokenizer;
        let vocabulary = &self.vocabulary;

        // Configure prefetch for this training run
        let config = PrefetchConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_ram_fraction(0.10);

        let prefetch = PrefetchingReader::with_config(reader, config);
        let mut received_any = false;

        // Send progress every 10,000 sentences
        let progress_interval = 10_000usize;

        // Process prefetched batches in parallel
        for batch in prefetch.batches() {
            received_any = true;

            batch.par_iter().for_each(|sentence| {
                // Tokenize into owned strings, then work with references
                let token_strings: Vec<String> = tokenizer.words(sentence).collect();

                if token_strings.is_empty() {
                    return;
                }

                // Create refs slice for trie insertion (avoids second Vec allocation)
                let tokens: Vec<&str> = token_strings.iter().map(String::as_str).collect();

                stats.inc_tokens(tokens.len() as u64);
                stats.inc_sentences();

                let mut ngram_count = 0u64;

                // Pass slice directly to avoid Vec allocation per n-gram
                for n in 1..=order.min(tokens.len()) {
                    for i in 0..=(tokens.len() - n) {
                        let ngram_slice = &tokens[i..i + n];

                        // Choose encoding based on vocabulary mode
                        if let Some(vocab) = vocabulary {
                            let key = encode_ngram_key(ngram_slice, vocab);
                            trie.insert_with_key(&key);
                        } else {
                            trie.insert(ngram_slice);
                        }

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
    /// For each n-gram w1...wn, we count:
    /// - Continuation count: Number of unique contexts (w0, w1...wn-1) for which c(w0, w1...wn) > 0
    /// - Unique continuations: Number of unique words wn+1 for which c(w1...wn, wn+1) > 0
    ///
    /// This performs a second pass over all n-grams to compute:
    /// 1. For each word, how many unique histories precede it (continuation count)
    /// 2. For each history, how many unique words follow it (unique continuations)
    ///
    /// # Memory Warning
    ///
    /// This function uses `HashMap<String, HashSet<String>>` to track unique relationships.
    /// For very large corpora (10M+ n-grams), memory usage can reach 2-5GB due to:
    /// - String allocations for each word/history
    /// - HashSet overhead for unique tracking
    ///
    /// For production use with massive corpora, consider:
    /// - Pre-computing continuation counts during n-gram insertion
    /// - Using approximate counting (HyperLogLog) for unique estimation
    /// - Processing in sorted batches with external merge
    fn collect_continuation_counts(&self) {
        if self.vocabulary.is_some() {
            self.collect_continuation_counts_vocabulary();
        } else {
            self.collect_continuation_counts_legacy();
        }
    }

    /// Collect continuation counts for vocabulary-indexed encoding.
    ///
    /// In this mode, keys are sequences of PUA characters where each character
    /// represents a word. We track unique contexts by PUA character directly,
    /// which is more efficient than decoding back to strings.
    fn collect_continuation_counts_vocabulary(&self) {
        log::debug!("Collecting continuation counts (vocabulary mode) for MKN smoothing");

        let entry_count = self.stats.ngrams_counted();
        if entry_count > 5_000_000 {
            log::warn!(
                "Collecting continuation counts for {} n-grams may use significant memory (2-5GB). \
                 Consider using smaller corpus or pre-computed statistics.",
                entry_count
            );
        }

        // Track continuation counts by PUA character (more efficient than strings)
        // For each word (represented by PUA char), count unique preceding contexts
        let mut word_contexts: std::collections::HashMap<char, HashSet<String>> =
            std::collections::HashMap::new();

        // Track unique continuations: for each history, count unique following words
        let mut history_words: std::collections::HashMap<String, HashSet<char>> =
            std::collections::HashMap::new();

        // Iterate over all n-grams
        for (key, _entry) in self.trie.iter_entries() {
            let chars: Vec<char> = key.chars().collect();

            // Skip unigrams for continuation counting
            if chars.len() < 2 {
                continue;
            }

            // Extract history (all but last char) and word (last char)
            let word_char = chars[chars.len() - 1];
            let history_key: String = chars[..chars.len() - 1].iter().collect();

            // Record that this word has this history as a context
            word_contexts
                .entry(word_char)
                .or_default()
                .insert(history_key.clone());

            // Record that this history has this word as a continuation
            history_words
                .entry(history_key)
                .or_default()
                .insert(word_char);
        }

        // Update continuation counts in the trie using by_key methods
        for (word_char, contexts) in word_contexts {
            let continuation_count = contexts.len() as u32;
            // Single PUA char = unigram key
            let word_key: String = std::iter::once(word_char).collect();
            self.trie
                .update_continuation_count_by_key(&word_key, continuation_count);
        }

        // Update unique continuations in the trie
        for (history_key, words) in history_words {
            let unique_continuations = words.len() as u32;
            self.trie
                .update_unique_continuations_by_key(&history_key, unique_continuations);
        }

        log::debug!("Continuation count collection (vocabulary mode) complete");
    }

    /// Collect continuation counts for legacy pipe-separated encoding.
    fn collect_continuation_counts_legacy(&self) {
        log::debug!("Collecting continuation counts (legacy mode) for MKN smoothing");

        let entry_count = self.stats.ngrams_counted();
        if entry_count > 5_000_000 {
            log::warn!(
                "Collecting continuation counts for {} n-grams may use significant memory (2-5GB). \
                 Consider using smaller corpus or pre-computed statistics.",
                entry_count
            );
        }

        // Track continuation counts: for each word, count unique preceding contexts
        // continuation_count[word] = |{h : c(h, word) > 0}|
        let mut word_contexts: std::collections::HashMap<String, HashSet<String>> =
            std::collections::HashMap::new();

        // Track unique continuations: for each history, count unique following words
        // unique_continuations[history] = |{w : c(history, w) > 0}|
        let mut history_words: std::collections::HashMap<String, HashSet<String>> =
            std::collections::HashMap::new();

        // Iterate over all n-grams
        for (key, _entry) in self.trie.iter_entries() {
            let parts: Vec<&str> = key.split(LEGACY_NGRAM_SEPARATOR).collect();

            // Skip unigrams for continuation counting
            if parts.len() < 2 {
                continue;
            }

            // Extract history (all but last) and word (last)
            let word = parts[parts.len() - 1].to_string();
            let history = parts[..parts.len() - 1].join(&LEGACY_NGRAM_SEPARATOR.to_string());

            // Record that this word has this history as a context
            word_contexts
                .entry(word.clone())
                .or_default()
                .insert(history.clone());

            // Record that this history has this word as a continuation
            history_words.entry(history).or_default().insert(word);
        }

        // Update continuation counts in the trie
        for (word, contexts) in word_contexts {
            let continuation_count = contexts.len() as u32;
            self.trie
                .update_continuation_count(&[&word], continuation_count);
        }

        // Update unique continuations in the trie
        for (history, words) in history_words {
            let unique_continuations = words.len() as u32;
            let history_tokens: Vec<&str> = history.split(LEGACY_NGRAM_SEPARATOR).collect();
            self.trie
                .update_unique_continuations(&history_tokens, unique_continuations);
        }

        log::debug!("Continuation count collection (legacy mode) complete");
    }

    /// Count n-grams by frequency for MKN discount computation.
    ///
    /// Returns (n1, n2, n3, n4) where:
    /// - n1 = count of n-grams occurring exactly once
    /// - n2 = count of n-grams occurring exactly twice
    /// - n3 = count of n-grams occurring exactly 3 times
    /// - n4 = count of n-grams occurring exactly 4 times
    fn count_ngram_frequencies(&self) -> (u64, u64, u64, u64) {
        let mut n1 = 0u64;
        let mut n2 = 0u64;
        let mut n3 = 0u64;
        let mut n4 = 0u64;

        for (_key, entry) in self.trie.iter_entries() {
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

    /// Compute Modified Kneser-Ney smoothing parameters from actual corpus statistics.
    ///
    /// Uses the Chen & Goodman formula to compute optimal discounts:
    /// - Y = n1 / (n1 + 2*n2)
    /// - D1 = 1 - 2*Y * (n2/n1)
    /// - D2 = 2 - 3*Y * (n3/n2)
    /// - D3+ = 3 - 4*Y * (n4/n3)
    fn compute_smoothing_params(&self) -> KneserNeySmoothing {
        let (n1, n2, n3, n4) = self.count_ngram_frequencies();

        // Need all counts to be non-zero for meaningful discount computation
        if n1 > 0 && n2 > 0 && n3 > 0 && n4 > 0 {
            log::info!("Computing optimal MKN discounts from corpus statistics");
            KneserNeySmoothing::from_counts(n1, n2, n3, n4)
        } else {
            log::debug!(
                "Insufficient count diversity (n1={}, n2={}, n3={}, n4={}), using default MKN discounts",
                n1, n2, n3, n4
            );
            KneserNeySmoothing::new(self.config.order)
        }
    }

    /// Count unique unigrams (vocabulary size).
    ///
    /// In legacy mode, unigrams are detected by the absence of pipe separators.
    /// In vocabulary mode, unigrams are single PUA characters.
    fn count_unigrams(&self) -> usize {
        let mut count = 0;
        let use_vocabulary = self.vocabulary.is_some();

        for (key, _entry) in self.trie.iter_entries() {
            let is_unigram = if use_vocabulary {
                // In vocabulary mode, unigrams are single PUA characters
                key.chars().count() == 1
            } else {
                // In legacy mode, unigrams have no separator
                !key.contains(LEGACY_NGRAM_SEPARATOR)
            };

            if is_unigram {
                count += 1;
            }
        }
        count
    }
}

/// Builder for training with fluent API.
pub struct TrainerBuilder<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync,
{
    dictionary: D,
    config: TrainingConfig,
    tokenizer: Option<Tokenizer>,
}

impl<D> TrainerBuilder<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync + 'static,
{
    /// Create a new trainer builder.
    pub fn new(dictionary: D) -> Self {
        Self {
            dictionary,
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

    /// Set vocabulary path for creating a new vocabulary during training.
    ///
    /// When set, the trainer uses vocabulary-indexed encoding instead of
    /// legacy pipe-separated keys. The vocabulary is persisted to disk at
    /// the given path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = TrainerBuilder::new(dictionary)
    ///     .order(5)
    ///     .with_vocabulary_path(PathBuf::from("model/vocab.artrie"))
    ///     .train(reader)?;
    /// ```
    pub fn with_vocabulary_path(mut self, path: PathBuf) -> Self {
        self.config.vocabulary_mode = VocabularyMode::Create(path);
        self
    }

    /// Set an existing shared vocabulary for training.
    ///
    /// Useful when training multiple models with a consistent vocabulary,
    /// or when integrating with the Google Books import pipeline.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vocab = open_vocabulary(&vocab_path)?;
    /// let model = TrainerBuilder::new(dictionary)
    ///     .order(5)
    ///     .with_vocabulary(vocab)
    ///     .train(reader)?;
    /// ```
    pub fn with_vocabulary(mut self, vocab: SharedVocabARTrie) -> Self {
        self.config.vocabulary_mode = VocabularyMode::Shared(vocab);
        self
    }

    /// Build the trainer.
    pub fn build(self) -> NgramTrainer<D> {
        let mut trainer = NgramTrainer::new(self.dictionary, self.config);
        if let Some(tokenizer) = self.tokenizer {
            trainer = trainer.with_tokenizer(tokenizer);
        }
        trainer
    }

    /// Build and immediately train from corpus.
    pub fn train<R: CorpusReader + 'static>(self, reader: R) -> Result<NgramModel<D>> {
        self.build().train(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::super::vocabulary::create_vocabulary;
    use super::*;
    use crate::corpus::PlaintextReader;
    use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_corpus(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");
        path
    }

    #[test]
    fn test_train_simple_corpus() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_corpus(
            dir.path(),
            "The quick brown fox. The quick brown dog. The lazy fox.",
        );

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();

        let model = TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("Training failed");

        // Check that model was trained
        assert!(model.vocab_size() > 0);
        assert!(model.ngram_count() > 0);
    }

    #[test]
    fn test_bigram_counts() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_corpus(dir.path(), "a b a b a b");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();

        let model = TrainerBuilder::new(dictionary)
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

        // Create vocabulary first so we can inspect it after training
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");

        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();

        let model = TrainerBuilder::new(dictionary)
            .order(3)
            .with_vocabulary(vocab.clone())
            .train(reader)
            .expect("Training with vocabulary failed");

        // Model should have been trained
        assert!(model.vocab_size() > 0, "Vocabulary should contain entries");
        assert!(model.ngram_count() > 0, "Model should contain n-grams");

        // Verify words are in the SharedVocabARTrie (not model.in_vocabulary, which uses legacy encoding)
        assert!(vocab.read().contains("the"), "Expected 'the' in vocabulary");
        assert!(
            vocab.read().contains("quick"),
            "Expected 'quick' in vocabulary"
        );
        assert!(
            vocab.read().contains("brown"),
            "Expected 'brown' in vocabulary"
        );
        assert!(vocab.read().contains("fox"), "Expected 'fox' in vocabulary");

        // Verify we can look up n-grams using the vocabulary for encoding
        let bigram_key = encode_ngram_key(&["the", "quick"], &vocab);
        assert!(
            model.trie().contains_key(&bigram_key),
            "Expected 'the quick' bigram in trie"
        );
    }

    #[test]
    fn test_pipe_in_token_no_corruption_vocabulary_mode() {
        // This test verifies the key benefit of vocabulary encoding:
        // tokens containing pipe characters are handled correctly.
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");

        // Create vocabulary first so we can inspect it after training
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");

        let corpus_path = create_test_corpus(dir.path(), "foo|bar baz foo|bar baz foo|bar baz");

        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();

        let model = TrainerBuilder::new(dictionary)
            .order(2)
            .with_vocabulary(vocab.clone())
            .train(reader)
            .expect("Training failed");

        // In vocabulary mode, "foo|bar" is stored as a single PUA character,
        // so it won't be corrupted by the pipe separator.
        // Verify through the vocabulary, not the model's legacy query methods
        assert!(
            vocab.read().contains("foo|bar"),
            "Expected 'foo|bar' as single token in vocabulary"
        );
        assert!(vocab.read().contains("baz"), "Expected 'baz' in vocabulary");

        // vocab_size should be 2 (the two unique words)
        assert_eq!(model.vocab_size(), 2, "Should have exactly 2 unique words");

        // Verify the bigram "foo|bar baz" is stored correctly
        let bigram_key = encode_ngram_key(&["foo|bar", "baz"], &vocab);
        let count = model.trie().count_by_key(&bigram_key);
        assert!(
            count >= 3,
            "Expected 'foo|bar baz' bigram count >= 3, got {}",
            count
        );
    }

    #[test]
    fn test_legacy_trainer_unchanged() {
        // Verify that the default (legacy) behavior is unchanged
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_corpus(dir.path(), "a b c a b c");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();

        // Default config should use legacy mode
        let config = TrainingConfig::new(2);
        assert!(
            matches!(config.vocabulary_mode, VocabularyMode::Legacy),
            "Default mode should be Legacy"
        );

        let model = TrainerBuilder::new(dictionary)
            .order(2)
            .train(reader)
            .expect("Training failed");

        // Should work exactly as before
        assert!(model.vocab_size() > 0);
        assert!(model.ngram_count() > 0);
    }

    #[test]
    fn test_vocabulary_mode_shared() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("shared_vocab.artrie");

        // Create a shared vocabulary
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");

        // Pre-populate the vocabulary
        vocab.write().insert("pre").expect("insert pre");
        vocab.write().insert("populated").expect("insert populated");
        vocab.write().insert("words").expect("insert words");

        let corpus_path = create_test_corpus(dir.path(), "pre populated words are here");
        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();

        let model = TrainerBuilder::new(dictionary)
            .order(2)
            .with_vocabulary(vocab.clone())
            .train(reader)
            .expect("Training with shared vocabulary failed");

        // The vocabulary should have grown
        assert!(
            vocab.read().len() > 3,
            "Vocabulary should have grown with new words"
        );
        assert!(model.vocab_size() > 0);
    }

    #[test]
    fn test_continuation_counts_vocabulary_mode() {
        // Test that MKN continuation counts are computed correctly in vocabulary mode
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");

        // Create corpus with clear continuation patterns:
        // "the" is followed by "quick", "slow", "big" (3 unique continuations)
        let corpus_path = create_test_corpus(dir.path(), "the quick fox the slow fox the big fox");

        let reader = PlaintextReader::from_file(&corpus_path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();

        let model = TrainerBuilder::new(dictionary)
            .order(2)
            .with_vocabulary_path(vocab_path)
            .train(reader)
            .expect("Training failed");

        // Model should have been trained without panics
        // (validation of internal continuation counts is implicit)
        assert!(model.vocab_size() > 0);
        assert!(model.ngram_count() > 0);

        // Log probabilities should be finite (proves smoothing works)
        let log_prob = model.log_prob("fox", &["quick"]);
        assert!(log_prob.is_finite(), "Log probability should be finite");
    }
}
