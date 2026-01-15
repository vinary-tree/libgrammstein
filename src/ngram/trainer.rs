//! N-gram model training with parallel corpus processing.
//!
//! This module provides the training pipeline for n-gram language models:
//! - Streaming corpus reading
//! - Parallel n-gram counting with Rayon
//! - Continuation count collection for Modified Kneser-Ney
//! - Progress reporting

use crate::corpus::{CorpusReader, PrefetchConfig, PrefetchingReader, Tokenizer};
use crate::ngram::smoothing::KneserNeySmoothing;
use crate::ngram::trie::{IterableDictionary, NGRAM_SEPARATOR};
use crate::ngram::{NgramEntry, NgramModel, NgramTrie};
use crate::Result;

use crossbeam_channel::Sender;
use liblevenshtein::dictionary::MutableMappedDictionary;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

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

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Maximum n-gram order (e.g., 5 for 5-grams).
    pub order: usize,

    /// Batch size for parallel processing.
    pub batch_size: usize,

    /// Minimum word frequency to include in vocabulary.
    pub min_word_freq: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            order: 5,
            batch_size: 10_000,
            min_word_freq: 1,
        }
    }
}

impl TrainingConfig {
    /// Create a new training configuration.
    pub fn new(order: usize) -> Self {
        Self {
            order,
            ..Default::default()
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
    pub fn new(dictionary: D, config: TrainingConfig) -> Self {
        let order = config.order;
        Self {
            trie: NgramTrie::new(dictionary, order),
            config,
            stats: TrainingStats::default(),
            tokenizer: Tokenizer::new(),
        }
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
    fn count_ngrams<R: CorpusReader + 'static>(&self, reader: R) -> Result<()> {
        let order = self.config.order;
        let trie = &self.trie;
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

                // Create refs slice for trie insertion (avoids second Vec allocation)
                let tokens: Vec<&str> = token_strings.iter().map(String::as_str).collect();

                stats.inc_tokens(tokens.len() as u64);
                stats.inc_sentences();

                let mut ngram_count = 0u64;

                // Extract and count n-grams of all orders up to max
                // Pass slice directly to avoid Vec allocation per n-gram
                for n in 1..=order.min(tokens.len()) {
                    for i in 0..=(tokens.len() - n) {
                        trie.insert(&tokens[i..i + n]);
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
                        trie.insert(&tokens[i..i + n]);
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
        log::debug!("Collecting continuation counts for MKN smoothing");

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
            let parts: Vec<&str> = key.split(NGRAM_SEPARATOR).collect();

            // Skip unigrams for continuation counting
            if parts.len() < 2 {
                continue;
            }

            // Extract history (all but last) and word (last)
            let word = parts[parts.len() - 1].to_string();
            let history = parts[..parts.len() - 1].join(&NGRAM_SEPARATOR.to_string());

            // Record that this word has this history as a context
            word_contexts
                .entry(word.clone())
                .or_default()
                .insert(history.clone());

            // Record that this history has this word as a continuation
            history_words
                .entry(history)
                .or_default()
                .insert(word);
        }

        // Update continuation counts in the trie
        for (word, contexts) in word_contexts {
            let continuation_count = contexts.len() as u32;
            self.trie.update_continuation_count(&[&word], continuation_count);
        }

        // Update unique continuations in the trie
        for (history, words) in history_words {
            let unique_continuations = words.len() as u32;
            let history_tokens: Vec<&str> = history.split(NGRAM_SEPARATOR).collect();
            self.trie.update_unique_continuations(&history_tokens, unique_continuations);
        }

        log::debug!("Continuation count collection complete");
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
            n1, n2, n3, n4
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
    fn count_unigrams(&self) -> usize {
        // Count entries that are unigrams (no separator in key)
        let mut count = 0;
        for (key, _entry) in self.trie.iter_entries() {
            if !key.contains(NGRAM_SEPARATOR) {
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
}
