//! Skip-gram training with negative sampling for subword embeddings.
//!
//! Implements FastText-style training:
//! - Skip-gram objective: Predict context words from center word
//! - Negative sampling: Efficient approximation of softmax
//! - Subword integration: Updates both word and subword embeddings
//! - Parallel training with Rayon

use super::bpe::{extract_subwords, hash_subword};
use super::model::{SubwordEmbedding, DEFAULT_BUCKET_COUNT, DEFAULT_EMBEDDING_DIM};
use crate::corpus::{CorpusReader, PrefetchConfig, PrefetchingReader};
use crate::Result;

use crossbeam_channel::Sender;
use ndarray::Array1;
use rand::prelude::*;

#[allow(unused_imports)] // Will be used for parallel training
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Vocabulary terms, per-term frequency counts, total token count, and the
/// collected sentences — the four outputs of the vocabulary-building pass,
/// reused across the subsequent multi-pass training epochs.
type VocabularyAndSentences = (Vec<String>, Vec<u64>, u64, Vec<String>);

/// Training configuration for embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Embedding dimension.
    pub dim: usize,

    /// Context window size (words on each side).
    pub window_size: usize,

    /// Minimum word frequency to include in vocabulary.
    pub min_count: u64,

    /// Number of negative samples per positive sample.
    pub neg_samples: usize,

    /// Number of training epochs.
    pub epochs: usize,

    /// Initial learning rate.
    pub learning_rate: f32,

    /// Number of subword hash buckets.
    pub bucket_count: usize,

    /// Minimum subword (character n-gram) length.
    pub min_subword_len: usize,

    /// Maximum subword (character n-gram) length.
    pub max_subword_len: usize,

    /// Subsampling threshold for frequent words.
    pub subsample_threshold: f32,

    /// Batch size for parallel processing.
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dim: DEFAULT_EMBEDDING_DIM,
            window_size: 5,
            min_count: 5,
            neg_samples: 5,
            epochs: 5,
            learning_rate: 0.05,
            bucket_count: DEFAULT_BUCKET_COUNT,
            min_subword_len: 3,
            max_subword_len: 6,
            subsample_threshold: 1e-4,
            batch_size: 10_000,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new configuration with specified dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    /// Set window size.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set minimum word count.
    pub fn with_min_count(mut self, count: u64) -> Self {
        self.min_count = count;
        self
    }

    /// Set number of negative samples.
    pub fn with_neg_samples(mut self, n: usize) -> Self {
        self.neg_samples = n;
        self
    }

    /// Set number of epochs.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
}

/// Training progress information.
#[derive(Debug, Clone)]
pub struct EmbeddingProgress {
    /// Current epoch.
    pub epoch: usize,

    /// Words processed in current epoch.
    pub words_processed: u64,

    /// Total words in corpus.
    pub total_words: u64,

    /// Current learning rate.
    pub learning_rate: f32,

    /// Training loss (if computed).
    pub loss: Option<f32>,
}

/// Training statistics with atomic counters.
#[derive(Default)]
struct TrainingStats {
    words_processed: AtomicU64,
    examples_processed: AtomicU64,
}

/// Negative sampling distribution.
///
/// Uses unigram distribution raised to the 3/4 power for sampling.
struct NegativeSampler {
    /// Sampling table for O(1) lookups.
    table: Vec<usize>,

    /// Table size.
    table_size: usize,
}

impl NegativeSampler {
    /// Create a new sampler from word frequencies.
    fn new(word_counts: &[u64], table_size: usize) -> Self {
        let mut table = Vec::with_capacity(table_size);

        // Compute smoothed frequencies (count^0.75)
        let smoothed: Vec<f64> = word_counts.iter().map(|&c| (c as f64).powf(0.75)).collect();
        let total: f64 = smoothed.iter().sum();

        // Fill table proportionally
        for (idx, &freq) in smoothed.iter().enumerate() {
            let proportion = freq / total;
            let slots = (proportion * table_size as f64) as usize;
            for _ in 0..slots.max(1) {
                if table.len() < table_size {
                    table.push(idx);
                }
            }
        }

        // Fill any remaining slots
        while table.len() < table_size {
            table.push(table.len() % word_counts.len());
        }

        Self { table, table_size }
    }

    /// Sample a negative word index.
    #[inline]
    fn sample<R: Rng>(&self, rng: &mut R) -> usize {
        let idx = rng.gen_range(0..self.table_size);
        self.table[idx]
    }

    /// Sample multiple negative word indices, excluding a specific word.
    fn sample_negatives<R: Rng>(&self, rng: &mut R, count: usize, exclude: usize) -> Vec<usize> {
        let mut negatives = Vec::with_capacity(count);
        while negatives.len() < count {
            let neg = self.sample(rng);
            if neg != exclude {
                negatives.push(neg);
            }
        }
        negatives
    }
}

/// Embedding trainer using skip-gram with negative sampling.
pub struct EmbeddingTrainer {
    /// Training configuration.
    config: EmbeddingConfig,
}

impl EmbeddingTrainer {
    /// Create a new trainer with configuration.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }

    /// Train embeddings from corpus using multi-pass streaming (memory-efficient).
    ///
    /// This method uses streaming to avoid loading the entire corpus into memory:
    /// - Pass 1: Build vocabulary (streaming word counts)
    /// - Pass 2+: For each epoch, re-stream the corpus for training
    ///
    /// # Arguments
    ///
    /// * `reader_factory` - A closure that creates a new CorpusReader for each pass
    ///
    /// # Returns
    ///
    /// Trained `SubwordEmbedding` model.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let trainer = EmbeddingTrainer::new(config);
    /// let path = Path::new("corpus.txt");
    /// let model = trainer.train_streaming(|| PlaintextReader::from_file(&path))?;
    /// ```
    pub fn train_streaming<F, R>(&self, reader_factory: F) -> Result<SubwordEmbedding>
    where
        F: Fn() -> Result<R>,
        R: CorpusReader + 'static,
    {
        // Phase 1: Build vocabulary with streaming (no sentence collection)
        log::info!("Building vocabulary (streaming pass 1)...");
        let (vocab, word_counts, total_words) =
            self.build_vocabulary_streaming(reader_factory()?)?;

        log::info!(
            "Vocabulary: {} words, {} total tokens",
            vocab.len(),
            total_words
        );

        // Phase 2: Initialize model
        let mut model =
            SubwordEmbedding::new(vocab.clone(), self.config.dim, self.config.bucket_count)
                .with_subword_range(self.config.min_subword_len, self.config.max_subword_len);

        self.initialize_embeddings(&mut model);

        // Phase 3: Create negative sampler
        let sampler = Arc::new(NegativeSampler::new(&word_counts, 10_000_000));

        // Phase 4: Train epochs with streaming (one pass per epoch)
        log::info!("Training {} epochs (streaming)...", self.config.epochs);
        for epoch in 0..self.config.epochs {
            let lr = self.config.learning_rate * (1.0 - epoch as f32 / self.config.epochs as f32);
            log::info!("Epoch {}/{}, lr={:.6}", epoch + 1, self.config.epochs, lr);

            // Create fresh reader for this epoch
            let reader = reader_factory()?;
            self.train_epoch_streaming(
                reader,
                &mut model,
                &vocab,
                &word_counts,
                total_words,
                &sampler,
                lr,
            )?;
        }

        Ok(model)
    }

    /// Build vocabulary using streaming (without collecting sentences).
    fn build_vocabulary_streaming<R: CorpusReader + 'static>(
        &self,
        reader: R,
    ) -> Result<(Vec<String>, Vec<u64>, u64)> {
        let mut word_counts: HashMap<String, u64> = HashMap::new();
        let mut total_words = 0u64;

        let config = PrefetchConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_ram_fraction(0.10);

        let prefetch = PrefetchingReader::with_config(reader, config);

        // Stream through corpus, counting words only (no sentence collection)
        for batch in prefetch.batches() {
            for sentence in batch {
                for word in sentence.split_whitespace() {
                    let word = word.to_lowercase();
                    *word_counts.entry(word).or_insert(0) += 1;
                    total_words += 1;
                }
            }
        }

        // Filter by minimum count and sort by frequency
        let mut vocab_entries: Vec<(String, u64)> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.config.min_count)
            .collect();

        vocab_entries.sort_by_key(|e| std::cmp::Reverse(e.1));

        let vocab: Vec<String> = vocab_entries.iter().map(|(w, _)| w.clone()).collect();
        let counts: Vec<u64> = vocab_entries.iter().map(|(_, c)| *c).collect();

        if vocab.is_empty() {
            return Err(crate::Error::EmptyCorpus);
        }

        Ok((vocab, counts, total_words))
    }

    /// Train a single epoch using streaming (memory-efficient).
    // Training kernel: the mutable model, the immutable corpus statistics
    // (vocabulary, counts, totals), the shared sampler, and the learning rate
    // are irreducible inputs; bundling them into a struct adds no clarity.
    #[allow(clippy::too_many_arguments)]
    fn train_epoch_streaming<R: CorpusReader + 'static>(
        &self,
        reader: R,
        model: &mut SubwordEmbedding,
        vocab: &[String],
        word_counts: &[u64],
        total_words: u64,
        sampler: &Arc<NegativeSampler>,
        lr: f32,
    ) -> Result<()> {
        let word_to_idx: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        let config = PrefetchConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_ram_fraction(0.10);

        let prefetch = PrefetchingReader::with_config(reader, config);
        let subsample_threshold = self.config.subsample_threshold;
        let stats = TrainingStats::default();

        for batch in prefetch.batches() {
            for sentence in batch {
                let words: Vec<&str> = sentence.split_whitespace().collect();
                let word_indices: Vec<Option<usize>> = words
                    .iter()
                    .map(|w| word_to_idx.get(w.to_lowercase().as_str()).copied())
                    .collect();

                for (pos, &opt_center_idx) in word_indices.iter().enumerate() {
                    let center_idx = match opt_center_idx {
                        Some(idx) => idx,
                        None => continue,
                    };

                    // Subsampling for frequent words
                    let center_count = word_counts[center_idx];
                    let freq = center_count as f32 / total_words as f32;
                    let keep_prob =
                        ((freq / subsample_threshold).sqrt() + 1.0) * (subsample_threshold / freq);

                    let mut rng = thread_rng();
                    if rng.gen::<f32>() > keep_prob {
                        continue;
                    }

                    // Dynamic window size
                    let window = rng.gen_range(1..=self.config.window_size);

                    // Get context words
                    let start = pos.saturating_sub(window);
                    let end = (pos + window + 1).min(words.len());

                    for (ctx_pos, ctx_slot) in word_indices.iter().enumerate().take(end).skip(start)
                    {
                        if ctx_pos == pos {
                            continue;
                        }

                        let ctx_idx = match *ctx_slot {
                            Some(idx) => idx,
                            None => continue,
                        };

                        // Skip-gram update
                        self.skipgram_update(
                            model,
                            center_idx,
                            ctx_idx,
                            &words[pos].to_lowercase(),
                            sampler,
                            lr,
                            &mut rng,
                        );

                        stats.examples_processed.fetch_add(1, Ordering::Relaxed);
                    }

                    stats.words_processed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        log::info!(
            "  Words processed: {}",
            stats.words_processed.load(Ordering::Relaxed)
        );

        Ok(())
    }

    /// Train embeddings from corpus (legacy method - collects sentences into memory).
    ///
    /// **Warning**: This method buffers all sentences into memory. For large corpora
    /// (>500MB), use [`train_streaming`](Self::train_streaming) instead to avoid OOM.
    ///
    /// # Arguments
    ///
    /// * `reader` - Corpus reader providing sentences (takes ownership)
    ///
    /// # Returns
    ///
    /// Trained `SubwordEmbedding` model.
    pub fn train<R: CorpusReader + 'static>(&self, reader: R) -> Result<SubwordEmbedding> {
        // Phase 1: Build vocabulary with prefetched streaming
        log::info!("Building vocabulary...");
        let (vocab, word_counts, total_words, sentences) =
            self.build_vocabulary_and_collect(reader)?;

        if sentences.len() > 1_000_000 {
            log::warn!(
                "Collected {} sentences into memory. For large corpora, use train_streaming() instead.",
                sentences.len()
            );
        }

        log::info!(
            "Vocabulary: {} words, {} total tokens, {} sentences",
            vocab.len(),
            total_words,
            sentences.len()
        );

        // Phase 2: Initialize model
        let mut model =
            SubwordEmbedding::new(vocab.clone(), self.config.dim, self.config.bucket_count)
                .with_subword_range(self.config.min_subword_len, self.config.max_subword_len);

        // Initialize embeddings with small random values
        self.initialize_embeddings(&mut model);

        // Phase 3: Create negative sampler
        let sampler = Arc::new(NegativeSampler::new(&word_counts, 10_000_000));

        // Phase 4: Train on collected sentences
        log::info!("Training {} epochs...", self.config.epochs);
        self.train_epochs_on_sentences(
            &sentences,
            &mut model,
            &vocab,
            &word_counts,
            total_words,
            &sampler,
            0,
        )?;

        Ok(model)
    }

    /// Train with progress reporting.
    pub fn train_with_progress<R: CorpusReader + 'static>(
        &self,
        reader: R,
        progress_tx: Sender<EmbeddingProgress>,
    ) -> Result<SubwordEmbedding> {
        // Phase 1: Build vocabulary with prefetched streaming
        let (vocab, word_counts, total_words, sentences) =
            self.build_vocabulary_and_collect(reader)?;

        // Phase 2: Initialize model
        let mut model =
            SubwordEmbedding::new(vocab.clone(), self.config.dim, self.config.bucket_count)
                .with_subword_range(self.config.min_subword_len, self.config.max_subword_len);

        self.initialize_embeddings(&mut model);

        // Phase 3: Create negative sampler
        let sampler = Arc::new(NegativeSampler::new(&word_counts, 10_000_000));

        // Phase 4: Train with progress
        self.train_epochs_on_sentences_with_progress(
            &sentences,
            &mut model,
            &vocab,
            &word_counts,
            total_words,
            &sampler,
            &progress_tx,
        )?;

        Ok(model)
    }

    /// Build vocabulary from corpus using prefetched streaming.
    ///
    /// Returns (vocabulary, word_counts, total_words, collected_sentences).
    /// Sentences are collected for subsequent multi-pass training.
    fn build_vocabulary_and_collect<R: CorpusReader + 'static>(
        &self,
        reader: R,
    ) -> Result<VocabularyAndSentences> {
        let mut word_counts: HashMap<String, u64> = HashMap::new();
        let mut total_words = 0u64;
        let mut sentences = Vec::new();

        // Configure prefetch for vocabulary building
        let config = PrefetchConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_ram_fraction(0.10);

        let prefetch = PrefetchingReader::with_config(reader, config);

        // Count words and collect sentences
        for batch in prefetch.batches() {
            for sentence in batch {
                for word in sentence.split_whitespace() {
                    let word = word.to_lowercase();
                    *word_counts.entry(word).or_insert(0) += 1;
                    total_words += 1;
                }
                sentences.push(sentence);
            }
        }

        // Filter by minimum count and sort by frequency
        let mut vocab_entries: Vec<(String, u64)> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.config.min_count)
            .collect();

        vocab_entries.sort_by_key(|e| std::cmp::Reverse(e.1));

        let vocab: Vec<String> = vocab_entries.iter().map(|(w, _)| w.clone()).collect();
        let counts: Vec<u64> = vocab_entries.iter().map(|(_, c)| *c).collect();

        if vocab.is_empty() {
            return Err(crate::Error::EmptyCorpus);
        }

        Ok((vocab, counts, total_words, sentences))
    }

    /// Continue training a previously-trained model for the remaining epochs.
    ///
    /// Unlike [`train`](Self::train), this does **not** rebuild the vocabulary or
    /// re-initialize the embeddings — it keeps the loaded model's weights and vocab.
    /// Checkpoints persist weights + vocabulary but not the ephemeral corpus statistics
    /// (the negative-sampling table and sub-sampling counts), so those are recovered with
    /// one corpus pass, aligned to the model's existing vocabulary to keep embedding-row
    /// indices consistent. Training then runs epochs `start_epoch .. epochs`, so the
    /// learning-rate schedule resumes at the correct point.
    pub fn train_continued<R: CorpusReader + 'static>(
        &self,
        mut model: SubwordEmbedding,
        start_epoch: usize,
        reader: R,
    ) -> Result<SubwordEmbedding> {
        if start_epoch >= self.config.epochs {
            log::info!(
                "Resume requested at epoch {} but the model already has {} epochs; nothing to do.",
                start_epoch,
                self.config.epochs
            );
            return Ok(model);
        }

        let vocab: Vec<String> = model.vocab().to_vec();
        log::info!(
            "Resuming training from epoch {} to {} ({} vocab words)...",
            start_epoch,
            self.config.epochs,
            vocab.len()
        );

        // Recover corpus statistics aligned to the model's existing vocabulary.
        let (word_counts, total_words, sentences) =
            self.collect_aligned_counts_and_sentences(reader, &vocab)?;

        let sampler = Arc::new(NegativeSampler::new(&word_counts, 10_000_000));

        self.train_epochs_on_sentences(
            &sentences,
            &mut model,
            &vocab,
            &word_counts,
            total_words,
            &sampler,
            start_epoch,
        )?;

        Ok(model)
    }

    /// Collect corpus statistics (per-word counts, total tokens, and the sentences)
    /// aligned to an existing `vocab` ordering. Words outside `vocab` count toward
    /// `total_words` (for sub-sampling) but get no per-word entry — mirroring
    /// [`build_vocabulary_and_collect`](Self::build_vocabulary_and_collect).
    fn collect_aligned_counts_and_sentences<R: CorpusReader + 'static>(
        &self,
        reader: R,
        vocab: &[String],
    ) -> Result<(Vec<u64>, u64, Vec<String>)> {
        let index_of: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        let mut counts = vec![0u64; vocab.len()];
        let mut total_words = 0u64;
        let mut sentences = Vec::new();

        let config = PrefetchConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_ram_fraction(0.10);
        let prefetch = PrefetchingReader::with_config(reader, config);

        for batch in prefetch.batches() {
            for sentence in batch {
                for word in sentence.split_whitespace() {
                    let word = word.to_lowercase();
                    total_words += 1;
                    if let Some(&i) = index_of.get(word.as_str()) {
                        counts[i] += 1;
                    }
                }
                sentences.push(sentence);
            }
        }

        Ok((counts, total_words, sentences))
    }

    /// Initialize embeddings with small random values.
    fn initialize_embeddings(&self, model: &mut SubwordEmbedding) {
        let mut rng = StdRng::seed_from_u64(42);
        let scale = 1.0 / self.config.dim as f32;

        // Initialize word embeddings
        let word_emb = model.word_embeddings_mut();
        for elem in word_emb.iter_mut() {
            *elem = (rng.gen::<f32>() - 0.5) * scale;
        }

        // Initialize subword embeddings
        let subword_emb = model.subword_embeddings_mut();
        for elem in subword_emb.iter_mut() {
            *elem = (rng.gen::<f32>() - 0.5) * scale;
        }
    }

    /// Train for multiple epochs on collected sentences.
    // Training kernel: mutable model + immutable corpus statistics + sampler +
    // starting epoch are irreducible inputs (see `train_epoch_streaming`).
    #[allow(clippy::too_many_arguments)]
    fn train_epochs_on_sentences(
        &self,
        sentences: &[String],
        model: &mut SubwordEmbedding,
        vocab: &[String],
        word_counts: &[u64],
        total_words: u64,
        sampler: &Arc<NegativeSampler>,
        start_epoch: usize,
    ) -> Result<()> {
        let word_to_idx: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        for epoch in start_epoch..self.config.epochs {
            let stats = TrainingStats::default();
            let lr = self.config.learning_rate * (1.0 - epoch as f32 / self.config.epochs as f32);

            log::info!("Epoch {}/{}, lr={:.6}", epoch + 1, self.config.epochs, lr);

            self.train_epoch_on_sentences(
                sentences,
                model,
                &word_to_idx,
                word_counts,
                total_words,
                sampler,
                lr,
                &stats,
            )?;

            log::info!(
                "  Words processed: {}",
                stats.words_processed.load(Ordering::Relaxed)
            );
        }

        Ok(())
    }

    /// Train for multiple epochs with progress reporting.
    // Training kernel: as `train_epochs_on_sentences`, plus a progress channel.
    #[allow(clippy::too_many_arguments)]
    fn train_epochs_on_sentences_with_progress(
        &self,
        sentences: &[String],
        model: &mut SubwordEmbedding,
        vocab: &[String],
        word_counts: &[u64],
        total_words: u64,
        sampler: &Arc<NegativeSampler>,
        progress_tx: &Sender<EmbeddingProgress>,
    ) -> Result<()> {
        let word_to_idx: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        for epoch in 0..self.config.epochs {
            let stats = TrainingStats::default();
            let lr = self.config.learning_rate * (1.0 - epoch as f32 / self.config.epochs as f32);

            self.train_epoch_on_sentences(
                sentences,
                model,
                &word_to_idx,
                word_counts,
                total_words,
                sampler,
                lr,
                &stats,
            )?;

            let _ = progress_tx.try_send(EmbeddingProgress {
                epoch: epoch + 1,
                words_processed: stats.words_processed.load(Ordering::Relaxed),
                total_words,
                learning_rate: lr,
                loss: None,
            });
        }

        Ok(())
    }

    /// Train a single epoch on collected sentences.
    // Training kernel: mutable model, vocabulary index, immutable corpus
    // statistics, sampler, learning rate, and per-epoch stats sink.
    #[allow(clippy::too_many_arguments)]
    fn train_epoch_on_sentences(
        &self,
        sentences: &[String],
        model: &mut SubwordEmbedding,
        word_to_idx: &HashMap<&str, usize>,
        word_counts: &[u64],
        total_words: u64,
        sampler: &Arc<NegativeSampler>,
        lr: f32,
        stats: &TrainingStats,
    ) -> Result<()> {
        let config = &self.config;
        let subsample_threshold = config.subsample_threshold;

        // Process sentences
        // Note: For full parallelism, we'd need thread-safe model updates.
        // This simplified version processes sequentially with parallel subword extraction.
        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let word_indices: Vec<Option<usize>> = words
                .iter()
                .map(|w| word_to_idx.get(w.to_lowercase().as_str()).copied())
                .collect();

            for (pos, &opt_center_idx) in word_indices.iter().enumerate() {
                let center_idx = match opt_center_idx {
                    Some(idx) => idx,
                    None => continue,
                };

                // Subsampling for frequent words
                let center_count = word_counts[center_idx];
                let freq = center_count as f32 / total_words as f32;
                let keep_prob =
                    ((freq / subsample_threshold).sqrt() + 1.0) * (subsample_threshold / freq);

                let mut rng = thread_rng();
                if rng.gen::<f32>() > keep_prob {
                    continue;
                }

                // Dynamic window size
                let window = rng.gen_range(1..=config.window_size);

                // Get context words
                let start = pos.saturating_sub(window);
                let end = (pos + window + 1).min(words.len());

                for (ctx_pos, ctx_slot) in word_indices.iter().enumerate().take(end).skip(start) {
                    if ctx_pos == pos {
                        continue;
                    }

                    let ctx_idx = match *ctx_slot {
                        Some(idx) => idx,
                        None => continue,
                    };

                    // Skip-gram update
                    self.skipgram_update(
                        model,
                        center_idx,
                        ctx_idx,
                        &words[pos].to_lowercase(),
                        sampler,
                        lr,
                        &mut rng,
                    );

                    stats.examples_processed.fetch_add(1, Ordering::Relaxed);
                }

                stats.words_processed.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    /// Perform a single skip-gram update with negative sampling.
    // Hot per-update SGD kernel: the model, the center/context indices, the
    // center word, the sampler, and the learning rate are per-update scalars
    // kept flat to avoid struct overhead in the inner training loop.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn skipgram_update<R: Rng>(
        &self,
        model: &mut SubwordEmbedding,
        center_idx: usize,
        context_idx: usize,
        center_word: &str,
        sampler: &NegativeSampler,
        lr: f32,
        rng: &mut R,
    ) {
        let dim = self.config.dim;
        let neg_samples = self.config.neg_samples;

        // Get center word embedding + subword embeddings
        let center_vec = self.get_input_vector(model, center_idx, center_word);

        // Sample negative words
        let negatives = sampler.sample_negatives(rng, neg_samples, context_idx);

        // Compute gradients
        let mut grad_in = Array1::<f32>::zeros(dim);

        // Positive sample (context word)
        let ctx_vec = model
            .embedding_by_index(context_idx)
            .expect("valid index")
            .to_owned();
        let dot = center_vec.dot(&ctx_vec);
        let score = Self::sigmoid(dot);
        let grad = (1.0 - score) * lr;

        // Update output (context) embedding
        model.update_word_embedding(context_idx, &(&center_vec * grad), 1.0);
        grad_in = grad_in + &ctx_vec * grad;

        // Negative samples
        for &neg_idx in &negatives {
            let neg_vec = model
                .embedding_by_index(neg_idx)
                .expect("valid index")
                .to_owned();
            let dot = center_vec.dot(&neg_vec);
            let score = Self::sigmoid(dot);
            let grad = -score * lr;

            model.update_word_embedding(neg_idx, &(&center_vec * grad), 1.0);
            grad_in = grad_in + &neg_vec * grad;
        }

        // Update input (center) embedding
        model.update_word_embedding(center_idx, &grad_in, 1.0);

        // Update subword embeddings
        let subwords = extract_subwords(
            center_word,
            self.config.min_subword_len,
            self.config.max_subword_len,
        );
        if !subwords.is_empty() {
            let subword_grad = &grad_in / subwords.len() as f32;
            for subword in &subwords {
                let bucket = hash_subword(subword, self.config.bucket_count);
                model.update_subword_embedding(bucket, &subword_grad, 1.0);
            }
        }
    }

    /// Get input vector for a word (word embedding + averaged subword embeddings).
    fn get_input_vector(
        &self,
        model: &SubwordEmbedding,
        word_idx: usize,
        word: &str,
    ) -> Array1<f32> {
        let word_vec = model
            .embedding_by_index(word_idx)
            .expect("valid index")
            .to_owned();

        let subwords = extract_subwords(
            word,
            self.config.min_subword_len,
            self.config.max_subword_len,
        );
        if subwords.is_empty() {
            return word_vec;
        }

        // Average subword embeddings
        let mut subword_sum = Array1::<f32>::zeros(self.config.dim);
        for subword in &subwords {
            let bucket = hash_subword(subword, self.config.bucket_count);
            if let Some(row) = model.embedding_by_index(bucket) {
                subword_sum = subword_sum + row;
            }
        }
        let subword_avg = subword_sum / subwords.len() as f32;

        // Combine word and subword vectors
        (word_vec + subword_avg) / 2.0
    }

    /// Sigmoid function.
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Builder for embedding training with fluent API.
pub struct EmbeddingTrainerBuilder {
    config: EmbeddingConfig,
}

impl EmbeddingTrainerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: EmbeddingConfig::default(),
        }
    }

    /// Set embedding dimension.
    pub fn dim(mut self, dim: usize) -> Self {
        self.config.dim = dim;
        self
    }

    /// Set window size.
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    /// Set minimum word count.
    pub fn min_count(mut self, count: u64) -> Self {
        self.config.min_count = count;
        self
    }

    /// Set number of negative samples.
    pub fn neg_samples(mut self, n: usize) -> Self {
        self.config.neg_samples = n;
        self
    }

    /// Set number of epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Build the trainer.
    pub fn build(self) -> EmbeddingTrainer {
        EmbeddingTrainer::new(self.config)
    }

    /// Build and train from corpus (legacy - buffers sentences).
    ///
    /// **Warning**: For large corpora (>500MB), use `train_streaming` instead.
    pub fn train<R: CorpusReader + 'static>(self, reader: R) -> Result<SubwordEmbedding> {
        self.build().train(reader)
    }

    /// Build and train from corpus using multi-pass streaming (memory-efficient).
    ///
    /// # Arguments
    ///
    /// * `reader_factory` - A closure that creates a new CorpusReader for each pass
    ///
    /// # Example
    ///
    /// ```ignore
    /// use libgrammstein::corpus::PlaintextReader;
    ///
    /// let path = std::path::Path::new("corpus.txt");
    /// let model = EmbeddingTrainerBuilder::new()
    ///     .dim(100)
    ///     .epochs(5)
    ///     .train_streaming(|| PlaintextReader::from_file(&path))?;
    /// ```
    pub fn train_streaming<F, R>(self, reader_factory: F) -> Result<SubwordEmbedding>
    where
        F: Fn() -> Result<R>,
        R: CorpusReader + 'static,
    {
        self.build().train_streaming(reader_factory)
    }

    /// Build and continue training an existing model for the remaining epochs.
    ///
    /// See [`EmbeddingTrainer::train_continued`].
    pub fn train_continued<R: CorpusReader + 'static>(
        self,
        model: SubwordEmbedding,
        start_epoch: usize,
        reader: R,
    ) -> Result<SubwordEmbedding> {
        self.build().train_continued(model, start_epoch, reader)
    }
}

impl Default for EmbeddingTrainerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_corpus(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");
        path
    }

    #[test]
    fn test_config_builder() {
        let config = EmbeddingConfig::new(100)
            .with_window_size(3)
            .with_min_count(2)
            .with_neg_samples(10)
            .with_epochs(3);

        assert_eq!(config.dim, 100);
        assert_eq!(config.window_size, 3);
        assert_eq!(config.min_count, 2);
        assert_eq!(config.neg_samples, 10);
        assert_eq!(config.epochs, 3);
    }

    #[test]
    fn test_negative_sampler() {
        let counts = vec![100, 50, 25, 10, 5];
        let sampler = NegativeSampler::new(&counts, 1000);

        let mut rng = StdRng::seed_from_u64(42);

        // Sample many times and check distribution roughly matches frequencies
        let mut sample_counts = vec![0usize; counts.len()];
        for _ in 0..10000 {
            let idx = sampler.sample(&mut rng);
            sample_counts[idx] += 1;
        }

        // First word (highest frequency) should be sampled most often
        assert!(sample_counts[0] > sample_counts[4]);
    }

    #[test]
    fn test_train_simple_corpus() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Create a corpus with repeated patterns
        let content = "the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox";

        let path = create_test_corpus(dir.path(), content);
        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let model = EmbeddingTrainerBuilder::new()
            .dim(10)
            .window_size(2)
            .min_count(1)
            .epochs(2)
            .train(reader)
            .expect("Training failed");

        // Check model was trained
        assert!(model.vocab_size() > 0);
        assert_eq!(model.dim(), 10);

        // Check we can get vectors
        let vec = model.word_vector("the");
        assert_eq!(vec.len(), 10);
    }

    #[test]
    fn test_sigmoid() {
        assert!((EmbeddingTrainer::sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(EmbeddingTrainer::sigmoid(10.0) > 0.99);
        assert!(EmbeddingTrainer::sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_train_continued_resumes_remaining_epochs() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        // Use a corpus with many distinct, low-frequency words across several lines so
        // word2vec sub-sampling reliably keeps some words to train (a tiny high-frequency
        // corpus can stochastically drop every word, making the "did training run?" check
        // flaky). Each word appears 3x (once per repeat) — low frequency, kept by sub-sampling.
        let mut content = String::new();
        for _ in 0..3 {
            for i in 0..60 {
                content.push_str(&format!("alpha{i} beta{i} gamma{i} "));
            }
            content.push('\n');
        }
        let path = create_test_corpus(dir.path(), &content);

        // Train one epoch from scratch.
        let model1 = EmbeddingTrainerBuilder::new()
            .dim(10)
            .window_size(2)
            .min_count(1)
            .epochs(1)
            .train(PlaintextReader::from_file(&path).expect("reader"))
            .expect("initial training failed");
        let vsz = model1.vocab_size();
        // Snapshot every word vector before resuming (checking all words is robust to
        // word2vec sub-sampling, which can skip the most frequent words like "the").
        let words: Vec<String> = model1.vocab().to_vec();
        // Use the uncached accessor: `word_vector` memoizes, and snapshotting would
        // otherwise populate a cache that survives into the resumed model and read stale.
        let before: Vec<Vec<f32>> = words
            .iter()
            .map(|w| model1.word_vector_uncached(w).to_vec())
            .collect();

        // Continue from epoch 1 to 3, keeping the model's weights + vocabulary.
        let model2 = EmbeddingTrainerBuilder::new()
            .dim(10)
            .window_size(2)
            .min_count(1)
            .epochs(3)
            .train_continued(
                model1,
                1,
                PlaintextReader::from_file(&path).expect("reader"),
            )
            .expect("continued training failed");

        // Vocabulary stays aligned (same size) and the remaining epochs ran (some moved).
        assert_eq!(model2.vocab_size(), vsz);
        let changed = words.iter().zip(&before).any(|(w, b)| {
            model2
                .word_vector_uncached(w)
                .iter()
                .zip(b)
                .any(|(x, y)| (x - y).abs() > 1e-9)
        });
        assert!(
            changed,
            "continued training should update at least one embedding"
        );

        // start_epoch >= epochs is a no-op that returns the model unchanged.
        let model3 = EmbeddingTrainerBuilder::new()
            .dim(10)
            .min_count(1)
            .epochs(3)
            .train_continued(
                model2,
                3,
                PlaintextReader::from_file(&path).expect("reader"),
            )
            .expect("no-op resume failed");
        assert_eq!(model3.vocab_size(), vsz);
    }
}
