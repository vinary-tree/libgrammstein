//! Prefetching corpus reader with auto-tuned buffering.
//!
//! Wraps any `CorpusReader` with a producer thread that reads sentences ahead
//! of consumption, decoupling I/O from processing.
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::corpus::{CorpusReader, PlaintextReader, PrefetchingReader};
//!
//! let reader = PlaintextReader::from_file("corpus.txt")?;
//! let prefetch = PrefetchingReader::new(reader);
//!
//! // Process batches in parallel
//! for batch in prefetch.batches() {
//!     batch.par_iter().for_each(|sentence| {
//!         process_sentence(sentence);
//!     });
//! }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::vec::IntoIter;

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use sysinfo::System;

use super::CorpusReader;

/// Global System instance for memory queries.
/// Using a singleton avoids expensive repeated initialization.
static SYSTEM: OnceLock<Mutex<System>> = OnceLock::new();

fn get_system() -> &'static Mutex<System> {
    SYSTEM.get_or_init(|| Mutex::new(System::new()))
}

/// Configuration for prefetching behavior.
#[derive(Clone, Debug)]
pub struct PrefetchConfig {
    /// Number of sentences per batch.
    /// Default: 10,000
    pub batch_size: usize,

    /// Number of batches to buffer ahead.
    /// If `auto_tune` is true, this is computed from available RAM.
    /// Default: 8
    pub buffer_batches: usize,

    /// Whether to auto-tune buffer size based on available RAM.
    /// Default: true
    pub auto_tune: bool,

    /// Fraction of available RAM to use for buffering.
    /// Only used when `auto_tune` is true.
    /// Default: 0.10 (10%)
    pub ram_fraction: f64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            batch_size: 10_000,
            buffer_batches: 8,
            auto_tune: true,
            ram_fraction: 0.10,
        }
    }
}

impl PrefetchConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the batch size (sentences per batch).
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the number of buffer batches (ignored if auto_tune is true).
    pub fn with_buffer_batches(mut self, batches: usize) -> Self {
        self.buffer_batches = batches;
        self
    }

    /// Enable or disable auto-tuning.
    pub fn with_auto_tune(mut self, enabled: bool) -> Self {
        self.auto_tune = enabled;
        self
    }

    /// Set the RAM fraction for auto-tuning.
    pub fn with_ram_fraction(mut self, fraction: f64) -> Self {
        self.ram_fraction = fraction.clamp(0.01, 0.50);
        self
    }

    /// Compute the effective buffer size in batches.
    fn effective_buffer_batches(&self) -> usize {
        if self.auto_tune {
            compute_buffer_batches(self.batch_size, self.ram_fraction)
        } else {
            self.buffer_batches.clamp(2, 64)
        }
    }
}

/// Default fallback memory when detection fails.
const DEFAULT_FALLBACK_MEMORY: usize = 8 * 1024 * 1024 * 1024; // 8 GB

/// Get available memory in bytes.
///
/// Uses the `sysinfo` crate for cross-platform support:
/// - Linux, macOS, Windows, FreeBSD, Android, iOS
///
/// Falls back to 8 GB if detection fails.
///
/// Uses a cached `System` instance per sysinfo best practices.
fn get_available_memory_bytes() -> usize {
    let mut sys = get_system().lock();
    sys.refresh_memory();

    let total = sys.total_memory() as usize;
    let used = sys.used_memory() as usize;

    if total > 0 {
        total.saturating_sub(used)
    } else {
        DEFAULT_FALLBACK_MEMORY
    }
}

/// Compute buffer batches based on available RAM.
fn compute_buffer_batches(batch_size: usize, ram_fraction: f64) -> usize {
    let available = get_available_memory_bytes();
    let target_bytes = (available as f64 * ram_fraction) as usize;

    // Estimate ~100 bytes per sentence average (conservative)
    let bytes_per_batch = batch_size * 100;

    let batches = target_bytes / bytes_per_batch.max(1);
    batches.clamp(2, 32)
}

/// Message types sent from producer to consumer.
enum PrefetchMessage {
    /// A batch of sentences.
    Batch(Vec<String>),
    /// Producer finished successfully.
    Done,
    /// Producer encountered an error.
    Error(String),
}

/// A prefetching wrapper for corpus readers.
///
/// Spawns a producer thread that reads sentences into batches ahead of
/// consumption, providing natural backpressure through a bounded channel.
pub struct PrefetchingReader {
    /// Receiver for batches from producer (Option for take() in batches()).
    rx: Option<Receiver<PrefetchMessage>>,

    /// Signal for early termination.
    stop_signal: Arc<AtomicBool>,

    /// Handle to producer thread (for join on drop).
    producer_handle: Option<JoinHandle<()>>,

    /// Current batch being iterated (for sentence-by-sentence access).
    current_batch: Option<IntoIter<String>>,

    /// Whether the producer has finished.
    exhausted: bool,

    /// Number of batches received (for statistics).
    batches_received: usize,

    /// Number of sentences yielded (for statistics).
    sentences_yielded: usize,
}

impl PrefetchingReader {
    /// Create a prefetching reader with default configuration.
    pub fn new<R>(reader: R) -> Self
    where
        R: CorpusReader + 'static,
    {
        Self::with_config(reader, PrefetchConfig::default())
    }

    /// Create a prefetching reader with custom configuration.
    pub fn with_config<R>(reader: R, config: PrefetchConfig) -> Self
    where
        R: CorpusReader + 'static,
    {
        let buffer_batches = config.effective_buffer_batches();
        let batch_size = config.batch_size;

        log::debug!(
            "PrefetchingReader: batch_size={}, buffer_batches={}, ram_fraction={:.1}%",
            batch_size,
            buffer_batches,
            config.ram_fraction * 100.0
        );

        let (tx, rx) = bounded::<PrefetchMessage>(buffer_batches);
        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_clone = stop_signal.clone();

        let producer_handle = thread::spawn(move || {
            producer_loop(reader, tx, stop_clone, batch_size);
        });

        Self {
            rx: Some(rx),
            stop_signal,
            producer_handle: Some(producer_handle),
            current_batch: None,
            exhausted: false,
            batches_received: 0,
            sentences_yielded: 0,
        }
    }

    /// Signal the producer to stop early.
    ///
    /// This is useful when you want to abort processing before
    /// consuming all sentences.
    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::Release);
    }

    /// Check if the producer has been signaled to stop.
    pub fn is_stopped(&self) -> bool {
        self.stop_signal.load(Ordering::Acquire)
    }

    /// Get the number of batches received so far.
    pub fn batches_received(&self) -> usize {
        self.batches_received
    }

    /// Get the number of sentences yielded so far.
    pub fn sentences_yielded(&self) -> usize {
        self.sentences_yielded
    }

    /// Convert to a batch iterator for parallel processing.
    ///
    /// This is more efficient than iterating sentence-by-sentence
    /// when using `rayon::par_iter()` on each batch.
    ///
    /// # Panics
    ///
    /// Panics if the reader has already been converted to batches or
    /// if iteration has started.
    pub fn batches(mut self) -> PrefetchBatchIterator {
        let rx = self.rx.take().expect("PrefetchingReader already consumed");
        let producer_handle = self.producer_handle.take();
        let stop_signal = self.stop_signal.clone();

        // Prevent Drop from trying to join/drain again
        self.exhausted = true;

        PrefetchBatchIterator {
            rx,
            stop_signal,
            producer_handle,
            exhausted: false,
            batches_received: self.batches_received,
        }
    }

    /// Receive the next batch from the producer.
    fn receive_batch(&mut self) -> Option<Vec<String>> {
        if self.exhausted {
            return None;
        }

        let rx = self.rx.as_ref()?;

        match rx.recv() {
            Ok(PrefetchMessage::Batch(batch)) => {
                self.batches_received += 1;
                Some(batch)
            }
            Ok(PrefetchMessage::Done) => {
                self.exhausted = true;
                None
            }
            Ok(PrefetchMessage::Error(e)) => {
                log::error!("Prefetch producer error: {}", e);
                self.exhausted = true;
                None
            }
            Err(_) => {
                // Channel closed unexpectedly
                self.exhausted = true;
                None
            }
        }
    }
}

impl Iterator for PrefetchingReader {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to get next sentence from current batch
            if let Some(ref mut batch_iter) = self.current_batch {
                if let Some(sentence) = batch_iter.next() {
                    self.sentences_yielded += 1;
                    return Some(sentence);
                }
            }

            // Current batch exhausted, get next batch
            match self.receive_batch() {
                Some(batch) => {
                    self.current_batch = Some(batch.into_iter());
                }
                None => {
                    return None;
                }
            }
        }
    }
}

impl Drop for PrefetchingReader {
    fn drop(&mut self) {
        // If rx was taken (by batches()), cleanup is handled by PrefetchBatchIterator
        if self.rx.is_none() {
            return;
        }

        // Signal producer to stop
        self.stop_signal.store(true, Ordering::Release);

        // Drain channel to unblock producer if it's waiting on send
        if let Some(ref rx) = self.rx {
            while rx.try_recv().is_ok() {}
        }

        // Wait for producer thread to finish
        if let Some(handle) = self.producer_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Iterator that yields batches of sentences for parallel processing.
pub struct PrefetchBatchIterator {
    rx: Receiver<PrefetchMessage>,
    stop_signal: Arc<AtomicBool>,
    producer_handle: Option<JoinHandle<()>>,
    exhausted: bool,
    batches_received: usize,
}

impl PrefetchBatchIterator {
    /// Signal the producer to stop early.
    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::Release);
    }

    /// Get the number of batches received so far.
    pub fn batches_received(&self) -> usize {
        self.batches_received
    }
}

impl Iterator for PrefetchBatchIterator {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        match self.rx.recv() {
            Ok(PrefetchMessage::Batch(batch)) => {
                self.batches_received += 1;
                Some(batch)
            }
            Ok(PrefetchMessage::Done) => {
                self.exhausted = true;
                None
            }
            Ok(PrefetchMessage::Error(e)) => {
                log::error!("Prefetch producer error: {}", e);
                self.exhausted = true;
                None
            }
            Err(_) => {
                self.exhausted = true;
                None
            }
        }
    }
}

impl Drop for PrefetchBatchIterator {
    fn drop(&mut self) {
        // Signal producer to stop
        self.stop_signal.store(true, Ordering::Release);

        // Drain channel to unblock producer
        while self.rx.try_recv().is_ok() {}

        // Wait for producer thread
        if let Some(handle) = self.producer_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Producer loop that reads from corpus and sends batches through channel.
fn producer_loop<R: CorpusReader>(
    reader: R,
    tx: Sender<PrefetchMessage>,
    stop_signal: Arc<AtomicBool>,
    batch_size: usize,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut batch = Vec::with_capacity(batch_size);

        for sentence in reader.sentences() {
            // Check for stop signal
            if stop_signal.load(Ordering::Acquire) {
                log::debug!("Prefetch producer: stop signal received");
                break;
            }

            batch.push(sentence);

            if batch.len() >= batch_size {
                // Send full batch
                if tx.send(PrefetchMessage::Batch(batch)).is_err() {
                    // Receiver dropped, stop producing
                    log::debug!("Prefetch producer: receiver dropped");
                    return;
                }
                batch = Vec::with_capacity(batch_size);
            }
        }

        // Send final partial batch if non-empty
        if !batch.is_empty() && !stop_signal.load(Ordering::Acquire) {
            let _ = tx.send(PrefetchMessage::Batch(batch));
        }

        // Signal completion
        let _ = tx.send(PrefetchMessage::Done);
    }));

    if let Err(panic) = result {
        let msg = if let Some(s) = panic.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic in prefetch producer".to_string()
        };
        log::error!("Prefetch producer panicked: {}", msg);
        let _ = tx.send(PrefetchMessage::Error(msg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock corpus reader for testing.
    struct MockReader {
        sentences: Arc<Mutex<Vec<String>>>,
    }

    impl MockReader {
        fn new(sentences: Vec<String>) -> Self {
            Self {
                sentences: Arc::new(Mutex::new(sentences)),
            }
        }
    }

    impl CorpusReader for MockReader {
        fn documents(&self) -> Box<dyn Iterator<Item = crate::corpus::Document> + Send + '_> {
            Box::new(std::iter::empty())
        }

        fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
            let sentences = self.sentences.lock().unwrap().clone();
            Box::new(sentences.into_iter())
        }
    }

    #[test]
    fn test_prefetch_config_defaults() {
        let config = PrefetchConfig::default();
        assert_eq!(config.batch_size, 10_000);
        assert!(config.auto_tune);
        assert!((config.ram_fraction - 0.10).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prefetch_config_builder() {
        let config = PrefetchConfig::new()
            .with_batch_size(5_000)
            .with_auto_tune(false)
            .with_buffer_batches(4)
            .with_ram_fraction(0.20);

        assert_eq!(config.batch_size, 5_000);
        assert!(!config.auto_tune);
        assert_eq!(config.buffer_batches, 4);
        assert!((config.ram_fraction - 0.20).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ram_fraction_clamping() {
        let config = PrefetchConfig::new().with_ram_fraction(0.90);
        assert!((config.ram_fraction - 0.50).abs() < f64::EPSILON);

        let config = PrefetchConfig::new().with_ram_fraction(0.001);
        assert!((config.ram_fraction - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prefetch_empty_corpus() {
        let reader = MockReader::new(vec![]);
        let prefetch = PrefetchingReader::new(reader);

        let sentences: Vec<String> = prefetch.collect();
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_prefetch_small_corpus() {
        let input = vec![
            "Hello world.".to_string(),
            "This is a test.".to_string(),
            "Rust is great.".to_string(),
        ];

        let reader = MockReader::new(input.clone());
        let config = PrefetchConfig::new()
            .with_batch_size(2)
            .with_auto_tune(false)
            .with_buffer_batches(2);

        let prefetch = PrefetchingReader::with_config(reader, config);
        let output: Vec<String> = prefetch.collect();

        assert_eq!(output, input);
    }

    #[test]
    fn test_prefetch_batch_iterator() {
        let input: Vec<String> = (0..100).map(|i| format!("Sentence {}", i)).collect();

        let reader = MockReader::new(input.clone());
        let config = PrefetchConfig::new()
            .with_batch_size(25)
            .with_auto_tune(false)
            .with_buffer_batches(2);

        let prefetch = PrefetchingReader::with_config(reader, config);
        let batches: Vec<Vec<String>> = prefetch.batches().collect();

        // Should have 4 batches of 25 each
        assert_eq!(batches.len(), 4);
        for batch in &batches {
            assert_eq!(batch.len(), 25);
        }

        // Flatten and compare
        let flattened: Vec<String> = batches.into_iter().flatten().collect();
        assert_eq!(flattened, input);
    }

    #[test]
    fn test_prefetch_partial_batch() {
        let input: Vec<String> = (0..7).map(|i| format!("Sentence {}", i)).collect();

        let reader = MockReader::new(input.clone());
        let config = PrefetchConfig::new()
            .with_batch_size(3)
            .with_auto_tune(false)
            .with_buffer_batches(2);

        let prefetch = PrefetchingReader::with_config(reader, config);
        let batches: Vec<Vec<String>> = prefetch.batches().collect();

        // Should have 3 batches: [3, 3, 1]
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 1);
    }

    #[test]
    fn test_prefetch_early_stop() {
        let input: Vec<String> = (0..1000).map(|i| format!("Sentence {}", i)).collect();

        let reader = MockReader::new(input);
        let config = PrefetchConfig::new()
            .with_batch_size(100)
            .with_auto_tune(false)
            .with_buffer_batches(2);

        let prefetch = PrefetchingReader::with_config(reader, config);

        // Take only first 50 sentences
        let output: Vec<String> = prefetch.take(50).collect();
        assert_eq!(output.len(), 50);
    }

    #[test]
    fn test_prefetch_drop_no_hang() {
        let input: Vec<String> = (0..10_000).map(|i| format!("Sentence {}", i)).collect();

        let reader = MockReader::new(input);
        let config = PrefetchConfig::new()
            .with_batch_size(100)
            .with_auto_tune(false)
            .with_buffer_batches(2);

        // Create and immediately drop without consuming
        let prefetch = PrefetchingReader::with_config(reader, config);
        drop(prefetch);

        // If we reach here without hanging, the test passes
    }

    #[test]
    fn test_get_available_memory() {
        let memory = get_available_memory_bytes();
        // Should return some reasonable value (at least 1 GB, at most 1 TB)
        assert!(memory >= 1024 * 1024 * 1024);
        assert!(memory <= 1024 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_compute_buffer_batches() {
        // With 10GB available and 10% fraction, targeting 1GB
        // batch_size=10000, ~100 bytes/sentence = 1MB per batch
        // 1GB / 1MB = 1000, clamped to 32
        let batches = compute_buffer_batches(10_000, 0.10);
        assert!(batches >= 2);
        assert!(batches <= 32);
    }
}
