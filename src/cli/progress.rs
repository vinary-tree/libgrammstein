//! Progress bar utilities for training and evaluation.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use indicatif::{HumanBytes, HumanDuration, MultiProgress, ProgressBar, ProgressStyle};

/// Progress update message.
#[derive(Debug, Clone)]
pub enum ProgressUpdate {
    /// Update sentence count.
    Sentences(u64),
    /// Update token count.
    Tokens(u64),
    /// Update bytes read.
    Bytes(u64),
    /// Update with all statistics.
    Stats {
        sentences: u64,
        tokens: u64,
        bytes: u64,
    },
    /// Update message.
    Message(String),
    /// Training complete.
    Finish,
}

/// Training statistics with lock-free atomic updates.
#[derive(Debug, Default)]
pub struct TrainingStats {
    /// Number of sentences processed.
    pub sentences_processed: AtomicU64,
    /// Number of tokens processed.
    pub tokens_processed: AtomicU64,
    /// Number of N-grams counted.
    pub ngrams_counted: AtomicU64,
    /// Bytes read from corpus.
    pub bytes_read: AtomicU64,
    /// Whether training is still running.
    pub is_running: AtomicBool,
}

impl TrainingStats {
    /// Create new training stats.
    pub fn new() -> Self {
        Self {
            sentences_processed: AtomicU64::new(0),
            tokens_processed: AtomicU64::new(0),
            ngrams_counted: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            is_running: AtomicBool::new(true),
        }
    }

    /// Increment sentences processed.
    pub fn inc_sentences(&self, count: u64) {
        self.sentences_processed.fetch_add(count, Ordering::Relaxed);
    }

    /// Increment tokens processed.
    pub fn inc_tokens(&self, count: u64) {
        self.tokens_processed.fetch_add(count, Ordering::Relaxed);
    }

    /// Increment N-grams counted.
    pub fn inc_ngrams(&self, count: u64) {
        self.ngrams_counted.fetch_add(count, Ordering::Relaxed);
    }

    /// Increment bytes read.
    pub fn inc_bytes(&self, count: u64) {
        self.bytes_read.fetch_add(count, Ordering::Relaxed);
    }

    /// Mark training as stopped.
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// Check if training is running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Get current sentences count.
    pub fn sentences(&self) -> u64 {
        self.sentences_processed.load(Ordering::Relaxed)
    }

    /// Get current tokens count.
    pub fn tokens(&self) -> u64 {
        self.tokens_processed.load(Ordering::Relaxed)
    }

    /// Get current N-grams count.
    pub fn ngrams(&self) -> u64 {
        self.ngrams_counted.load(Ordering::Relaxed)
    }

    /// Get current bytes read.
    pub fn bytes(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }
}

/// Progress display with percentage, ETA, and statistics.
pub struct TrainingProgress {
    pb: ProgressBar,
    start_time: Instant,
    total: Option<u64>,
}

impl TrainingProgress {
    /// Create new training progress bar.
    pub fn new(total_estimate: Option<u64>) -> Self {
        let pb = if let Some(total) = total_estimate {
            ProgressBar::new(total)
        } else {
            ProgressBar::new_spinner()
        };

        // Rich progress bar with percentage, speed, ETA
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                     {pos}/{len} ({percent}%) | {msg}",
                )
                .expect("Invalid progress template")
                .progress_chars("=>-"),
        );

        Self {
            pb,
            start_time: Instant::now(),
            total: total_estimate,
        }
    }

    /// Create a hidden progress bar (for --quiet mode).
    pub fn hidden() -> Self {
        Self {
            pb: ProgressBar::hidden(),
            start_time: Instant::now(),
            total: None,
        }
    }

    /// Update progress with current statistics.
    pub fn update(&self, sentences: u64, tokens: u64, bytes: u64) {
        self.pb.set_position(sentences);

        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let sentences_per_sec = sentences as f64 / elapsed;
            let tokens_per_sec = tokens as f64 / elapsed;

            // Calculate ETA if we know the total
            let eta_msg = if let Some(total) = self.total {
                if sentences > 0 {
                    let remaining = total.saturating_sub(sentences);
                    let eta_secs = remaining as f64 / sentences_per_sec;
                    format!(
                        " ETA: {}",
                        HumanDuration(Duration::from_secs_f64(eta_secs))
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            self.pb.set_message(format!(
                "{:.0} sent/s | {:.0} tok/s | {} read{}",
                sentences_per_sec,
                tokens_per_sec,
                HumanBytes(bytes),
                eta_msg
            ));
        }
    }

    /// Set a custom message.
    pub fn set_message(&self, message: &str) {
        self.pb.set_message(message.to_string());
    }

    /// Set the total count.
    pub fn set_total(&mut self, total: u64) {
        self.pb.set_length(total);
        self.total = Some(total);
    }

    /// Show completion summary.
    pub fn finish(&self, sentences: u64, tokens: u64, ngrams: u64) {
        self.pb.finish_with_message(format!(
            "Complete: {} sentences, {} tokens, {} n-grams in {}",
            sentences,
            tokens,
            ngrams,
            HumanDuration(self.start_time.elapsed())
        ));
    }

    /// Finish with a custom message.
    pub fn finish_with_message(&self, message: &str) {
        self.pb.finish_with_message(message.to_string());
    }

    /// Abandon progress bar (for errors).
    pub fn abandon(&self) {
        self.pb.abandon();
    }
}

/// Multi-stage progress for embedding training.
pub struct EmbeddingProgress {
    #[allow(dead_code)]
    multi: MultiProgress,
    epoch_bar: ProgressBar,
    batch_bar: ProgressBar,
    start_time: Instant,
}

impl EmbeddingProgress {
    /// Create new embedding training progress display.
    pub fn new(epochs: u32, batches_per_epoch: u64) -> Self {
        let multi = MultiProgress::new();

        let epoch_bar = multi.add(ProgressBar::new(epochs as u64));
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("Epoch {pos}/{len} [{bar:30}] {percent}% | Loss: {msg}")
                .expect("Invalid epoch progress template"),
        );

        let batch_bar = multi.add(ProgressBar::new(batches_per_epoch));
        batch_bar.set_style(
            ProgressStyle::default_bar()
                .template("  Batch [{bar:25}] {pos}/{len} | {per_sec} | ETA: {eta}")
                .expect("Invalid batch progress template"),
        );

        Self {
            multi,
            epoch_bar,
            batch_bar,
            start_time: Instant::now(),
        }
    }

    /// Start a new epoch.
    pub fn start_epoch(&self, epoch: u32) {
        self.epoch_bar.set_position(epoch as u64);
        self.batch_bar.reset();
    }

    /// Update batch progress.
    pub fn update_batch(&self, batch: u64, loss: f64) {
        self.batch_bar.set_position(batch);
        self.epoch_bar.set_message(format!("{:.4}", loss));
    }

    /// Finish an epoch.
    pub fn finish_epoch(&self, _epoch: u32, epoch_loss: f64) {
        self.epoch_bar.set_message(format!("{:.4}", epoch_loss));
        self.batch_bar.finish_and_clear();
    }

    /// Finish all training.
    pub fn finish(&self) {
        self.epoch_bar.finish_with_message(format!(
            "Complete in {}",
            HumanDuration(self.start_time.elapsed())
        ));
    }
}

/// Reactive progress reporter with backpressure.
///
/// Uses a bounded channel to prevent memory blowup while
/// allowing non-blocking updates from training threads.
pub struct ProgressReporter {
    tx: Sender<ProgressUpdate>,
    _handle: std::thread::JoinHandle<()>,
}

impl ProgressReporter {
    /// Create new progress reporter with a progress bar.
    pub fn new(pb: ProgressBar) -> Self {
        let (tx, rx) = bounded::<ProgressUpdate>(1000);

        let handle = std::thread::spawn(move || {
            Self::progress_loop(rx, pb);
        });

        Self { tx, _handle: handle }
    }

    /// Progress update loop (runs in dedicated thread).
    fn progress_loop(rx: Receiver<ProgressUpdate>, pb: ProgressBar) {
        loop {
            match rx.recv() {
                Ok(ProgressUpdate::Sentences(n)) => pb.set_position(n),
                Ok(ProgressUpdate::Message(msg)) => pb.set_message(msg),
                Ok(ProgressUpdate::Finish) => {
                    pb.finish();
                    break;
                }
                Ok(ProgressUpdate::Stats {
                    sentences,
                    tokens: _,
                    bytes: _,
                }) => {
                    pb.set_position(sentences);
                }
                Ok(ProgressUpdate::Tokens(_)) => {}
                Ok(ProgressUpdate::Bytes(_)) => {}
                Err(_) => break, // Channel closed
            }
        }
    }

    /// Non-blocking update (drops if channel full).
    pub fn update(&self, update: ProgressUpdate) {
        let _ = self.tx.try_send(update); // Never blocks training
    }

    /// Try to receive without blocking.
    pub fn try_recv(&self) -> Result<(), TryRecvError> {
        // This is a sender, so we can't receive
        Ok(())
    }

    /// Finish progress reporting.
    pub fn finish(&self) {
        let _ = self.tx.send(ProgressUpdate::Finish);
    }
}

/// Model statistics for completion summary.
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Number of sentences processed.
    pub sentences: u64,
    /// Number of tokens processed.
    pub tokens: u64,
    /// Number of N-grams in model.
    pub ngrams: u64,
}

/// Create a simple spinner for indeterminate progress.
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Invalid spinner template"),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Set up interrupt handler for graceful shutdown.
pub fn setup_interrupt_handler(stats: Arc<TrainingStats>) {
    let _ = ctrlc::set_handler(move || {
        stats.stop();
        eprintln!("\n\nInterrupt received. Finishing current batch...");
    });
}
