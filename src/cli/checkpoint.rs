//! Checkpoint management for resumable training.
//!
//! This module provides checkpoint infrastructure for N-gram and embedding training:
//! - Training state metadata serialization
//! - Checkpoint file management (save/load/prune)
//! - Integration with NgramAccumulator's WAL-based persistence

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::cli::error::{CliError, CliResult};

/// Checkpoint version for compatibility checking.
const CHECKPOINT_VERSION: u32 = 1;

/// N-gram training checkpoint metadata.
///
/// Note: The actual n-gram counts are stored in the PersistentARTrie (accumulator).
/// This metadata tracks training progress and configuration for resume capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramCheckpoint {
    /// Checkpoint format version.
    pub version: u32,

    /// Training configuration.
    pub config: NgramCheckpointConfig,

    /// Current training state.
    pub state: NgramTrainingState,

    /// Path to the accumulator file (PersistentARTrie).
    pub accumulator_path: PathBuf,

    /// Number of unique n-grams at checkpoint time.
    pub unique_ngrams: usize,

    /// Timestamp of checkpoint creation.
    pub created_at: DateTime<Utc>,
}

/// N-gram training configuration stored in checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramCheckpointConfig {
    /// N-gram order.
    pub order: usize,

    /// Minimum n-gram count threshold.
    pub min_count: u64,

    /// Corpus path.
    pub corpus_path: String,

    /// Whether to lowercase tokens.
    pub lowercase: bool,
}

/// Training state for N-gram models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NgramTrainingState {
    /// Number of sentences processed.
    pub sentences_processed: u64,

    /// Number of tokens processed.
    pub tokens_processed: u64,

    /// Number of bytes read from corpus.
    pub bytes_read: u64,

    /// Total bytes in corpus (if known).
    pub total_bytes: Option<u64>,

    /// Elapsed training time in seconds (excluding pauses).
    pub elapsed_secs: f64,

    /// Position in corpus for resume.
    pub corpus_position: CorpusPosition,
}

/// Position in corpus for resume.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusPosition {
    /// For file-based corpus: current file index (in directory).
    pub file_index: usize,

    /// For file-based corpus: line number within current file.
    pub line_number: usize,

    /// For URL-based corpus: byte offset.
    pub byte_offset: u64,
}

/// Checkpoint manager for training operations.
pub struct CheckpointManager {
    /// Directory for checkpoint files.
    checkpoint_dir: PathBuf,

    /// Maximum checkpoints to keep (older ones deleted).
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(checkpoint_dir: &Path, max_checkpoints: usize) -> CliResult<Self> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(checkpoint_dir).map_err(|e| {
            CliError::io(format!(
                "Failed to create checkpoint directory {}: {}",
                checkpoint_dir.display(),
                e
            ))
        })?;

        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            max_checkpoints,
        })
    }

    /// Get path to accumulator file.
    pub fn accumulator_path(&self) -> PathBuf {
        self.checkpoint_dir.join("ngram_hot.artrie")
    }

    /// Save checkpoint atomically (write to temp, then rename).
    pub fn save_ngram_checkpoint(&self, checkpoint: &NgramCheckpoint) -> CliResult<PathBuf> {
        let name = format!("ngram_ckpt_{}", checkpoint.state.sentences_processed);
        let temp_path = self.checkpoint_dir.join(format!("{}.tmp", name));
        let final_path = self.checkpoint_dir.join(format!("{}.bin", name));

        // Write to temp file with zstd compression
        let file = File::create(&temp_path)
            .map_err(|e| CliError::io(format!("Failed to create checkpoint file: {}", e)))?;
        let writer = BufWriter::new(file);
        let encoder = zstd::Encoder::new(writer, 3)
            .map_err(|e| CliError::io(format!("Failed to create zstd encoder: {}", e)))?
            .auto_finish();
        crate::bincode_compat::serialize_into(encoder, checkpoint)
            .map_err(|e| CliError::io(format!("Failed to serialize checkpoint: {}", e)))?;

        // Atomic rename
        fs::rename(&temp_path, &final_path)
            .map_err(|e| CliError::io(format!("Failed to finalize checkpoint: {}", e)))?;

        // Update "latest" symlink
        let latest = self.checkpoint_dir.join("latest.bin");
        let _ = fs::remove_file(&latest);
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(&final_path, &latest);
        }
        #[cfg(not(unix))]
        {
            // On Windows, copy instead of symlink
            let _ = fs::copy(&final_path, &latest);
        }

        // Prune old checkpoints if needed
        self.prune_old_checkpoints()?;

        Ok(final_path)
    }

    /// Load checkpoint from file.
    pub fn load_ngram_checkpoint(&self, name: &str) -> CliResult<NgramCheckpoint> {
        let path = if name == "latest" {
            self.checkpoint_dir.join("latest.bin")
        } else if name.ends_with(".bin") {
            PathBuf::from(name)
        } else {
            self.checkpoint_dir.join(format!("{}.bin", name))
        };

        if !path.exists() {
            return Err(CliError::file_not_found(&path));
        }

        let file = File::open(&path)
            .map_err(|e| CliError::io(format!("Failed to open checkpoint: {}", e)))?;
        let reader = BufReader::new(file);
        let decoder = zstd::Decoder::new(reader)
            .map_err(|e| CliError::io(format!("Failed to create zstd decoder: {}", e)))?;
        let checkpoint: NgramCheckpoint = crate::bincode_compat::deserialize_from(decoder)
            .map_err(|e| CliError::io(format!("Failed to deserialize checkpoint: {}", e)))?;

        // Version check
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(CliError::unsupported(format!(
                "Checkpoint version {} not supported (expected {})",
                checkpoint.version, CHECKPOINT_VERSION
            )));
        }

        Ok(checkpoint)
    }

    /// List available checkpoints.
    pub fn list_checkpoints(&self) -> CliResult<Vec<CheckpointInfo>> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| CliError::io(format!("Failed to read checkpoint directory: {}", e)))?
        {
            let entry = entry.map_err(|e| CliError::io(format!("Directory read error: {}", e)))?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "bin")
                && path
                    .file_stem()
                    .is_some_and(|s| s.to_string_lossy().starts_with("ngram_ckpt_"))
            {
                let metadata = fs::metadata(&path)
                    .map_err(|e| CliError::io(format!("Failed to read metadata: {}", e)))?;

                checkpoints.push(CheckpointInfo {
                    path,
                    size: metadata.len(),
                    modified: metadata.modified().ok().map(DateTime::<Utc>::from),
                });
            }
        }

        // Sort by modification time (newest first)
        checkpoints.sort_by_key(|c| std::cmp::Reverse(c.modified));

        Ok(checkpoints)
    }

    /// Prune old checkpoints, keeping only max_checkpoints newest.
    fn prune_old_checkpoints(&self) -> CliResult<()> {
        let checkpoints = self.list_checkpoints()?;

        // Keep newest max_checkpoints, delete the rest
        for checkpoint in checkpoints.into_iter().skip(self.max_checkpoints) {
            log::debug!("Pruning old checkpoint: {}", checkpoint.path.display());
            let _ = fs::remove_file(&checkpoint.path);
        }

        Ok(())
    }
}

/// Information about an available checkpoint.
#[derive(Debug)]
pub struct CheckpointInfo {
    /// Path to checkpoint file.
    pub path: PathBuf,

    /// File size in bytes.
    pub size: u64,

    /// Modification time.
    pub modified: Option<DateTime<Utc>>,
}

// =============================================================================
// Embedding Checkpoint Support
// =============================================================================

/// Embedding training checkpoint.
///
/// Includes the model state for epoch-based resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCheckpoint {
    /// Checkpoint format version.
    pub version: u32,

    /// Training configuration.
    pub config: EmbeddingCheckpointConfig,

    /// Current training state.
    pub state: EmbeddingTrainingState,

    /// Path to the saved model file.
    pub model_path: PathBuf,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Timestamp of checkpoint creation.
    pub created_at: DateTime<Utc>,
}

/// Embedding training configuration stored in checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCheckpointConfig {
    /// Embedding dimension.
    pub dim: usize,

    /// Context window size.
    pub window: usize,

    /// Minimum word frequency.
    pub min_count: u64,

    /// Negative samples per word.
    pub neg_samples: usize,

    /// Total training epochs.
    pub epochs: u32,

    /// Initial learning rate.
    pub learning_rate: f64,

    /// Corpus path.
    pub corpus_path: String,
}

/// Training state for embedding models (epoch-based).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingTrainingState {
    /// Completed epochs (0-indexed).
    pub completed_epochs: u32,

    /// Words processed in current/last epoch.
    pub words_processed: u64,

    /// Total words in corpus.
    pub total_words: u64,

    /// Current learning rate.
    pub current_learning_rate: f64,

    /// Loss history per epoch.
    pub loss_history: Vec<f64>,

    /// Elapsed training time in seconds.
    pub elapsed_secs: f64,
}

impl CheckpointManager {
    /// Save embedding checkpoint.
    pub fn save_embedding_checkpoint(
        &self,
        checkpoint: &EmbeddingCheckpoint,
    ) -> CliResult<PathBuf> {
        let name = format!("embedding_epoch_{}", checkpoint.state.completed_epochs);
        let temp_path = self.checkpoint_dir.join(format!("{}.tmp", name));
        let final_path = self.checkpoint_dir.join(format!("{}.bin", name));

        // Write to temp file with zstd compression
        let file = File::create(&temp_path)
            .map_err(|e| CliError::io(format!("Failed to create checkpoint file: {}", e)))?;
        let writer = BufWriter::new(file);
        let encoder = zstd::Encoder::new(writer, 3)
            .map_err(|e| CliError::io(format!("Failed to create zstd encoder: {}", e)))?
            .auto_finish();
        crate::bincode_compat::serialize_into(encoder, checkpoint)
            .map_err(|e| CliError::io(format!("Failed to serialize checkpoint: {}", e)))?;

        // Atomic rename
        fs::rename(&temp_path, &final_path)
            .map_err(|e| CliError::io(format!("Failed to finalize checkpoint: {}", e)))?;

        // Update embedding "latest" symlink
        let latest = self.checkpoint_dir.join("embedding_latest.bin");
        let _ = fs::remove_file(&latest);
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(&final_path, &latest);
        }
        #[cfg(not(unix))]
        {
            let _ = fs::copy(&final_path, &latest);
        }

        Ok(final_path)
    }

    /// Load embedding checkpoint.
    pub fn load_embedding_checkpoint(&self, name: &str) -> CliResult<EmbeddingCheckpoint> {
        let path = if name == "latest" {
            self.checkpoint_dir.join("embedding_latest.bin")
        } else if name.ends_with(".bin") {
            PathBuf::from(name)
        } else {
            self.checkpoint_dir.join(format!("{}.bin", name))
        };

        if !path.exists() {
            return Err(CliError::file_not_found(&path));
        }

        let file = File::open(&path)
            .map_err(|e| CliError::io(format!("Failed to open checkpoint: {}", e)))?;
        let reader = BufReader::new(file);
        let decoder = zstd::Decoder::new(reader)
            .map_err(|e| CliError::io(format!("Failed to create zstd decoder: {}", e)))?;
        let checkpoint: EmbeddingCheckpoint = crate::bincode_compat::deserialize_from(decoder)
            .map_err(|e| CliError::io(format!("Failed to deserialize checkpoint: {}", e)))?;

        // Version check
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(CliError::unsupported(format!(
                "Checkpoint version {} not supported (expected {})",
                checkpoint.version, CHECKPOINT_VERSION
            )));
        }

        Ok(checkpoint)
    }

    /// Get path for embedding model checkpoints.
    pub fn embedding_model_path(&self, epoch: u32) -> PathBuf {
        self.checkpoint_dir
            .join(format!("embedding_model_epoch_{}.bin", epoch))
    }
}

/// Training timer that can be paused and resumed.
pub struct TrainingTimer {
    start: Instant,
    elapsed_before_pause: f64,
    paused_at: Option<Instant>,
}

impl TrainingTimer {
    /// Create a new training timer.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            elapsed_before_pause: 0.0,
            paused_at: None,
        }
    }

    /// Resume timer with elapsed time from checkpoint.
    pub fn resume_from(elapsed_secs: f64) -> Self {
        Self {
            start: Instant::now(),
            elapsed_before_pause: elapsed_secs,
            paused_at: None,
        }
    }

    /// Get total elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        if let Some(paused) = self.paused_at {
            self.elapsed_before_pause + (paused - self.start).as_secs_f64()
        } else {
            self.elapsed_before_pause + self.start.elapsed().as_secs_f64()
        }
    }

    /// Pause the timer.
    pub fn pause(&mut self) {
        if self.paused_at.is_none() {
            self.paused_at = Some(Instant::now());
        }
    }

    /// Resume the timer.
    pub fn resume(&mut self) {
        if let Some(paused) = self.paused_at.take() {
            self.elapsed_before_pause += (paused - self.start).as_secs_f64();
            self.start = Instant::now();
        }
    }
}

impl Default for TrainingTimer {
    fn default() -> Self {
        Self::new()
    }
}
