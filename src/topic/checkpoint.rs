//! Checkpoint support for resumable topic extraction.
//!
//! This module provides checkpoint and resume functionality for long-running
//! topic extraction jobs. Checkpoints are saved as compact bincode files.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{Result, TopicError};

/// Extraction phase for checkpointing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractionPhase {
    /// Computing pairwise distance matrix.
    DistanceMatrix,
    /// Performing agglomerative clustering.
    Clustering,
    /// Building vocabulary for c-TF-IDF.
    VocabularyBuild,
    /// Extracting topic keywords.
    KeywordExtraction,
    /// Generating topic descriptions.
    Summarization,
    /// Extraction complete.
    Complete,
}

impl ExtractionPhase {
    /// Check if extraction is complete.
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }

    /// Get a human-readable description of the phase.
    pub fn description(&self) -> &'static str {
        match self {
            Self::DistanceMatrix => "Computing distance matrix",
            Self::Clustering => "Performing hierarchical clustering",
            Self::VocabularyBuild => "Building vocabulary",
            Self::KeywordExtraction => "Extracting topic keywords",
            Self::Summarization => "Generating topic descriptions",
            Self::Complete => "Complete",
        }
    }
}

/// Checkpoint for resuming topic extraction.
///
/// Contains all state needed to resume extraction from any phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopicExtractionCheckpoint {
    /// Checkpoint format version for compatibility.
    pub version: u32,
    /// Current extraction phase.
    pub phase: ExtractionPhase,
    /// Current iteration within phase (e.g., merge step in clustering).
    pub iteration: u64,
    /// Linkage matrix entries (cluster1, cluster2, distance, size).
    pub linkage_matrix: Vec<(u32, u32, f32, u32)>,
    /// Condensed distance matrix (upper triangular).
    /// None if distance matrix phase is not complete.
    pub distance_matrix: Option<Vec<f32>>,
    /// Current cluster assignments (document -> cluster).
    pub cluster_assignments: Vec<u32>,
    /// Extracted vocabulary.
    pub vocabulary: Vec<String>,
    /// Per-cluster term frequencies.
    /// Outer Vec: clusters, Inner Vec: term counts.
    pub term_frequencies: Vec<Vec<u32>>,
    /// Unix timestamp of checkpoint creation.
    pub timestamp: u64,
    /// CRC-64 checksum for integrity verification.
    pub checksum: u64,
    /// Number of documents being processed.
    pub num_documents: usize,
    /// Embedding dimension.
    pub embedding_dim: usize,
}

impl TopicExtractionCheckpoint {
    /// Current checkpoint format version.
    pub const VERSION: u32 = 1;

    /// Create a new checkpoint at the start of extraction.
    pub fn new(num_documents: usize, embedding_dim: usize) -> Self {
        Self {
            version: Self::VERSION,
            phase: ExtractionPhase::DistanceMatrix,
            iteration: 0,
            linkage_matrix: Vec::new(),
            distance_matrix: None,
            cluster_assignments: (0..num_documents as u32).collect(),
            vocabulary: Vec::new(),
            term_frequencies: Vec::new(),
            timestamp: current_timestamp(),
            checksum: 0,
            num_documents,
            embedding_dim,
        }
    }

    /// Create checkpoint at distance matrix completion.
    pub fn with_distance_matrix(
        num_documents: usize,
        embedding_dim: usize,
        distances: Vec<f32>,
    ) -> Self {
        Self {
            version: Self::VERSION,
            phase: ExtractionPhase::Clustering,
            iteration: 0,
            linkage_matrix: Vec::new(),
            distance_matrix: Some(distances),
            cluster_assignments: (0..num_documents as u32).collect(),
            vocabulary: Vec::new(),
            term_frequencies: Vec::new(),
            timestamp: current_timestamp(),
            checksum: 0,
            num_documents,
            embedding_dim,
        }
    }

    /// Update checkpoint for clustering progress.
    pub fn update_clustering(&mut self, iteration: u64, linkage: &[(u32, u32, f32, u32)]) {
        self.phase = ExtractionPhase::Clustering;
        self.iteration = iteration;
        self.linkage_matrix = linkage.to_vec();
        self.timestamp = current_timestamp();
    }

    /// Update checkpoint for vocabulary building.
    pub fn update_vocabulary(&mut self, vocabulary: Vec<String>) {
        self.phase = ExtractionPhase::VocabularyBuild;
        self.vocabulary = vocabulary;
        self.timestamp = current_timestamp();
    }

    /// Update checkpoint for keyword extraction.
    pub fn update_keyword_extraction(
        &mut self,
        term_frequencies: Vec<Vec<u32>>,
        cluster_assignments: Vec<u32>,
    ) {
        self.phase = ExtractionPhase::KeywordExtraction;
        self.term_frequencies = term_frequencies;
        self.cluster_assignments = cluster_assignments;
        self.timestamp = current_timestamp();
    }

    /// Mark extraction as complete.
    pub fn mark_complete(&mut self) {
        self.phase = ExtractionPhase::Complete;
        self.timestamp = current_timestamp();
    }

    /// Compute checksum and update the checkpoint.
    pub fn compute_checksum(&mut self) {
        // Simple CRC-64 using polynomial division
        let data = self.checksum_data();
        self.checksum = crc64(&data);
    }

    /// Verify checkpoint integrity.
    pub fn verify_checksum(&self) -> bool {
        let data = self.checksum_data();
        crc64(&data) == self.checksum
    }

    /// Get data for checksum computation (excludes checksum field).
    fn checksum_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.version.to_le_bytes());
        data.push(self.phase as u8);
        data.extend_from_slice(&self.iteration.to_le_bytes());
        data.extend_from_slice(&self.num_documents.to_le_bytes());
        data.extend_from_slice(&self.embedding_dim.to_le_bytes());
        data.extend_from_slice(&(self.linkage_matrix.len() as u64).to_le_bytes());
        data.extend_from_slice(&(self.vocabulary.len() as u64).to_le_bytes());
        data
    }

    /// Save checkpoint to file.
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write to temporary file first, then rename for atomicity
        let temp_path = path.with_extension("tmp");
        {
            let file = File::create(&temp_path)?;
            let writer = BufWriter::new(file);
            bincode::serialize_into(writer, self)?;
        }

        // Atomic rename
        std::fs::rename(&temp_path, path)?;
        Ok(())
    }

    /// Load checkpoint from file.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint: Self = bincode::deserialize_from(reader)?;

        // Verify version
        if checkpoint.version != Self::VERSION {
            return Err(TopicError::CheckpointError(format!(
                "Incompatible checkpoint version: {} (expected {})",
                checkpoint.version,
                Self::VERSION
            )));
        }

        // Verify checksum
        if !checkpoint.verify_checksum() {
            return Err(TopicError::CheckpointError(
                "Checkpoint checksum verification failed".to_string(),
            ));
        }

        Ok(checkpoint)
    }

    /// Check if a checkpoint file exists.
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }

    /// Get the size of the condensed distance matrix for N documents.
    pub const fn condensed_distance_size(n: usize) -> usize {
        n * (n - 1) / 2
    }

    /// Convert (i, j) indices to condensed matrix index.
    pub const fn condensed_index(i: usize, j: usize, n: usize) -> usize {
        debug_assert!(i < j);
        n * i - i * (i + 1) / 2 + j - i - 1
    }
}

/// Get current Unix timestamp.
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Simple CRC-64 computation (ECMA-182 polynomial).
fn crc64(data: &[u8]) -> u64 {
    const POLY: u64 = 0x42F0_E1EB_A9EA_3693;
    let mut crc: u64 = !0;

    for &byte in data {
        crc ^= u64::from(byte) << 56;
        for _ in 0..8 {
            if crc & (1 << 63) != 0 {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
        }
    }

    !crc
}

/// Progress tracker for topic extraction.
#[derive(Clone, Debug)]
pub struct ExtractionProgress {
    /// Current phase.
    pub phase: ExtractionPhase,
    /// Current step within phase.
    pub current_step: usize,
    /// Total steps in phase.
    pub total_steps: usize,
    /// Elapsed time in seconds.
    pub elapsed_seconds: f64,
    /// Estimated remaining time in seconds.
    pub estimated_remaining: Option<f64>,
}

impl ExtractionProgress {
    /// Create a new progress tracker.
    pub fn new(phase: ExtractionPhase, total_steps: usize) -> Self {
        Self {
            phase,
            current_step: 0,
            total_steps,
            elapsed_seconds: 0.0,
            estimated_remaining: None,
        }
    }

    /// Update progress.
    pub fn update(&mut self, step: usize, elapsed: f64) {
        self.current_step = step;
        self.elapsed_seconds = elapsed;

        if step > 0 {
            let rate = elapsed / step as f64;
            let remaining = (self.total_steps - step) as f64 * rate;
            self.estimated_remaining = Some(remaining);
        }
    }

    /// Get progress as a percentage (0.0 to 100.0).
    pub fn percentage(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.current_step as f64 / self.total_steps as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_new() {
        let checkpoint = TopicExtractionCheckpoint::new(100, 768);
        assert_eq!(checkpoint.version, TopicExtractionCheckpoint::VERSION);
        assert_eq!(checkpoint.phase, ExtractionPhase::DistanceMatrix);
        assert_eq!(checkpoint.num_documents, 100);
        assert_eq!(checkpoint.cluster_assignments.len(), 100);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("checkpoint.bin");

        let mut checkpoint = TopicExtractionCheckpoint::new(50, 768);
        checkpoint.update_clustering(10, &[(0, 1, 0.5, 2), (2, 3, 0.6, 2)]);
        checkpoint.compute_checksum();
        checkpoint.save(&path).expect("save failed");

        let loaded = TopicExtractionCheckpoint::load(&path).expect("load failed");
        assert_eq!(loaded.phase, ExtractionPhase::Clustering);
        assert_eq!(loaded.iteration, 10);
        assert_eq!(loaded.linkage_matrix.len(), 2);
        assert!(loaded.verify_checksum());
    }

    #[test]
    fn test_checkpoint_phases() {
        let mut checkpoint = TopicExtractionCheckpoint::new(10, 768);

        assert!(!checkpoint.phase.is_complete());
        assert_eq!(checkpoint.phase.description(), "Computing distance matrix");

        checkpoint.mark_complete();
        assert!(checkpoint.phase.is_complete());
    }

    #[test]
    fn test_condensed_index() {
        let n = 5;
        // Distance matrix for 5 items has 10 entries
        assert_eq!(TopicExtractionCheckpoint::condensed_distance_size(n), 10);

        // Index (0,1) should be 0
        assert_eq!(TopicExtractionCheckpoint::condensed_index(0, 1, n), 0);
        // Index (0,4) should be 3
        assert_eq!(TopicExtractionCheckpoint::condensed_index(0, 4, n), 3);
        // Index (3,4) should be 9
        assert_eq!(TopicExtractionCheckpoint::condensed_index(3, 4, n), 9);
    }

    #[test]
    fn test_crc64() {
        let data = b"hello world";
        let crc = crc64(data);
        assert_ne!(crc, 0);

        // Same data should produce same CRC
        assert_eq!(crc, crc64(data));

        // Different data should produce different CRC
        let different = b"hello world!";
        assert_ne!(crc, crc64(different));
    }

    #[test]
    fn test_progress_tracker() {
        let mut progress = ExtractionProgress::new(ExtractionPhase::Clustering, 100);

        assert_eq!(progress.percentage(), 0.0);

        progress.update(50, 10.0);
        assert_eq!(progress.percentage(), 50.0);
        assert!(progress.estimated_remaining.is_some());

        let remaining = progress.estimated_remaining.unwrap();
        assert!((remaining - 10.0).abs() < 0.01); // 50 steps left, 0.2s per step = 10s
    }

    #[test]
    fn test_extraction_phase_description() {
        assert_eq!(
            ExtractionPhase::DistanceMatrix.description(),
            "Computing distance matrix"
        );
        assert_eq!(
            ExtractionPhase::Clustering.description(),
            "Performing hierarchical clustering"
        );
        assert_eq!(ExtractionPhase::Complete.description(), "Complete");
    }
}
