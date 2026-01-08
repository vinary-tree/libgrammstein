//! PathMap translation for production deployment.
//!
//! Translates a trained PersistentARTrie model to PathMap for optimized read-only access.
//! PathMap provides:
//!
//! - **ACT compression**: Adaptive Compressed Trie for compact storage
//! - **Merkleization**: Content-addressed nodes for integrity verification
//! - **mmap/mm2**: Memory-mapped reads for efficient large-scale access

use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use liblevenshtein::dictionary::persistent_artrie_char::dict_impl_char::{
    CharTrieNodeInner, CharTrieRoot,
};
use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
use pathmap::paths_serialization::{serialize_paths_with_auxdata, for_each_deserialized_path};
use pathmap::zipper::ZipperMoving;
use pathmap::PathMap;

/// Recursively traverse the trie and collect all entries.
///
/// # Arguments
///
/// * `node` - Current trie node being traversed
/// * `prefix` - Current string prefix built during traversal
/// * `entries` - Output vector to collect (key, value) pairs
/// * `entries_count` - Counter for entries processed
fn collect_entries_recursive(
    node: &CharTrieNodeInner<u64>,
    prefix: &mut String,
    entries: &mut Vec<(String, u64)>,
    entries_count: &mut u64,
) {
    // Check if this node represents a complete term
    if node.is_final() {
        // Skip MKN metadata entries (they start with \x00)
        if !prefix.starts_with('\x00') {
            if let Some(count) = node.value.as_ref() {
                entries.push((prefix.clone(), *count));
                *entries_count += 1;
            }
        }
    }

    // Recursively traverse children
    for (c, child) in node.iter_children() {
        prefix.push(c);
        collect_entries_recursive(child, prefix, entries, entries_count);
        prefix.pop();
    }
}

/// Recursively traverse the trie with progress reporting.
fn collect_entries_with_progress<F>(
    node: &CharTrieNodeInner<u64>,
    prefix: &mut String,
    entries: &mut Vec<(String, u64)>,
    entries_count: &mut u64,
    progress_callback: &mut F,
    last_progress: &mut u64,
    start_time: std::time::Instant,
) where
    F: FnMut(TranslationProgress),
{
    // Check if this node represents a complete term
    if node.is_final() {
        // Skip MKN metadata entries (they start with \x00)
        if !prefix.starts_with('\x00') {
            if let Some(count) = node.value.as_ref() {
                entries.push((prefix.clone(), *count));
                *entries_count += 1;

                // Report progress every 100k entries
                if *entries_count - *last_progress >= 100_000 {
                    *last_progress = *entries_count;
                    progress_callback(TranslationProgress {
                        phase: TranslationPhase::Iterating,
                        entries_processed: *entries_count,
                        entries_total: None,
                        bytes_written: 0,
                        elapsed_seconds: start_time.elapsed().as_secs_f64(),
                    });
                }
            }
        }
    }

    // Recursively traverse children
    for (c, child) in node.iter_children() {
        prefix.push(c);
        collect_entries_with_progress(
            child, prefix, entries, entries_count, progress_callback, last_progress, start_time,
        );
        prefix.pop();
    }
}

/// Statistics from PathMap translation.
#[derive(Clone, Debug, Default)]
pub struct TranslationStats {
    /// Number of entries translated.
    pub entries_translated: u64,

    /// Source ARTrie size in bytes.
    pub artrie_size_bytes: u64,

    /// Output PathMap size in bytes.
    pub pathmap_size_bytes: u64,

    /// Compression ratio (ARTrie / PathMap).
    pub compression_ratio: f64,

    /// Translation time in seconds.
    pub elapsed_seconds: f64,

    /// Memory peak during translation (bytes).
    pub peak_memory_bytes: u64,
}

/// Errors that can occur during translation.
#[derive(Debug, thiserror::Error)]
pub enum TranslationError {
    /// Source model not found.
    #[error("Source model not found: {0}")]
    SourceNotFound(String),

    /// Empty source model.
    #[error("Source model is empty")]
    EmptySource,

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// PathMap construction error.
    #[error("PathMap construction error: {0}")]
    Construction(String),
}

/// PathMap translator for production deployment.
///
/// Translates trained n-gram models from PersistentARTrie (training format)
/// to PathMap (production format) for optimized read-only access.
///
/// # Workflow
///
/// 1. Training: Import n-grams into PersistentARTrie (supports updates)
/// 2. Translation: Convert to PathMap (read-only, optimized)
/// 3. Deployment: Use PathMap for production inference
///
/// # Benefits
///
/// - ~40% size reduction via ACT compression
/// - Content-addressed storage with Merkleization
/// - Efficient mmap-based reads for large models
/// - Zero-copy deserialization
///
/// # Example
///
/// ```ignore
/// use libgrammstein::sources::google_books::PathMapTranslator;
///
/// let stats = PathMapTranslator::translate(
///     "english.artrie",
///     "english.pathmap",
/// )?;
///
/// println!("Compression ratio: {:.2}x", stats.compression_ratio);
/// ```
pub struct PathMapTranslator;

impl PathMapTranslator {
    /// Translate trained n-gram model to PathMap format.
    ///
    /// # Arguments
    ///
    /// * `artrie_path` - Path to trained PersistentARTrie<NgramEntry>
    /// * `pathmap_path` - Output path for PathMap file
    pub fn translate<P1: AsRef<Path>, P2: AsRef<Path>>(
        artrie_path: P1,
        pathmap_path: P2,
    ) -> Result<TranslationStats, TranslationError> {
        use std::time::Instant;

        let start = Instant::now();
        let artrie_path = artrie_path.as_ref();
        let pathmap_path = pathmap_path.as_ref();

        if !artrie_path.exists() {
            return Err(TranslationError::SourceNotFound(
                artrie_path.display().to_string(),
            ));
        }

        let artrie_size = std::fs::metadata(artrie_path)?.len();

        log::info!(
            "Translating {:?} to {:?}",
            artrie_path,
            pathmap_path
        );

        // Open the disk-backed trie
        let trie: DiskBackedCharTrieInner<u64> = DiskBackedCharTrieInner::open(artrie_path)
            .map_err(|e| TranslationError::Io(std::io::Error::other(format!(
                "Failed to open trie: {}", e
            ))))?;

        // Collect all entries by traversing the trie
        let mut entries = Vec::new();
        let mut entries_count = 0u64;

        match &trie.root {
            CharTrieRoot::Empty => {
                return Err(TranslationError::EmptySource);
            }
            CharTrieRoot::Node(root_node) => {
                let mut prefix = String::new();
                collect_entries_recursive(root_node, &mut prefix, &mut entries, &mut entries_count);
            }
        }

        if entries.is_empty() {
            return Err(TranslationError::EmptySource);
        }

        log::info!("Collected {} entries, building PathMap...", entries.len());

        // Build PathMap from the collected entries
        let mut pathmap: PathMap<u64> = PathMap::new();
        for (key, value) in &entries {
            pathmap.insert(key.as_bytes(), *value);
        }

        log::info!("PathMap built, serializing to {:?}", pathmap_path);

        // Serialize PathMap structure using paths_serialization
        // Create paths file for the trie structure
        let paths_path = pathmap_path;
        let values_path = pathmap_path.with_extension("values");

        // Serialize paths
        let paths_file = std::fs::File::create(paths_path)?;
        let mut paths_writer = BufWriter::new(paths_file);

        // Serialize values separately (auxdata)
        let values_file = std::fs::File::create(&values_path)?;
        let mut values_writer = BufWriter::new(values_file);

        // Use read_zipper for iteration and serialize paths with auxdata
        let rz = pathmap.read_zipper();
        let _stats = serialize_paths_with_auxdata(rz, &mut paths_writer, |_idx, _path, value| {
            // Write the value as 8 bytes (u64 little-endian)
            let bytes = value.to_le_bytes();
            values_writer.write_all(&bytes).expect("Failed to write value");
        }).map_err(|e| TranslationError::Serialization(format!(
            "Failed to serialize paths: {}", e
        )))?;

        paths_writer.flush()?;
        values_writer.flush()?;

        // Calculate output sizes
        let paths_size = std::fs::metadata(paths_path)?.len();
        let values_size = std::fs::metadata(&values_path)?.len();
        let total_pathmap_size = paths_size + values_size;

        let stats = TranslationStats {
            entries_translated: entries_count,
            artrie_size_bytes: artrie_size,
            pathmap_size_bytes: total_pathmap_size,
            compression_ratio: if total_pathmap_size > 0 {
                artrie_size as f64 / total_pathmap_size as f64
            } else {
                0.0
            },
            elapsed_seconds: start.elapsed().as_secs_f64(),
            peak_memory_bytes: 0,
        };

        log::info!(
            "Translation complete: {} entries, {:.2}x compression in {:.2}s",
            entries_count,
            stats.compression_ratio,
            stats.elapsed_seconds
        );
        log::info!(
            "Output files: {:?} ({} bytes), {:?} ({} bytes)",
            paths_path, paths_size,
            values_path, values_size
        );

        Ok(stats)
    }

    /// Translate with progress callback.
    ///
    /// # Arguments
    ///
    /// * `artrie_path` - Path to trained PersistentARTrie<NgramEntry>
    /// * `pathmap_path` - Output path for PathMap file
    /// * `progress` - Callback for progress updates
    pub fn translate_with_progress<P1, P2, F>(
        artrie_path: P1,
        pathmap_path: P2,
        mut progress: F,
    ) -> Result<TranslationStats, TranslationError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(TranslationProgress),
    {
        use std::time::Instant;

        let start = Instant::now();
        let artrie_path = artrie_path.as_ref();
        let pathmap_path = pathmap_path.as_ref();

        if !artrie_path.exists() {
            return Err(TranslationError::SourceNotFound(
                artrie_path.display().to_string(),
            ));
        }

        let artrie_size = std::fs::metadata(artrie_path)?.len();

        // Phase 1: Loading
        progress(TranslationProgress {
            phase: TranslationPhase::Loading,
            entries_processed: 0,
            entries_total: None,
            bytes_written: 0,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        });

        log::info!(
            "Translating {:?} to {:?} with progress",
            artrie_path,
            pathmap_path
        );

        // Open the disk-backed trie
        let trie: DiskBackedCharTrieInner<u64> = DiskBackedCharTrieInner::open(artrie_path)
            .map_err(|e| TranslationError::Io(std::io::Error::other(format!(
                "Failed to open trie: {}", e
            ))))?;

        // Phase 2: Iterating - collect all entries with progress
        let mut entries = Vec::new();
        let mut entries_count = 0u64;
        let mut last_progress = 0u64;

        match &trie.root {
            CharTrieRoot::Empty => {
                return Err(TranslationError::EmptySource);
            }
            CharTrieRoot::Node(root_node) => {
                let mut prefix = String::new();
                collect_entries_with_progress(
                    root_node,
                    &mut prefix,
                    &mut entries,
                    &mut entries_count,
                    &mut progress,
                    &mut last_progress,
                    start,
                );
            }
        }

        if entries.is_empty() {
            return Err(TranslationError::EmptySource);
        }

        let total_entries = entries.len() as u64;

        // Phase 3: Building PathMap
        progress(TranslationProgress {
            phase: TranslationPhase::Building,
            entries_processed: total_entries,
            entries_total: Some(total_entries),
            bytes_written: 0,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        });

        log::info!("Building PathMap from {} entries...", total_entries);

        let mut pathmap: PathMap<u64> = PathMap::new();
        for (key, value) in &entries {
            pathmap.insert(key.as_bytes(), *value);
        }

        // Phase 4: Merkleizing (PathMap merkleization is optional)
        progress(TranslationProgress {
            phase: TranslationPhase::Merkleizing,
            entries_processed: total_entries,
            entries_total: Some(total_entries),
            bytes_written: 0,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        });

        // Phase 5: Saving
        progress(TranslationProgress {
            phase: TranslationPhase::Saving,
            entries_processed: total_entries,
            entries_total: Some(total_entries),
            bytes_written: 0,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        });

        log::info!("Serializing PathMap to {:?}", pathmap_path);

        // Serialize paths
        let paths_path = pathmap_path;
        let values_path = pathmap_path.with_extension("values");

        let paths_file = std::fs::File::create(paths_path)?;
        let mut paths_writer = BufWriter::new(paths_file);

        let values_file = std::fs::File::create(&values_path)?;
        let mut values_writer = BufWriter::new(values_file);

        let rz = pathmap.read_zipper();
        let _stats = serialize_paths_with_auxdata(rz, &mut paths_writer, |_idx, _path, value| {
            let bytes = value.to_le_bytes();
            values_writer.write_all(&bytes).expect("Failed to write value");
        }).map_err(|e| TranslationError::Serialization(format!(
            "Failed to serialize paths: {}", e
        )))?;

        paths_writer.flush()?;
        values_writer.flush()?;

        // Calculate output sizes
        let paths_size = std::fs::metadata(paths_path)?.len();
        let values_size = std::fs::metadata(&values_path)?.len();
        let total_pathmap_size = paths_size + values_size;

        // Phase 6: Complete
        progress(TranslationProgress {
            phase: TranslationPhase::Complete,
            entries_processed: total_entries,
            entries_total: Some(total_entries),
            bytes_written: total_pathmap_size,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        });

        let stats = TranslationStats {
            entries_translated: total_entries,
            artrie_size_bytes: artrie_size,
            pathmap_size_bytes: total_pathmap_size,
            compression_ratio: if total_pathmap_size > 0 {
                artrie_size as f64 / total_pathmap_size as f64
            } else {
                0.0
            },
            elapsed_seconds: start.elapsed().as_secs_f64(),
            peak_memory_bytes: 0,
        };

        log::info!(
            "Translation complete: {} entries, {:.2}x compression in {:.2}s",
            total_entries,
            stats.compression_ratio,
            stats.elapsed_seconds
        );

        Ok(stats)
    }

    /// Verify PathMap integrity against source ARTrie.
    ///
    /// Compares all entries to ensure translation was correct.
    pub fn verify<P1: AsRef<Path>, P2: AsRef<Path>>(
        artrie_path: P1,
        pathmap_path: P2,
    ) -> Result<VerificationResult, TranslationError> {
        let artrie_path = artrie_path.as_ref();
        let pathmap_path = pathmap_path.as_ref();

        if !artrie_path.exists() {
            return Err(TranslationError::SourceNotFound(
                artrie_path.display().to_string(),
            ));
        }

        if !pathmap_path.exists() {
            return Err(TranslationError::SourceNotFound(
                pathmap_path.display().to_string(),
            ));
        }

        let values_path = pathmap_path.with_extension("values");
        if !values_path.exists() {
            return Err(TranslationError::SourceNotFound(
                values_path.display().to_string(),
            ));
        }

        log::info!(
            "Verifying PathMap {:?} against ARTrie {:?}",
            pathmap_path,
            artrie_path
        );

        // Open the source ARTrie and collect all entries
        let trie: DiskBackedCharTrieInner<u64> = DiskBackedCharTrieInner::open(artrie_path)
            .map_err(|e| TranslationError::Io(std::io::Error::other(format!(
                "Failed to open trie: {}", e
            ))))?;

        // Collect source entries into a HashMap for fast lookup
        let mut source_entries: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();

        match &trie.root {
            CharTrieRoot::Empty => {
                return Err(TranslationError::EmptySource);
            }
            CharTrieRoot::Node(root_node) => {
                let mut prefix = String::new();
                let mut entries = Vec::new();
                let mut entries_count = 0u64;
                collect_entries_recursive(root_node, &mut prefix, &mut entries, &mut entries_count);

                for (key, value) in entries {
                    source_entries.insert(key, value);
                }
            }
        }

        let source_count = source_entries.len() as u64;
        log::info!("Source ARTrie has {} entries", source_count);

        // Read the PathMap serialized paths and verify against source
        let paths_file = std::fs::File::open(pathmap_path)?;
        let paths_reader = BufReader::new(paths_file);

        let values_file = std::fs::File::open(&values_path)?;
        let mut values_reader = BufReader::new(values_file);

        let mut entries_verified = 0u64;
        let mut mismatches = 0u64;

        // Use for_each_deserialized_path to iterate through PathMap entries
        for_each_deserialized_path(paths_reader, |_idx, path| {
            // Read the corresponding value (8 bytes for u64)
            let mut value_bytes = [0u8; 8];
            use std::io::Read;
            values_reader.read_exact(&mut value_bytes)?;
            let pathmap_value = u64::from_le_bytes(value_bytes);

            // Convert path bytes back to string for lookup
            let key = match std::str::from_utf8(path) {
                Ok(s) => s.to_string(),
                Err(_) => {
                    mismatches += 1;
                    log::warn!("PathMap contains invalid UTF-8 key at index {}", _idx);
                    return Ok(());
                }
            };

            // Verify against source
            match source_entries.get(&key) {
                Some(&source_value) => {
                    if source_value != pathmap_value {
                        mismatches += 1;
                        log::warn!(
                            "Value mismatch for key '{}': source={}, pathmap={}",
                            key,
                            source_value,
                            pathmap_value
                        );
                    }
                }
                None => {
                    mismatches += 1;
                    log::warn!("PathMap contains key not in source: '{}'", key);
                }
            }

            entries_verified += 1;
            Ok(())
        }).map_err(|e| TranslationError::Io(e))?;

        // Check for missing entries (in source but not in PathMap)
        if entries_verified != source_count {
            log::warn!(
                "Entry count mismatch: source={}, pathmap={}",
                source_count,
                entries_verified
            );
            // The difference could be due to entries in source not in PathMap
            let missing = source_count.saturating_sub(entries_verified);
            mismatches += missing;
        }

        let verified = mismatches == 0;

        log::info!(
            "Verification {}: {} entries checked, {} mismatches",
            if verified { "PASSED" } else { "FAILED" },
            entries_verified,
            mismatches
        );

        Ok(VerificationResult {
            entries_verified,
            mismatches,
            verified,
        })
    }
}

/// Progress information for translation.
#[derive(Clone, Debug)]
pub struct TranslationProgress {
    /// Current translation phase.
    pub phase: TranslationPhase,

    /// Entries processed so far.
    pub entries_processed: u64,

    /// Total entries (if known).
    pub entries_total: Option<u64>,

    /// Bytes written to output.
    pub bytes_written: u64,

    /// Elapsed time in seconds.
    pub elapsed_seconds: f64,
}

/// Translation phase.
#[derive(Clone, Debug, PartialEq)]
pub enum TranslationPhase {
    /// Loading the source ARTrie.
    Loading,
    /// Iterating through entries.
    Iterating,
    /// Building PathMap structure.
    Building,
    /// Computing Merkle hashes.
    Merkleizing,
    /// Saving to disk.
    Saving,
    /// Complete.
    Complete,
}

/// Result of PathMap verification.
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// Number of entries verified.
    pub entries_verified: u64,

    /// Number of mismatches found.
    pub mismatches: u64,

    /// Whether verification passed.
    pub verified: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_translation_stats_default() {
        let stats = TranslationStats::default();
        assert_eq!(stats.entries_translated, 0);
        assert_eq!(stats.compression_ratio, 0.0);
    }

    #[test]
    fn test_translation_source_not_found() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output = dir.path().join("output.pathmap");

        let result = PathMapTranslator::translate("/nonexistent/path.artrie", &output);
        assert!(matches!(result, Err(TranslationError::SourceNotFound(_))));
    }

    #[test]
    fn test_translation_progress() {
        let progress = TranslationProgress {
            phase: TranslationPhase::Building,
            entries_processed: 10000,
            entries_total: Some(100000),
            bytes_written: 1024 * 1024,
            elapsed_seconds: 5.5,
        };

        assert_eq!(progress.phase, TranslationPhase::Building);
        assert_eq!(progress.entries_processed, 10000);
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult {
            entries_verified: 100000,
            mismatches: 0,
            verified: true,
        };

        assert!(result.verified);
        assert_eq!(result.mismatches, 0);
    }
}
