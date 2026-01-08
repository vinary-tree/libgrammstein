//! Dictionary extraction from n-gram models.
//!
//! Extracts a `DoubleArrayTrieChar` dictionary from an existing n-gram model's 1-grams.
//! This is a post-processing step that can be run anytime after LM training.
//!
//! The dictionary can be used with `liblevenshtein` for fast lexical correction
//! and spelling suggestions.

use std::io::Write;
use std::path::Path;

use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;
use liblevenshtein::dictionary::persistent_artrie_char::dict_impl_char::{
    CharTrieNodeInner, CharTrieRoot,
};
use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;

/// Recursively traverse the trie and collect unigrams (single words without spaces).
///
/// # Arguments
///
/// * `node` - Current trie node being traversed
/// * `prefix` - Current string prefix built during traversal
/// * `min_count` - Minimum frequency threshold for inclusion
/// * `words` - Output vector to collect matching words
/// * `total_unigrams` - Counter for total 1-grams examined
/// * `filtered_count` - Counter for 1-grams filtered by threshold
fn collect_unigrams_recursive(
    node: &CharTrieNodeInner<u64>,
    prefix: &mut String,
    min_count: u64,
    words: &mut Vec<String>,
    total_unigrams: &mut u64,
    filtered_count: &mut u64,
) {
    // Check if this node represents a complete term
    if node.is_final() {
        // Check if this is a unigram (no whitespace = single word)
        // Also skip MKN metadata entries (they start with \x00)
        if !prefix.starts_with('\x00') && !prefix.contains(char::is_whitespace) {
            *total_unigrams += 1;

            // Get the count value stored at this node
            if let Some(count) = node.value.as_ref() {
                if *count >= min_count {
                    words.push(prefix.clone());
                } else {
                    *filtered_count += 1;
                }
            } else {
                // If no count value, skip
                *filtered_count += 1;
            }
        }
    }

    // Recursively traverse children
    for (c, child) in node.iter_children() {
        prefix.push(c);
        collect_unigrams_recursive(child, prefix, min_count, words, total_unigrams, filtered_count);
        prefix.pop();
    }
}

/// Recursively traverse the trie and collect unigrams with progress reporting.
///
/// # Arguments
///
/// * `node` - Current trie node being traversed
/// * `prefix` - Current string prefix built during traversal
/// * `min_count` - Minimum frequency threshold for inclusion
/// * `words` - Output vector to collect matching words
/// * `total_unigrams` - Counter for total 1-grams examined
/// * `filtered_count` - Counter for 1-grams filtered by threshold
/// * `progress_callback` - Called periodically with progress updates
/// * `last_progress` - Tracks when we last called the progress callback
fn collect_unigrams_with_progress<F>(
    node: &CharTrieNodeInner<u64>,
    prefix: &mut String,
    min_count: u64,
    words: &mut Vec<String>,
    total_unigrams: &mut u64,
    filtered_count: &mut u64,
    progress_callback: &mut F,
    last_progress: &mut u64,
    start_time: std::time::Instant,
) where
    F: FnMut(ExtractionProgress),
{
    // Check if this node represents a complete term
    if node.is_final() {
        // Check if this is a unigram (no whitespace = single word)
        // Also skip MKN metadata entries (they start with \x00)
        if !prefix.starts_with('\x00') && !prefix.contains(char::is_whitespace) {
            *total_unigrams += 1;

            // Get the count value stored at this node
            if let Some(count) = node.value.as_ref() {
                if *count >= min_count {
                    words.push(prefix.clone());
                } else {
                    *filtered_count += 1;
                }
            } else {
                *filtered_count += 1;
            }

            // Report progress every 100k words
            if *total_unigrams - *last_progress >= 100_000 {
                *last_progress = *total_unigrams;
                progress_callback(ExtractionProgress {
                    phase: ExtractionPhase::Filtering,
                    items_processed: *total_unigrams,
                    items_total: None,
                    items_accepted: words.len() as u64,
                    elapsed_seconds: start_time.elapsed().as_secs_f64(),
                    words_processed: *total_unigrams,
                });
            }
        }
    }

    // Recursively traverse children
    for (c, child) in node.iter_children() {
        prefix.push(c);
        collect_unigrams_with_progress(
            child,
            prefix,
            min_count,
            words,
            total_unigrams,
            filtered_count,
            progress_callback,
            last_progress,
            start_time,
        );
        prefix.pop();
    }
}

/// Statistics from dictionary extraction.
#[derive(Clone, Debug, Default)]
pub struct ExtractionStats {
    /// Total 1-grams examined.
    pub total_unigrams: u64,

    /// 1-grams meeting frequency threshold.
    pub filtered_unigrams: u64,

    /// Final vocabulary size in dictionary.
    pub vocabulary_size: u64,

    /// Source model size in bytes.
    pub source_size_bytes: u64,

    /// Output dictionary size in bytes.
    pub dictionary_size_bytes: u64,

    /// Extraction time in seconds.
    pub elapsed_seconds: f64,

    /// Words extracted (alias for vocabulary_size for CLI compatibility).
    pub words_extracted: u64,

    /// Words filtered out (alias for filtered_unigrams for CLI compatibility).
    pub words_filtered: u64,

    /// Dictionary size in bytes (alias for dictionary_size_bytes for CLI compatibility).
    pub dict_size_bytes: u64,
}

/// Errors that can occur during extraction.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    /// Model file not found.
    #[error("Model file not found: {0}")]
    ModelNotFound(String),

    /// No 1-grams found in model.
    #[error("No 1-grams found in model")]
    NoUnigrams,

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Dictionary construction error.
    #[error("Dictionary construction error: {0}")]
    Construction(String),
}

/// Dictionary extractor for building lexical dictionaries from n-gram models.
///
/// The extractor reads 1-grams from a trained PersistentARTrie model and builds
/// a `DoubleArrayTrieChar` for fast read-only lexical lookup.
///
/// # Use Cases
///
/// - Spelling correction with Levenshtein automata
/// - Word completion and suggestions
/// - Vocabulary validation
///
/// # Example
///
/// ```ignore
/// use libgrammstein::sources::google_books::DictionaryExtractor;
///
/// // Extract dictionary with minimum frequency threshold
/// let stats = DictionaryExtractor::extract_to_file(
///     "english.artrie",
///     "english.dict",
///     100,  // Only include words with count >= 100
/// )?;
///
/// println!("Extracted {} words", stats.vocabulary_size);
/// ```
pub struct DictionaryExtractor;

impl DictionaryExtractor {
    /// Extract dictionary from n-gram model's 1-grams.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to PersistentARTrie n-gram model
    /// * `min_count` - Minimum frequency threshold (filters rare/misspelled words)
    ///
    /// # Returns
    ///
    /// A vector of words meeting the threshold, sorted alphabetically.
    pub fn extract_words<P: AsRef<Path>>(
        model_path: P,
        min_count: u64,
    ) -> Result<(Vec<String>, ExtractionStats), ExtractionError> {
        use std::time::Instant;

        let start = Instant::now();
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(ExtractionError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        let source_size = std::fs::metadata(model_path)?.len();

        log::info!(
            "Extracting dictionary from {:?} with min_count={}",
            model_path,
            min_count
        );

        // Open the disk-backed trie
        let trie: DiskBackedCharTrieInner<u64> = DiskBackedCharTrieInner::open(model_path)
            .map_err(|e| ExtractionError::Io(std::io::Error::other(format!(
                "Failed to open trie: {}", e
            ))))?;

        // Collect unigrams by traversing the trie
        let mut words = Vec::new();
        let mut total_unigrams = 0u64;
        let mut filtered_unigrams = 0u64;

        match &trie.root {
            CharTrieRoot::Empty => {
                log::warn!("Trie is empty, no words to extract");
            }
            CharTrieRoot::Node(root_node) => {
                let mut prefix = String::new();
                collect_unigrams_recursive(
                    root_node,
                    &mut prefix,
                    min_count,
                    &mut words,
                    &mut total_unigrams,
                    &mut filtered_unigrams,
                );
            }
        }

        // Sort words alphabetically
        words.sort();

        log::info!(
            "Extracted {} words from {} unigrams ({} filtered) in {:.2}s",
            words.len(),
            total_unigrams,
            filtered_unigrams,
            start.elapsed().as_secs_f64()
        );

        let stats = ExtractionStats {
            total_unigrams,
            filtered_unigrams,
            vocabulary_size: words.len() as u64,
            source_size_bytes: source_size,
            dictionary_size_bytes: 0,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            words_extracted: words.len() as u64,
            words_filtered: filtered_unigrams,
            dict_size_bytes: 0,
        };

        Ok((words, stats))
    }

    /// Extract dictionary and save to disk.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to PersistentARTrie n-gram model
    /// * `output_path` - Output path for dictionary file
    /// * `min_count` - Minimum frequency threshold
    pub fn extract_to_file<P: AsRef<Path>>(
        model_path: P,
        output_path: P,
        min_count: u64,
    ) -> Result<ExtractionStats, ExtractionError> {
        use std::time::Instant;

        let start = Instant::now();
        let output_path = output_path.as_ref();

        // Extract words from the model
        let (words, mut stats) = Self::extract_words(&model_path, min_count)?;

        if words.is_empty() {
            return Err(ExtractionError::NoUnigrams);
        }

        log::info!("Building DoubleArrayTrieChar from {} words", words.len());

        // Build DoubleArrayTrieChar from the sorted word list
        // DoubleArrayTrieChar::from_terms handles sorting and deduplication internally
        let dict: DoubleArrayTrieChar<()> = DoubleArrayTrieChar::from_terms(&words);

        log::info!("Serializing dictionary to {:?}", output_path);

        // Serialize to bytes using bincode
        let bytes = bincode::serialize(&dict)
            .map_err(|e| ExtractionError::Serialization(format!(
                "Failed to serialize dictionary: {}", e
            )))?;

        // Write to file
        let mut file = std::fs::File::create(output_path)?;
        file.write_all(&bytes)?;
        file.flush()?;

        // Update stats with dictionary size
        let dictionary_size = bytes.len() as u64;
        stats.dictionary_size_bytes = dictionary_size;
        stats.dict_size_bytes = dictionary_size;
        stats.elapsed_seconds = start.elapsed().as_secs_f64();

        log::info!(
            "Dictionary extraction complete: {} words, {} bytes in {:.2}s",
            stats.vocabulary_size,
            stats.dictionary_size_bytes,
            stats.elapsed_seconds
        );

        Ok(stats)
    }

    /// Extract dictionary with progress callback.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to PersistentARTrie n-gram model
    /// * `output_path` - Output path for dictionary file
    /// * `min_count` - Minimum frequency threshold
    /// * `progress` - Callback for progress updates
    pub fn extract_with_progress<P1, P2, F>(
        model_path: P1,
        output_path: P2,
        min_count: u64,
        mut progress: F,
    ) -> Result<ExtractionStats, ExtractionError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(ExtractionProgress),
    {
        use std::time::Instant;

        let start = Instant::now();
        let model_path = model_path.as_ref();
        let output_path = output_path.as_ref();

        if !model_path.exists() {
            return Err(ExtractionError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        let source_size = std::fs::metadata(model_path)?.len();

        // Phase 1: Loading
        progress(ExtractionProgress {
            phase: ExtractionPhase::Loading,
            items_processed: 0,
            items_total: None,
            items_accepted: 0,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            words_processed: 0,
        });

        // Open the disk-backed trie
        let trie: DiskBackedCharTrieInner<u64> = DiskBackedCharTrieInner::open(model_path)
            .map_err(|e| ExtractionError::Io(std::io::Error::other(format!(
                "Failed to open trie: {}", e
            ))))?;

        // Phase 2: Filtering - traverse trie and collect unigrams with progress
        let mut words = Vec::new();
        let mut total_unigrams = 0u64;
        let mut filtered_unigrams = 0u64;
        let mut last_progress = 0u64;

        match &trie.root {
            CharTrieRoot::Empty => {
                log::warn!("Trie is empty, no words to extract");
            }
            CharTrieRoot::Node(root_node) => {
                let mut prefix = String::new();
                collect_unigrams_with_progress(
                    root_node,
                    &mut prefix,
                    min_count,
                    &mut words,
                    &mut total_unigrams,
                    &mut filtered_unigrams,
                    &mut progress,
                    &mut last_progress,
                    start,
                );
            }
        }

        // Sort words alphabetically
        words.sort();

        if words.is_empty() {
            return Err(ExtractionError::NoUnigrams);
        }

        // Phase 3: Building
        progress(ExtractionProgress {
            phase: ExtractionPhase::Building,
            items_processed: total_unigrams,
            items_total: Some(total_unigrams),
            items_accepted: words.len() as u64,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            words_processed: total_unigrams,
        });

        // Build DoubleArrayTrieChar from the sorted word list
        let dict: DoubleArrayTrieChar<()> = DoubleArrayTrieChar::from_terms(&words);

        // Phase 4: Saving
        progress(ExtractionProgress {
            phase: ExtractionPhase::Saving,
            items_processed: total_unigrams,
            items_total: Some(total_unigrams),
            items_accepted: words.len() as u64,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            words_processed: total_unigrams,
        });

        // Serialize and write to file
        let bytes = bincode::serialize(&dict)
            .map_err(|e| ExtractionError::Serialization(format!(
                "Failed to serialize dictionary: {}", e
            )))?;

        let mut file = std::fs::File::create(output_path)?;
        file.write_all(&bytes)?;
        file.flush()?;

        let dictionary_size = bytes.len() as u64;

        // Phase 5: Complete
        progress(ExtractionProgress {
            phase: ExtractionPhase::Complete,
            items_processed: total_unigrams,
            items_total: Some(total_unigrams),
            items_accepted: words.len() as u64,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            words_processed: total_unigrams,
        });

        let stats = ExtractionStats {
            total_unigrams,
            filtered_unigrams,
            vocabulary_size: words.len() as u64,
            source_size_bytes: source_size,
            dictionary_size_bytes: dictionary_size,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            words_extracted: words.len() as u64,
            words_filtered: filtered_unigrams,
            dict_size_bytes: dictionary_size,
        };

        Ok(stats)
    }

    /// Extract dictionary to file with progress callback.
    ///
    /// Alias for `extract_with_progress` for CLI compatibility.
    pub fn extract_to_file_with_progress<P1, P2, F>(
        model_path: P1,
        output_path: P2,
        min_count: u64,
        progress: F,
    ) -> Result<ExtractionStats, ExtractionError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(ExtractionProgress),
    {
        Self::extract_with_progress(model_path, output_path, min_count, progress)
    }
}

/// Progress information for dictionary extraction.
#[derive(Clone, Debug)]
pub struct ExtractionProgress {
    /// Current extraction phase.
    pub phase: ExtractionPhase,

    /// Items processed so far.
    pub items_processed: u64,

    /// Total items (if known).
    pub items_total: Option<u64>,

    /// Items accepted (meeting threshold).
    pub items_accepted: u64,

    /// Elapsed time in seconds.
    pub elapsed_seconds: f64,

    /// Words processed (alias for items_processed for CLI compatibility).
    pub words_processed: u64,
}

/// Extraction phase.
#[derive(Clone, Debug, PartialEq)]
pub enum ExtractionPhase {
    /// Loading the n-gram model.
    Loading,
    /// Filtering 1-grams by frequency.
    Filtering,
    /// Building the DoubleArrayTrieChar.
    Building,
    /// Saving to disk.
    Saving,
    /// Complete.
    Complete,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_extraction_stats_default() {
        let stats = ExtractionStats::default();
        assert_eq!(stats.total_unigrams, 0);
        assert_eq!(stats.vocabulary_size, 0);
    }

    #[test]
    fn test_extraction_model_not_found() {
        let result = DictionaryExtractor::extract_words("/nonexistent/path.artrie", 100);
        assert!(matches!(result, Err(ExtractionError::ModelNotFound(_))));
    }

    #[test]
    fn test_extraction_progress() {
        let progress = ExtractionProgress {
            phase: ExtractionPhase::Filtering,
            items_processed: 1000,
            items_total: Some(10000),
            items_accepted: 500,
            elapsed_seconds: 1.5,
            words_processed: 1000,
        };

        assert_eq!(progress.phase, ExtractionPhase::Filtering);
        assert_eq!(progress.items_processed, 1000);
        assert_eq!(progress.items_accepted, 500);
    }
}
