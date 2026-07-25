//! Dictionary building from extracted word frequencies.
//!
//! Converts word frequency data into a compact spelling dictionary
//! optimized for read-only lookups during WFST rescoring.
//!
//! # Serialization
//!
//! Binary serialization (`.dict`) requires the `serde-extras` feature.
//! Text-based import/export (tab-separated format) is always available.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "serde-extras")]
use std::io::Read;

use serde::{Deserialize, Serialize};

use super::extractor::WordExtractor;
use super::types::{DictionaryMetadata, WordEntry};

/// Error type for dictionary operations.
#[derive(Debug)]
pub enum DictionaryError {
    /// I/O error.
    Io(std::io::Error),
    /// Serialization error.
    Serialization(String),
    /// Invalid dictionary format.
    InvalidFormat(String),
}

impl std::fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for DictionaryError {}

impl From<std::io::Error> for DictionaryError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type for dictionary operations.
pub type Result<T> = std::result::Result<T, DictionaryError>;

/// Builder for creating spelling dictionaries.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::dictionary::{DictionaryBuilder, WordExtractor};
///
/// let extractor = WordExtractor::new();
/// extractor.add_sentence("The quick brown fox.");
///
/// let dictionary = DictionaryBuilder::new()
///     .min_frequency(1)
///     .language("en")
///     .build_from_extractor(&extractor)?;
///
/// dictionary.save("words.dict")?;
/// ```
#[derive(Debug, Clone)]
pub struct DictionaryBuilder {
    /// Minimum frequency threshold for inclusion.
    min_frequency: u64,
    /// Language code (BCP 47).
    language: String,
    /// Source description.
    source: Option<String>,
    /// Whether to compute log probabilities.
    compute_log_probs: bool,
}

impl Default for DictionaryBuilder {
    fn default() -> Self {
        Self {
            min_frequency: 1,
            language: "en".to_string(),
            source: None,
            compute_log_probs: true,
        }
    }
}

impl DictionaryBuilder {
    /// Create a new dictionary builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum frequency threshold.
    pub fn min_frequency(mut self, min_frequency: u64) -> Self {
        self.min_frequency = min_frequency;
        self
    }

    /// Set the language code.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Set the source description.
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set whether to compute log probabilities.
    pub fn compute_log_probs(mut self, compute: bool) -> Self {
        self.compute_log_probs = compute;
        self
    }

    /// Build a dictionary from a word extractor.
    pub fn build_from_extractor(&self, extractor: &WordExtractor) -> Result<SpellingDictionary> {
        let total_tokens = extractor.total_tokens();
        let total_f64 = total_tokens as f64;

        // Get entries above frequency threshold
        let mut entries: Vec<WordEntry> = extractor
            .entries_filtered(self.min_frequency)
            .into_iter()
            .map(|mut e| {
                if self.compute_log_probs && total_f64 > 0.0 {
                    e.log_prob = Some((e.frequency as f64 / total_f64).ln());
                }
                e
            })
            .collect();

        // Sort by frequency descending for consistent ordering
        entries.sort_by_key(|e| std::cmp::Reverse(e.frequency));

        // Build word-to-index map for fast lookups
        let word_index: HashMap<String, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.word.clone(), i))
            .collect();

        let metadata = DictionaryMetadata {
            version: 1,
            language: self.language.clone(),
            word_count: entries.len(),
            total_tokens,
            min_frequency: self.min_frequency,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            source: self.source.clone(),
        };

        Ok(SpellingDictionary {
            metadata,
            entries,
            word_index,
        })
    }

    /// Build a dictionary from raw word counts.
    pub fn build_from_counts(&self, counts: HashMap<String, u64>) -> Result<SpellingDictionary> {
        let total_tokens: u64 = counts.values().sum();
        let total_f64 = total_tokens as f64;

        // Filter and create entries
        let mut entries: Vec<WordEntry> = counts
            .into_iter()
            .filter(|(_, freq)| *freq >= self.min_frequency)
            .map(|(word, frequency)| {
                let log_prob = if self.compute_log_probs && total_f64 > 0.0 {
                    Some((frequency as f64 / total_f64).ln())
                } else {
                    None
                };
                WordEntry {
                    word,
                    frequency,
                    log_prob,
                }
            })
            .collect();

        // Sort by frequency descending
        entries.sort_by_key(|e| std::cmp::Reverse(e.frequency));

        // Build word-to-index map
        let word_index: HashMap<String, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.word.clone(), i))
            .collect();

        let metadata = DictionaryMetadata {
            version: 1,
            language: self.language.clone(),
            word_count: entries.len(),
            total_tokens,
            min_frequency: self.min_frequency,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            source: self.source.clone(),
        };

        Ok(SpellingDictionary {
            metadata,
            entries,
            word_index,
        })
    }
}

/// A spelling dictionary optimized for read-only lookups.
///
/// Contains word entries sorted by frequency with precomputed
/// log probabilities for language model scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellingDictionary {
    /// Dictionary metadata.
    pub metadata: DictionaryMetadata,
    /// Word entries sorted by frequency (descending).
    entries: Vec<WordEntry>,
    /// Word to index mapping for O(1) lookups.
    #[serde(skip)]
    word_index: HashMap<String, usize>,
}

impl SpellingDictionary {
    /// Load a dictionary from a binary file.
    ///
    /// Requires the `serde-extras` feature for bincode serialization.
    #[cfg(feature = "serde-extras")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"DICT" {
            return Err(DictionaryError::InvalidFormat(
                "Invalid magic number".to_string(),
            ));
        }

        // Read version
        let mut version = [0u8; 4];
        reader.read_exact(&mut version)?;
        let version = u32::from_le_bytes(version);
        if version != 1 {
            return Err(DictionaryError::InvalidFormat(format!(
                "Unsupported version: {}",
                version
            )));
        }

        // Read the rest as bincode
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let mut dict: SpellingDictionary = crate::bincode_compat::deserialize(&data)
            .map_err(|e| DictionaryError::Serialization(e.to_string()))?;

        // Rebuild word index
        dict.word_index = dict
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.word.clone(), i))
            .collect();

        Ok(dict)
    }

    /// Save the dictionary to a binary file.
    ///
    /// Requires the `serde-extras` feature for bincode serialization.
    #[cfg(feature = "serde-extras")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic number
        writer.write_all(b"DICT")?;

        // Write version
        writer.write_all(&1u32.to_le_bytes())?;

        // Write bincode data
        let data = crate::bincode_compat::serialize(self)
            .map_err(|e| DictionaryError::Serialization(e.to_string()))?;
        writer.write_all(&data)?;

        Ok(())
    }

    /// Check if a word is in the dictionary.
    pub fn contains(&self, word: &str) -> bool {
        self.word_index.contains_key(word)
    }

    /// Get the entry for a word.
    pub fn get(&self, word: &str) -> Option<&WordEntry> {
        self.word_index.get(word).map(|&i| &self.entries[i])
    }

    /// Get the frequency of a word.
    pub fn frequency(&self, word: &str) -> Option<u64> {
        self.get(word).map(|e| e.frequency)
    }

    /// Get the log probability of a word.
    pub fn log_prob(&self, word: &str) -> Option<f64> {
        self.get(word).and_then(|e| e.log_prob)
    }

    /// Get the rank of a word (0 = most frequent).
    pub fn rank(&self, word: &str) -> Option<usize> {
        self.word_index.get(word).copied()
    }

    /// Get the number of words in the dictionary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the total token count from the source corpus.
    pub fn total_tokens(&self) -> u64 {
        self.metadata.total_tokens
    }

    /// Get all words in the dictionary.
    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.word.as_str())
    }

    /// Get all entries in the dictionary (sorted by frequency).
    pub fn entries(&self) -> &[WordEntry] {
        &self.entries
    }

    /// Get the top N most frequent words.
    pub fn top_n(&self, n: usize) -> &[WordEntry] {
        let end = n.min(self.entries.len());
        &self.entries[..end]
    }

    /// Get words with frequency in a given range.
    pub fn words_in_frequency_range(&self, min: u64, max: u64) -> Vec<&WordEntry> {
        self.entries
            .iter()
            .filter(|e| e.frequency >= min && e.frequency <= max)
            .collect()
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &DictionaryMetadata {
        &self.metadata
    }

    /// Merge another dictionary into this one.
    ///
    /// Words present in both dictionaries have their frequencies summed.
    pub fn merge(&mut self, other: &SpellingDictionary) {
        // Create a map for efficient merging
        let mut freq_map: HashMap<String, u64> = self
            .entries
            .iter()
            .map(|e| (e.word.clone(), e.frequency))
            .collect();

        // Add frequencies from other dictionary
        for entry in &other.entries {
            *freq_map.entry(entry.word.clone()).or_insert(0) += entry.frequency;
        }

        // Rebuild entries
        let total_tokens: u64 = freq_map.values().sum();
        let total_f64 = total_tokens as f64;

        let mut entries: Vec<WordEntry> = freq_map
            .into_iter()
            .map(|(word, frequency)| {
                let log_prob = if total_f64 > 0.0 {
                    Some((frequency as f64 / total_f64).ln())
                } else {
                    None
                };
                WordEntry {
                    word,
                    frequency,
                    log_prob,
                }
            })
            .collect();

        entries.sort_by_key(|e| std::cmp::Reverse(e.frequency));

        // Rebuild index
        self.word_index = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.word.clone(), i))
            .collect();

        // Update metadata
        self.metadata.word_count = entries.len();
        self.metadata.total_tokens = total_tokens;
        self.metadata.created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.entries = entries;
    }

    /// Export to a plain text file (word frequency format).
    pub fn export_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for entry in &self.entries {
            writeln!(writer, "{}\t{}", entry.word, entry.frequency)?;
        }

        Ok(())
    }

    /// Import from a plain text file (word frequency format).
    pub fn import_text<P: AsRef<Path>>(path: P, language: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut counts = HashMap::new();
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                if let Ok(freq) = parts[1].trim().parse::<u64>() {
                    counts.insert(parts[0].to_string(), freq);
                }
            }
        }

        DictionaryBuilder::new()
            .language(language)
            .build_from_counts(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_from_counts() {
        let mut counts = HashMap::new();
        counts.insert("the".to_string(), 100);
        counts.insert("quick".to_string(), 50);
        counts.insert("brown".to_string(), 25);
        counts.insert("fox".to_string(), 10);
        counts.insert("rare".to_string(), 1);

        let dict = DictionaryBuilder::new()
            .min_frequency(5)
            .language("en")
            .build_from_counts(counts)
            .unwrap();

        assert_eq!(dict.len(), 4); // 'rare' filtered out
        assert!(dict.contains("the"));
        assert!(dict.contains("fox"));
        assert!(!dict.contains("rare"));

        // Check ordering (by frequency)
        let entries = dict.entries();
        assert_eq!(entries[0].word, "the");
        assert_eq!(entries[1].word, "quick");
        assert_eq!(entries[2].word, "brown");
        assert_eq!(entries[3].word, "fox");
    }

    #[test]
    fn test_build_from_extractor() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("the quick brown fox jumps over the lazy dog");
        extractor.add_sentence("the fox is quick and brown");

        let dict = DictionaryBuilder::new()
            .min_frequency(2)
            .build_from_extractor(&extractor)
            .unwrap();

        // Only words appearing >= 2 times
        assert!(dict.contains("the")); // 3 times
        assert!(dict.contains("quick")); // 2 times
        assert!(dict.contains("brown")); // 2 times
        assert!(dict.contains("fox")); // 2 times
        assert!(!dict.contains("lazy")); // 1 time
    }

    #[test]
    fn test_frequency_and_rank() {
        let mut counts = HashMap::new();
        counts.insert("a".to_string(), 100);
        counts.insert("b".to_string(), 50);
        counts.insert("c".to_string(), 25);

        let dict = DictionaryBuilder::new().build_from_counts(counts).unwrap();

        assert_eq!(dict.frequency("a"), Some(100));
        assert_eq!(dict.frequency("b"), Some(50));
        assert_eq!(dict.frequency("c"), Some(25));
        assert_eq!(dict.frequency("d"), None);

        assert_eq!(dict.rank("a"), Some(0));
        assert_eq!(dict.rank("b"), Some(1));
        assert_eq!(dict.rank("c"), Some(2));
        assert_eq!(dict.rank("d"), None);
    }

    #[test]
    fn test_log_probs() {
        let mut counts = HashMap::new();
        counts.insert("a".to_string(), 100);
        counts.insert("b".to_string(), 100);

        let dict = DictionaryBuilder::new()
            .compute_log_probs(true)
            .build_from_counts(counts)
            .unwrap();

        // Total = 200, each word has probability 0.5
        // ln(0.5) = -ln(2) ≈ -0.693147
        let log_prob = dict.log_prob("a").unwrap();
        assert!((log_prob - (-std::f64::consts::LN_2)).abs() < 0.001);
    }

    #[test]
    fn test_top_n() {
        let mut counts = HashMap::new();
        counts.insert("a".to_string(), 100);
        counts.insert("b".to_string(), 50);
        counts.insert("c".to_string(), 25);
        counts.insert("d".to_string(), 10);
        counts.insert("e".to_string(), 5);

        let dict = DictionaryBuilder::new().build_from_counts(counts).unwrap();

        let top3 = dict.top_n(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].word, "a");
        assert_eq!(top3[1].word, "b");
        assert_eq!(top3[2].word, "c");
    }

    #[test]
    fn test_merge() {
        let mut counts1 = HashMap::new();
        counts1.insert("a".to_string(), 100);
        counts1.insert("b".to_string(), 50);

        let mut counts2 = HashMap::new();
        counts2.insert("b".to_string(), 30);
        counts2.insert("c".to_string(), 20);

        let mut dict1 = DictionaryBuilder::new().build_from_counts(counts1).unwrap();

        let dict2 = DictionaryBuilder::new().build_from_counts(counts2).unwrap();

        dict1.merge(&dict2);

        assert_eq!(dict1.frequency("a"), Some(100));
        assert_eq!(dict1.frequency("b"), Some(80)); // 50 + 30
        assert_eq!(dict1.frequency("c"), Some(20));
    }

    #[test]
    #[cfg(feature = "serde-extras")]
    fn test_save_load() {
        let mut counts = HashMap::new();
        counts.insert("hello".to_string(), 100);
        counts.insert("world".to_string(), 50);

        let dict = DictionaryBuilder::new()
            .language("en")
            .source("test")
            .build_from_counts(counts)
            .unwrap();

        // Save to temp file
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.dict");

        dict.save(&path).unwrap();
        assert!(path.exists());

        // Load and verify
        let loaded = SpellingDictionary::load(&path).unwrap();
        assert_eq!(loaded.len(), dict.len());
        assert_eq!(loaded.frequency("hello"), Some(100));
        assert_eq!(loaded.frequency("world"), Some(50));
        assert_eq!(loaded.metadata().language, "en");
    }

    #[test]
    fn test_export_import_text() {
        let mut counts = HashMap::new();
        counts.insert("hello".to_string(), 100);
        counts.insert("world".to_string(), 50);

        let dict = DictionaryBuilder::new().build_from_counts(counts).unwrap();

        // Export to temp file
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("words.txt");

        dict.export_text(&path).unwrap();

        // Import and verify
        let imported = SpellingDictionary::import_text(&path, "en").unwrap();
        assert_eq!(imported.len(), 2);
        assert_eq!(imported.frequency("hello"), Some(100));
        assert_eq!(imported.frequency("world"), Some(50));
    }

    #[test]
    fn test_words_in_frequency_range() {
        let mut counts = HashMap::new();
        counts.insert("a".to_string(), 100);
        counts.insert("b".to_string(), 50);
        counts.insert("c".to_string(), 25);
        counts.insert("d".to_string(), 10);

        let dict = DictionaryBuilder::new().build_from_counts(counts).unwrap();

        let range = dict.words_in_frequency_range(20, 60);
        assert_eq!(range.len(), 2); // b (50) and c (25)
    }
}
