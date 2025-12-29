//! Types for dictionary extraction and representation.

use serde::{Deserialize, Serialize};

/// Metadata about a spelling dictionary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryMetadata {
    /// Version of the dictionary format.
    pub version: u32,
    /// Language code (BCP 47).
    pub language: String,
    /// Number of words in the dictionary.
    pub word_count: usize,
    /// Total token count from corpus.
    pub total_tokens: u64,
    /// Minimum frequency threshold used.
    pub min_frequency: u64,
    /// Timestamp when dictionary was created (Unix epoch seconds).
    pub created_at: u64,
    /// Source corpus description (optional).
    pub source: Option<String>,
}

impl Default for DictionaryMetadata {
    fn default() -> Self {
        Self {
            version: 1,
            language: "en".to_string(),
            word_count: 0,
            total_tokens: 0,
            min_frequency: 1,
            created_at: 0,
            source: None,
        }
    }
}

/// A word entry with frequency information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordEntry {
    /// The word itself.
    pub word: String,
    /// Raw frequency count.
    pub frequency: u64,
    /// Log probability (optional, computed after extraction).
    pub log_prob: Option<f64>,
}

impl WordEntry {
    /// Create a new word entry.
    pub fn new(word: String, frequency: u64) -> Self {
        Self {
            word,
            frequency,
            log_prob: None,
        }
    }

    /// Create a word entry with log probability.
    pub fn with_log_prob(word: String, frequency: u64, log_prob: f64) -> Self {
        Self {
            word,
            frequency,
            log_prob: Some(log_prob),
        }
    }
}

/// Statistics about dictionary extraction.
#[derive(Debug, Clone, Default)]
pub struct DictionaryStats {
    /// Total words extracted (before filtering).
    pub total_words: usize,
    /// Words kept after frequency filtering.
    pub words_kept: usize,
    /// Words filtered out.
    pub words_filtered: usize,
    /// Total token occurrences.
    pub total_tokens: u64,
    /// Number of sentences processed.
    pub sentences_processed: usize,
}

impl std::fmt::Display for DictionaryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dictionary Extraction Statistics:")?;
        writeln!(f, "  Sentences processed: {}", self.sentences_processed)?;
        writeln!(f, "  Total tokens:        {}", self.total_tokens)?;
        writeln!(f, "  Unique words:        {}", self.total_words)?;
        writeln!(f, "  Words kept:          {}", self.words_kept)?;
        writeln!(f, "  Words filtered:      {}", self.words_filtered)?;
        if self.total_words > 0 {
            let keep_rate = 100.0 * self.words_kept as f64 / self.total_words as f64;
            writeln!(f, "  Keep rate:           {:.1}%", keep_rate)?;
        }
        Ok(())
    }
}
