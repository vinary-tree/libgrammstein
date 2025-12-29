//! Word extraction from corpora.
//!
//! Provides concurrent word counting during corpus processing using
//! thread-safe data structures.

use std::collections::HashMap;

use dashmap::DashMap;
use gxhash::GxBuildHasher;
use rayon::prelude::*;

use super::types::{DictionaryStats, WordEntry};

/// Configuration for word extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Minimum word length to include.
    pub min_word_length: usize,
    /// Maximum word length to include.
    pub max_word_length: usize,
    /// Whether to lowercase words.
    pub lowercase: bool,
    /// Whether to filter out words with digits.
    pub filter_digits: bool,
    /// Whether to filter out words with special characters.
    pub filter_special: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_word_length: 1,
            max_word_length: 50,
            lowercase: true,
            filter_digits: false,
            filter_special: true,
        }
    }
}

/// Word extractor for building frequency dictionaries.
///
/// Uses a concurrent hash map for thread-safe word counting during
/// parallel corpus processing.
pub struct WordExtractor {
    /// Word counts.
    counts: DashMap<String, u64, GxBuildHasher>,
    /// Configuration.
    config: ExtractionConfig,
    /// Total tokens seen.
    total_tokens: std::sync::atomic::AtomicU64,
    /// Sentences processed.
    sentences_processed: std::sync::atomic::AtomicUsize,
}

impl WordExtractor {
    /// Create a new word extractor with default configuration.
    pub fn new() -> Self {
        Self::with_config(ExtractionConfig::default())
    }

    /// Create a new word extractor with custom configuration.
    pub fn with_config(config: ExtractionConfig) -> Self {
        Self {
            counts: DashMap::with_hasher(GxBuildHasher::default()),
            config,
            total_tokens: std::sync::atomic::AtomicU64::new(0),
            sentences_processed: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Add a sentence to the extractor.
    pub fn add_sentence(&self, sentence: &str) {
        self.sentences_processed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        for word in sentence.split_whitespace() {
            self.add_word(word);
        }
    }

    /// Add a single word to the extractor.
    pub fn add_word(&self, word: &str) {
        let normalized = self.normalize_word(word);

        if let Some(normalized) = normalized {
            self.total_tokens
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            *self.counts.entry(normalized).or_insert(0) += 1;
        }
    }

    /// Process multiple sentences in parallel.
    pub fn add_sentences_parallel<'a, I>(&self, sentences: I)
    where
        I: ParallelIterator<Item = &'a str>,
    {
        sentences.for_each(|s| self.add_sentence(s));
    }

    /// Normalize a word according to configuration.
    fn normalize_word(&self, word: &str) -> Option<String> {
        // Trim punctuation from edges
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());

        if word.is_empty() {
            return None;
        }

        // Length check
        let char_count = word.chars().count();
        if char_count < self.config.min_word_length || char_count > self.config.max_word_length {
            return None;
        }

        // Filter digits
        if self.config.filter_digits && word.chars().any(|c| c.is_ascii_digit()) {
            return None;
        }

        // Filter special characters
        if self.config.filter_special && word.chars().any(|c| !c.is_alphanumeric()) {
            return None;
        }

        // Lowercase
        let normalized = if self.config.lowercase {
            word.to_lowercase()
        } else {
            word.to_string()
        };

        Some(normalized)
    }

    /// Get the frequency of a word.
    pub fn get_frequency(&self, word: &str) -> u64 {
        self.counts.get(word).map(|v| *v).unwrap_or(0)
    }

    /// Get the total number of unique words.
    pub fn unique_word_count(&self) -> usize {
        self.counts.len()
    }

    /// Get the total number of tokens processed.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the number of sentences processed.
    pub fn sentences_processed(&self) -> usize {
        self.sentences_processed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get all word entries, sorted by frequency (descending).
    pub fn entries_by_frequency(&self) -> Vec<WordEntry> {
        let mut entries: Vec<WordEntry> = self
            .counts
            .iter()
            .map(|e| WordEntry::new(e.key().clone(), *e.value()))
            .collect();

        entries.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        entries
    }

    /// Get word entries with frequency >= min_frequency.
    pub fn entries_filtered(&self, min_frequency: u64) -> Vec<WordEntry> {
        let total = self.total_tokens() as f64;

        self.counts
            .iter()
            .filter(|e| *e.value() >= min_frequency)
            .map(|e| {
                let log_prob = if total > 0.0 {
                    (*e.value() as f64 / total).ln()
                } else {
                    f64::NEG_INFINITY
                };
                WordEntry::with_log_prob(e.key().clone(), *e.value(), log_prob)
            })
            .collect()
    }

    /// Get extraction statistics.
    pub fn stats(&self, min_frequency: u64) -> DictionaryStats {
        let total_words = self.counts.len();
        let words_kept = self.counts.iter().filter(|e| *e.value() >= min_frequency).count();

        DictionaryStats {
            total_words,
            words_kept,
            words_filtered: total_words - words_kept,
            total_tokens: self.total_tokens(),
            sentences_processed: self.sentences_processed(),
        }
    }

    /// Merge another extractor into this one.
    pub fn merge(&self, other: &WordExtractor) {
        for entry in other.counts.iter() {
            *self.counts.entry(entry.key().clone()).or_insert(0) += *entry.value();
        }
        self.total_tokens.fetch_add(
            other.total_tokens.load(std::sync::atomic::Ordering::Relaxed),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.sentences_processed.fetch_add(
            other
                .sentences_processed
                .load(std::sync::atomic::Ordering::Relaxed),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Clear all data.
    pub fn clear(&self) {
        self.counts.clear();
        self.total_tokens
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.sentences_processed
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Export to a standard HashMap (for serialization).
    pub fn to_hashmap(&self) -> HashMap<String, u64> {
        self.counts.iter().map(|e| (e.key().clone(), *e.value())).collect()
    }
}

impl Default for WordExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extraction() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("The quick brown fox jumps over the lazy dog.");
        extractor.add_sentence("The fox is quick.");

        assert_eq!(extractor.get_frequency("the"), 3);
        assert_eq!(extractor.get_frequency("fox"), 2);
        assert_eq!(extractor.get_frequency("quick"), 2);
        assert_eq!(extractor.get_frequency("lazy"), 1);
    }

    #[test]
    fn test_case_normalization() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("Hello World HELLO");

        assert_eq!(extractor.get_frequency("hello"), 2);
        assert_eq!(extractor.get_frequency("world"), 1);
    }

    #[test]
    fn test_punctuation_stripping() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("Hello, world! How are you?");

        assert_eq!(extractor.get_frequency("hello"), 1);
        assert_eq!(extractor.get_frequency("world"), 1);
        assert_eq!(extractor.get_frequency("you"), 1);
    }

    #[test]
    fn test_filter_digits() {
        let config = ExtractionConfig {
            filter_digits: true,
            ..Default::default()
        };
        let extractor = WordExtractor::with_config(config);

        extractor.add_sentence("Hello 123 world test1");

        assert_eq!(extractor.get_frequency("hello"), 1);
        assert_eq!(extractor.get_frequency("world"), 1);
        assert_eq!(extractor.get_frequency("123"), 0);
        assert_eq!(extractor.get_frequency("test1"), 0);
    }

    #[test]
    fn test_entries_by_frequency() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("a a a b b c");

        let entries = extractor.entries_by_frequency();

        assert_eq!(entries[0].word, "a");
        assert_eq!(entries[0].frequency, 3);
        assert_eq!(entries[1].word, "b");
        assert_eq!(entries[1].frequency, 2);
        assert_eq!(entries[2].word, "c");
        assert_eq!(entries[2].frequency, 1);
    }

    #[test]
    fn test_entries_filtered() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("a a a b b c");

        let entries = extractor.entries_filtered(2);

        assert_eq!(entries.len(), 2); // Only 'a' and 'b' have freq >= 2
    }

    #[test]
    fn test_stats() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("hello world hello");
        extractor.add_sentence("world test");

        let stats = extractor.stats(2);

        assert_eq!(stats.sentences_processed, 2);
        assert_eq!(stats.total_tokens, 5);
        assert_eq!(stats.total_words, 3);
        assert_eq!(stats.words_kept, 2); // 'hello' and 'world' have freq >= 2
    }

    #[test]
    fn test_merge() {
        let extractor1 = WordExtractor::new();
        let extractor2 = WordExtractor::new();

        extractor1.add_sentence("hello world");
        extractor2.add_sentence("world test");

        extractor1.merge(&extractor2);

        assert_eq!(extractor1.get_frequency("hello"), 1);
        assert_eq!(extractor1.get_frequency("world"), 2);
        assert_eq!(extractor1.get_frequency("test"), 1);
    }

    #[test]
    fn test_unicode() {
        let extractor = WordExtractor::new();

        extractor.add_sentence("Héllo wörld 你好 世界");

        assert_eq!(extractor.get_frequency("héllo"), 1);
        assert_eq!(extractor.get_frequency("wörld"), 1);
        assert_eq!(extractor.get_frequency("你好"), 1);
        assert_eq!(extractor.get_frequency("世界"), 1);
    }
}
