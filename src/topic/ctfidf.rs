//! Class-based TF-IDF (c-TF-IDF) for topic keyword extraction.
//!
//! This module implements c-TF-IDF, a modified TF-IDF algorithm designed for
//! extracting representative keywords from topic clusters.
//!
//! # Algorithm
//!
//! c-TF-IDF computes term importance per topic (class) rather than per document:
//!
//! ```text
//! c-TF-IDF(t, c) = tf(t, c) * log(1 + avg_docs / df(t))
//! ```
//!
//! Where:
//! - `tf(t, c)` = frequency of term t in topic c (optionally with sublinear scaling)
//! - `avg_docs` = average number of words per topic
//! - `df(t)` = number of topics containing term t

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use dashmap::DashMap;
use rayon::prelude::*;

use super::config::CtfidfConfig;
use super::{Result, TopicError};

/// Vocabulary with atomic term counting.
pub struct AtomicVocabulary {
    /// Term to index mapping.
    term_to_idx: DashMap<String, usize>,
    /// Index to term mapping.
    idx_to_term: parking_lot::RwLock<Vec<String>>,
    /// Document frequency for each term (number of documents containing term).
    doc_freq: parking_lot::RwLock<Vec<AtomicUsize>>,
    /// Next available index.
    next_idx: AtomicUsize,
    /// Configuration.
    config: CtfidfConfig,
}

impl AtomicVocabulary {
    /// Create a new vocabulary.
    pub fn new(config: CtfidfConfig) -> Self {
        Self {
            term_to_idx: DashMap::new(),
            idx_to_term: parking_lot::RwLock::new(Vec::new()),
            doc_freq: parking_lot::RwLock::new(Vec::new()),
            next_idx: AtomicUsize::new(0),
            config,
        }
    }

    /// Get or create index for a term (thread-safe).
    pub fn get_or_insert(&self, term: &str) -> Option<usize> {
        // Check term length constraints
        if term.len() < self.config.min_term_length || term.len() > self.config.max_term_length {
            return None;
        }

        // Try to get existing index
        if let Some(idx) = self.term_to_idx.get(term) {
            return Some(*idx);
        }

        // Insert new term
        let idx = self.next_idx.fetch_add(1, Ordering::SeqCst);

        // Use entry API to handle race conditions
        let entry = self.term_to_idx.entry(term.to_string());
        let final_idx = match entry {
            dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
            dashmap::mapref::entry::Entry::Vacant(e) => {
                e.insert(idx);
                // Extend storage
                let mut idx_to_term = self.idx_to_term.write();
                let mut doc_freq = self.doc_freq.write();
                while idx_to_term.len() <= idx {
                    idx_to_term.push(String::new());
                    doc_freq.push(AtomicUsize::new(0));
                }
                idx_to_term[idx] = term.to_string();
                idx
            }
        };

        Some(final_idx)
    }

    /// Get index for a term.
    pub fn get(&self, term: &str) -> Option<usize> {
        self.term_to_idx.get(term).map(|r| *r)
    }

    /// Get term for an index.
    pub fn get_term(&self, idx: usize) -> Option<String> {
        let idx_to_term = self.idx_to_term.read();
        idx_to_term.get(idx).cloned()
    }

    /// Increment document frequency for a term.
    pub fn increment_doc_freq(&self, idx: usize) {
        let doc_freq = self.doc_freq.read();
        if idx < doc_freq.len() {
            doc_freq[idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get document frequency for a term.
    pub fn doc_frequency(&self, idx: usize) -> usize {
        let doc_freq = self.doc_freq.read();
        if idx < doc_freq.len() {
            doc_freq[idx].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Number of terms in vocabulary.
    pub fn len(&self) -> usize {
        self.term_to_idx.len()
    }

    /// Check if vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.term_to_idx.is_empty()
    }

    /// Get all terms as a sorted vector.
    pub fn terms(&self) -> Vec<String> {
        let idx_to_term = self.idx_to_term.read();
        idx_to_term
            .iter()
            .filter(|t| !t.is_empty())
            .cloned()
            .collect()
    }

    /// Filter vocabulary by document frequency thresholds.
    ///
    /// Returns indices of terms that pass the filter.
    pub fn filter_by_df(&self, num_topics: usize) -> Vec<usize> {
        let doc_freq = self.doc_freq.read();
        let max_df = (self.config.max_df_ratio * num_topics as f32) as usize;

        doc_freq
            .iter()
            .enumerate()
            .filter(|(_, df)| {
                let freq = df.load(Ordering::Relaxed);
                freq >= self.config.min_df && freq <= max_df
            })
            .map(|(idx, _)| idx)
            .collect()
    }
}

/// Topic term frequencies using atomic counters.
pub struct TopicTermFrequencies {
    /// Term frequencies per topic.
    /// Outer: topic index, Inner: term index -> count.
    frequencies: Vec<DashMap<usize, AtomicUsize>>,
    /// Total word count per topic.
    topic_word_counts: Vec<AtomicUsize>,
    /// Number of topics.
    num_topics: usize,
}

impl TopicTermFrequencies {
    /// Create new topic term frequencies.
    pub fn new(num_topics: usize) -> Self {
        Self {
            frequencies: (0..num_topics).map(|_| DashMap::new()).collect(),
            topic_word_counts: (0..num_topics).map(|_| AtomicUsize::new(0)).collect(),
            num_topics,
        }
    }

    /// Increment term frequency for a topic (thread-safe).
    pub fn increment(&self, topic_idx: usize, term_idx: usize) {
        if topic_idx < self.num_topics {
            let topic_freqs = &self.frequencies[topic_idx];
            topic_freqs
                .entry(term_idx)
                .or_insert_with(|| AtomicUsize::new(0))
                .fetch_add(1, Ordering::Relaxed);
            self.topic_word_counts[topic_idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get term frequency for a topic.
    pub fn get(&self, topic_idx: usize, term_idx: usize) -> usize {
        if topic_idx < self.num_topics {
            self.frequencies[topic_idx]
                .get(&term_idx)
                .map(|v| v.load(Ordering::Relaxed))
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Get total word count for a topic.
    pub fn topic_word_count(&self, topic_idx: usize) -> usize {
        if topic_idx < self.num_topics {
            self.topic_word_counts[topic_idx].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Get average word count across all topics.
    pub fn average_word_count(&self) -> f64 {
        let total: usize = self
            .topic_word_counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        if self.num_topics > 0 {
            total as f64 / self.num_topics as f64
        } else {
            0.0
        }
    }

    /// Number of topics.
    pub fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Convert to dense matrix for checkpointing.
    pub fn to_dense(&self, vocab_size: usize) -> Vec<Vec<u32>> {
        self.frequencies
            .iter()
            .map(|topic_freqs| {
                let mut row = vec![0u32; vocab_size];
                for entry in topic_freqs.iter() {
                    let idx = *entry.key();
                    let count = entry.value().load(Ordering::Relaxed);
                    if idx < vocab_size {
                        row[idx] = count as u32;
                    }
                }
                row
            })
            .collect()
    }

    /// Create from dense matrix (for checkpoint restore).
    pub fn from_dense(dense: &[Vec<u32>]) -> Self {
        let num_topics = dense.len();
        let frequencies: Vec<DashMap<usize, AtomicUsize>> = dense
            .iter()
            .map(|row| {
                let map = DashMap::new();
                for (idx, &count) in row.iter().enumerate() {
                    if count > 0 {
                        map.insert(idx, AtomicUsize::new(count as usize));
                    }
                }
                map
            })
            .collect();

        let topic_word_counts: Vec<AtomicUsize> = dense
            .iter()
            .map(|row| {
                let total: u32 = row.iter().sum();
                AtomicUsize::new(total as usize)
            })
            .collect();

        Self {
            frequencies,
            topic_word_counts,
            num_topics,
        }
    }
}

/// c-TF-IDF keyword extractor.
pub struct CtfIdf {
    /// Configuration.
    config: CtfidfConfig,
    /// Vocabulary.
    vocabulary: AtomicVocabulary,
    /// Topic term frequencies.
    term_frequencies: Option<TopicTermFrequencies>,
}

impl CtfIdf {
    /// Create a new c-TF-IDF extractor.
    pub fn new(config: CtfidfConfig) -> Self {
        Self {
            vocabulary: AtomicVocabulary::new(config.clone()),
            config,
            term_frequencies: None,
        }
    }

    /// Tokenize text into terms.
    ///
    /// Simple whitespace tokenizer with basic normalization.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                // Lowercase and remove punctuation
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Build vocabulary from documents with their topic assignments.
    ///
    /// Uses parallel processing with atomic counters.
    pub fn build_vocabulary(
        &mut self,
        documents: &[String],
        topic_assignments: &[u32],
    ) -> Result<()> {
        if documents.len() != topic_assignments.len() {
            return Err(TopicError::CtfidfError(format!(
                "Document count ({}) != assignment count ({})",
                documents.len(),
                topic_assignments.len()
            )));
        }

        let num_topics = *topic_assignments.iter().max().unwrap_or(&0) as usize + 1;
        let term_frequencies = TopicTermFrequencies::new(num_topics);

        // Process documents in parallel
        documents
            .par_iter()
            .zip(topic_assignments.par_iter())
            .for_each(|(doc, &topic)| {
                let tokens = Self::tokenize(doc);
                let mut seen_terms = std::collections::HashSet::new();

                for token in tokens {
                    if let Some(idx) = self.vocabulary.get_or_insert(&token) {
                        // Increment term frequency for this topic
                        term_frequencies.increment(topic as usize, idx);

                        // Track document frequency (only count once per document)
                        if seen_terms.insert(idx) {
                            self.vocabulary.increment_doc_freq(idx);
                        }
                    }
                }
            });

        self.term_frequencies = Some(term_frequencies);
        Ok(())
    }

    /// Compute c-TF-IDF scores for a topic.
    ///
    /// Returns term indices and their c-TF-IDF scores.
    pub fn compute_ctfidf(&self, topic_idx: usize) -> Vec<(usize, f64)> {
        let Some(term_freqs) = &self.term_frequencies else {
            return Vec::new();
        };

        let num_topics = term_freqs.num_topics();
        let avg_words = term_freqs.average_word_count();
        let topic_word_count = term_freqs.topic_word_count(topic_idx);

        if topic_word_count == 0 {
            return Vec::new();
        }

        // Get valid terms (pass document frequency filters)
        let valid_terms = self.vocabulary.filter_by_df(num_topics);

        let mut scores: Vec<(usize, f64)> = valid_terms
            .iter()
            .filter_map(|&term_idx| {
                let tf = term_freqs.get(topic_idx, term_idx);
                if tf == 0 {
                    return None;
                }

                // Sublinear TF scaling
                let scaled_tf = if self.config.sublinear_tf {
                    1.0 + (tf as f64).ln()
                } else {
                    tf as f64
                };

                // Normalize by topic word count
                let normalized_tf = scaled_tf / topic_word_count as f64;

                // Document frequency component
                let df = self.vocabulary.doc_frequency(term_idx).max(1);
                let idf = (1.0 + avg_words / df as f64).ln();

                let score = normalized_tf * idf;
                Some((term_idx, score))
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
    }

    /// Extract top keywords for a topic.
    pub fn extract_keywords(&self, topic_idx: usize) -> Vec<(String, f32)> {
        let scores = self.compute_ctfidf(topic_idx);

        scores
            .into_iter()
            .take(self.config.num_keywords)
            .filter_map(|(term_idx, score)| {
                self.vocabulary
                    .get_term(term_idx)
                    .map(|term| (term, score as f32))
            })
            .collect()
    }

    /// Extract keywords for all topics.
    pub fn extract_all_keywords(&self) -> Vec<Vec<(String, f32)>> {
        let Some(term_freqs) = &self.term_frequencies else {
            return Vec::new();
        };

        (0..term_freqs.num_topics())
            .map(|topic_idx| self.extract_keywords(topic_idx))
            .collect()
    }

    /// Get the vocabulary.
    pub fn vocabulary(&self) -> &AtomicVocabulary {
        &self.vocabulary
    }

    /// Get term frequencies (if built).
    pub fn term_frequencies(&self) -> Option<&TopicTermFrequencies> {
        self.term_frequencies.as_ref()
    }

    /// Get configuration.
    pub fn config(&self) -> &CtfidfConfig {
        &self.config
    }

    /// Export vocabulary terms for checkpointing.
    pub fn export_vocabulary(&self) -> Vec<String> {
        self.vocabulary.terms()
    }

    /// Export term frequencies for checkpointing.
    pub fn export_term_frequencies(&self) -> Option<Vec<Vec<u32>>> {
        self.term_frequencies
            .as_ref()
            .map(|tf| tf.to_dense(self.vocabulary.len()))
    }
}

/// Format keywords as a comma-separated string.
pub fn format_keywords(keywords: &[(String, f32)]) -> String {
    keywords
        .iter()
        .map(|(term, _)| term.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Format keywords with scores.
pub fn format_keywords_with_scores(keywords: &[(String, f32)]) -> String {
    keywords
        .iter()
        .map(|(term, score)| format!("{} ({:.3})", term, score))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = CtfIdf::tokenize("Hello, World! This is a test.");
        assert_eq!(
            tokens,
            vec!["hello", "world", "this", "is", "a", "test"]
        );

        let tokens = CtfIdf::tokenize("Machine-learning  and  AI!");
        assert_eq!(tokens, vec!["machinelearning", "and", "ai"]);
    }

    #[test]
    fn test_atomic_vocabulary() {
        let config = CtfidfConfig::default();
        let vocab = AtomicVocabulary::new(config);

        let idx1 = vocab.get_or_insert("hello").unwrap();
        let idx2 = vocab.get_or_insert("world").unwrap();
        let idx3 = vocab.get_or_insert("hello").unwrap(); // Duplicate

        assert_eq!(idx1, idx3); // Same term gets same index
        assert_ne!(idx1, idx2);
        assert_eq!(vocab.len(), 2);

        assert_eq!(vocab.get_term(idx1), Some("hello".to_string()));
        assert_eq!(vocab.get("hello"), Some(idx1));
    }

    #[test]
    fn test_vocabulary_length_filter() {
        let config = CtfidfConfig {
            min_term_length: 3,
            max_term_length: 10,
            ..Default::default()
        };
        let vocab = AtomicVocabulary::new(config);

        // Too short
        assert!(vocab.get_or_insert("ab").is_none());
        // Too long
        assert!(vocab.get_or_insert("verylongterm").is_none());
        // Just right
        assert!(vocab.get_or_insert("hello").is_some());
    }

    #[test]
    fn test_topic_term_frequencies() {
        let ttf = TopicTermFrequencies::new(3);

        ttf.increment(0, 5); // Topic 0, term 5
        ttf.increment(0, 5); // Again
        ttf.increment(1, 5); // Topic 1, term 5
        ttf.increment(2, 10); // Topic 2, term 10

        assert_eq!(ttf.get(0, 5), 2);
        assert_eq!(ttf.get(1, 5), 1);
        assert_eq!(ttf.get(2, 5), 0);
        assert_eq!(ttf.get(2, 10), 1);

        assert_eq!(ttf.topic_word_count(0), 2);
        assert_eq!(ttf.topic_word_count(1), 1);
        assert_eq!(ttf.topic_word_count(2), 1);
    }

    #[test]
    fn test_ctfidf_basic() {
        let config = CtfidfConfig {
            num_keywords: 5,
            min_df: 1,
            min_term_length: 2,
            ..Default::default()
        };
        let mut ctfidf = CtfIdf::new(config);

        let documents = vec![
            "machine learning algorithms neural networks".to_string(),
            "deep learning neural networks training".to_string(),
            "data science statistics analysis".to_string(),
            "data mining clustering classification".to_string(),
        ];
        let assignments = vec![0, 0, 1, 1]; // Two topics

        ctfidf
            .build_vocabulary(&documents, &assignments)
            .expect("build failed");

        // Topic 0 should have machine learning keywords
        let keywords_0 = ctfidf.extract_keywords(0);
        assert!(!keywords_0.is_empty());

        // Topic 1 should have data science keywords
        let keywords_1 = ctfidf.extract_keywords(1);
        assert!(!keywords_1.is_empty());

        // Keywords should be different between topics
        let terms_0: std::collections::HashSet<_> =
            keywords_0.iter().map(|(t, _)| t.clone()).collect();
        let terms_1: std::collections::HashSet<_> =
            keywords_1.iter().map(|(t, _)| t.clone()).collect();

        // Should have some unique keywords in each topic
        assert!(
            terms_0.difference(&terms_1).next().is_some()
                || terms_1.difference(&terms_0).next().is_some()
        );
    }

    #[test]
    fn test_ctfidf_extract_all() {
        let config = CtfidfConfig {
            num_keywords: 3,
            min_df: 1,
            min_term_length: 2,
            ..Default::default()
        };
        let mut ctfidf = CtfIdf::new(config);

        let documents = vec![
            "alpha beta gamma".to_string(),
            "alpha beta delta".to_string(),
            "epsilon zeta eta".to_string(),
        ];
        let assignments = vec![0, 0, 1];

        ctfidf
            .build_vocabulary(&documents, &assignments)
            .expect("build failed");

        let all_keywords = ctfidf.extract_all_keywords();
        assert_eq!(all_keywords.len(), 2); // Two topics
    }

    #[test]
    fn test_format_keywords() {
        let keywords = vec![
            ("machine".to_string(), 0.5),
            ("learning".to_string(), 0.3),
            ("neural".to_string(), 0.2),
        ];

        let formatted = format_keywords(&keywords);
        assert_eq!(formatted, "machine, learning, neural");

        let with_scores = format_keywords_with_scores(&keywords);
        assert!(with_scores.contains("0.500"));
        assert!(with_scores.contains("learning"));
    }

    #[test]
    fn test_term_frequencies_to_dense() {
        let ttf = TopicTermFrequencies::new(2);
        ttf.increment(0, 0);
        ttf.increment(0, 0);
        ttf.increment(0, 2);
        ttf.increment(1, 1);

        let dense = ttf.to_dense(4);
        assert_eq!(dense.len(), 2);
        assert_eq!(dense[0][0], 2);
        assert_eq!(dense[0][2], 1);
        assert_eq!(dense[1][1], 1);
    }

    #[test]
    fn test_term_frequencies_from_dense() {
        let dense = vec![vec![2, 0, 1, 0], vec![0, 1, 0, 0]];

        let ttf = TopicTermFrequencies::from_dense(&dense);
        assert_eq!(ttf.num_topics(), 2);
        assert_eq!(ttf.get(0, 0), 2);
        assert_eq!(ttf.get(0, 2), 1);
        assert_eq!(ttf.get(1, 1), 1);
        assert_eq!(ttf.topic_word_count(0), 3);
        assert_eq!(ttf.topic_word_count(1), 1);
    }

    #[test]
    fn test_export_vocabulary() {
        let config = CtfidfConfig::default();
        let mut ctfidf = CtfIdf::new(config);

        let documents = vec!["hello world".to_string(), "world test".to_string()];
        let assignments = vec![0, 0];

        ctfidf
            .build_vocabulary(&documents, &assignments)
            .expect("build failed");

        let vocab = ctfidf.export_vocabulary();
        assert!(vocab.contains(&"hello".to_string()));
        assert!(vocab.contains(&"world".to_string()));
        assert!(vocab.contains(&"test".to_string()));
    }
}
