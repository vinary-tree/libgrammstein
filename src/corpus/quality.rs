//! Quality filtering for corpus sentences.
//!
//! This module provides configurable quality filtering to remove low-quality
//! sentences from training corpora. Filtering criteria include:
//!
//! - Minimum word count
//! - Maximum word repetition ratio
//! - Terminal punctuation requirement
//! - Minimum character entropy
//! - Minimum alphabetic character ratio
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::corpus::QualityFilter;
//!
//! let filter = QualityFilter::builder()
//!     .min_words(5)
//!     .max_word_repetition(0.3)
//!     .require_terminal_punct(true)
//!     .build();
//!
//! let sentence = "The quick brown fox jumps over the lazy dog.";
//! if filter.is_quality(sentence) {
//!     println!("High quality: {}", sentence);
//! }
//! ```

use std::collections::HashMap;

/// Quality metrics computed for a sentence.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Number of words in the sentence.
    pub word_count: usize,
    /// Maximum repetition ratio of any single word.
    pub max_word_repetition: f32,
    /// Whether the sentence ends with terminal punctuation.
    pub has_terminal_punct: bool,
    /// Shannon entropy of characters (bits).
    pub char_entropy: f32,
    /// Ratio of alphabetic characters to total characters.
    pub alpha_ratio: f32,
    /// Average word length.
    pub avg_word_length: f32,
}

impl QualityMetrics {
    /// Compute quality metrics for a sentence.
    pub fn compute(sentence: &str) -> Self {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let word_count = words.len();

        // Compute word repetition
        let max_word_repetition = if word_count > 0 {
            let mut word_counts: HashMap<&str, usize> = HashMap::new();
            for word in &words {
                *word_counts.entry(*word).or_insert(0) += 1;
            }
            let max_count = word_counts.values().max().copied().unwrap_or(0);
            max_count as f32 / word_count as f32
        } else {
            0.0
        };

        // Check terminal punctuation
        let has_terminal_punct = sentence
            .trim()
            .chars()
            .last()
            .map(|c| matches!(c, '.' | '!' | '?' | '。' | '！' | '？'))
            .unwrap_or(false);

        // Compute character entropy
        let char_entropy = Self::compute_char_entropy(sentence);

        // Compute alphabetic ratio
        let total_chars = sentence.chars().count();
        let alpha_chars = sentence.chars().filter(|c| c.is_alphabetic()).count();
        let alpha_ratio = if total_chars > 0 {
            alpha_chars as f32 / total_chars as f32
        } else {
            0.0
        };

        // Compute average word length
        let avg_word_length = if word_count > 0 {
            let total_word_chars: usize = words.iter().map(|w| w.chars().count()).sum();
            total_word_chars as f32 / word_count as f32
        } else {
            0.0
        };

        Self {
            word_count,
            max_word_repetition,
            has_terminal_punct,
            char_entropy,
            alpha_ratio,
            avg_word_length,
        }
    }

    /// Compute Shannon entropy of character distribution.
    fn compute_char_entropy(text: &str) -> f32 {
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        let mut total = 0usize;

        for c in text.chars() {
            if !c.is_whitespace() {
                *char_counts
                    .entry(c.to_lowercase().next().unwrap_or(c))
                    .or_insert(0) += 1;
                total += 1;
            }
        }

        if total == 0 {
            return 0.0;
        }

        let total_f = total as f32;
        let mut entropy = 0.0f32;

        for &count in char_counts.values() {
            if count > 0 {
                let p = count as f32 / total_f;
                entropy -= p * p.log2();
            }
        }

        entropy
    }
}

/// Configurable quality filter for corpus sentences.
///
/// Use [`QualityFilter::builder()`] to create a filter with custom settings,
/// or [`QualityFilter::default()`] for reasonable defaults.
#[derive(Debug, Clone)]
pub struct QualityFilter {
    /// Minimum number of words required.
    min_words: usize,
    /// Maximum number of words allowed (0 = unlimited).
    max_words: usize,
    /// Maximum repetition ratio for any single word.
    max_word_repetition: f32,
    /// Whether to require terminal punctuation.
    require_terminal_punct: bool,
    /// Minimum character entropy (bits).
    min_char_entropy: f32,
    /// Minimum ratio of alphabetic characters.
    min_alpha_ratio: f32,
    /// Minimum average word length.
    min_avg_word_length: f32,
    /// Maximum average word length.
    max_avg_word_length: f32,
}

impl Default for QualityFilter {
    fn default() -> Self {
        Self {
            min_words: 5,
            max_words: 0, // No limit
            max_word_repetition: 0.3,
            require_terminal_punct: false,
            min_char_entropy: 3.0,
            min_alpha_ratio: 0.7,
            min_avg_word_length: 2.0,
            max_avg_word_length: 20.0,
        }
    }
}

impl QualityFilter {
    /// Create a new builder for configuring a quality filter.
    pub fn builder() -> QualityFilterBuilder {
        QualityFilterBuilder::new()
    }

    /// Create a strict filter that enforces all quality criteria.
    pub fn strict() -> Self {
        Self {
            min_words: 8,
            max_words: 100,
            max_word_repetition: 0.2,
            require_terminal_punct: true,
            min_char_entropy: 3.5,
            min_alpha_ratio: 0.78, // Accounts for spaces diluting the ratio
            min_avg_word_length: 3.0,
            max_avg_word_length: 15.0,
        }
    }

    /// Create a lenient filter with minimal requirements.
    pub fn lenient() -> Self {
        Self {
            min_words: 3,
            max_words: 0,
            max_word_repetition: 0.5,
            require_terminal_punct: false,
            min_char_entropy: 2.0,
            min_alpha_ratio: 0.5,
            min_avg_word_length: 1.5,
            max_avg_word_length: 25.0,
        }
    }

    /// Check if a sentence passes all quality filters.
    pub fn is_quality(&self, sentence: &str) -> bool {
        let metrics = QualityMetrics::compute(sentence);
        self.check_metrics(&metrics)
    }

    /// Check if computed metrics pass all quality filters.
    pub fn check_metrics(&self, metrics: &QualityMetrics) -> bool {
        // Word count check
        if metrics.word_count < self.min_words {
            return false;
        }
        if self.max_words > 0 && metrics.word_count > self.max_words {
            return false;
        }

        // Word repetition check
        if metrics.max_word_repetition > self.max_word_repetition {
            return false;
        }

        // Terminal punctuation check
        if self.require_terminal_punct && !metrics.has_terminal_punct {
            return false;
        }

        // Character entropy check
        if metrics.char_entropy < self.min_char_entropy {
            return false;
        }

        // Alphabetic ratio check
        if metrics.alpha_ratio < self.min_alpha_ratio {
            return false;
        }

        // Average word length checks
        if metrics.avg_word_length < self.min_avg_word_length {
            return false;
        }
        if metrics.avg_word_length > self.max_avg_word_length {
            return false;
        }

        true
    }

    /// Filter an iterator of sentences, keeping only high-quality ones.
    pub fn filter<'a, I>(&'a self, sentences: I) -> impl Iterator<Item = String> + 'a
    where
        I: Iterator<Item = String> + 'a,
    {
        sentences.filter(move |s| self.is_quality(s))
    }

    /// Filter sentences with detailed rejection reasons.
    pub fn filter_with_reasons<'a, I>(
        &'a self,
        sentences: I,
    ) -> impl Iterator<Item = (String, Option<RejectionReason>)> + 'a
    where
        I: Iterator<Item = String> + 'a,
    {
        sentences.map(move |s| {
            let reason = self.rejection_reason(&s);
            (s, reason)
        })
    }

    /// Get the reason a sentence was rejected, or None if it passes.
    pub fn rejection_reason(&self, sentence: &str) -> Option<RejectionReason> {
        let metrics = QualityMetrics::compute(sentence);

        if metrics.word_count < self.min_words {
            return Some(RejectionReason::TooFewWords {
                count: metrics.word_count,
                minimum: self.min_words,
            });
        }

        if self.max_words > 0 && metrics.word_count > self.max_words {
            return Some(RejectionReason::TooManyWords {
                count: metrics.word_count,
                maximum: self.max_words,
            });
        }

        if metrics.max_word_repetition > self.max_word_repetition {
            return Some(RejectionReason::ExcessiveRepetition {
                ratio: metrics.max_word_repetition,
                maximum: self.max_word_repetition,
            });
        }

        if self.require_terminal_punct && !metrics.has_terminal_punct {
            return Some(RejectionReason::MissingTerminalPunct);
        }

        if metrics.char_entropy < self.min_char_entropy {
            return Some(RejectionReason::LowEntropy {
                entropy: metrics.char_entropy,
                minimum: self.min_char_entropy,
            });
        }

        if metrics.alpha_ratio < self.min_alpha_ratio {
            return Some(RejectionReason::LowAlphaRatio {
                ratio: metrics.alpha_ratio,
                minimum: self.min_alpha_ratio,
            });
        }

        if metrics.avg_word_length < self.min_avg_word_length {
            return Some(RejectionReason::ShortWords {
                avg_length: metrics.avg_word_length,
                minimum: self.min_avg_word_length,
            });
        }

        if metrics.avg_word_length > self.max_avg_word_length {
            return Some(RejectionReason::LongWords {
                avg_length: metrics.avg_word_length,
                maximum: self.max_avg_word_length,
            });
        }

        None
    }

    /// Get statistics about filtering results.
    pub fn compute_stats<I>(&self, sentences: I) -> QualityStats
    where
        I: Iterator<Item = String>,
    {
        let mut stats = QualityStats::default();

        for sentence in sentences {
            stats.total += 1;
            match self.rejection_reason(&sentence) {
                None => stats.passed += 1,
                Some(reason) => {
                    stats.rejected += 1;
                    match reason {
                        RejectionReason::TooFewWords { .. } => stats.too_few_words += 1,
                        RejectionReason::TooManyWords { .. } => stats.too_many_words += 1,
                        RejectionReason::ExcessiveRepetition { .. } => {
                            stats.excessive_repetition += 1
                        }
                        RejectionReason::MissingTerminalPunct => stats.missing_punct += 1,
                        RejectionReason::LowEntropy { .. } => stats.low_entropy += 1,
                        RejectionReason::LowAlphaRatio { .. } => stats.low_alpha_ratio += 1,
                        RejectionReason::ShortWords { .. } => stats.short_words += 1,
                        RejectionReason::LongWords { .. } => stats.long_words += 1,
                    }
                }
            }
        }

        stats
    }
}

/// Reason a sentence was rejected by the quality filter.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectionReason {
    /// Sentence has too few words.
    TooFewWords { count: usize, minimum: usize },
    /// Sentence has too many words.
    TooManyWords { count: usize, maximum: usize },
    /// Excessive word repetition.
    ExcessiveRepetition { ratio: f32, maximum: f32 },
    /// Missing terminal punctuation.
    MissingTerminalPunct,
    /// Character entropy too low.
    LowEntropy { entropy: f32, minimum: f32 },
    /// Alphabetic character ratio too low.
    LowAlphaRatio { ratio: f32, minimum: f32 },
    /// Average word length too short.
    ShortWords { avg_length: f32, minimum: f32 },
    /// Average word length too long.
    LongWords { avg_length: f32, maximum: f32 },
}

impl std::fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewWords { count, minimum } => {
                write!(f, "Too few words: {} (minimum: {})", count, minimum)
            }
            Self::TooManyWords { count, maximum } => {
                write!(f, "Too many words: {} (maximum: {})", count, maximum)
            }
            Self::ExcessiveRepetition { ratio, maximum } => {
                write!(
                    f,
                    "Excessive repetition: {:.2} (maximum: {:.2})",
                    ratio, maximum
                )
            }
            Self::MissingTerminalPunct => write!(f, "Missing terminal punctuation"),
            Self::LowEntropy { entropy, minimum } => {
                write!(
                    f,
                    "Low entropy: {:.2} bits (minimum: {:.2})",
                    entropy, minimum
                )
            }
            Self::LowAlphaRatio { ratio, minimum } => {
                write!(f, "Low alpha ratio: {:.2} (minimum: {:.2})", ratio, minimum)
            }
            Self::ShortWords {
                avg_length,
                minimum,
            } => {
                write!(
                    f,
                    "Words too short: {:.2} avg (minimum: {:.2})",
                    avg_length, minimum
                )
            }
            Self::LongWords {
                avg_length,
                maximum,
            } => {
                write!(
                    f,
                    "Words too long: {:.2} avg (maximum: {:.2})",
                    avg_length, maximum
                )
            }
        }
    }
}

/// Statistics about quality filtering results.
#[derive(Debug, Clone, Default)]
pub struct QualityStats {
    /// Total sentences processed.
    pub total: usize,
    /// Sentences that passed filtering.
    pub passed: usize,
    /// Sentences that were rejected.
    pub rejected: usize,
    /// Rejected for too few words.
    pub too_few_words: usize,
    /// Rejected for too many words.
    pub too_many_words: usize,
    /// Rejected for excessive repetition.
    pub excessive_repetition: usize,
    /// Rejected for missing punctuation.
    pub missing_punct: usize,
    /// Rejected for low entropy.
    pub low_entropy: usize,
    /// Rejected for low alpha ratio.
    pub low_alpha_ratio: usize,
    /// Rejected for short words.
    pub short_words: usize,
    /// Rejected for long words.
    pub long_words: usize,
}

impl QualityStats {
    /// Get pass rate as a percentage.
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            100.0 * self.passed as f64 / self.total as f64
        }
    }
}

impl std::fmt::Display for QualityStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quality Filtering Statistics:")?;
        writeln!(f, "  Total:     {}", self.total)?;
        writeln!(f, "  Passed:    {} ({:.1}%)", self.passed, self.pass_rate())?;
        writeln!(f, "  Rejected:  {}", self.rejected)?;
        if self.rejected > 0 {
            writeln!(f, "  Rejection breakdown:")?;
            if self.too_few_words > 0 {
                writeln!(f, "    - Too few words:    {}", self.too_few_words)?;
            }
            if self.too_many_words > 0 {
                writeln!(f, "    - Too many words:   {}", self.too_many_words)?;
            }
            if self.excessive_repetition > 0 {
                writeln!(f, "    - Repetition:       {}", self.excessive_repetition)?;
            }
            if self.missing_punct > 0 {
                writeln!(f, "    - Missing punct:    {}", self.missing_punct)?;
            }
            if self.low_entropy > 0 {
                writeln!(f, "    - Low entropy:      {}", self.low_entropy)?;
            }
            if self.low_alpha_ratio > 0 {
                writeln!(f, "    - Low alpha ratio:  {}", self.low_alpha_ratio)?;
            }
            if self.short_words > 0 {
                writeln!(f, "    - Short words:      {}", self.short_words)?;
            }
            if self.long_words > 0 {
                writeln!(f, "    - Long words:       {}", self.long_words)?;
            }
        }
        Ok(())
    }
}

/// Builder for configuring a QualityFilter.
#[derive(Debug, Clone)]
pub struct QualityFilterBuilder {
    filter: QualityFilter,
}

impl QualityFilterBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            filter: QualityFilter::default(),
        }
    }

    /// Set minimum number of words.
    pub fn min_words(mut self, min: usize) -> Self {
        self.filter.min_words = min;
        self
    }

    /// Set maximum number of words (0 = unlimited).
    pub fn max_words(mut self, max: usize) -> Self {
        self.filter.max_words = max;
        self
    }

    /// Set maximum word repetition ratio.
    pub fn max_word_repetition(mut self, max: f32) -> Self {
        self.filter.max_word_repetition = max;
        self
    }

    /// Set whether terminal punctuation is required.
    pub fn require_terminal_punct(mut self, require: bool) -> Self {
        self.filter.require_terminal_punct = require;
        self
    }

    /// Set minimum character entropy.
    pub fn min_char_entropy(mut self, min: f32) -> Self {
        self.filter.min_char_entropy = min;
        self
    }

    /// Set minimum alphabetic character ratio.
    pub fn min_alpha_ratio(mut self, min: f32) -> Self {
        self.filter.min_alpha_ratio = min;
        self
    }

    /// Set minimum average word length.
    pub fn min_avg_word_length(mut self, min: f32) -> Self {
        self.filter.min_avg_word_length = min;
        self
    }

    /// Set maximum average word length.
    pub fn max_avg_word_length(mut self, max: f32) -> Self {
        self.filter.max_avg_word_length = max;
        self
    }

    /// Build the quality filter.
    pub fn build(self) -> QualityFilter {
        self.filter
    }
}

impl Default for QualityFilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metrics() {
        let sentence = "The quick brown fox jumps over the lazy dog.";
        let metrics = QualityMetrics::compute(sentence);

        assert_eq!(metrics.word_count, 9);
        assert!(metrics.has_terminal_punct);
        assert!(metrics.char_entropy > 3.0);
        // Alpha ratio includes spaces in denominator, so ~35/45 = 0.78
        assert!(
            metrics.alpha_ratio > 0.7,
            "alpha_ratio: {}",
            metrics.alpha_ratio
        );
    }

    #[test]
    fn test_low_entropy() {
        let sentence = "aaaa aaaa aaaa aaaa aaaa";
        let metrics = QualityMetrics::compute(sentence);

        assert!(
            metrics.char_entropy < 1.0,
            "Entropy should be very low for repeated chars"
        );
    }

    #[test]
    fn test_high_repetition() {
        let sentence = "the the the the the the the quick";
        let metrics = QualityMetrics::compute(sentence);

        assert!(
            metrics.max_word_repetition > 0.5,
            "Repetition should be high"
        );
    }

    #[test]
    fn test_filter_default() {
        let filter = QualityFilter::default();

        // Good sentence should pass
        assert!(filter.is_quality("The quick brown fox jumps over the lazy dog."));

        // Too short should fail
        assert!(!filter.is_quality("Hello world."));
    }

    #[test]
    fn test_filter_strict() {
        let filter = QualityFilter::strict();

        // Good sentence with terminal punct should pass
        let good = "The quick brown fox jumps over the lazy dog in the forest.";
        assert!(
            filter.is_quality(good),
            "Good sentence failed: {:?}",
            filter.rejection_reason(good)
        );

        // Missing terminal punct should fail
        let no_punct = "The quick brown fox jumps over the lazy dog";
        assert!(
            !filter.is_quality(no_punct),
            "Should have failed without terminal punct"
        );
    }

    #[test]
    fn test_filter_lenient() {
        let filter = QualityFilter::lenient();

        // Short sentence should pass with lenient filter
        assert!(filter.is_quality("Hello world again."));
    }

    #[test]
    fn test_rejection_reasons() {
        let filter = QualityFilter::default();

        let reason = filter.rejection_reason("Hi there.");
        assert!(matches!(reason, Some(RejectionReason::TooFewWords { .. })));

        // Use lenient thresholds to avoid other rejections
        let filter_punct = QualityFilter::builder()
            .min_words(3)
            .min_char_entropy(1.0)
            .min_alpha_ratio(0.5)
            .max_word_repetition(0.5)
            .require_terminal_punct(true)
            .build();
        let reason = filter_punct.rejection_reason("Hello world today");
        assert!(
            matches!(reason, Some(RejectionReason::MissingTerminalPunct)),
            "Expected MissingTerminalPunct, got {:?}",
            reason
        );
    }

    #[test]
    fn test_builder() {
        let filter = QualityFilter::builder()
            .min_words(3)
            .max_words(20)
            .max_word_repetition(0.4)
            .require_terminal_punct(false)
            .min_char_entropy(2.5)
            .min_alpha_ratio(0.6)
            .build();

        assert!(filter.is_quality("Hello world today."));
    }

    #[test]
    fn test_quality_stats() {
        let filter = QualityFilter::default();
        let sentences = vec![
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "Hi.".to_string(),
            "Another good sentence with enough words here.".to_string(),
            "Short one.".to_string(),
        ];

        let stats = filter.compute_stats(sentences.into_iter());

        assert_eq!(stats.total, 4);
        assert_eq!(stats.passed, 2);
        assert_eq!(stats.rejected, 2);
        assert_eq!(stats.too_few_words, 2);
    }

    #[test]
    fn test_unicode_support() {
        // Lenient filter for CJK text (no spaces between words)
        // CJK text is tokenized as a single "word" by whitespace, so repetition = 1.0
        let filter = QualityFilter::builder()
            .min_words(1)
            .min_char_entropy(1.0)
            .min_alpha_ratio(0.5)
            .min_avg_word_length(1.0)
            .max_word_repetition(1.0) // CJK has no word boundaries
            .build();

        // Chinese sentence (single "word" by whitespace tokenization)
        let chinese = "这是一个测试句子。";
        assert!(
            filter.is_quality(chinese),
            "Chinese failed: {:?}",
            filter.rejection_reason(chinese)
        );

        // Japanese sentence
        let japanese = "これは日本語のテストです。";
        assert!(
            filter.is_quality(japanese),
            "Japanese failed: {:?}",
            filter.rejection_reason(japanese)
        );

        // CJK with spaces should work with normal repetition settings
        let filter_normal = QualityFilter::builder()
            .min_words(3)
            .min_char_entropy(1.0)
            .min_alpha_ratio(0.5)
            .build();
        let chinese_spaced = "这是 一个 测试 句子。";
        assert!(
            filter_normal.is_quality(chinese_spaced),
            "Spaced Chinese failed: {:?}",
            filter_normal.rejection_reason(chinese_spaced)
        );
    }
}
