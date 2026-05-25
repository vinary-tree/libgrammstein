//! Text tokenization utilities.
//!
//! Provides sentence and word tokenization for corpus processing.

use regex::Regex;
use std::sync::LazyLock;

/// Regex for sentence boundary detection.
static SENTENCE_BOUNDARY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[.!?]+\s+").expect("Invalid sentence boundary regex"));

/// Regex for word tokenization.
static WORD_BOUNDARY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\s\p{P}]+").expect("Invalid word boundary regex"));

/// Text tokenizer for sentence and word segmentation.
#[derive(Clone, Debug, Default)]
pub struct Tokenizer {
    /// Minimum sentence length in characters.
    min_sentence_length: usize,

    /// Minimum word length in characters.
    min_word_length: usize,

    /// Whether to lowercase tokens.
    lowercase: bool,
}

impl Tokenizer {
    /// Create a new tokenizer with default settings.
    pub fn new() -> Self {
        Self {
            min_sentence_length: 10,
            min_word_length: 1,
            lowercase: true,
        }
    }

    /// Set minimum sentence length.
    pub fn with_min_sentence_length(mut self, length: usize) -> Self {
        self.min_sentence_length = length;
        self
    }

    /// Set minimum word length.
    pub fn with_min_word_length(mut self, length: usize) -> Self {
        self.min_word_length = length;
        self
    }

    /// Enable or disable lowercasing.
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Split text into sentences.
    ///
    /// Uses simple regex-based sentence boundary detection.
    pub fn sentences<'a>(&self, text: &'a str) -> impl Iterator<Item = String> + 'a {
        let min_len = self.min_sentence_length;
        let lowercase = self.lowercase;

        SENTENCE_BOUNDARY
            .split(text)
            .filter(move |s| s.len() >= min_len)
            .map(move |s| {
                let s = s.trim();
                if lowercase {
                    s.to_lowercase()
                } else {
                    s.to_string()
                }
            })
    }

    /// Split text into words.
    ///
    /// Splits on whitespace and punctuation.
    pub fn words<'a>(&self, text: &'a str) -> impl Iterator<Item = String> + 'a {
        let min_len = self.min_word_length;
        let lowercase = self.lowercase;

        WORD_BOUNDARY
            .split(text)
            .filter(move |w| w.len() >= min_len)
            .map(move |w| {
                if lowercase {
                    w.to_lowercase()
                } else {
                    w.to_string()
                }
            })
    }

    /// Tokenize text into word tokens.
    ///
    /// Returns a vector of words.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.words(text).collect()
    }

    /// Tokenize text and return references to original positions.
    ///
    /// Useful when you need to map back to original text.
    pub fn tokenize_with_spans<'a>(&self, text: &'a str) -> Vec<(&'a str, usize, usize)> {
        let mut tokens = Vec::new();
        let mut last_end = 0;

        for mat in WORD_BOUNDARY.find_iter(text) {
            if last_end < mat.start() {
                let token = &text[last_end..mat.start()];
                if token.len() >= self.min_word_length {
                    tokens.push((token, last_end, mat.start()));
                }
            }
            last_end = mat.end();
        }

        // Don't forget the last token
        if last_end < text.len() {
            let token = &text[last_end..];
            if token.len() >= self.min_word_length {
                tokens.push((token, last_end, text.len()));
            }
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_tokenization() {
        let tokenizer = Tokenizer::new().with_min_sentence_length(5);
        let text = "Hello world. This is a test! How are you?";
        let sentences: Vec<_> = tokenizer.sentences(text).collect();

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "hello world");
        assert_eq!(sentences[1], "this is a test");
        assert_eq!(sentences[2], "how are you?");
    }

    #[test]
    fn test_word_tokenization() {
        let tokenizer = Tokenizer::new();
        let text = "Hello, world! This is a test.";
        let words: Vec<_> = tokenizer.words(text).collect();

        assert_eq!(words, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_no_lowercase() {
        let tokenizer = Tokenizer::new().with_lowercase(false);
        let text = "Hello World";
        let words: Vec<_> = tokenizer.words(text).collect();

        assert_eq!(words, vec!["Hello", "World"]);
    }
}
