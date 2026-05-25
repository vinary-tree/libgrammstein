//! Text normalization utilities.
//!
//! Provides Unicode normalization and text cleaning for corpus processing.

use regex::Regex;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

/// Regex for collapsing multiple whitespace.
static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("Invalid whitespace regex"));

/// Regex for removing control characters.
static CONTROL_CHARS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").expect("Invalid control chars regex")
});

/// Text normalizer for preprocessing corpus text.
#[derive(Clone, Debug, Default)]
pub struct Normalizer {
    /// Apply Unicode NFC normalization.
    nfc: bool,

    /// Remove control characters.
    remove_control_chars: bool,

    /// Collapse multiple whitespace to single space.
    collapse_whitespace: bool,

    /// Strip leading/trailing whitespace.
    strip: bool,
}

impl Normalizer {
    /// Create a new normalizer with default settings.
    pub fn new() -> Self {
        Self {
            nfc: true,
            remove_control_chars: true,
            collapse_whitespace: true,
            strip: true,
        }
    }

    /// Enable or disable NFC normalization.
    pub fn with_nfc(mut self, nfc: bool) -> Self {
        self.nfc = nfc;
        self
    }

    /// Enable or disable control character removal.
    pub fn with_remove_control_chars(mut self, remove: bool) -> Self {
        self.remove_control_chars = remove;
        self
    }

    /// Enable or disable whitespace collapsing.
    pub fn with_collapse_whitespace(mut self, collapse: bool) -> Self {
        self.collapse_whitespace = collapse;
        self
    }

    /// Enable or disable stripping.
    pub fn with_strip(mut self, strip: bool) -> Self {
        self.strip = strip;
        self
    }

    /// Normalize a string according to the configured options.
    pub fn normalize(&self, text: &str) -> String {
        let mut result = if self.nfc {
            text.nfc().collect::<String>()
        } else {
            text.to_string()
        };

        if self.remove_control_chars {
            result = CONTROL_CHARS.replace_all(&result, "").to_string();
        }

        if self.collapse_whitespace {
            result = WHITESPACE.replace_all(&result, " ").to_string();
        }

        if self.strip {
            result = result.trim().to_string();
        }

        result
    }

    /// Normalize in place, modifying the string.
    pub fn normalize_in_place(&self, text: &mut String) {
        *text = self.normalize(text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_collapse() {
        let normalizer = Normalizer::new();
        let text = "hello    world\n\n\tfoo";
        let normalized = normalizer.normalize(text);
        assert_eq!(normalized, "hello world foo");
    }

    #[test]
    fn test_control_char_removal() {
        let normalizer = Normalizer::new();
        let text = "hello\x00world\x1F";
        let normalized = normalizer.normalize(text);
        assert_eq!(normalized, "helloworld");
    }

    #[test]
    fn test_nfc_normalization() {
        let normalizer = Normalizer::new();
        // é as e + combining acute (NFD) should become é (NFC)
        let text = "caf\u{0065}\u{0301}";
        let normalized = normalizer.normalize(text);
        assert_eq!(normalized, "café");
    }

    #[test]
    fn test_strip() {
        let normalizer = Normalizer::new();
        let text = "  hello world  ";
        let normalized = normalizer.normalize(text);
        assert_eq!(normalized, "hello world");
    }
}
