//! Text preprocessing for corpus normalization.
//!
//! This module provides configurable text preprocessing to normalize text
//! before training language models. Preprocessing steps include:
//!
//! - Number normalization (digits -> `<NUM>`)
//! - URL normalization (URLs -> `<URL>`)
//! - Email normalization (emails -> `<EMAIL>`)
//! - Contraction expansion ("can't" -> "cannot")
//! - Unicode normalization (NFC/NFD/NFKC/NFKD)
//! - Whitespace normalization
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::corpus::TextPreprocessor;
//!
//! let preprocessor = TextPreprocessor::builder()
//!     .normalize_numbers(true)
//!     .normalize_urls(true)
//!     .expand_contractions(true)
//!     .build();
//!
//! let text = "I can't visit https://example.com in 2024.";
//! let normalized = preprocessor.process(text);
//! // Result: "I cannot visit <URL> in <NUM>."
//! ```

use lazy_static::lazy_static;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

/// Special tokens used for normalization.
pub mod tokens {
    /// Token for normalized numbers.
    pub const NUM: &str = "<NUM>";
    /// Token for normalized URLs.
    pub const URL: &str = "<URL>";
    /// Token for normalized emails.
    pub const EMAIL: &str = "<EMAIL>";
    /// Token for normalized usernames/mentions.
    pub const USER: &str = "<USER>";
    /// Token for normalized hashtags.
    pub const HASHTAG: &str = "<HASHTAG>";
    /// Token for unknown/rare words.
    pub const UNK: &str = "<UNK>";
}

lazy_static! {
    // URL pattern - matches http(s), ftp, file URLs
    static ref URL_REGEX: Regex = Regex::new(
        r"(?i)(?:https?|ftp|file)://[^\s<>]+"
    ).expect("Invalid URL regex");

    // Email pattern
    static ref EMAIL_REGEX: Regex = Regex::new(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    ).expect("Invalid email regex");

    // Number pattern - matches integers, decimals, percentages, ordinals
    static ref NUMBER_REGEX: Regex = Regex::new(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|st|nd|rd|th)?\b"
    ).expect("Invalid number regex");

    // Twitter-style username
    static ref USERNAME_REGEX: Regex = Regex::new(
        r"@[a-zA-Z0-9_]+"
    ).expect("Invalid username regex");

    // Hashtag pattern
    static ref HASHTAG_REGEX: Regex = Regex::new(
        r"#[a-zA-Z0-9_]+"
    ).expect("Invalid hashtag regex");

    // Multiple whitespace
    static ref MULTI_SPACE_REGEX: Regex = Regex::new(r"\s+").expect("Invalid whitespace regex");
}

/// English contractions and their expansions.
const CONTRACTIONS: &[(&str, &str)] = &[
    // Negations
    ("can't", "cannot"),
    ("cannot", "cannot"),
    ("won't", "will not"),
    ("n't", " not"),
    // Be verbs
    ("'m", " am"),
    ("'re", " are"),
    ("'s", " is"), // Note: also possessive, context-dependent
    // Have verbs
    ("'ve", " have"),
    ("'d", " would"), // Also "had", context-dependent
    ("'ll", " will"),
    // Informal
    ("gonna", "going to"),
    ("gotta", "got to"),
    ("wanna", "want to"),
    ("lemme", "let me"),
    ("gimme", "give me"),
    ("kinda", "kind of"),
    ("sorta", "sort of"),
    ("dunno", "do not know"),
    ("'cause", "because"),
    ("'til", "until"),
    // Possessive contractions (less common to expand)
    ("let's", "let us"),
    ("that's", "that is"),
    ("there's", "there is"),
    ("here's", "here is"),
    ("what's", "what is"),
    ("who's", "who is"),
    ("how's", "how is"),
    ("where's", "where is"),
    ("it's", "it is"),
];

/// Unicode normalization form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnicodeNorm {
    /// No Unicode normalization.
    None,
    /// NFC (Canonical Decomposition, followed by Canonical Composition).
    #[default]
    Nfc,
    /// NFD (Canonical Decomposition).
    Nfd,
    /// NFKC (Compatibility Decomposition, followed by Canonical Composition).
    Nfkc,
    /// NFKD (Compatibility Decomposition).
    Nfkd,
}

/// Text preprocessor for corpus normalization.
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    /// Whether to normalize numbers to <NUM>.
    normalize_numbers: bool,
    /// Whether to normalize URLs to <URL>.
    normalize_urls: bool,
    /// Whether to normalize emails to <EMAIL>.
    normalize_emails: bool,
    /// Whether to normalize @mentions to <USER>.
    normalize_usernames: bool,
    /// Whether to normalize #hashtags to <HASHTAG>.
    normalize_hashtags: bool,
    /// Whether to expand contractions.
    expand_contractions: bool,
    /// Whether to lowercase text.
    lowercase: bool,
    /// Unicode normalization form.
    unicode_norm: UnicodeNorm,
    /// Whether to normalize whitespace.
    normalize_whitespace: bool,
    /// Whether to strip leading/trailing whitespace.
    strip: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            normalize_numbers: true,
            normalize_urls: true,
            normalize_emails: true,
            normalize_usernames: false,
            normalize_hashtags: false,
            expand_contractions: false,
            lowercase: false,
            unicode_norm: UnicodeNorm::Nfc,
            normalize_whitespace: true,
            strip: true,
        }
    }
}

impl TextPreprocessor {
    /// Create a new preprocessor with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> TextPreprocessorBuilder {
        TextPreprocessorBuilder::new()
    }

    /// Create a minimal preprocessor (only whitespace normalization).
    pub fn minimal() -> Self {
        Self {
            normalize_numbers: false,
            normalize_urls: false,
            normalize_emails: false,
            normalize_usernames: false,
            normalize_hashtags: false,
            expand_contractions: false,
            lowercase: false,
            unicode_norm: UnicodeNorm::Nfc,
            normalize_whitespace: true,
            strip: true,
        }
    }

    /// Create an aggressive preprocessor (all normalizations enabled).
    pub fn aggressive() -> Self {
        Self {
            normalize_numbers: true,
            normalize_urls: true,
            normalize_emails: true,
            normalize_usernames: true,
            normalize_hashtags: true,
            expand_contractions: true,
            lowercase: true,
            unicode_norm: UnicodeNorm::Nfkc,
            normalize_whitespace: true,
            strip: true,
        }
    }

    /// Process a text string.
    pub fn process(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Unicode normalization (do first for consistent matching)
        result = self.apply_unicode_norm(&result);

        // URL normalization (before email to avoid partial matches)
        if self.normalize_urls {
            result = URL_REGEX.replace_all(&result, tokens::URL).to_string();
        }

        // Email normalization
        if self.normalize_emails {
            result = EMAIL_REGEX.replace_all(&result, tokens::EMAIL).to_string();
        }

        // Username normalization
        if self.normalize_usernames {
            result = USERNAME_REGEX
                .replace_all(&result, tokens::USER)
                .to_string();
        }

        // Hashtag normalization
        if self.normalize_hashtags {
            result = HASHTAG_REGEX
                .replace_all(&result, tokens::HASHTAG)
                .to_string();
        }

        // Number normalization
        if self.normalize_numbers {
            result = NUMBER_REGEX.replace_all(&result, tokens::NUM).to_string();
        }

        // Contraction expansion
        if self.expand_contractions {
            result = self.expand_contractions_in(&result);
        }

        // Lowercase
        if self.lowercase {
            result = result.to_lowercase();
        }

        // Whitespace normalization
        if self.normalize_whitespace {
            result = MULTI_SPACE_REGEX.replace_all(&result, " ").to_string();
        }

        // Strip
        if self.strip {
            result = result.trim().to_string();
        }

        result
    }

    /// Process multiple texts.
    pub fn process_batch<'a, I>(&'a self, texts: I) -> impl Iterator<Item = String> + 'a
    where
        I: Iterator<Item = &'a str> + 'a,
    {
        texts.map(move |t| self.process(t))
    }

    /// Apply Unicode normalization.
    fn apply_unicode_norm(&self, text: &str) -> String {
        match self.unicode_norm {
            UnicodeNorm::None => text.to_string(),
            UnicodeNorm::Nfc => text.nfc().collect(),
            UnicodeNorm::Nfd => text.nfd().collect(),
            UnicodeNorm::Nfkc => text.nfkc().collect(),
            UnicodeNorm::Nfkd => text.nfkd().collect(),
        }
    }

    /// Expand contractions in text.
    fn expand_contractions_in(&self, text: &str) -> String {
        let mut result = text.to_string();

        for (contraction, expansion) in CONTRACTIONS {
            // Case-insensitive replacement
            let pattern = format!(r"(?i){}", regex::escape(contraction));
            if let Ok(re) = Regex::new(&pattern) {
                result = re.replace_all(&result, *expansion).to_string();
            }
        }

        result
    }
}

/// Builder for TextPreprocessor.
#[derive(Debug, Clone)]
pub struct TextPreprocessorBuilder {
    preprocessor: TextPreprocessor,
}

impl TextPreprocessorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            preprocessor: TextPreprocessor::default(),
        }
    }

    /// Set whether to normalize numbers.
    pub fn normalize_numbers(mut self, enable: bool) -> Self {
        self.preprocessor.normalize_numbers = enable;
        self
    }

    /// Set whether to normalize URLs.
    pub fn normalize_urls(mut self, enable: bool) -> Self {
        self.preprocessor.normalize_urls = enable;
        self
    }

    /// Set whether to normalize emails.
    pub fn normalize_emails(mut self, enable: bool) -> Self {
        self.preprocessor.normalize_emails = enable;
        self
    }

    /// Set whether to normalize @mentions.
    pub fn normalize_usernames(mut self, enable: bool) -> Self {
        self.preprocessor.normalize_usernames = enable;
        self
    }

    /// Set whether to normalize #hashtags.
    pub fn normalize_hashtags(mut self, enable: bool) -> Self {
        self.preprocessor.normalize_hashtags = enable;
        self
    }

    /// Set whether to expand contractions.
    pub fn expand_contractions(mut self, enable: bool) -> Self {
        self.preprocessor.expand_contractions = enable;
        self
    }

    /// Set whether to lowercase text.
    pub fn lowercase(mut self, enable: bool) -> Self {
        self.preprocessor.lowercase = enable;
        self
    }

    /// Set Unicode normalization form.
    pub fn unicode_norm(mut self, form: UnicodeNorm) -> Self {
        self.preprocessor.unicode_norm = form;
        self
    }

    /// Set whether to normalize whitespace.
    pub fn normalize_whitespace(mut self, enable: bool) -> Self {
        self.preprocessor.normalize_whitespace = enable;
        self
    }

    /// Set whether to strip leading/trailing whitespace.
    pub fn strip(mut self, enable: bool) -> Self {
        self.preprocessor.strip = enable;
        self
    }

    /// Build the preprocessor.
    pub fn build(self) -> TextPreprocessor {
        self.preprocessor
    }
}

impl Default for TextPreprocessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preprocessing pipeline that combines multiple steps.
#[derive(Debug, Clone)]
pub struct PreprocessingPipeline {
    /// Text preprocessor.
    preprocessor: TextPreprocessor,
    /// Optional quality filter.
    quality_filter: Option<super::QualityFilter>,
    /// Optional deduplicator.
    dedup_mode: Option<super::DeduplicationMode>,
}

impl PreprocessingPipeline {
    /// Create a new pipeline with default settings.
    pub fn new() -> Self {
        Self {
            preprocessor: TextPreprocessor::default(),
            quality_filter: None,
            dedup_mode: None,
        }
    }

    /// Create a builder for the pipeline.
    pub fn builder() -> PreprocessingPipelineBuilder {
        PreprocessingPipelineBuilder::new()
    }

    /// Process a sentence through the pipeline.
    ///
    /// Returns `Some(processed_text)` if the sentence passes all filters,
    /// `None` if it should be filtered out.
    pub fn process(&self, text: &str) -> Option<String> {
        // Preprocess first
        let processed = self.preprocessor.process(text);

        // Quality filter
        if let Some(ref filter) = self.quality_filter {
            if !filter.is_quality(&processed) {
                return None;
            }
        }

        Some(processed)
    }

    /// Process a batch of sentences through the pipeline.
    ///
    /// Returns an iterator of processed sentences that pass all filters.
    /// Includes deduplication if configured.
    pub fn process_batch<'a, I>(&'a self, texts: I) -> Box<dyn Iterator<Item = String> + 'a>
    where
        I: Iterator<Item = String> + 'a,
    {
        let processed = texts.filter_map(move |t| self.process(&t));

        if let Some(mode) = &self.dedup_mode {
            let mut dedup = super::Deduplicator::new(*mode);
            Box::new(processed.filter(move |s| dedup.is_unique(s)))
        } else {
            Box::new(processed)
        }
    }
}

impl Default for PreprocessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for PreprocessingPipeline.
#[derive(Debug, Clone)]
pub struct PreprocessingPipelineBuilder {
    pipeline: PreprocessingPipeline,
}

impl PreprocessingPipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            pipeline: PreprocessingPipeline::new(),
        }
    }

    /// Set the text preprocessor.
    pub fn preprocessor(mut self, preprocessor: TextPreprocessor) -> Self {
        self.pipeline.preprocessor = preprocessor;
        self
    }

    /// Set the quality filter.
    pub fn quality_filter(mut self, filter: super::QualityFilter) -> Self {
        self.pipeline.quality_filter = Some(filter);
        self
    }

    /// Set the deduplication mode.
    pub fn deduplication(mut self, mode: super::DeduplicationMode) -> Self {
        self.pipeline.dedup_mode = Some(mode);
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> PreprocessingPipeline {
        self.pipeline
    }
}

impl Default for PreprocessingPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_normalization() {
        let pp = TextPreprocessor::builder()
            .normalize_urls(true)
            .normalize_numbers(false)
            .build();

        assert_eq!(
            pp.process("Visit https://example.com for more."),
            "Visit <URL> for more."
        );

        assert_eq!(
            pp.process("See http://foo.bar/baz?q=1 and https://a.b"),
            "See <URL> and <URL>"
        );
    }

    #[test]
    fn test_email_normalization() {
        let pp = TextPreprocessor::builder()
            .normalize_emails(true)
            .normalize_urls(false)
            .normalize_numbers(false)
            .build();

        assert_eq!(
            pp.process("Contact user@example.com for help."),
            "Contact <EMAIL> for help."
        );
    }

    #[test]
    fn test_number_normalization() {
        let pp = TextPreprocessor::builder()
            .normalize_numbers(true)
            .normalize_urls(false)
            .build();

        assert_eq!(pp.process("I have 42 apples."), "I have <NUM> apples.");
        assert_eq!(pp.process("It costs $1,234.56"), "It costs $<NUM>");
        assert_eq!(pp.process("The 1st place winner"), "The <NUM> place winner");
    }

    #[test]
    fn test_contraction_expansion() {
        let pp = TextPreprocessor::builder()
            .expand_contractions(true)
            .normalize_numbers(false)
            .normalize_urls(false)
            .build();

        assert_eq!(pp.process("I can't do it."), "I cannot do it.");
        assert_eq!(pp.process("They won't come."), "They will not come.");
        assert_eq!(pp.process("I'm going home."), "I am going home.");
    }

    #[test]
    fn test_lowercase() {
        let pp = TextPreprocessor::builder()
            .lowercase(true)
            .normalize_numbers(false)
            .normalize_urls(false)
            .build();

        assert_eq!(pp.process("Hello WORLD!"), "hello world!");
    }

    #[test]
    fn test_whitespace_normalization() {
        let pp = TextPreprocessor::builder()
            .normalize_whitespace(true)
            .normalize_numbers(false)
            .normalize_urls(false)
            .build();

        assert_eq!(
            pp.process("  Multiple   spaces   here  "),
            "Multiple spaces here"
        );
    }

    #[test]
    fn test_username_normalization() {
        let pp = TextPreprocessor::builder()
            .normalize_usernames(true)
            .normalize_numbers(false)
            .normalize_urls(false)
            .build();

        assert_eq!(
            pp.process("Hey @user123, check this out!"),
            "Hey <USER>, check this out!"
        );
    }

    #[test]
    fn test_hashtag_normalization() {
        let pp = TextPreprocessor::builder()
            .normalize_hashtags(true)
            .normalize_numbers(false)
            .normalize_urls(false)
            .build();

        assert_eq!(
            pp.process("Loving this #coding life!"),
            "Loving this <HASHTAG> life!"
        );
    }

    #[test]
    fn test_combined_preprocessing() {
        let pp = TextPreprocessor::aggressive();

        let text = "Hey @user, I can't visit https://example.com in 2024! #excited";
        let result = pp.process(text);

        // Aggressive mode lowercases everything, including tokens
        assert!(result.contains("<user>"), "Expected <user> in: {}", result);
        assert!(result.contains("<url>"), "Expected <url> in: {}", result);
        assert!(result.contains("<num>"), "Expected <num> in: {}", result);
        assert!(
            result.contains("<hashtag>"),
            "Expected <hashtag> in: {}",
            result
        );
        assert!(
            result.contains("cannot"),
            "Expected 'cannot' in: {}",
            result
        );
        assert_eq!(result, result.to_lowercase());
    }

    #[test]
    fn test_minimal_preprocessor() {
        let pp = TextPreprocessor::minimal();

        let text = "  Hello  123  world@example.com  ";
        let result = pp.process(text);

        assert_eq!(result, "Hello 123 world@example.com");
    }

    #[test]
    fn test_unicode_normalization() {
        let pp = TextPreprocessor::builder()
            .unicode_norm(UnicodeNorm::Nfc)
            .normalize_numbers(false)
            .normalize_urls(false)
            .build();

        // Combining character (e + combining acute = e with accent)
        let composed = "cafe\u{0301}"; // cafe + combining acute
        let result = pp.process(composed);

        // After NFC, the combining character should be composed
        assert_eq!(result.chars().count(), 4); // c-a-f-e with accent as one char
    }

    #[test]
    fn test_pipeline() {
        use super::super::{DeduplicationMode, QualityFilter};

        let pipeline = PreprocessingPipeline::builder()
            .preprocessor(TextPreprocessor::default())
            .quality_filter(QualityFilter::builder().min_words(3).build())
            .deduplication(DeduplicationMode::Normalized)
            .build();

        // Short sentence should be filtered out
        assert!(pipeline.process("Hi.").is_none());

        // Good sentence should pass
        let result = pipeline.process("This is a good sentence with enough words.");
        assert!(result.is_some());
    }

    #[test]
    fn test_batch_processing() {
        let pp = TextPreprocessor::builder()
            .normalize_numbers(true)
            .normalize_urls(false)
            .build();

        let texts = ["I have 5 apples.", "You have 10 oranges."];
        let results: Vec<String> = pp.process_batch(texts.iter().copied()).collect();

        assert_eq!(results[0], "I have <NUM> apples.");
        assert_eq!(results[1], "You have <NUM> oranges.");
    }
}
