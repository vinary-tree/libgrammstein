//! Language metadata for Google Books N-grams.
//!
//! Provides URL patterns, file prefixes, and metadata for supported languages.

use std::collections::HashMap;
use lazy_static::lazy_static;

/// Metadata for a supported language.
#[derive(Clone, Debug)]
pub struct LanguageMetadata {
    /// BCP-47 language tag (e.g., "en", "de").
    pub tag: &'static str,

    /// Google Books corpus ID (e.g., "eng", "ger").
    pub corpus_id: &'static str,

    /// Display name.
    pub name: &'static str,

    /// Whether this language uses Latin script.
    pub latin_script: bool,

    /// Notes about special handling.
    pub notes: Option<&'static str>,
}

/// Base URL for Google Books N-grams.
pub const BASE_URL: &str = "https://storage.googleapis.com/books/ngrams/books";

/// Current version of the n-gram dataset.
pub const VERSION: &str = "20120701";

lazy_static! {
    /// Supported languages and their metadata.
    pub static ref SUPPORTED_LANGUAGES: HashMap<&'static str, LanguageMetadata> = {
        let mut m = HashMap::new();

        m.insert("en", LanguageMetadata {
            tag: "en",
            corpus_id: "eng",
            name: "English",
            latin_script: true,
            notes: None,
        });

        m.insert("en-fiction", LanguageMetadata {
            tag: "en-fiction",
            corpus_id: "eng-fiction",
            name: "English Fiction",
            latin_script: true,
            notes: Some("Subset of English corpus from fiction works"),
        });

        m.insert("de", LanguageMetadata {
            tag: "de",
            corpus_id: "ger",
            name: "German",
            latin_script: true,
            notes: None,
        });

        m.insert("fr", LanguageMetadata {
            tag: "fr",
            corpus_id: "fre",
            name: "French",
            latin_script: true,
            notes: None,
        });

        m.insert("es", LanguageMetadata {
            tag: "es",
            corpus_id: "spa",
            name: "Spanish",
            latin_script: true,
            notes: None,
        });

        m.insert("it", LanguageMetadata {
            tag: "it",
            corpus_id: "ita",
            name: "Italian",
            latin_script: true,
            notes: None,
        });

        m.insert("ru", LanguageMetadata {
            tag: "ru",
            corpus_id: "rus",
            name: "Russian",
            latin_script: false,
            notes: Some("Cyrillic script"),
        });

        m.insert("he", LanguageMetadata {
            tag: "he",
            corpus_id: "heb",
            name: "Hebrew",
            latin_script: false,
            notes: Some("Right-to-left script"),
        });

        m.insert("zh", LanguageMetadata {
            tag: "zh",
            corpus_id: "chi-sim",
            name: "Chinese (Simplified)",
            latin_script: false,
            notes: Some("Character-based, no word boundaries"),
        });

        m
    };

    /// Two-letter prefixes for 2-5 grams.
    ///
    /// Lazily built once: 676 ("aa".."zz") + 2 ("other", "punctuation")
    /// entries. The previous version stored these as owned `String`s;
    /// switching to `&'static str` via a `Box::leak` of the concatenated
    /// buffer means callers that need `&str` borrow directly with no
    /// allocation per call.
    pub static ref MULTIGRAM_PREFIXES: Vec<&'static str> = {
        // Pre-size the joined buffer: 676 prefixes × 2 chars + 2 special
        // prefixes ("other" 5 + "punctuation" 11) = 1352 + 16 = 1368 bytes.
        let mut buf = String::with_capacity(1368);
        let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(678);
        for c1 in 'a'..='z' {
            for c2 in 'a'..='z' {
                let start = buf.len();
                buf.push(c1);
                buf.push(c2);
                offsets.push((start, buf.len()));
            }
        }
        let other_start = buf.len();
        buf.push_str("other");
        offsets.push((other_start, buf.len()));
        let punct_start = buf.len();
        buf.push_str("punctuation");
        offsets.push((punct_start, buf.len()));

        // Leak once; the resulting &'static str backs every &str slice below.
        let leaked: &'static str = Box::leak(buf.into_boxed_str());
        offsets.into_iter().map(|(s, e)| &leaked[s..e]).collect()
    };
}

/// Single-letter prefixes for 1-grams.
///
/// Replaces the lazy_static + match-arm-from-char workaround that
/// previously required a 26-arm `match c { 'a' => "a", ... }` to coerce
/// `char` to `&'static str`. A const slice is simpler, allocation-free,
/// and lets the compiler verify length at compile time.
pub static UNIGRAM_PREFIXES: &[&str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "other",
];

/// Get the URL for a specific n-gram file.
///
/// # Arguments
///
/// * `language` - Language tag (e.g., "en", "de")
/// * `order` - N-gram order (1-5)
/// * `prefix` - File prefix (e.g., "a" for 1-grams, "aa" for higher orders)
///
/// # Returns
///
/// Full URL to the gzipped n-gram file.
pub fn get_file_url(language: &str, order: u8, prefix: &str) -> Option<String> {
    let metadata = SUPPORTED_LANGUAGES.get(language)?;

    Some(format!(
        "{}/googlebooks-{}-all-{}gram-{}-{}.gz",
        BASE_URL,
        metadata.corpus_id,
        order,
        VERSION,
        prefix
    ))
}

/// Get all file URLs for a specific language and order.
pub fn get_order_urls(language: &str, order: u8) -> Option<Vec<String>> {
    let metadata = SUPPORTED_LANGUAGES.get(language)?;

    let prefixes: &[&str] = if order == 1 {
        UNIGRAM_PREFIXES
    } else {
        MULTIGRAM_PREFIXES.as_slice()
    };

    let urls: Vec<String> = prefixes
        .iter()
        .map(|prefix| {
            format!(
                "{}/googlebooks-{}-all-{}gram-{}-{}.gz",
                BASE_URL,
                metadata.corpus_id,
                order,
                VERSION,
                prefix
            )
        })
        .collect();

    Some(urls)
}

/// Get all prefixes for a specific order.
pub fn get_prefixes(order: u8) -> Vec<String> {
    if order == 1 {
        UNIGRAM_PREFIXES.iter().map(|s| s.to_string()).collect()
    } else {
        MULTIGRAM_PREFIXES.iter().map(|s| s.to_string()).collect()
    }
}

/// Validates that a prefix is valid for the given n-gram order.
///
/// # Arguments
///
/// * `order` - N-gram order (1-5)
/// * `prefix` - The prefix to validate (e.g., "j" for 1-grams, "th" for 2-5 grams)
///
/// # Returns
///
/// `true` if the prefix is valid for the given order, `false` otherwise.
///
/// # Examples
///
/// ```
/// use libgrammstein::sources::google_books::is_valid_prefix;
///
/// // Valid 1-gram prefixes
/// assert!(is_valid_prefix(1, "a"));
/// assert!(is_valid_prefix(1, "z"));
/// assert!(is_valid_prefix(1, "other"));
///
/// // Invalid for 1-grams (two-letter prefixes)
/// assert!(!is_valid_prefix(1, "th"));
///
/// // Valid 2-5 gram prefixes
/// assert!(is_valid_prefix(2, "th"));
/// assert!(is_valid_prefix(3, "aa"));
/// assert!(is_valid_prefix(5, "punctuation"));
///
/// // Invalid for 2-5 grams (single-letter prefixes)
/// assert!(!is_valid_prefix(2, "t"));
/// ```
pub fn is_valid_prefix(order: u8, prefix: &str) -> bool {
    if order == 1 {
        UNIGRAM_PREFIXES.contains(&prefix)
    } else {
        MULTIGRAM_PREFIXES.contains(&prefix)
    }
}

/// Check if a language is supported.
pub fn is_supported(language: &str) -> bool {
    SUPPORTED_LANGUAGES.contains_key(language)
}

/// Simplified language info for CLI usage.
#[derive(Clone, Debug)]
pub struct LanguageInfo {
    /// Language tag.
    pub tag: String,
    /// Display name.
    pub name: String,
    /// Google Books corpus ID.
    pub corpus_id: String,
}

impl LanguageInfo {
    /// Get language info from a language code.
    pub fn from_code(code: &str) -> Option<Self> {
        let metadata = SUPPORTED_LANGUAGES.get(code)?;
        Some(Self {
            tag: metadata.tag.to_string(),
            name: metadata.name.to_string(),
            corpus_id: metadata.corpus_id.to_string(),
        })
    }
}

/// Get metadata for a language.
pub fn get_metadata(language: &str) -> Option<&'static LanguageMetadata> {
    SUPPORTED_LANGUAGES.get(language)
}

/// List all supported language tags.
pub fn list_languages() -> Vec<&'static str> {
    SUPPORTED_LANGUAGES.keys().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_languages() {
        assert!(is_supported("en"));
        assert!(is_supported("de"));
        assert!(is_supported("fr"));
        assert!(!is_supported("invalid"));
    }

    #[test]
    fn test_get_file_url() {
        let url = get_file_url("en", 1, "a").unwrap();
        assert_eq!(
            url,
            "https://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-1gram-20120701-a.gz"
        );

        let url = get_file_url("en", 5, "aa").unwrap();
        assert_eq!(
            url,
            "https://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-5gram-20120701-aa.gz"
        );
    }

    #[test]
    fn test_unigram_prefixes() {
        assert_eq!(UNIGRAM_PREFIXES.len(), 27); // a-z + other
        assert_eq!(UNIGRAM_PREFIXES[0], "a");
        assert_eq!(UNIGRAM_PREFIXES[25], "z");
        assert_eq!(UNIGRAM_PREFIXES[26], "other");
    }

    #[test]
    fn test_multigram_prefixes() {
        // 26*26 = 676 + 2 (other, punctuation) = 678
        assert_eq!(MULTIGRAM_PREFIXES.len(), 678);
        assert_eq!(MULTIGRAM_PREFIXES[0], "aa");
        assert_eq!(MULTIGRAM_PREFIXES[675], "zz");
        assert_eq!(MULTIGRAM_PREFIXES[676], "other");
        assert_eq!(MULTIGRAM_PREFIXES[677], "punctuation");
    }

    #[test]
    fn test_get_prefixes() {
        let unigram_prefixes = get_prefixes(1);
        assert_eq!(unigram_prefixes.len(), 27);

        let bigram_prefixes = get_prefixes(2);
        assert_eq!(bigram_prefixes.len(), 678);
    }

    #[test]
    fn test_german_url() {
        let url = get_file_url("de", 3, "abc").unwrap();
        assert!(url.contains("googlebooks-ger-all-3gram"));
    }

    #[test]
    fn test_is_valid_prefix_unigrams() {
        // Valid 1-gram prefixes
        assert!(is_valid_prefix(1, "a"));
        assert!(is_valid_prefix(1, "j"));
        assert!(is_valid_prefix(1, "z"));
        assert!(is_valid_prefix(1, "other"));

        // Invalid for 1-grams (two-letter prefixes)
        assert!(!is_valid_prefix(1, "th"));
        assert!(!is_valid_prefix(1, "aa"));
        assert!(!is_valid_prefix(1, "punctuation"));

        // Invalid for 1-grams (not a valid prefix at all)
        assert!(!is_valid_prefix(1, "invalid"));
        assert!(!is_valid_prefix(1, ""));
    }

    #[test]
    fn test_is_valid_prefix_multigrams() {
        // Valid 2-5 gram prefixes
        assert!(is_valid_prefix(2, "th"));
        assert!(is_valid_prefix(2, "aa"));
        assert!(is_valid_prefix(2, "zz"));
        assert!(is_valid_prefix(2, "other"));
        assert!(is_valid_prefix(2, "punctuation"));

        // Valid for higher orders too
        assert!(is_valid_prefix(3, "th"));
        assert!(is_valid_prefix(4, "aa"));
        assert!(is_valid_prefix(5, "punctuation"));

        // Invalid for 2-5 grams (single-letter prefixes)
        assert!(!is_valid_prefix(2, "t"));
        assert!(!is_valid_prefix(3, "a"));

        // Invalid for all orders
        assert!(!is_valid_prefix(2, "invalid"));
        assert!(!is_valid_prefix(5, ""));
    }
}
