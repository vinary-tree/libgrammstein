//! BCP 47 language tag parsing and manipulation.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error type for language tag parsing.
#[derive(Error, Debug)]
pub enum LanguageTagError {
    /// Empty language tag provided.
    #[error("Empty language tag")]
    Empty,

    /// Invalid language code.
    #[error("Invalid language code: {0}")]
    InvalidLanguage(String),

    /// Invalid script code.
    #[error("Invalid script code: {0}")]
    InvalidScript(String),

    /// Invalid region code.
    #[error("Invalid region code: {0}")]
    InvalidRegion(String),

    /// Parse error from unic-langid.
    #[error("Language tag parse error: {0}")]
    Parse(String),
}

/// BCP 47 language tag with optional dialect/region.
///
/// Represents a language identifier following the BCP 47 standard,
/// commonly used for identifying languages in internationalization.
///
/// # Examples
///
/// ```ignore
/// use libgrammstein::language::LanguageTag;
///
/// // Parse various language tags
/// let en_us: LanguageTag = "en-US".parse().unwrap();
/// let zh_hans: LanguageTag = "zh-Hans".parse().unwrap();
/// let pt_br: LanguageTag = "pt-BR".parse().unwrap();
///
/// // Access components
/// assert_eq!(en_us.language(), "en");
/// assert_eq!(en_us.region(), Some("US"));
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LanguageTag {
    /// Primary language (ISO 639-1 or 639-3).
    language: String,

    /// Optional script (ISO 15924).
    script: Option<String>,

    /// Optional region (ISO 3166-1 alpha-2).
    region: Option<String>,

    /// Optional variant subtag.
    variant: Option<String>,
}

impl LanguageTag {
    /// Create a new language tag with just the primary language.
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into().to_lowercase(),
            script: None,
            region: None,
            variant: None,
        }
    }

    /// Create a language tag with language and region.
    pub fn with_region(language: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            language: language.into().to_lowercase(),
            script: None,
            region: Some(region.into().to_uppercase()),
            variant: None,
        }
    }

    /// Create a language tag with language and script.
    pub fn with_script(language: impl Into<String>, script: impl Into<String>) -> Self {
        let script_str = script.into();
        // Capitalize first letter, lowercase rest (Title case for scripts)
        let script_normalized = if !script_str.is_empty() {
            let mut chars = script_str.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().to_string() + &chars.as_str().to_lowercase()
                }
                None => String::new(),
            }
        } else {
            String::new()
        };

        Self {
            language: language.into().to_lowercase(),
            script: Some(script_normalized),
            region: None,
            variant: None,
        }
    }

    /// Parse a BCP 47 language tag string.
    pub fn parse(tag: &str) -> Result<Self, LanguageTagError> {
        if tag.is_empty() {
            return Err(LanguageTagError::Empty);
        }

        // Use unic-langid for parsing
        let langid: unic_langid::LanguageIdentifier = tag
            .parse()
            .map_err(|e: unic_langid::LanguageIdentifierError| {
                LanguageTagError::Parse(e.to_string())
            })?;

        Ok(Self {
            language: langid.language.to_string(),
            script: langid.script.map(|s| s.to_string()),
            region: langid.region.map(|r| r.to_string()),
            variant: None, // unic-langid doesn't expose variants directly in the simple API
        })
    }

    /// Get the primary language code.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Get the script code, if any.
    pub fn script(&self) -> Option<&str> {
        self.script.as_deref()
    }

    /// Get the region code, if any.
    pub fn region(&self) -> Option<&str> {
        self.region.as_deref()
    }

    /// Get the variant code, if any.
    pub fn variant(&self) -> Option<&str> {
        self.variant.as_deref()
    }

    /// Get the base language tag without dialect/region.
    pub fn base(&self) -> LanguageTag {
        LanguageTag {
            language: self.language.clone(),
            script: None,
            region: None,
            variant: None,
        }
    }

    /// Check if this tag matches another tag (with fallback semantics).
    ///
    /// Returns true if:
    /// - Tags are exactly equal, OR
    /// - This tag is a more specific version of `other` (same language, this has region)
    pub fn matches(&self, other: &LanguageTag) -> bool {
        if self == other {
            return true;
        }

        // Check if languages match
        if self.language != other.language {
            return false;
        }

        // If other has no script/region, any tag with same language matches
        if other.script.is_none() && other.region.is_none() {
            return true;
        }

        // If scripts are specified, they must match
        if other.script.is_some() && self.script != other.script {
            return false;
        }

        // If regions are specified, they must match
        if other.region.is_some() && self.region != other.region {
            return false;
        }

        true
    }

    /// Convert to a directory path component.
    ///
    /// Returns a string suitable for use in file paths, e.g., "en/en-US".
    pub fn to_path(&self) -> String {
        let mut path = self.language.clone();
        if self.script.is_some() || self.region.is_some() {
            path.push('/');
            path.push_str(&self.to_string());
        }
        path
    }
}

impl fmt::Display for LanguageTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.language)?;
        if let Some(ref script) = self.script {
            write!(f, "-{}", script)?;
        }
        if let Some(ref region) = self.region {
            write!(f, "-{}", region)?;
        }
        if let Some(ref variant) = self.variant {
            write!(f, "-{}", variant)?;
        }
        Ok(())
    }
}

impl FromStr for LanguageTag {
    type Err = LanguageTagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LanguageTag::parse(s)
    }
}

/// Well-known Wikipedia dump URLs by language.
pub const WIKIPEDIA_URLS: &[(&str, &str)] = &[
    (
        "en",
        "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "simple",
        "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2",
    ),
    (
        "de",
        "https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2",
    ),
    (
        "fr",
        "https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "es",
        "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles.xml.bz2",
    ),
    (
        "pt",
        "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "it",
        "https://dumps.wikimedia.org/itwiki/latest/itwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "ru",
        "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "zh",
        "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "ja",
        "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2",
    ),
    (
        "ko",
        "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2",
    ),
    (
        "ar",
        "https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "nl",
        "https://dumps.wikimedia.org/nlwiki/latest/nlwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "pl",
        "https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles.xml.bz2",
    ),
    (
        "sv",
        "https://dumps.wikimedia.org/svwiki/latest/svwiki-latest-pages-articles.xml.bz2",
    ),
];

/// Get the Wikipedia dump URL for a language.
pub fn wikipedia_dump_url(lang: &str) -> String {
    format!(
        "https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2",
        lang, lang
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let tag: LanguageTag = "en".parse().unwrap();
        assert_eq!(tag.language(), "en");
        assert_eq!(tag.script(), None);
        assert_eq!(tag.region(), None);
    }

    #[test]
    fn test_parse_with_region() {
        let tag: LanguageTag = "en-US".parse().unwrap();
        assert_eq!(tag.language(), "en");
        assert_eq!(tag.region(), Some("US"));
    }

    #[test]
    fn test_parse_with_script() {
        let tag: LanguageTag = "zh-Hans".parse().unwrap();
        assert_eq!(tag.language(), "zh");
        assert_eq!(tag.script(), Some("Hans"));
    }

    #[test]
    fn test_display() {
        let tag = LanguageTag::with_region("en", "US");
        assert_eq!(tag.to_string(), "en-US");

        let tag = LanguageTag::with_script("zh", "Hans");
        assert_eq!(tag.to_string(), "zh-Hans");
    }

    #[test]
    fn test_matches() {
        let en = LanguageTag::new("en");
        let en_us = LanguageTag::with_region("en", "US");
        let en_gb = LanguageTag::with_region("en", "GB");
        let de = LanguageTag::new("de");

        // Same language matches
        assert!(en_us.matches(&en));
        assert!(en_gb.matches(&en));

        // Different regions don't match specific tags
        assert!(!en_us.matches(&en_gb));

        // Different languages don't match
        assert!(!de.matches(&en));
    }

    #[test]
    fn test_to_path() {
        let en = LanguageTag::new("en");
        assert_eq!(en.to_path(), "en");

        let en_us = LanguageTag::with_region("en", "US");
        assert_eq!(en_us.to_path(), "en/en-US");
    }
}
