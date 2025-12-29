//! Language-aware tokenization.

use unicode_segmentation::UnicodeSegmentation;

use super::LanguageTag;

/// Trait for tokenizing text into words/tokens.
pub trait Tokenizer: Send + Sync {
    /// Tokenize the given text into a vector of tokens.
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str>;

    /// Tokenize the given text, returning an iterator.
    fn tokenize_iter<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = &'a str> + 'a>;
}

/// Simple whitespace-based tokenizer.
///
/// This is the default tokenizer for most Western languages that use
/// space-separated words.
#[derive(Clone, Debug, Default)]
pub struct WhitespaceTokenizer {
    /// Whether to lowercase tokens.
    lowercase: bool,
}

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a tokenizer with a specific locale (currently unused but reserved for future).
    pub fn with_locale(_locale: &str) -> Self {
        Self::default()
    }

    /// Set whether to lowercase tokens.
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split_whitespace().collect()
    }

    fn tokenize_iter<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = &'a str> + 'a> {
        Box::new(text.split_whitespace())
    }
}

/// Unicode word tokenizer using UAX #29 word boundaries.
///
/// This tokenizer respects Unicode word boundaries and is suitable for
/// most languages with word-like units.
#[derive(Clone, Debug, Default)]
pub struct UnicodeWordTokenizer;

impl UnicodeWordTokenizer {
    /// Create a new Unicode word tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for UnicodeWordTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.unicode_words().collect()
    }

    fn tokenize_iter<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = &'a str> + 'a> {
        Box::new(text.unicode_words())
    }
}

/// Character-based tokenizer for CJK and similar languages.
///
/// Segments text into individual characters, filtering out whitespace
/// and punctuation.
#[derive(Clone, Debug, Default)]
pub struct CharacterTokenizer {
    /// Include punctuation in output.
    include_punctuation: bool,
}

impl CharacterTokenizer {
    /// Create a new character tokenizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include punctuation.
    pub fn include_punctuation(mut self, include: bool) -> Self {
        self.include_punctuation = include;
        self
    }
}

impl Tokenizer for CharacterTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.graphemes(true)
            .filter(|g| {
                let c = g.chars().next().unwrap_or(' ');
                !c.is_whitespace() && (self.include_punctuation || !c.is_ascii_punctuation())
            })
            .collect()
    }

    fn tokenize_iter<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = &'a str> + 'a> {
        let include_punct = self.include_punctuation;
        Box::new(text.graphemes(true).filter(move |g| {
            let c = g.chars().next().unwrap_or(' ');
            !c.is_whitespace() && (include_punct || !c.is_ascii_punctuation())
        }))
    }
}

/// Create a tokenizer appropriate for the given language.
///
/// Returns a boxed tokenizer implementing the `Tokenizer` trait.
///
/// # Language-specific behavior
///
/// - CJK languages (Chinese, Japanese, Korean): Character-based tokenization
/// - Thai, Khmer, Lao: Character-based tokenization (no word boundaries)
/// - Most other languages: Whitespace/Unicode word tokenization
pub fn create_tokenizer(lang: &LanguageTag) -> Box<dyn Tokenizer> {
    match lang.language() {
        // CJK languages: character-based
        "zh" | "ja" | "ko" => Box::new(CharacterTokenizer::new()),

        // Southeast Asian languages without spaces
        "th" | "km" | "lo" | "my" => Box::new(CharacterTokenizer::new()),

        // Default: Unicode word tokenizer
        _ => Box::new(UnicodeWordTokenizer::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("The quick brown fox");
        assert_eq!(tokens, vec!["The", "quick", "brown", "fox"]);
    }

    #[test]
    fn test_unicode_word_tokenizer() {
        let tokenizer = UnicodeWordTokenizer::new();
        let tokens = tokenizer.tokenize("Hello, world! How are you?");
        assert_eq!(tokens, vec!["Hello", "world", "How", "are", "you"]);
    }

    #[test]
    fn test_character_tokenizer() {
        let tokenizer = CharacterTokenizer::new();
        let tokens = tokenizer.tokenize("hello world");
        assert_eq!(
            tokens,
            vec!["h", "e", "l", "l", "o", "w", "o", "r", "l", "d"]
        );
    }

    #[test]
    fn test_create_tokenizer_english() {
        let lang = LanguageTag::new("en");
        let tokenizer = create_tokenizer(&lang);
        let tokens = tokenizer.tokenize("Hello world");
        assert_eq!(tokens, vec!["Hello", "world"]);
    }

    #[test]
    fn test_create_tokenizer_chinese() {
        let lang = LanguageTag::new("zh");
        let tokenizer = create_tokenizer(&lang);
        // For CJK, we use character tokenization
        let tokens = tokenizer.tokenize("hello");
        assert_eq!(tokens, vec!["h", "e", "l", "l", "o"]);
    }
}
