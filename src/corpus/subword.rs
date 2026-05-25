//! BPE subword tokenization using the HuggingFace tokenizers library.
//!
//! This module provides BPE (Byte-Pair Encoding) subword tokenization
//! for handling out-of-vocabulary words and morphologically rich languages.
//!
//! # Features
//!
//! - Train BPE tokenizer on corpus
//! - Load pre-trained tokenizer vocabulary
//! - Encode text to subword tokens/IDs
//! - Decode subword IDs back to text
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::corpus::SubwordTokenizer;
//!
//! // Train a new tokenizer
//! let mut tokenizer = SubwordTokenizer::train_bpe(&["Hello world", "Hello there"], 1000)?;
//!
//! // Encode text
//! let tokens = tokenizer.encode("Hello world");
//! println!("Tokens: {:?}", tokens);
//!
//! // Decode back
//! let text = tokenizer.decode(&tokens);
//! println!("Decoded: {}", text);
//! ```

use std::path::Path;

use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::normalizers::Sequence as NormalizerSequence;
use tokenizers::normalizers::{Lowercase, StripAccents, NFD};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{AddedToken, Tokenizer};

/// Error type for subword tokenization operations.
#[derive(Debug)]
pub enum SubwordError {
    /// Tokenizer error from the underlying library.
    Tokenizer(String),
    /// I/O error.
    Io(std::io::Error),
    /// Training error.
    Training(String),
}

impl std::fmt::Display for SubwordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenizer(msg) => write!(f, "Tokenizer error: {}", msg),
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Training(msg) => write!(f, "Training error: {}", msg),
        }
    }
}

impl std::error::Error for SubwordError {}

impl From<std::io::Error> for SubwordError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type for subword tokenization operations.
pub type Result<T> = std::result::Result<T, SubwordError>;

/// Special tokens for BPE tokenization.
pub mod special_tokens {
    /// Unknown token for OOV words.
    pub const UNK: &str = "<unk>";
    /// Padding token.
    pub const PAD: &str = "<pad>";
    /// Beginning of sentence token.
    pub const BOS: &str = "<s>";
    /// End of sentence token.
    pub const EOS: &str = "</s>";
}

/// Configuration for BPE tokenizer training.
#[derive(Debug, Clone)]
pub struct BpeConfig {
    /// Vocabulary size (number of merge operations + base vocabulary).
    pub vocab_size: usize,
    /// Minimum frequency for a pair to be merged.
    pub min_frequency: u64,
    /// Whether to lowercase text before tokenization.
    pub lowercase: bool,
    /// Whether to strip accents.
    pub strip_accents: bool,
    /// Special tokens to include.
    pub special_tokens: Vec<String>,
    /// Whether to show progress during training.
    pub show_progress: bool,
}

impl Default for BpeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 2,
            lowercase: true,
            strip_accents: false,
            special_tokens: vec![
                special_tokens::UNK.to_string(),
                special_tokens::PAD.to_string(),
                special_tokens::BOS.to_string(),
                special_tokens::EOS.to_string(),
            ],
            show_progress: true,
        }
    }
}

/// BPE subword tokenizer.
pub struct SubwordTokenizer {
    tokenizer: Tokenizer,
}

impl SubwordTokenizer {
    /// Create a new tokenizer from a pre-trained tokenizer.
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Train a BPE tokenizer on the given texts.
    pub fn train_bpe<I, S>(texts: I, config: BpeConfig) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        use tokenizers::models::TrainerWrapper;

        // Create BPE trainer
        let trainer = BpeTrainer::builder()
            .vocab_size(config.vocab_size)
            .min_frequency(config.min_frequency)
            .show_progress(config.show_progress)
            .special_tokens(
                config
                    .special_tokens
                    .iter()
                    .map(|s| AddedToken::from(s.clone(), true))
                    .collect(),
            )
            .build();

        // Create tokenizer with BPE model
        let mut tokenizer = Tokenizer::new(BPE::default());

        // Set up normalizer
        let mut normalizers = Vec::new();
        normalizers.push(tokenizers::NormalizerWrapper::NFD(NFD));
        if config.strip_accents {
            normalizers.push(tokenizers::NormalizerWrapper::StripAccents(StripAccents));
        }
        if config.lowercase {
            normalizers.push(tokenizers::NormalizerWrapper::Lowercase(Lowercase));
        }
        if !normalizers.is_empty() {
            tokenizer.with_normalizer(Some(NormalizerSequence::new(normalizers)));
        }

        // Set up pre-tokenizer (whitespace splitting)
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        // Collect texts for training
        let text_refs: Vec<String> = texts.into_iter().map(|s| s.as_ref().to_string()).collect();

        if text_refs.is_empty() {
            return Err(SubwordError::Training(
                "No training texts provided".to_string(),
            ));
        }

        // Train the tokenizer using the trainer
        let mut wrapper_trainer = TrainerWrapper::BpeTrainer(trainer);
        tokenizer
            .train(&mut wrapper_trainer, text_refs.iter().map(|s| s.as_str()))
            .map_err(|e| SubwordError::Tokenizer(e.to_string()))?;

        Ok(Self { tokenizer })
    }

    /// Train a BPE tokenizer with default configuration.
    pub fn train_bpe_default<I, S>(texts: I, vocab_size: usize) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let config = BpeConfig {
            vocab_size,
            ..Default::default()
        };
        Self::train_bpe(texts, config)
    }

    /// Load a tokenizer from a JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|e| SubwordError::Tokenizer(e.to_string()))?;
        Ok(Self { tokenizer })
    }

    /// Save the tokenizer to a JSON file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.tokenizer
            .save(path.as_ref(), false)
            .map_err(|e| SubwordError::Tokenizer(e.to_string()))?;
        Ok(())
    }

    /// Encode text into subword token strings.
    pub fn encode(&self, text: &str) -> Vec<String> {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding.get_tokens().to_vec(),
            Err(_) => vec![],
        }
    }

    /// Encode text into subword token IDs.
    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(_) => vec![],
        }
    }

    /// Encode text with special tokens (BOS/EOS).
    pub fn encode_with_special(&self, text: &str) -> Vec<String> {
        match self.tokenizer.encode(text, true) {
            Ok(encoding) => encoding.get_tokens().to_vec(),
            Err(_) => vec![],
        }
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self.tokenizer.decode(ids, true) {
            Ok(text) => text,
            Err(_) => String::new(),
        }
    }

    /// Decode tokens back to text.
    pub fn decode_tokens(&self, tokens: &[String]) -> String {
        // Convert tokens to IDs first
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|t| self.tokenizer.token_to_id(t))
            .collect();
        self.decode(&ids)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get the ID for a token.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Get the token for an ID.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Check if a token is in the vocabulary.
    pub fn contains(&self, token: &str) -> bool {
        self.tokenizer.token_to_id(token).is_some()
    }

    /// Get the underlying tokenizer (for advanced usage).
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get a mutable reference to the underlying tokenizer.
    pub fn inner_mut(&mut self) -> &mut Tokenizer {
        &mut self.tokenizer
    }
}

/// Iterator adapter for tokenizing a stream of sentences.
pub struct TokenizedSentences<'a, I> {
    sentences: I,
    tokenizer: &'a SubwordTokenizer,
}

impl<'a, I> TokenizedSentences<'a, I> {
    /// Create a new tokenized sentences iterator.
    pub fn new(sentences: I, tokenizer: &'a SubwordTokenizer) -> Self {
        Self {
            sentences,
            tokenizer,
        }
    }
}

impl<'a, I, S> Iterator for TokenizedSentences<'a, I>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sentences
            .next()
            .map(|s| self.tokenizer.encode(s.as_ref()))
    }
}

/// Extension trait for iterators to add tokenization.
pub trait TokenizeExt<'a>: Sized {
    /// Tokenize each sentence in the iterator.
    fn tokenize(self, tokenizer: &'a SubwordTokenizer) -> TokenizedSentences<'a, Self>;
}

impl<'a, I, S> TokenizeExt<'a> for I
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    fn tokenize(self, tokenizer: &'a SubwordTokenizer) -> TokenizedSentences<'a, Self> {
        TokenizedSentences::new(self, tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_texts() -> Vec<&'static str> {
        vec![
            "The quick brown fox jumps over the lazy dog.",
            "Hello world! How are you today?",
            "Machine learning is transforming technology.",
            "Natural language processing enables computers to understand text.",
            "The fox jumped over the fence.",
            "Hello there, how is everything going?",
            "Deep learning models learn representations.",
            "Text processing is fundamental to NLP.",
        ]
    }

    #[test]
    fn test_train_bpe() {
        let config = BpeConfig {
            vocab_size: 100,
            min_frequency: 1,
            show_progress: false,
            ..Default::default()
        };

        let tokenizer = SubwordTokenizer::train_bpe(sample_texts(), config);
        assert!(tokenizer.is_ok());

        let tokenizer = tokenizer.unwrap();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_encode_decode() {
        let config = BpeConfig {
            vocab_size: 100,
            min_frequency: 1,
            show_progress: false,
            ..Default::default()
        };

        let tokenizer = SubwordTokenizer::train_bpe(sample_texts(), config).unwrap();

        let text = "hello world";
        let tokens = tokenizer.encode(text);
        assert!(!tokens.is_empty());

        let ids = tokenizer.encode_ids(text);
        assert!(!ids.is_empty());
        assert_eq!(tokens.len(), ids.len());

        // Decode should produce something close to original
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_vocab_operations() {
        let config = BpeConfig {
            vocab_size: 100,
            min_frequency: 1,
            show_progress: false,
            ..Default::default()
        };

        let tokenizer = SubwordTokenizer::train_bpe(sample_texts(), config).unwrap();

        // Special tokens should be in vocabulary
        assert!(tokenizer.contains(special_tokens::UNK));
        assert!(tokenizer.contains(special_tokens::PAD));

        // Token to ID and back
        if let Some(unk_id) = tokenizer.token_to_id(special_tokens::UNK) {
            assert_eq!(
                tokenizer.id_to_token(unk_id),
                Some(special_tokens::UNK.to_string())
            );
        }
    }

    #[test]
    fn test_tokenize_iterator() {
        let config = BpeConfig {
            vocab_size: 100,
            min_frequency: 1,
            show_progress: false,
            ..Default::default()
        };

        let tokenizer = SubwordTokenizer::train_bpe(sample_texts(), config).unwrap();

        let sentences = vec!["hello world", "test sentence"];
        let tokenized: Vec<Vec<String>> = sentences.iter().tokenize(&tokenizer).collect();

        assert_eq!(tokenized.len(), 2);
        assert!(!tokenized[0].is_empty());
        assert!(!tokenized[1].is_empty());
    }

    #[test]
    fn test_empty_training() {
        let config = BpeConfig {
            vocab_size: 100,
            min_frequency: 1,
            show_progress: false,
            ..Default::default()
        };

        let result = SubwordTokenizer::train_bpe(Vec::<&str>::new(), config);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_load() {
        let config = BpeConfig {
            vocab_size: 100,
            min_frequency: 1,
            show_progress: false,
            ..Default::default()
        };

        let tokenizer = SubwordTokenizer::train_bpe(sample_texts(), config).unwrap();

        // Save to temp file
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("tokenizer.json");

        tokenizer.save(&path).unwrap();
        assert!(path.exists());

        // Load and verify
        let loaded = SubwordTokenizer::load(&path).unwrap();
        assert_eq!(loaded.vocab_size(), tokenizer.vocab_size());

        // Encoding should produce same results
        let text = "hello world";
        assert_eq!(loaded.encode(text), tokenizer.encode(text));
    }
}
