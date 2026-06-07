//! Language support module.
//!
//! This module provides language-related types and utilities for multi-language
//! model training and organization.

mod detection;
mod registry;
mod tag;
mod tokenizer;

pub use detection::{detect_from_sentences, detect_language, LanguageDetectionError};
pub use registry::{ModelEntry, ModelRegistry};
pub use tag::{wikipedia_dump_url, LanguageTag, LanguageTagError, WIKIPEDIA_URLS};
pub use tokenizer::{create_tokenizer, CharacterTokenizer, Tokenizer, WhitespaceTokenizer};
