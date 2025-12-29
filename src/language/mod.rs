//! Language support module.
//!
//! This module provides language-related types and utilities for multi-language
//! model training and organization.

mod detection;
mod registry;
mod tag;
mod tokenizer;

pub use detection::{detect_language, LanguageDetectionError};
pub use registry::{ModelEntry, ModelRegistry};
pub use tag::{LanguageTag, LanguageTagError};
pub use tokenizer::{create_tokenizer, Tokenizer, WhitespaceTokenizer};
