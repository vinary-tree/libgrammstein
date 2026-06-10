//! Language-specific implementations.
//!
//! This module provides implementations of the [`CodeLanguage`] trait
//! for various programming languages.
//!
//! ## Supported Languages
//!
//! ### Mainstream Languages (feature: `code-mainstream`)
//! - **Python** (`code-python`): Full support with type hints
//! - **Rust** (`code-rust`): Full support with macro awareness
//! - **JavaScript** (`code-javascript`): ES6+ support

// Mainstream languages
#[cfg(feature = "code-python")]
pub mod python;

#[cfg(feature = "code-rust")]
pub mod rust_lang;

#[cfg(feature = "code-javascript")]
pub mod javascript;

// Re-exports for mainstream languages
#[cfg(feature = "code-python")]
pub use python::Python;

#[cfg(feature = "code-rust")]
pub use rust_lang::Rust;

#[cfg(feature = "code-javascript")]
pub use javascript::JavaScript;
