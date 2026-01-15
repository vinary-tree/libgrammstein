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
//!
//! ### Domain-Specific Languages
//! - **Rholang** (`code-rholang`): Process algebra for blockchain
//! - **MeTTa** (`code-metta`): Knowledge representation and reasoning

// Mainstream languages
#[cfg(feature = "code-python")]
pub mod python;

#[cfg(feature = "code-rust")]
pub mod rust_lang;

#[cfg(feature = "code-javascript")]
pub mod javascript;

// Domain-specific languages
#[cfg(feature = "code-rholang")]
pub mod rholang;

#[cfg(feature = "code-metta")]
pub mod metta;

// Re-exports for mainstream languages
#[cfg(feature = "code-python")]
pub use python::Python;

#[cfg(feature = "code-rust")]
pub use rust_lang::Rust;

#[cfg(feature = "code-javascript")]
pub use javascript::JavaScript;

// Re-exports for domain-specific languages
#[cfg(feature = "code-rholang")]
pub use rholang::Rholang;

#[cfg(feature = "code-metta")]
pub use metta::MeTTa;
