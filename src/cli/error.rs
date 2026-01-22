//! CLI-specific error types and formatting.

use std::path::PathBuf;

use console::style;
use thiserror::Error;

/// CLI-specific error type.
#[derive(Error, Debug)]
pub enum CliError {
    /// Library error from libgrammstein.
    #[error("{0}")]
    Library(#[from] crate::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File not found.
    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },

    /// Invalid argument.
    #[error("Invalid argument: {message}")]
    InvalidArgument { message: String },

    /// Model loading error.
    #[error("Failed to load model from {path}: {reason}")]
    ModelLoad { path: PathBuf, reason: String },

    /// Corpus error.
    #[error("Corpus error: {message}")]
    Corpus { message: String },

    /// Checkpoint error.
    #[error("Checkpoint error: {message}")]
    Checkpoint { message: String },

    /// Training interrupted.
    #[error("Training interrupted")]
    Interrupted,

    /// Training error.
    #[error("Training error: {message}")]
    Training { message: String },

    /// Generic I/O error with context.
    #[error("I/O error: {message}")]
    IoError { message: String },

    /// REPL error.
    #[error("REPL error: {message}")]
    Repl { message: String },

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Language detection error.
    #[error("Language detection error: {message}")]
    LanguageDetection { message: String },

    /// Unsupported operation.
    #[error("Unsupported: {message}")]
    Unsupported { message: String },
}

/// Result type for CLI operations.
pub type CliResult<T> = Result<T, CliError>;

impl CliError {
    /// Create a file not found error.
    pub fn file_not_found(path: impl Into<PathBuf>) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    /// Create an invalid argument error.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            message: message.into(),
        }
    }

    /// Create a model load error.
    pub fn model_load(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::ModelLoad {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Create a corpus error.
    pub fn corpus(message: impl Into<String>) -> Self {
        Self::Corpus {
            message: message.into(),
        }
    }

    /// Create a checkpoint error.
    pub fn checkpoint(message: impl Into<String>) -> Self {
        Self::Checkpoint {
            message: message.into(),
        }
    }

    /// Create a REPL error.
    pub fn repl(message: impl Into<String>) -> Self {
        Self::Repl {
            message: message.into(),
        }
    }

    /// Create an unsupported error.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    /// Create a training error.
    pub fn training(message: impl Into<String>) -> Self {
        Self::Training {
            message: message.into(),
        }
    }

    /// Create an I/O error with context.
    pub fn io(message: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }

    /// Print error with formatting for terminal.
    pub fn print_error(&self) {
        eprintln!("{}: {}", style("error").red().bold(), self);
    }
}

/// Print a warning message.
pub fn print_warning(message: &str) {
    eprintln!("{}: {}", style("warning").yellow().bold(), message);
}

/// Print an info message.
pub fn print_info(message: &str) {
    eprintln!("{}: {}", style("info").blue().bold(), message);
}

/// Print a success message.
pub fn print_success(message: &str) {
    eprintln!("{}: {}", style("success").green().bold(), message);
}
