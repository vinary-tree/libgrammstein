//! CLI module for the grammstein command-line tool.
//!
//! Provides subcommands for training, evaluating, and querying language models.

pub mod args;
pub mod checkpoint;
pub mod commands;
pub mod error;
pub mod output;
pub mod progress;

#[cfg(feature = "google-books")]
pub mod tui;

pub use args::{Cli, Commands};
pub use checkpoint::CheckpointManager;
pub use error::{CliError, CliResult};
