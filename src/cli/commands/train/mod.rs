//! Training command implementations.
//!
//! The four `train` subcommands (`ngram`, `embedding`, `hybrid`,
//! `import-google-books`) each live in their own sibling module so this file
//! stays a thin dispatcher. The shared corpus-reader factory used by
//! multiple subcommands lives in `corpus_reader`.

use crate::cli::args::TrainCommands;
use crate::cli::error::CliResult;

mod corpus_reader;
mod embedding;
mod hybrid;
mod ngram;

#[cfg(feature = "google-books")]
mod google_books;

// create_corpus_reader is pub(super) in corpus_reader.rs; siblings access via `super::corpus_reader::create_corpus_reader`.

/// Run the train command.
pub fn run(cmd: TrainCommands, verbose: bool, quiet: bool) -> CliResult<()> {
    match cmd {
        TrainCommands::Ngram(args) => ngram::train_ngram(args, verbose, quiet),
        TrainCommands::Embedding(args) => embedding::train_embedding(args, verbose, quiet),
        TrainCommands::Hybrid(args) => hybrid::train_hybrid(args, verbose, quiet),
        #[cfg(feature = "google-books")]
        TrainCommands::ImportGoogleBooks(args) => {
            google_books::import_google_books(args, verbose, quiet)
        }
    }
}
