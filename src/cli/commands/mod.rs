//! CLI command implementations.

pub mod corpus;
pub mod convert;
pub mod eval;
pub mod models;
pub mod query;
pub mod repl;
pub mod train;

use crate::cli::args::{Commands, Cli};
use crate::cli::error::CliResult;

/// Dispatch and run the appropriate command.
pub fn run(cli: Cli) -> CliResult<()> {
    match cli.command {
        Commands::Train(cmd) => train::run(cmd, cli.verbose, cli.quiet),
        Commands::Eval(cmd) => eval::run(cmd, cli.verbose, cli.quiet),
        Commands::Query(cmd) => query::run(cmd, cli.verbose),
        Commands::Models(cmd) => models::run(cmd, cli.verbose),
        Commands::Corpus(cmd) => corpus::run(cmd, cli.verbose),
        Commands::Convert(cmd) => convert::run(cmd, cli.verbose),
        Commands::Repl(args) => repl::run(args),
    }
}
