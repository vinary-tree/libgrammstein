//! grammstein - CLI for training, evaluating, and experimenting with language models.
//!
//! This is the main entry point for the grammstein CLI tool, which provides
//! a unified interface for working with N-gram, embedding, and hybrid language models.

use clap::Parser;

use libgrammstein::cli::{commands, Cli};

fn main() {
    // Parse command-line arguments
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    init_logging(cli.verbose, cli.quiet);

    // Run the command
    if let Err(e) = commands::run(cli) {
        // Error is already formatted by CliError::Display
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

/// Initialize logging with appropriate level based on flags.
fn init_logging(verbose: bool, quiet: bool) {
    let level = if quiet {
        log::LevelFilter::Error
    } else if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::new()
        .filter_level(level)
        .format_timestamp(None)
        .format_module_path(false)
        .init();
}
