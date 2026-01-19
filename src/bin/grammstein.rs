//! grammstein - CLI for training, evaluating, and experimenting with language models.
//!
//! This is the main entry point for the grammstein CLI tool, which provides
//! a unified interface for working with N-gram, embedding, and hybrid language models.

use clap::Parser;

use libgrammstein::cli::{commands, Cli};

fn main() {
    // Parse command-line arguments
    let cli = Cli::parse();

    // Check if we should skip env_logger initialization
    // (TUI commands will set up their own file-based logging)
    let skip_env_logger = should_skip_env_logger(&cli);

    if !skip_env_logger {
        // Initialize logging based on verbosity
        init_logging(cli.verbose, cli.quiet);
    }

    // Run the command
    if let Err(e) = commands::run(cli) {
        // Error is already formatted by CliError::Display
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

/// Check if we should skip env_logger for commands that use TUI.
///
/// The TUI sets up its own file-based logging via tracing_subscriber,
/// and Rust's log crate only allows one global logger. If env_logger
/// is initialized first, the TUI's LogTracer cannot take over, causing
/// log output to corrupt the TUI display.
fn should_skip_env_logger(cli: &Cli) -> bool {
    use libgrammstein::cli::Commands;

    // Import-google-books uses TUI unless --quiet or --no-progress is set
    #[cfg(feature = "google-books")]
    if let Commands::Train(ref train) = cli.command {
        use libgrammstein::cli::args::TrainCommands;
        if let TrainCommands::ImportGoogleBooks(ref args) = train {
            // TUI is active when not quiet and progress is enabled
            return !cli.quiet && !args.resources.no_progress;
        }
    }

    false
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
