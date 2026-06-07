//! `train import-google-books` command: large-scale Google Books n-gram import.

#![cfg(feature = "google-books")]

use crate::cli::args::ImportGoogleBooksArgs;
use crate::cli::error::{CliError, CliResult};

pub(super) fn import_google_books(
    args: ImportGoogleBooksArgs,
    verbose: bool,
    quiet: bool,
) -> CliResult<()> {
    use crate::cli::args::ShardingModeArg;
    use crate::sources::google_books::{
        GoogleBooksConfig, GoogleBooksImporter, LanguageInfo, ShardingMode, ShardingOptions,
    };

    if verbose {
        eprintln!("Importing Google Books N-grams");
        eprintln!("  Language: {}", args.language);
        eprintln!("  Orders: {}-{}", args.min_order, args.max_order);
        eprintln!("  Output: {}", args.output.display());
        eprintln!("  Min count: {}", args.min_count);
        if let Some(min_year) = args.min_year {
            eprintln!("  Min year: {}", min_year);
        }
        if let Some(max_year) = args.max_year {
            eprintln!("  Max year: {}", max_year);
        }
        if args.cache_files {
            eprintln!("  Cache files: enabled");
        }
    }

    // Validate language
    let lang_info = LanguageInfo::from_code(&args.language).ok_or_else(|| {
        CliError::unsupported(format!(
            "Unsupported language: {}. Available: en, de, fr, es, it, ru, he, zh",
            args.language
        ))
    })?;

    if !quiet {
        eprintln!(
            "Importing {} n-grams ({}-grams to {}-grams)",
            lang_info.name, args.min_order, args.max_order
        );
    }

    // Validate prefix if specified
    if let Some(ref prefix) = args.prefix {
        use crate::sources::google_books::is_valid_prefix;

        // Check that the prefix is valid for at least one order in the range
        let valid_for_any_order =
            (args.min_order..=args.max_order).any(|order| is_valid_prefix(order, prefix));

        if !valid_for_any_order {
            return Err(CliError::unsupported(format!(
                "Prefix '{}' is not valid for n-gram orders {}-{}. \
                 Valid prefixes for 1-grams: a-z, other. \
                 Valid prefixes for 2-5 grams: aa-zz, other, punctuation.",
                prefix, args.min_order, args.max_order
            )));
        }

        if !quiet {
            eprintln!("  Prefix filter: {}", prefix);
        }
    }

    // Build configuration
    let year_range = match (args.min_year, args.max_year) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, 2020)),
        (None, Some(max)) => Some((1800, max)),
        (None, None) => None,
    };

    let config = GoogleBooksConfig {
        language: args.language.clone(),
        orders: args.min_order..=args.max_order,
        min_count: args.min_count,
        year_range,
        output_path: args.output.clone(),
        vocabulary_path: None, // Use default derived from output_path
        buffer_pool_size: 256, // 64MB default
        parallel_downloads: args.parallel,
        progress_interval: 100_000,
        skip_pos_tags: args.skip_pos_tags,
        sharding: match args.sharding {
            ShardingModeArg::Enabled => ShardingMode::Enabled(ShardingOptions::default()),
            ShardingModeArg::Disabled => ShardingMode::Disabled,
        },
        tx_chunk_size: args.tx_chunk_size,
        prefix: args.prefix.clone(),
        cache_files: args.cache_files,
    };

    // Create importer (resume from checkpoint if one exists)
    let mut importer = GoogleBooksImporter::resume_or_start(config)
        .map_err(|e| CliError::io(format!("Failed to create importer: {}", e)))?;

    // Apply user-specified lock-free flush threshold if provided
    if let Some(threshold) = args.lockfree_flush_threshold {
        importer.set_lockfree_flush_threshold(threshold);
    }

    // Check for local files vs HTTP
    if let Some(ref local_dir) = args.local_files {
        // Setup progress indicator for local file imports (simple spinner)
        // NOTE: Only created here to avoid interfering with ratatui TUI in HTTP path
        let local_progress = if quiet || args.resources.no_progress {
            None
        } else {
            let pb = indicatif::ProgressBar::new_spinner();
            pb.set_style(
                indicatif::ProgressStyle::default_spinner()
                    .template("{spinner:.green} [{elapsed_precise}] {msg}")
                    .expect("Invalid progress template"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message(format!(
                "Importing from local files: {}",
                local_dir.display()
            ));
            Some(pb)
        };

        // Verify the directory exists and has gzip files
        let file_count = std::fs::read_dir(local_dir)
            .map_err(|e| CliError::io(format!("Failed to read directory: {}", e)))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "gz"))
            .count();

        if file_count == 0 {
            return Err(CliError::corpus(format!(
                "No .gz files found in {}",
                local_dir.display()
            )));
        }

        if verbose {
            eprintln!("Found {} gzip files", file_count);
        }

        // Import files from directory
        let stats = importer
            .import_files(local_dir, |progress_info| {
                if let Some(ref pb) = local_progress {
                    pb.set_message(format!(
                        "Order {}: {} n-grams ({}/{} files, prefix: {})",
                        progress_info.current_order,
                        progress_info.total_ngrams,
                        progress_info.files_completed,
                        progress_info.total_files,
                        progress_info.current_prefix
                    ));
                }
            })
            .map_err(|e| CliError::io(format!("Import failed: {}", e)))?;

        if let Some(ref pb) = local_progress {
            pb.finish_and_clear();
        }

        if !quiet {
            print_import_stats(&stats, &args.output);
        }
    } else {
        // Import from HTTP using reactive TUI architecture
        use crate::sources::google_books::{ImportCommand, ImportEvent};

        let show_tui = !quiet && !args.resources.no_progress;

        // Run async import
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(args.parallel)
            .enable_all()
            .build()
            .map_err(|e| CliError::io(format!("Failed to create async runtime: {}", e)))?;

        let stats = if show_tui {
            // Use ratatui TUI for rich interactive display
            use crate::cli::tui::ImportTui;
            use tracing_appender::rolling::{RollingFileAppender, Rotation};
            use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

            // Create event/command channels
            let (event_tx, _) = tokio::sync::broadcast::channel::<ImportEvent>(1024);
            let (command_tx, command_rx) = tokio::sync::mpsc::channel::<ImportCommand>(16);

            // Subscribe to events for TUI
            let tui_event_rx = event_tx.subscribe();

            // Subscribe to events for file-based logging (separate from TUI)
            let mut log_rx = event_tx.subscribe();
            let log_task = runtime.spawn(async move {
                while let Ok(event) = log_rx.recv().await {
                    match event {
                        ImportEvent::OrderStarted { order, total_files } => {
                            log::info!("[EVENT] Started order {} ({} files)", order, total_files);
                        }
                        ImportEvent::OrderProgress { order, files_completed, total_files, ngrams_processed, .. } => {
                            log::debug!(
                                "[EVENT] Order {} progress: {}/{} files, {} n-grams",
                                order, files_completed, total_files, ngrams_processed
                            );
                        }
                        ImportEvent::OrderCompleted { order, ngram_count, duration } => {
                            log::info!(
                                "[EVENT] Completed order {}: {} n-grams in {:.1}s",
                                order, ngram_count, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::ImportCompleted { total_ngrams, duration } => {
                            log::info!(
                                "[EVENT] Import completed: {} n-grams in {:.1}s",
                                total_ngrams, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::MergeStarted { shard_count, .. } => {
                            log::info!("[EVENT] Starting merge of {} shards...", shard_count);
                        }
                        ImportEvent::MergeCompleted { total_ngrams, bytes_written, duration } => {
                            log::info!(
                                "[EVENT] Merge completed: {} n-grams ({} bytes) in {:.1}s",
                                total_ngrams, bytes_written, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::MknStarted { source, estimated_ngrams } => {
                            log::info!(
                                "[EVENT] Computing MKN statistics from {} (~{} n-grams)...",
                                source, estimated_ngrams
                            );
                        }
                        ImportEvent::MknCompleted { continuation_entries, frequency_entries, duration } => {
                            log::info!(
                                "[EVENT] MKN statistics complete: {} continuation, {} frequency entries in {:.1}s",
                                continuation_entries, frequency_entries, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::MknFailed { error } => {
                            log::error!("[EVENT] MKN computation failed: {}", error);
                        }
                        ImportEvent::AllWorkCompleted { total_ngrams, total_duration, .. } => {
                            log::info!(
                                "[EVENT] All work completed: {} n-grams in {:.1}s",
                                total_ngrams, total_duration.as_secs_f64()
                            );
                        }
                        ImportEvent::WorkerRetrying { prefix, attempt, error, .. } => {
                            log::warn!("[EVENT] Retry {}: {} - {}", attempt, prefix, error);
                        }
                        _ => {}
                    }
                }
            });

            // Clone event_tx for the import task
            let event_tx_clone = event_tx.clone();

            // Capture keep_shards flag
            let keep_shards = args.keep_shards;

            // Run import in background
            let import_handle = runtime.spawn(async move {
                importer
                    .import_http_reactive(event_tx_clone, command_rx, keep_shards)
                    .await
            });

            // Set up file-based tracing for debugging (works even when TUI is active)
            let log_dir = args
                .output
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| std::env::temp_dir().join("grammstein-logs"));
            std::fs::create_dir_all(&log_dir).ok();

            let file_appender =
                RollingFileAppender::new(Rotation::NEVER, &log_dir, "import-debug.log");
            let (non_blocking, tracing_guard) = tracing_appender::non_blocking(file_appender);

            // Use try_init() to avoid panic if already initialized
            let _ = tracing_subscriber::registry()
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_writer(non_blocking)
                        .with_ansi(false)
                        .with_target(true),
                )
                .with(
                    tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                        "libgrammstein::sources::google_books=debug,libgrammstein::cli::tui=debug"
                            .parse()
                            .expect("valid filter")
                    }),
                )
                .try_init();

            // Note: tracing-subscriber with 'fmt' feature automatically sets up
            // a LogTracer to bridge log:: calls to tracing. No explicit init needed.

            // Install panic hook to capture panics to the log file.
            // When TUI is active, panics write directly to stderr which can corrupt
            // terminal output and cause panic information to be lost.
            let default_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |panic_info| {
                // Log the panic to tracing (which goes to the file)
                let location = panic_info
                    .location()
                    .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                    .unwrap_or_else(|| "unknown".to_string());

                let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic payload".to_string()
                };

                // Use error! to ensure it goes to the log file
                tracing::error!(
                    target: "panic",
                    location = %location,
                    message = %message,
                    backtrace = %std::backtrace::Backtrace::force_capture(),
                    "PANIC occurred"
                );

                // Call the default panic hook to write to stderr as normal
                default_hook(panic_info);
            }));

            // Print log location to stderr before TUI takes over the terminal
            let log_path = log_dir.join("import-debug.log");
            eprintln!("Debug logs will be written to: {}", log_path.display());
            log::info!("Debug logs being written to: {}", log_path.display());

            // Note: We no longer suppress log level during TUI because:
            // 1. tracing_subscriber writes to file, not stderr (no TUI corruption)
            // 2. Suppressing log level prevents log:: calls from reaching the tracing bridge

            // Create and run TUI (blocks until import completes or user quits)
            let tui = ImportTui::new(
                command_tx,
                &args.language,
                args.min_order,
                args.max_order,
                args.parallel,
            );

            let tui_result = tui.run(tui_event_rx);

            // Abort the logging task before dropping tracing
            log_task.abort();

            // Flush and drop file-based tracing
            drop(tracing_guard);

            // Check TUI result - if user cancelled (returned Ok(false)), abort the import
            let user_cancelled = match &tui_result {
                Ok(completed) => !completed, // false means user quit before completion
                Err(_) => false,             // Error in TUI, not user cancellation
            };

            if user_cancelled {
                // User pressed Q - wait for import to finish gracefully
                // (The import task will call save_checkpoint() when it sees cancelled=true)
                log::info!("User cancelled, waiting for checkpoint to complete...");

                // Wait for import to finish normally (with generous timeout)
                let import_result = runtime.block_on(async {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(60), // 60 seconds for checkpoint
                        import_handle,
                    )
                    .await
                });

                match import_result {
                    Ok(Ok(Ok(_))) => log::info!("Import cancelled and checkpoint saved"),
                    Ok(Ok(Err(e))) => {
                        // Import error (e.g., Interrupted) - checkpoint was saved
                        log::info!("Import stopped: {} (checkpoint saved)", e);
                    }
                    Ok(Err(e)) => log::error!("Import task panicked: {}", e),
                    Err(_) => {
                        log::warn!("Checkpoint timeout - import task may not have saved progress");
                        // Note: import_handle is consumed by the timeout, cannot abort
                    }
                }

                return Ok(());
            }

            // Handle TUI error
            if let Err(e) = tui_result {
                return Err(CliError::io(format!("TUI error: {}", e)));
            }

            // Wait for import to finish (normal completion path)
            let import_result = runtime.block_on(async { import_handle.await });

            // Handle import result
            let import_result =
                import_result.map_err(|e| CliError::io(format!("Import task failed: {}", e)))?;
            import_result.map_err(|e| CliError::io(format!("HTTP import failed: {}", e)))?
        } else {
            // Quiet mode: use simple logging without TUI
            let (event_tx, _) = tokio::sync::broadcast::channel::<ImportEvent>(1024);
            let (_command_tx, command_rx) = tokio::sync::mpsc::channel::<ImportCommand>(16);

            // Clone event_tx for the import task
            let event_tx_clone = event_tx.clone();

            // Capture keep_shards flag
            let keep_shards = args.keep_shards;

            // Simple logging subscriber
            let mut log_rx = event_tx.subscribe();
            let log_task = runtime.spawn(async move {
                while let Ok(event) = log_rx.recv().await {
                    match event {
                        ImportEvent::OrderStarted { order, total_files } => {
                            log::info!("Started order {} ({} files)", order, total_files);
                        }
                        ImportEvent::OrderCompleted { order, ngram_count, duration } => {
                            log::info!(
                                "Completed order {}: {} n-grams in {:.1}s",
                                order, ngram_count, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::ImportCompleted { total_ngrams, duration } => {
                            log::info!(
                                "Import completed: {} n-grams in {:.1}s",
                                total_ngrams, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::MergeStarted { shard_count, .. } => {
                            log::info!("Starting merge of {} shards...", shard_count);
                        }
                        ImportEvent::MergeCompleted { total_ngrams, bytes_written, duration } => {
                            log::info!(
                                "Merge completed: {} n-grams ({} bytes) in {:.1}s",
                                total_ngrams, bytes_written, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::MknStarted { source, estimated_ngrams } => {
                            log::info!(
                                "Computing MKN statistics from {} (~{} n-grams)...",
                                source, estimated_ngrams
                            );
                        }
                        ImportEvent::MknCompleted { continuation_entries, frequency_entries, duration } => {
                            log::info!(
                                "MKN statistics complete: {} continuation, {} frequency entries in {:.1}s",
                                continuation_entries, frequency_entries, duration.as_secs_f64()
                            );
                        }
                        ImportEvent::MknFailed { error } => {
                            log::error!("MKN computation failed: {}", error);
                        }
                        ImportEvent::AllWorkCompleted { total_ngrams, total_duration, .. } => {
                            log::info!(
                                "All work completed: {} n-grams in {:.1}s",
                                total_ngrams, total_duration.as_secs_f64()
                            );
                        }
                        ImportEvent::WorkerRetrying { prefix, attempt, error, .. } => {
                            log::warn!("Retry {}: {} - {}", attempt, prefix, error);
                        }
                        _ => {}
                    }
                }
            });

            let result = runtime
                .block_on(async {
                    importer
                        .import_http_reactive(event_tx_clone, command_rx, keep_shards)
                        .await
                })
                .map_err(|e| CliError::io(format!("HTTP import failed: {}", e)))?;

            log_task.abort();
            result
        };

        if !quiet {
            print_import_stats(&stats, &args.output);
        }
    }

    Ok(())
}

/// Print import statistics.
#[cfg(feature = "google-books")]
fn print_import_stats(stats: &crate::sources::google_books::ImportStats, output: &std::path::Path) {
    use console::style;

    println!();
    println!("{} Google Books import complete", style("✓").green().bold());
    println!();
    println!("  Total n-grams:  {}", stats.total_ngrams);
    println!("  Unique n-grams: {}", stats.unique_ngrams);
    println!("  Files processed: {}", stats.files_processed);
    println!("  Duration: {:.2}s", stats.elapsed_seconds);
    println!();
    println!("  Output: {}", output.display());
}
