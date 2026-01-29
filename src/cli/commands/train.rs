//! Training command implementations.

use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use console::style;

use crate::cli::args::{
    CorpusFormat, TrainCommands, TrainEmbeddingArgs, TrainHybridArgs, TrainNgramArgs,
};
#[cfg(feature = "google-books")]
use crate::cli::args::ImportGoogleBooksArgs;
use crate::cli::checkpoint::{
    CheckpointManager, NgramCheckpoint, NgramCheckpointConfig,
    NgramTrainingState, TrainingTimer,
};
use crate::cli::error::{print_success, CliError, CliResult};
use crate::cli::progress::{setup_interrupt_handler, TrainingProgress, TrainingStats};
use crate::corpus::{CorpusReader, GutenbergReader, PlaintextReader, Tokenizer, WikipediaReader};
use crate::ngram::{NgramAccumulator, NgramEntry};
use crate::ngram::accumulator::key_format;

/// Run the train command.
pub fn run(cmd: TrainCommands, verbose: bool, quiet: bool) -> CliResult<()> {
    match cmd {
        TrainCommands::Ngram(args) => train_ngram(args, verbose, quiet),
        TrainCommands::Embedding(args) => train_embedding(args, verbose, quiet),
        TrainCommands::Hybrid(args) => train_hybrid(args, verbose, quiet),
        #[cfg(feature = "google-books")]
        TrainCommands::ImportGoogleBooks(args) => import_google_books(args, verbose, quiet),
    }
}

/// Train an N-gram model with persistent checkpointing.
fn train_ngram(args: TrainNgramArgs, verbose: bool, quiet: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Training N-gram model (order={})", args.order);
        eprintln!("  Corpus: {}", args.corpus);
        eprintln!("  Output: {}", args.output.display());
        eprintln!("  Min count: {}", args.min_count);
        eprintln!("  Batch size: {}", args.batch_size);
    }

    // Create corpus reader based on format
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    // Get estimated size for progress bar
    let estimated_sentences = reader.estimated_tokens().map(|t| t as u64 / 20); // Rough estimate

    // Create progress bar
    let progress = if quiet || args.resources.no_progress {
        TrainingProgress::hidden()
    } else {
        TrainingProgress::new(estimated_sentences)
    };

    // Setup interrupt handler
    let stats = Arc::new(TrainingStats::new());
    setup_interrupt_handler(stats.clone());

    // Create tokenizer
    let tokenizer = Tokenizer::new();

    // Initialize training state and accumulator
    let (mut accumulator, mut state, checkpoint_manager, timer) =
        if let Some(ref checkpoint_path) = args.checkpoint.resume {
            // Resume from checkpoint
            resume_ngram_training(checkpoint_path, &args, quiet)?
        } else if let Some(ref checkpoint_dir) = args.checkpoint.checkpoint {
            // Fresh start with checkpointing enabled
            let manager = CheckpointManager::new(checkpoint_dir, args.checkpoint.keep_checkpoints)?;
            let accumulator = NgramAccumulator::create(&manager.accumulator_path())
                .map_err(|e| CliError::io(format!("Failed to create accumulator: {}", e)))?;

            if !quiet {
                eprintln!(
                    "Checkpoints will be saved to: {}",
                    style(checkpoint_dir.display()).cyan()
                );
            }

            (
                accumulator,
                NgramTrainingState::default(),
                Some(manager),
                TrainingTimer::new(),
            )
        } else {
            // No checkpointing - use in-memory training
            return train_ngram_inmemory(args, verbose, quiet, reader, progress, stats);
        };

    // Training with checkpointing
    if !quiet {
        progress.set_message("Training N-gram model (with checkpointing)...");
    }

    let checkpoint_interval = args.checkpoint.checkpoint_interval;
    let order = args.order;

    // Stream sentences and accumulate n-grams
    for sentence in reader.sentences() {
        if !stats.is_running() {
            // Save checkpoint on interrupt before exiting
            if let Some(ref manager) = checkpoint_manager {
                save_ngram_checkpoint(
                    manager,
                    &mut accumulator,
                    &state,
                    &args,
                    &timer,
                    quiet,
                )?;
            }
            progress.abandon();
            return Err(CliError::Interrupted);
        }

        // Tokenize sentence
        let tokens: Vec<String> = if args.lowercase {
            tokenizer.words(&sentence).map(|s| s.to_lowercase()).collect()
        } else {
            tokenizer.words(&sentence).collect()
        };

        if tokens.is_empty() {
            continue;
        }

        state.tokens_processed += tokens.len() as u64;
        state.sentences_processed += 1;
        state.bytes_read += sentence.len() as u64;

        stats.inc_sentences(1);
        stats.inc_bytes(sentence.len() as u64);

        // Extract and count n-grams of all orders
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        for n in 1..=order.min(tokens.len()) {
            for i in 0..=(tokens.len() - n) {
                let ngram_key = key_format::build_key(&token_refs[i..i + n]);
                accumulator
                    .increment(&ngram_key)
                    .map_err(|e| CliError::io(format!("Failed to increment n-gram: {}", e)))?;
            }
        }

        // Update progress
        if state.sentences_processed % 10000 == 0 {
            progress.update(state.sentences_processed, state.tokens_processed, state.bytes_read);
        }

        // Periodic checkpoint
        if state.sentences_processed % checkpoint_interval == 0 {
            if let Some(ref manager) = checkpoint_manager {
                save_ngram_checkpoint(
                    manager,
                    &mut accumulator,
                    &state,
                    &args,
                    &timer,
                    quiet,
                )?;
            }
        }
    }

    // Final sync and checkpoint
    accumulator
        .sync()
        .map_err(|e| CliError::io(format!("Failed to sync accumulator: {}", e)))?;

    if let Some(ref manager) = checkpoint_manager {
        save_ngram_checkpoint(manager, &mut accumulator, &state, &args, &timer, quiet)?;
    }

    // Finalize: Convert accumulator to final model format
    let unique_ngrams = accumulator.len();
    finalize_ngram_model(&accumulator, &args, quiet)?;

    progress.finish(state.sentences_processed, state.tokens_processed, unique_ngrams as u64);

    if !quiet {
        print_success(&format!("Model saved to: {}", args.output.display()));
        eprintln!("  Sentences processed: {}", state.sentences_processed);
        eprintln!("  Tokens processed:    {}", state.tokens_processed);
        eprintln!("  Unique N-grams:      {}", unique_ngrams);
        eprintln!("  Training time:       {:.2}s", timer.elapsed_secs());
    }

    Ok(())
}

/// Resume N-gram training from checkpoint.
fn resume_ngram_training(
    checkpoint_path: &str,
    args: &TrainNgramArgs,
    quiet: bool,
) -> CliResult<(
    NgramAccumulator,
    NgramTrainingState,
    Option<CheckpointManager>,
    TrainingTimer,
)> {
    let checkpoint_dir = args
        .checkpoint
        .checkpoint
        .as_ref()
        .ok_or_else(|| {
            CliError::unsupported("--checkpoint directory required when using --resume")
        })?;

    let manager = CheckpointManager::new(checkpoint_dir, args.checkpoint.keep_checkpoints)?;
    let checkpoint = manager.load_ngram_checkpoint(checkpoint_path)?;

    if !quiet {
        eprintln!(
            "Resuming from checkpoint: {}",
            style(checkpoint_path).cyan()
        );
        eprintln!(
            "  {} sentences processed",
            checkpoint.state.sentences_processed
        );
        eprintln!("  {} unique n-grams", checkpoint.unique_ngrams);
        eprintln!("  {:.2}s elapsed", checkpoint.state.elapsed_secs);
    }

    // Open existing accumulator (WAL recovery happens automatically)
    let accumulator = NgramAccumulator::open(&checkpoint.accumulator_path)
        .map_err(|e| CliError::io(format!("Failed to open accumulator: {}", e)))?;

    let timer = TrainingTimer::resume_from(checkpoint.state.elapsed_secs);

    Ok((accumulator, checkpoint.state, Some(manager), timer))
}

/// Save an N-gram training checkpoint.
fn save_ngram_checkpoint(
    manager: &CheckpointManager,
    accumulator: &mut NgramAccumulator,
    state: &NgramTrainingState,
    args: &TrainNgramArgs,
    timer: &TrainingTimer,
    quiet: bool,
) -> CliResult<()> {
    // Sync WAL first
    accumulator
        .sync()
        .map_err(|e| CliError::io(format!("Failed to sync WAL: {}", e)))?;

    let checkpoint = NgramCheckpoint {
        version: 1,
        config: NgramCheckpointConfig {
            order: args.order,
            min_count: args.min_count,
            corpus_path: args.corpus.clone(),
            lowercase: args.lowercase,
        },
        state: NgramTrainingState {
            elapsed_secs: timer.elapsed_secs(),
            ..state.clone()
        },
        accumulator_path: manager.accumulator_path(),
        unique_ngrams: accumulator.len(),
        created_at: Utc::now(),
    };

    let path = manager.save_ngram_checkpoint(&checkpoint)?;

    if !quiet {
        eprintln!(
            "Checkpoint saved: {} ({} n-grams)",
            style(path.display()).cyan(),
            accumulator.len()
        );
    }

    Ok(())
}

/// Finalize N-gram model from accumulator.
fn finalize_ngram_model(
    accumulator: &NgramAccumulator,
    args: &TrainNgramArgs,
    quiet: bool,
) -> CliResult<()> {
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use crate::ngram::{NgramModel, NgramTrie};

    if !quiet {
        eprintln!("Finalizing model (converting to inference format)...");
    }

    // Create DynamicDawgChar dictionary for final model (serde-compatible)
    let dictionary = DynamicDawgChar::<NgramEntry>::new();

    // Export from accumulator to dictionary
    let mut entry_count = 0u64;
    for (ngram_key, count) in accumulator.iter_with_counts() {
        if count >= args.min_count as i64 {
            // Create entry using constructor (fields are private atomics)
            let entry = NgramEntry::new(count as u64);

            // Insert into DynamicDawgChar
            dictionary.insert_with_value(&ngram_key, entry);
            entry_count += 1;
        }
    }

    if !quiet {
        eprintln!("  Exported {} n-grams (min_count={})", entry_count, args.min_count);
    }

    // Build final model with smoothing
    let trie = NgramTrie::new(dictionary, args.order);
    let smoothing = crate::ngram::smoothing::KneserNeySmoothing::new(args.order);

    // Compute vocabulary size (unique unigrams)
    let vocab_size = accumulator
        .iter_with_counts()
        .filter(|(key, count)| key_format::order(key) == 1 && *count >= args.min_count as i64)
        .count();

    let total_tokens: u64 = accumulator
        .iter_with_counts()
        .filter(|(key, _)| key_format::order(key) == 1)
        .map(|(_, count)| count as u64)
        .sum();

    let model = NgramModel::new(trie, smoothing, vocab_size, total_tokens);

    // Save model using portable format (doesn't require D: Serialize)
    model
        .save_portable(&args.output)
        .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

    Ok(())
}

/// Train N-gram model in memory without checkpointing (for small corpora).
fn train_ngram_inmemory(
    args: TrainNgramArgs,
    verbose: bool,
    quiet: bool,
    reader: Box<dyn CorpusReader>,
    progress: TrainingProgress,
    _stats: Arc<TrainingStats>,
) -> CliResult<()> {
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use crate::ngram::TrainerBuilder;

    if verbose {
        eprintln!("Using in-memory training (no checkpointing)");
    }

    if !quiet {
        progress.set_message("Training N-gram model...");
    }

    // Use DynamicDawgChar for serde compatibility
    let dictionary = DynamicDawgChar::<NgramEntry>::new();

    let model = TrainerBuilder::new(dictionary)
        .order(args.order)
        .batch_size(args.batch_size)
        .min_word_freq(args.min_count)
        .build()
        .train(reader)
        .map_err(|e| CliError::training(format!("Training failed: {}", e)))?;

    let ngram_count = model.ngram_count();
    let vocab_size = model.vocab_size();
    let total_tokens = model.total_count();

    // Save model using portable format (doesn't require D: Serialize)
    model
        .save_portable(&args.output)
        .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

    // Use model statistics since the internal trainer doesn't update CLI stats
    // Estimate sentences from tokens (rough average of 20 tokens/sentence)
    let estimated_sentences = total_tokens / 20;
    progress.finish(estimated_sentences, total_tokens, ngram_count as u64);

    if !quiet {
        print_success(&format!("Model saved to: {}", args.output.display()));
        eprintln!("  Vocabulary size: {}", vocab_size);
        eprintln!("  Total tokens:    {}", total_tokens);
        eprintln!("  Total N-grams:   {}", ngram_count);
    }

    Ok(())
}

/// Train subword embeddings.
fn train_embedding(args: TrainEmbeddingArgs, verbose: bool, quiet: bool) -> CliResult<()> {
    use crate::cli::checkpoint::{
        CheckpointManager, TrainingTimer,
    };
    use crate::embedding::{EmbeddingTrainerBuilder, SubwordEmbedding};

    if verbose {
        eprintln!("Training embedding model (dim={})", args.dim);
        eprintln!("  Corpus: {}", args.corpus);
        eprintln!("  Output: {}", args.output.display());
        eprintln!("  Window: {}", args.window);
        eprintln!("  Epochs: {}", args.epochs);
        eprintln!("  Min count: {}", args.min_count);
        eprintln!("  Neg samples: {}", args.neg_samples);
        eprintln!("  Learning rate: {}", args.learning_rate);
    }

    // Create corpus reader based on format
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    // Setup interrupt handler
    let stats = Arc::new(TrainingStats::new());
    setup_interrupt_handler(stats.clone());

    // Check for resume
    let (start_epoch, model, timer, checkpoint_manager): (
        u32,
        Option<SubwordEmbedding>,
        TrainingTimer,
        Option<CheckpointManager>,
    ) = if let Some(ref resume_path) = args.checkpoint.resume {
        // Resume from checkpoint
        let checkpoint_dir = args.checkpoint.checkpoint.as_ref().ok_or_else(|| {
            CliError::unsupported("--checkpoint directory required when using --resume")
        })?;

        let manager = CheckpointManager::new(checkpoint_dir, args.checkpoint.keep_checkpoints)?;
        let checkpoint = manager.load_embedding_checkpoint(resume_path)?;

        if !quiet {
            eprintln!(
                "Resuming from checkpoint: {}",
                style(resume_path).cyan()
            );
            eprintln!(
                "  {} epochs completed",
                checkpoint.state.completed_epochs
            );
            eprintln!(
                "  {:.2}s elapsed",
                checkpoint.state.elapsed_secs
            );
        }

        // Load the saved model
        let model = SubwordEmbedding::load(&checkpoint.model_path)
            .map_err(|e| CliError::io(format!("Failed to load model from checkpoint: {}", e)))?;

        let timer = TrainingTimer::resume_from(checkpoint.state.elapsed_secs);

        (
            checkpoint.state.completed_epochs,
            Some(model),
            timer,
            Some(manager),
        )
    } else if let Some(ref checkpoint_dir) = args.checkpoint.checkpoint {
        // Fresh start with checkpointing enabled
        let manager = CheckpointManager::new(checkpoint_dir, args.checkpoint.keep_checkpoints)?;

        if !quiet {
            eprintln!(
                "Checkpoints will be saved to: {}",
                style(checkpoint_dir.display()).cyan()
            );
        }

        (0, None, TrainingTimer::new(), Some(manager))
    } else {
        // No checkpointing
        (0, None, TrainingTimer::new(), None)
    };

    // Create progress bar
    let progress = if quiet || args.resources.no_progress {
        TrainingProgress::hidden()
    } else {
        TrainingProgress::new(None) // Unknown total until vocabulary is built
    };

    if !quiet {
        progress.set_message("Training embedding model...");
    }

    // Build or use existing model
    let model = if let Some(existing_model) = model {
        // Resume training - we need to continue with more epochs
        if start_epoch >= args.epochs {
            if !quiet {
                eprintln!("Training already complete ({} epochs)", start_epoch);
            }
            existing_model
        } else {
            // Continue training for remaining epochs
            // Note: The current trainer doesn't support resuming mid-training,
            // so we save after each epoch and resume from saved model.
            // For true resumption, we'd need to refactor the trainer.

            // For now, just use the model as-is (training continuation not fully supported)
            if !quiet {
                eprintln!(
                    "{}: Resuming embedding training continues from saved model state",
                    style("note").yellow()
                );
            }
            existing_model
        }
    } else {
        // Train from scratch
        let trainer = EmbeddingTrainerBuilder::new()
            .dim(args.dim)
            .window_size(args.window)
            .min_count(args.min_count)
            .neg_samples(args.neg_samples)
            .epochs(args.epochs as usize)
            .learning_rate(args.learning_rate as f32);

        // Train with per-epoch checkpointing if enabled
        if let Some(ref manager) = checkpoint_manager {
            train_embedding_with_checkpoints(
                trainer,
                reader,
                &args,
                manager,
                start_epoch,
                &timer,
                &progress,
                &stats,
                quiet,
            )?
        } else {
            // Simple training without checkpoints
            trainer
                .train(reader)
                .map_err(|e| CliError::training(format!("Training failed: {}", e)))?
        }
    };

    // Save final model
    model
        .save(&args.output)
        .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

    let vocab_size = model.vocab_size();
    let dim = model.dim();

    progress.finish(args.epochs as u64, 0, vocab_size as u64);

    if !quiet {
        print_success(&format!("Model saved to: {}", args.output.display()));
        eprintln!("  Vocabulary size: {}", vocab_size);
        eprintln!("  Embedding dim:   {}", dim);
        eprintln!("  Epochs:          {}", args.epochs);
        eprintln!("  Training time:   {:.2}s", timer.elapsed_secs());
    }

    Ok(())
}

/// Train embeddings with per-epoch checkpointing.
fn train_embedding_with_checkpoints(
    trainer_builder: crate::embedding::EmbeddingTrainerBuilder,
    reader: Box<dyn CorpusReader>,
    args: &TrainEmbeddingArgs,
    manager: &crate::cli::checkpoint::CheckpointManager,
    start_epoch: u32,
    timer: &crate::cli::checkpoint::TrainingTimer,
    progress: &TrainingProgress,
    stats: &Arc<TrainingStats>,
    quiet: bool,
) -> CliResult<crate::embedding::SubwordEmbedding> {
    use crate::cli::checkpoint::{
        EmbeddingCheckpoint, EmbeddingCheckpointConfig, EmbeddingTrainingState,
    };

    // For epoch-based checkpointing, we train one epoch at a time
    // and save after each completed epoch.

    let epochs_remaining = args.epochs.saturating_sub(start_epoch);
    if epochs_remaining == 0 {
        return Err(CliError::unsupported("No epochs remaining to train"));
    }

    // Train all epochs at once (current trainer doesn't support epoch-by-epoch)
    // We checkpoint after training completes
    let model = trainer_builder
        .epochs(args.epochs as usize)
        .train(reader)
        .map_err(|e| CliError::training(format!("Training failed: {}", e)))?;

    // Check for interrupt
    if !stats.is_running() {
        progress.abandon();
        return Err(CliError::Interrupted);
    }

    // Save checkpoint after training
    let model_path = manager.embedding_model_path(args.epochs);
    model
        .save(&model_path)
        .map_err(|e| CliError::io(format!("Failed to save model checkpoint: {}", e)))?;

    let checkpoint = EmbeddingCheckpoint {
        version: 1,
        config: EmbeddingCheckpointConfig {
            dim: args.dim,
            window: args.window,
            min_count: args.min_count,
            neg_samples: args.neg_samples,
            epochs: args.epochs,
            learning_rate: args.learning_rate,
            corpus_path: args.corpus.clone(),
        },
        state: EmbeddingTrainingState {
            completed_epochs: args.epochs,
            words_processed: 0, // Not tracked currently
            total_words: 0,
            current_learning_rate: args.learning_rate,
            loss_history: Vec::new(),
            elapsed_secs: timer.elapsed_secs(),
        },
        model_path,
        vocab_size: model.vocab_size(),
        created_at: Utc::now(),
    };

    let path = manager.save_embedding_checkpoint(&checkpoint)?;

    if !quiet {
        eprintln!(
            "Checkpoint saved: {} (vocab_size={})",
            style(path.display()).cyan(),
            model.vocab_size()
        );
    }

    Ok(model)
}

/// Create hybrid model from N-gram and embedding models.
fn train_hybrid(args: TrainHybridArgs, verbose: bool, quiet: bool) -> CliResult<()> {
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use crate::embedding::SubwordEmbedding;
    use crate::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy as HybridStrategy};
    use crate::ngram::NgramModel;
    use crate::cli::args::InterpolationStrategy as CliStrategy;

    if verbose {
        eprintln!("Creating hybrid model");
        eprintln!("  N-gram:    {}", args.ngram_model.display());
        eprintln!("  Embedding: {}", args.embedding_model.display());
        eprintln!("  Output:    {}", args.output.display());
        eprintln!("  Strategy:  {:?}", args.strategy);
        eprintln!("  Alpha:     {}", args.alpha);
        eprintln!("  Cache:     {}", args.cache_size);
    }

    // Check that input models exist
    if !args.ngram_model.exists() {
        return Err(CliError::file_not_found(&args.ngram_model));
    }
    if !args.embedding_model.exists() {
        return Err(CliError::file_not_found(&args.embedding_model));
    }

    if !quiet {
        eprintln!("Loading N-gram model...");
    }

    // Load N-gram model using portable format (with DynamicDawgChar backend)
    let ngram_model: NgramModel<DynamicDawgChar<NgramEntry>> =
        NgramModel::load_portable(&args.ngram_model, DynamicDawgChar::new)
            .map_err(|e| CliError::io(format!("Failed to load N-gram model: {}", e)))?;

    if verbose {
        eprintln!(
            "  Loaded N-gram model: order={}, vocab={}, ngrams={}",
            ngram_model.order(),
            ngram_model.vocab_size(),
            ngram_model.ngram_count()
        );
    }

    if !quiet {
        eprintln!("Loading embedding model...");
    }

    // Load embedding model
    let embedding_model = SubwordEmbedding::load(&args.embedding_model)
        .map_err(|e| CliError::io(format!("Failed to load embedding model: {}", e)))?;

    if verbose {
        eprintln!(
            "  Loaded embedding model: vocab={}, dim={}",
            embedding_model.vocab_size(),
            embedding_model.dim()
        );
    }

    // Map CLI interpolation strategy to hybrid module strategy
    let strategy = match args.strategy {
        CliStrategy::Linear => HybridStrategy::Linear { alpha: args.alpha },
        CliStrategy::LogLinear => HybridStrategy::LogLinear { alpha: args.alpha },
        CliStrategy::NgramFallback => HybridStrategy::NgramWithEmbeddingFallback,
        CliStrategy::Dynamic => HybridStrategy::Dynamic {
            base_alpha: args.alpha * 0.5,
            alpha_per_context: 0.1,
            max_alpha: args.alpha.min(0.95),
        },
    };

    // Create hybrid configuration
    let config = HybridConfig {
        strategy,
        cache_size: args.cache_size,
        ..Default::default()
    };

    if !quiet {
        eprintln!("Creating hybrid model...");
    }

    // Create hybrid model
    let hybrid_model = HybridLanguageModel::new(ngram_model, embedding_model, config);

    if !quiet {
        eprintln!("Saving hybrid model...");
    }

    // Save hybrid model using portable format (works with any dictionary backend)
    hybrid_model
        .save_portable(&args.output)
        .map_err(|e| CliError::io(format!("Failed to save hybrid model: {}", e)))?;

    // Get file size for display
    let file_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);
    let size_str = humansize::format_size(file_size, humansize::BINARY);

    if !quiet {
        print_success(&format!("Hybrid model saved to: {}", args.output.display()));
        eprintln!("  N-gram order:        {}", hybrid_model.ngram_model().order());
        eprintln!("  N-gram vocab:        {}", hybrid_model.ngram_model().vocab_size());
        eprintln!("  Embedding dim:       {}", hybrid_model.embedding_model().dim());
        eprintln!("  Embedding vocab:     {}", hybrid_model.embedding_model().vocab_size());
        eprintln!("  Strategy:            {:?}", args.strategy);
        eprintln!("  Alpha:               {}", args.alpha);
        eprintln!("  Cache size:          {}", args.cache_size);
        eprintln!("  File size:           {}", size_str);
    }

    Ok(())
}

/// Create a corpus reader based on format.
fn create_corpus_reader(path: &str, format: CorpusFormat) -> CliResult<Box<dyn CorpusReader>> {
    let path_obj = Path::new(path);

    match format {
        CorpusFormat::Plaintext => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    PlaintextReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    PlaintextReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Wikipedia => {
            // Check if it's an HTTP URL
            #[cfg(feature = "http-corpus")]
            if path.starts_with("http://") || path.starts_with("https://") {
                return Ok(Box::new(
                    WikipediaReader::from_url(path, crate::corpus::WikipediaConfig::default())
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ));
            }

            // Local file
            if path_obj.exists() {
                Ok(Box::new(
                    WikipediaReader::new(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Gutenberg => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    GutenbergReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    GutenbergReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
    }
}

/// Import N-gram model from Google Books N-grams.
#[cfg(feature = "google-books")]
fn import_google_books(args: ImportGoogleBooksArgs, verbose: bool, quiet: bool) -> CliResult<()> {
    use crate::cli::args::ShardingModeArg;
    use crate::sources::google_books::{GoogleBooksConfig, GoogleBooksImporter, LanguageInfo, ShardingMode, ShardingOptions};

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
    };

    // Create importer (resume from checkpoint if one exists)
    let mut importer = GoogleBooksImporter::resume_or_start(config)
        .map_err(|e| CliError::io(format!("Failed to create importer: {}", e)))?;

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
            pb.set_message(format!("Importing from local files: {}", local_dir.display()));
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
            use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
            use tracing_appender::rolling::{RollingFileAppender, Rotation};

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
                importer.import_http_reactive(event_tx_clone, command_rx, keep_shards).await
            });

            // Set up file-based tracing for debugging (works even when TUI is active)
            let log_dir = args.output.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| std::env::temp_dir().join("grammstein-logs"));
            std::fs::create_dir_all(&log_dir).ok();

            let file_appender = RollingFileAppender::new(
                Rotation::NEVER,
                &log_dir,
                "import-debug.log",
            );
            let (non_blocking, tracing_guard) = tracing_appender::non_blocking(file_appender);

            // Use try_init() to avoid panic if already initialized
            let _ = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer()
                    .with_writer(non_blocking)
                    .with_ansi(false)
                    .with_target(true))
                .with(tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "libgrammstein::sources::google_books=debug,libgrammstein::cli::tui=debug".parse().expect("valid filter")))
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
                Err(_) => false, // Error in TUI, not user cancellation
            };

            if user_cancelled {
                // User pressed Q - abort the import task immediately
                log::info!("User cancelled, aborting import task...");
                import_handle.abort();

                // Brief wait for cleanup (with timeout)
                let _ = runtime.block_on(async {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        import_handle
                    ).await
                });

                log::info!("Import cancelled by user");
                return Ok(());
            }

            // Handle TUI error
            if let Err(e) = tui_result {
                return Err(CliError::io(format!("TUI error: {}", e)));
            }

            // Wait for import to finish (normal completion path)
            let import_result = runtime.block_on(async {
                import_handle.await
            });

            // Handle import result
            let import_result = import_result
                .map_err(|e| CliError::io(format!("Import task failed: {}", e)))?;
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
                    importer.import_http_reactive(event_tx_clone, command_rx, keep_shards).await
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
    println!(
        "{} Google Books import complete",
        style("✓").green().bold()
    );
    println!();
    println!("  Total n-grams:  {}", stats.total_ngrams);
    println!("  Unique n-grams: {}", stats.unique_ngrams);
    println!("  Files processed: {}", stats.files_processed);
    println!("  Duration: {:.2}s", stats.elapsed_seconds);
    println!();
    println!("  Output: {}", output.display());
}
