//! `train embedding` command: subword-embedding training (with optional checkpoints).

use std::sync::Arc;

use chrono::Utc;
use console::style;

use crate::cli::args::TrainEmbeddingArgs;
use crate::cli::error::{print_success, CliError, CliResult};
use crate::cli::progress::{setup_interrupt_handler, TrainingProgress, TrainingStats};
use crate::corpus::CorpusReader;

use super::corpus_reader::create_corpus_reader;

pub(super) fn train_embedding(
    args: TrainEmbeddingArgs,
    verbose: bool,
    quiet: bool,
) -> CliResult<()> {
    use crate::cli::checkpoint::{CheckpointManager, TrainingTimer};
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
            eprintln!("Resuming from checkpoint: {}", style(resume_path).cyan());
            eprintln!("  {} epochs completed", checkpoint.state.completed_epochs);
            eprintln!("  {:.2}s elapsed", checkpoint.state.elapsed_secs);
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
            // Continue training for the remaining epochs from the loaded model state,
            // keeping its weights + vocabulary. The ephemeral corpus statistics (negative
            // sampling / sub-sampling) are recovered inside `train_continued` with one
            // pass aligned to the model's vocabulary.
            if !quiet {
                eprintln!(
                    "{}: Resuming embedding training from epoch {} to {}",
                    style("note").cyan(),
                    start_epoch,
                    args.epochs
                );
            }
            EmbeddingTrainerBuilder::new()
                .dim(args.dim)
                .window_size(args.window)
                .min_count(args.min_count)
                .neg_samples(args.neg_samples)
                .epochs(args.epochs as usize)
                .learning_rate(args.learning_rate as f32)
                .train_continued(existing_model, start_epoch as usize, reader)
                .map_err(|e| CliError::training(format!("Resumed training failed: {}", e)))?
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
