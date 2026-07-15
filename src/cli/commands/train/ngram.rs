//! `train ngram` command: persistent checkpointed n-gram training.

use std::sync::Arc;

use chrono::Utc;
use console::style;

use crate::cli::args::TrainNgramArgs;
use crate::cli::checkpoint::{
    CheckpointManager, NgramCheckpoint, NgramCheckpointConfig, NgramTrainingState, TrainingTimer,
};
use crate::cli::error::{print_success, CliError, CliResult};
use crate::cli::progress::{setup_interrupt_handler, TrainingProgress, TrainingStats};
use crate::corpus::{CorpusReader, Tokenizer};
use crate::ngram::accumulator::key_format;
use crate::ngram::{NgramAccumulator, NgramEntry};

use super::corpus_reader::create_corpus_reader;

pub(super) fn train_ngram(args: TrainNgramArgs, verbose: bool, quiet: bool) -> CliResult<()> {
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
                save_ngram_checkpoint(manager, &mut accumulator, &state, &args, &timer, quiet)?;
            }
            progress.abandon();
            return Err(CliError::Interrupted);
        }

        // Tokenize sentence
        let tokens: Vec<String> = if args.lowercase {
            tokenizer
                .words(&sentence)
                .map(|s| s.to_lowercase())
                .collect()
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
            progress.update(
                state.sentences_processed,
                state.tokens_processed,
                state.bytes_read,
            );
        }

        // Periodic checkpoint
        if state.sentences_processed % checkpoint_interval == 0 {
            if let Some(ref manager) = checkpoint_manager {
                save_ngram_checkpoint(manager, &mut accumulator, &state, &args, &timer, quiet)?;
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

    progress.finish(
        state.sentences_processed,
        state.tokens_processed,
        unique_ngrams as u64,
    );

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
    let checkpoint_dir = args.checkpoint.checkpoint.as_ref().ok_or_else(|| {
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

/// Finalize the n-gram model from the accumulator.
///
/// Reads the accumulator's persisted (`'|'`-joined) count keys once — via the
/// accumulator's own `key_format::parse_key` — and re-keys them onto a
/// byte-native [`TermIdStore`] (LEB128 term-id keys), which is then saved in the
/// portable term-id format. No `'|'`-delimited store is ever live.
fn finalize_ngram_model(
    accumulator: &NgramAccumulator,
    args: &TrainNgramArgs,
    quiet: bool,
) -> CliResult<()> {
    use crate::ngram::store::{MutableNgramStore, TermIdStore};
    use crate::ngram::vocabulary::create_vocabulary;
    use crate::ngram::NgramModel;
    use libdictenstein::dynamic_dawg::DynamicDawg;

    if !quiet {
        eprintln!("Finalizing model (converting to term-id inference format)...");
    }

    // Fresh ephemeral vocabulary + in-memory byte-native store. The vocabulary
    // is embedded into the portable model on save, so the temp directory only
    // needs to outlive `save_portable` (it does — dropped at function return).
    let vocab_dir = tempfile::TempDir::new()
        .map_err(|e| CliError::io(format!("Failed to create vocabulary directory: {}", e)))?;
    let vocabulary = create_vocabulary(&vocab_dir.path().join("vocabulary"))
        .map_err(|e| CliError::io(format!("Failed to create vocabulary: {}", e)))?;
    let store = TermIdStore::new(DynamicDawg::<NgramEntry>::new(), vocabulary, args.order);

    // Export from accumulator to the term-id store.
    let mut entry_count = 0u64;
    for (ngram_key, count) in accumulator.iter_with_counts() {
        if count >= args.min_count as i64 {
            let tokens = key_format::parse_key(&ngram_key);
            store.bump(&tokens, count as u64);
            entry_count += 1;
        }
    }

    if !quiet {
        eprintln!(
            "  Exported {} n-grams (min_count={})",
            entry_count, args.min_count
        );
    }

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

    let model = NgramModel::new(store, smoothing, vocab_size, total_tokens);

    // Save model using the portable term-id format (embeds the vocabulary).
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
    use crate::ngram::TrainerBuilder;
    use libdictenstein::dynamic_dawg::DynamicDawg;

    if verbose {
        eprintln!("Using in-memory training (no checkpointing)");
    }

    if !quiet {
        progress.set_message("Training N-gram model...");
    }

    // In-memory byte-native backend for the term-id store.
    let dictionary = DynamicDawg::<NgramEntry>::new();

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
