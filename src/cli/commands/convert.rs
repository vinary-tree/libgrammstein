//! Conversion command implementations.

use std::path::Path;

use console::style;

use crate::cli::args::{ConvertCommands, ConvertInfoArgs, ConvertToStaticArgs};
#[cfg(feature = "google-books")]
use crate::cli::args::{ConvertToPathmapArgs, ExtractDictArgs};
use crate::cli::error::{CliError, CliResult};

/// Run the convert command.
pub fn run(cmd: ConvertCommands, verbose: bool) -> CliResult<()> {
    match cmd {
        ConvertCommands::ToStatic(args) => convert_to_static(args, verbose),
        #[cfg(feature = "google-books")]
        ConvertCommands::ToPathmap(args) => convert_to_pathmap(args, verbose),
        #[cfg(feature = "google-books")]
        ConvertCommands::ExtractDict(args) => extract_dict(args, verbose),
        ConvertCommands::Info(args) => convert_info(args, verbose),
    }
}

/// Detected source model type.
enum SourceModel {
    Hybrid,
    Ngram,
    Embedding,
}

/// Detect what type of model we're dealing with.
fn detect_source_type(path: &Path) -> Option<SourceModel> {
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use crate::ngram::{NgramEntry, NgramModel};
    use crate::hybrid::HybridLanguageModel;
    use crate::embedding::SubwordEmbedding;

    // Try hybrid first
    if HybridLanguageModel::load_portable(path, DynamicDawgChar::<NgramEntry>::new).is_ok() {
        return Some(SourceModel::Hybrid);
    }

    // Try n-gram
    if NgramModel::load_portable(path, DynamicDawgChar::<NgramEntry>::new).is_ok() {
        return Some(SourceModel::Ngram);
    }

    // Try embedding
    if SubwordEmbedding::load(path).is_ok() {
        return Some(SourceModel::Embedding);
    }

    None
}

/// Convert model to static format for faster inference.
///
/// This converts models from the DynamicDawgChar backend (which supports
/// updates) to a portable format optimized for fast loading and inference.
///
/// Note: Full DoubleArrayTrie conversion requires additional library support.
/// This currently re-exports the model in optimized portable format.
fn convert_to_static(args: ConvertToStaticArgs, verbose: bool) -> CliResult<()> {
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use crate::ngram::{NgramEntry, NgramModel};
    use crate::hybrid::HybridLanguageModel;
    use crate::embedding::SubwordEmbedding;

    if !args.input.exists() {
        return Err(CliError::file_not_found(&args.input));
    }

    if verbose {
        eprintln!("Converting model to static format");
        eprintln!("  Input:  {}", args.input.display());
        eprintln!("  Output: {}", args.output.display());
    }

    // Detect model type
    let source_type = detect_source_type(&args.input).ok_or_else(|| {
        CliError::model_load(
            args.input.clone(),
            "Failed to detect model type (unknown format)".to_string(),
        )
    })?;

    eprintln!("Loading model...");

    match source_type {
        SourceModel::Hybrid => {
            let model = HybridLanguageModel::load_portable(
                &args.input,
                DynamicDawgChar::<NgramEntry>::new,
            )
            .map_err(|e| CliError::model_load(args.input.clone(), e.to_string()))?;

            if verbose {
                eprintln!("  Type: HybridLanguageModel");
                eprintln!("  N-gram order: {}", model.ngram_model().order());
                eprintln!("  N-gram vocab: {}", model.ngram_model().vocab_size());
                eprintln!("  Embedding dim: {}", model.embedding_model().dim());
            }

            eprintln!("Saving to portable format...");
            model
                .save_portable(&args.output)
                .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

            print_conversion_success(&args.input, &args.output, "hybrid");
        }
        SourceModel::Ngram => {
            let model = NgramModel::load_portable(
                &args.input,
                DynamicDawgChar::<NgramEntry>::new,
            )
            .map_err(|e| CliError::model_load(args.input.clone(), e.to_string()))?;

            if verbose {
                eprintln!("  Type: NgramModel");
                eprintln!("  Order: {}", model.order());
                eprintln!("  Vocab size: {}", model.vocab_size());
                eprintln!("  N-grams: {}", model.ngram_count());
            }

            eprintln!("Saving to portable format...");
            model
                .save_portable(&args.output)
                .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

            print_conversion_success(&args.input, &args.output, "ngram");
        }
        SourceModel::Embedding => {
            let model = SubwordEmbedding::load(&args.input)
                .map_err(|e| CliError::model_load(args.input.clone(), e.to_string()))?;

            if verbose {
                eprintln!("  Type: SubwordEmbedding");
                eprintln!("  Dimension: {}", model.dim());
                eprintln!("  Vocab size: {}", model.vocab_size());
            }

            eprintln!("Saving embedding model...");
            model
                .save(&args.output)
                .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

            print_conversion_success(&args.input, &args.output, "embedding");
        }
    }

    Ok(())
}

/// Print conversion success message with file sizes.
fn print_conversion_success(input: &Path, output: &Path, model_type: &str) {
    let input_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let output_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    println!();
    println!(
        "{} Converted {} model successfully",
        style("✓").green().bold(),
        model_type
    );
    println!();
    println!(
        "  Input:  {} ({})",
        input.display(),
        humansize::format_size(input_size, humansize::BINARY)
    );
    println!(
        "  Output: {} ({})",
        output.display(),
        humansize::format_size(output_size, humansize::BINARY)
    );

    if output_size < input_size {
        let savings = input_size - output_size;
        let percent = (savings as f64 / input_size as f64) * 100.0;
        println!(
            "  Saved:  {} ({:.1}% reduction)",
            humansize::format_size(savings, humansize::BINARY),
            percent
        );
    }

    println!();
    println!(
        "{}: The portable format is optimized for fast loading.",
        style("note").cyan()
    );
    println!(
        "    For maximum inference speed, use DoubleArrayTrie backend (requires library support)."
    );
}

/// Display model info (delegates to models info command).
fn convert_info(args: ConvertInfoArgs, verbose: bool) -> CliResult<()> {
    if !args.model.exists() {
        return Err(CliError::file_not_found(&args.model));
    }

    // Delegate to models info
    crate::cli::commands::models::run(
        crate::cli::args::ModelsCommands::Info(crate::cli::args::ModelsInfoArgs {
            model: args.model,
            json: false,
        }),
        verbose,
    )
}

/// Translate trained model to PathMap format for production deployment.
#[cfg(feature = "google-books")]
fn convert_to_pathmap(args: ConvertToPathmapArgs, verbose: bool) -> CliResult<()> {
    use crate::sources::google_books::{PathMapTranslator, TranslationPhase};

    if !args.input.exists() {
        return Err(CliError::file_not_found(&args.input));
    }

    if verbose {
        eprintln!("Translating to PathMap format");
        eprintln!("  Input:  {}", args.input.display());
        eprintln!("  Output: {}", args.output.display());
    }

    eprintln!("Translating model to PathMap format...");

    // Create progress indicator
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .expect("Invalid progress template"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Translate with progress
    let stats = PathMapTranslator::translate_with_progress(&args.input, &args.output, |progress| {
        let phase_str = match progress.phase {
            TranslationPhase::Loading => "Loading source model",
            TranslationPhase::Iterating => "Iterating entries",
            TranslationPhase::Building => "Building PathMap",
            TranslationPhase::Merkleizing => "Computing Merkle hashes",
            TranslationPhase::Saving => "Saving to disk",
            TranslationPhase::Complete => "Complete",
        };
        pb.set_message(format!(
            "{}: {} entries processed",
            phase_str, progress.entries_processed
        ));
    })
    .map_err(|e| CliError::io(format!("Translation failed: {}", e)))?;

    pb.finish_and_clear();

    // Verify if requested
    if args.verify {
        eprintln!("Verifying translation integrity...");
        let verification = PathMapTranslator::verify(&args.input, &args.output)
            .map_err(|e| CliError::io(format!("Verification failed: {}", e)))?;

        if !verification.verified {
            return Err(CliError::io(format!(
                "Verification failed: {} mismatches found",
                verification.mismatches
            )));
        }
        eprintln!("  Verified {} entries", verification.entries_verified);
    }

    // Print success
    println!();
    println!(
        "{} PathMap translation complete",
        style("✓").green().bold()
    );
    println!();
    println!("  Entries translated: {}", stats.entries_translated);
    println!(
        "  Source size: {}",
        humansize::format_size(stats.artrie_size_bytes, humansize::BINARY)
    );
    println!(
        "  Output size: {}",
        humansize::format_size(stats.pathmap_size_bytes, humansize::BINARY)
    );
    if stats.compression_ratio > 0.0 {
        println!("  Compression ratio: {:.2}x", stats.compression_ratio);
    }
    println!("  Duration: {:.2}s", stats.elapsed_seconds);
    println!();
    println!("  Output: {}", args.output.display());

    Ok(())
}

/// Extract dictionary from n-gram model's 1-grams.
#[cfg(feature = "google-books")]
fn extract_dict(args: ExtractDictArgs, verbose: bool) -> CliResult<()> {
    use crate::sources::google_books::{DictionaryExtractor, ExtractionPhase};

    if !args.model.exists() {
        return Err(CliError::file_not_found(&args.model));
    }

    if verbose {
        eprintln!("Extracting dictionary from n-gram model");
        eprintln!("  Model: {}", args.model.display());
        eprintln!("  Output: {}", args.output.display());
        eprintln!("  Min count: {}", args.min_count);
    }

    eprintln!("Extracting vocabulary from 1-grams...");

    // Create progress indicator
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .expect("Invalid progress template"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Extract with progress
    let stats = DictionaryExtractor::extract_to_file_with_progress(
        &args.model,
        &args.output,
        args.min_count,
        |progress| {
            let phase_str = match progress.phase {
                ExtractionPhase::Loading => "Loading model",
                ExtractionPhase::Filtering => "Filtering vocabulary",
                ExtractionPhase::Building => "Building dictionary",
                ExtractionPhase::Saving => "Saving to disk",
                ExtractionPhase::Complete => "Complete",
            };
            pb.set_message(format!(
                "{}: {} words processed",
                phase_str, progress.words_processed
            ));
        },
    )
    .map_err(|e| CliError::io(format!("Extraction failed: {}", e)))?;

    pb.finish_and_clear();

    // Print success
    println!();
    println!(
        "{} Dictionary extraction complete",
        style("✓").green().bold()
    );
    println!();
    println!("  Words extracted: {}", stats.words_extracted);
    println!("  Words filtered: {}", stats.words_filtered);
    println!(
        "  Dictionary size: {}",
        humansize::format_size(stats.dict_size_bytes, humansize::BINARY)
    );
    println!("  Duration: {:.2}s", stats.elapsed_seconds);
    println!();
    println!("  Output: {}", args.output.display());

    Ok(())
}
