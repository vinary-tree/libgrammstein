//! Conversion command implementations.

use std::path::Path;

use console::style;

use crate::cli::args::{
    ConvertCommands, ConvertInfoArgs, ConvertToStaticArgs, ConvertToTermidArgs,
};
#[cfg(feature = "google-books")]
use crate::cli::args::{ConvertToPathmapArgs, ExtractDictArgs};
use crate::cli::error::{CliError, CliResult};

/// Run the convert command.
pub fn run(cmd: ConvertCommands, verbose: bool) -> CliResult<()> {
    match cmd {
        ConvertCommands::ToStatic(args) => convert_to_static(args, verbose),
        ConvertCommands::ToTermid(args) => convert_to_termid(args, verbose),
        #[cfg(feature = "google-books")]
        ConvertCommands::ToPathmap(args) => convert_to_pathmap(args, verbose),
        #[cfg(feature = "google-books")]
        ConvertCommands::ExtractDict(args) => extract_dict(args, verbose),
        ConvertCommands::Info(args) => convert_info(args, verbose),
    }
}

/// Transcode any portable n-gram model to the byte-native term-id (LEB128)
/// portable format (parallel to `--to-static`).
///
/// A legacy `'|'`-joined model is losslessly re-keyed onto term-id byte keys
/// (with a freshly assigned, embedded, fingerprinted vocabulary); an
/// already-term-id model is normalized. Hybrid and embedding inputs are rejected
/// with a clear message (only the n-gram half is term-id-encoded).
fn convert_to_termid(args: ConvertToTermidArgs, verbose: bool) -> CliResult<()> {
    use crate::ngram::{InMemoryTermIdModel, NgramEntry};
    use libdictenstein::dynamic_dawg::DynamicDawg;

    if !args.input.exists() {
        return Err(CliError::file_not_found(&args.input));
    }

    if verbose {
        eprintln!("Transcoding model to byte-native term-id format");
        eprintln!("  Input:  {}", args.input.display());
        eprintln!("  Output: {}", args.output.display());
    }

    // Load through the unified term-id loader (dispatches on the source encoding
    // and transcodes legacy keys into term-id byte keys).
    let model: InMemoryTermIdModel =
        InMemoryTermIdModel::load_portable(&args.input, DynamicDawg::<NgramEntry>::new).map_err(
            |e| {
                CliError::model_load(
                    args.input.clone(),
                    format!(
                        "failed to load as an n-gram model (hybrid/embedding inputs are not \
                         term-id-transcodable): {e}"
                    ),
                )
            },
        )?;

    if verbose {
        eprintln!(
            "  Order: {}, vocab: {}, n-grams: {}",
            model.order(),
            model.vocab_size(),
            model.ngram_count()
        );
    }

    let ngrams = model.ngram_count();
    model
        .save_portable(&args.output)
        .map_err(|e| CliError::io(format!("Failed to save term-id model: {}", e)))?;

    // Validate: reloading the output yields the same n-gram count.
    let reloaded: InMemoryTermIdModel =
        InMemoryTermIdModel::load_portable(&args.output, DynamicDawg::<NgramEntry>::new)
            .map_err(|e| CliError::io(format!("term-id conversion validation failed: {}", e)))?;
    if reloaded.ngram_count() != ngrams {
        return Err(CliError::io(
            "term-id conversion validation failed: n-gram count differs".to_string(),
        ));
    }

    print_conversion_success(&args.input, &args.output, "ngram (byte-native term-id)");
    Ok(())
}

/// Detected source model type.
enum SourceModel {
    Hybrid,
    Ngram,
    Embedding,
}

/// Detect what type of model we're dealing with.
fn detect_source_type(path: &Path) -> Option<SourceModel> {
    use crate::embedding::SubwordEmbedding;
    use crate::hybrid::HybridLanguageModel;
    use crate::ngram::NgramEntry;
    use libdictenstein::dynamic_dawg::DynamicDawg;

    // Try hybrid first
    if HybridLanguageModel::load_portable(path, DynamicDawg::<NgramEntry>::new).is_ok() {
        return Some(SourceModel::Hybrid);
    }

    // Try n-gram
    if crate::ngram::InMemoryTermIdModel::load_portable(path, DynamicDawg::<NgramEntry>::new)
        .is_ok()
    {
        return Some(SourceModel::Ngram);
    }

    // Try embedding
    if SubwordEmbedding::load(path).is_ok() {
        return Some(SourceModel::Embedding);
    }

    None
}

/// Convert a model to the normalized inference format.
///
/// The fast read-only inference format is the byte-native term-id portable model
/// (LEB128 term-id keys + embedded vocabulary). This normalizes any n-gram/hybrid
/// input to that encoding (transcoding a legacy `'|'`-joined source in the
/// process) and re-saves; embedding models are re-saved verbatim.
fn convert_to_static(args: ConvertToStaticArgs, verbose: bool) -> CliResult<()> {
    use crate::embedding::SubwordEmbedding;
    use crate::hybrid::HybridLanguageModel;
    use crate::ngram::{InMemoryTermIdModel, NgramEntry};
    use libdictenstein::dynamic_dawg::DynamicDawg;

    if !args.input.exists() {
        return Err(CliError::file_not_found(&args.input));
    }

    if verbose {
        eprintln!("Converting model to the normalized term-id inference format");
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
            let model =
                HybridLanguageModel::load_portable(&args.input, DynamicDawg::<NgramEntry>::new)
                    .map_err(|e| CliError::model_load(args.input.clone(), e.to_string()))?;

            if verbose {
                eprintln!("  Type: HybridLanguageModel");
                eprintln!("  N-gram order: {}", model.ngram_model().order());
                eprintln!("  N-gram vocab: {}", model.ngram_model().vocab_size());
                eprintln!("  Embedding dim: {}", model.embedding_model().dim());
            }

            eprintln!("Saving to term-id portable format...");
            model
                .save_portable(&args.output)
                .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

            print_conversion_success(&args.input, &args.output, "hybrid");
        }
        SourceModel::Ngram => {
            let model =
                InMemoryTermIdModel::load_portable(&args.input, DynamicDawg::<NgramEntry>::new)
                    .map_err(|e| CliError::model_load(args.input.clone(), e.to_string()))?;

            if verbose {
                eprintln!("  Type: NgramModel");
                eprintln!("  Order: {}", model.order());
                eprintln!("  Vocab size: {}", model.vocab_size());
                eprintln!("  N-grams: {}", model.ngram_count());
            }

            eprintln!("Saving to term-id portable format...");
            model
                .save_portable(&args.output)
                .map_err(|e| CliError::io(format!("Failed to save model: {}", e)))?;

            print_conversion_success(&args.input, &args.output, "ngram (byte-native term-id)");
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
        "{}: The output is a byte-native term-id portable model.",
        style("note").cyan()
    );
    println!("    Load it with InMemoryTermIdModel::load_portable (DynamicDawg backend).");
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
    println!("{} PathMap translation complete", style("✓").green().bold());
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
