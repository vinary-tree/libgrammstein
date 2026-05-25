//! `train hybrid` command: combine n-gram and embedding models.

use crate::cli::args::TrainHybridArgs;
use crate::cli::error::{print_success, CliError, CliResult};
use crate::ngram::NgramEntry;

pub(super) fn train_hybrid(args: TrainHybridArgs, verbose: bool, quiet: bool) -> CliResult<()> {
    use crate::cli::args::InterpolationStrategy as CliStrategy;
    use crate::embedding::SubwordEmbedding;
    use crate::hybrid::{
        HybridConfig, HybridLanguageModel, InterpolationStrategy as HybridStrategy,
    };
    use crate::ngram::NgramModel;
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

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
        eprintln!(
            "  N-gram order:        {}",
            hybrid_model.ngram_model().order()
        );
        eprintln!(
            "  N-gram vocab:        {}",
            hybrid_model.ngram_model().vocab_size()
        );
        eprintln!(
            "  Embedding dim:       {}",
            hybrid_model.embedding_model().dim()
        );
        eprintln!(
            "  Embedding vocab:     {}",
            hybrid_model.embedding_model().vocab_size()
        );
        eprintln!("  Strategy:            {:?}", args.strategy);
        eprintln!("  Alpha:               {}", args.alpha);
        eprintln!("  Cache size:          {}", args.cache_size);
        eprintln!("  File size:           {}", size_str);
    }

    Ok(())
}
