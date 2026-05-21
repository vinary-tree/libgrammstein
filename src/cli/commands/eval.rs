//! Evaluation command implementations.

use std::path::Path;
use std::time::Instant;

use console::style;
use comfy_table::{Table, presets::UTF8_FULL};

use crate::cli::args::{CorpusFormat, EvalCommands, EvalCompareArgs, EvalPerplexityArgs};
use crate::cli::error::{print_success, CliError, CliResult};
use crate::corpus::{CorpusReader, PlaintextReader, Tokenizer};

/// Run the eval command.
pub fn run(cmd: EvalCommands, verbose: bool, quiet: bool) -> CliResult<()> {
    match cmd {
        EvalCommands::Perplexity(args) => eval_perplexity(args, verbose, quiet),
        EvalCommands::Compare(args) => eval_compare(args, verbose, quiet),
    }
}

/// Result of perplexity evaluation.
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// Total perplexity over corpus.
    pub perplexity: f64,
    /// Total log probability.
    pub log_probability: f64,
    /// Number of sentences evaluated.
    pub sentences: u64,
    /// Number of tokens evaluated.
    pub tokens: u64,
    /// Number of out-of-vocabulary tokens.
    pub oov_tokens: u64,
    /// Per-sentence perplexities (if requested).
    pub per_sentence: Option<Vec<f64>>,
    /// Evaluation time in seconds.
    pub elapsed_secs: f64,
}

/// Trait for models that can compute perplexity.
trait PerplexityModel {
    /// Compute log probability for a sentence (list of tokens).
    fn sentence_log_prob(&self, tokens: &[&str]) -> f64;

    /// Check if a word is in vocabulary.
    fn in_vocabulary(&self, word: &str) -> bool;

    /// Get model description.
    fn description(&self) -> String;
}

/// N-gram model wrapper for perplexity computation.
struct NgramPerplexityModel {
    model: crate::ngram::NgramModel<liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<crate::ngram::NgramEntry>>,
}

impl PerplexityModel for NgramPerplexityModel {
    fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        self.model.sentence_log_prob(tokens)
    }

    fn in_vocabulary(&self, word: &str) -> bool {
        self.model.in_vocabulary(word)
    }

    fn description(&self) -> String {
        format!(
            "N-gram (order={}, vocab={})",
            self.model.order(),
            self.model.vocab_size()
        )
    }
}

/// Hybrid model wrapper for perplexity computation.
struct HybridPerplexityModel {
    model: crate::hybrid::HybridLanguageModel<liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<crate::ngram::NgramEntry>>,
}

impl PerplexityModel for HybridPerplexityModel {
    fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        self.model.sentence_log_prob(tokens)
    }

    fn in_vocabulary(&self, word: &str) -> bool {
        self.model.ngram_model().in_vocabulary(word)
    }

    fn description(&self) -> String {
        format!(
            "Hybrid (order={}, ngram_vocab={}, emb_vocab={})",
            self.model.ngram_model().order(),
            self.model.ngram_model().vocab_size(),
            self.model.embedding_model().vocab_size()
        )
    }
}

/// Load a model for perplexity evaluation.
fn load_model_for_perplexity(path: &Path) -> CliResult<Box<dyn PerplexityModel>> {
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use crate::ngram::NgramModel;
    use crate::hybrid::HybridLanguageModel;

    // Try to load as hybrid model first (more complex)
    if let Ok(model) = HybridLanguageModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(Box::new(HybridPerplexityModel { model }));
    }

    // Try to load as N-gram model
    if let Ok(model) = NgramModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(Box::new(NgramPerplexityModel { model }));
    }

    Err(CliError::model_load(
        path.to_path_buf(),
        "Failed to load model (unknown format or corrupted file)".to_string(),
    ))
}

/// Compute perplexity on a corpus.
fn compute_perplexity(
    model: &dyn PerplexityModel,
    reader: &dyn CorpusReader,
    per_sentence: bool,
) -> PerplexityResult {
    let start = Instant::now();
    let tokenizer = Tokenizer::new();

    let mut total_log_prob = 0.0f64;
    let mut total_tokens = 0u64;
    let mut total_sentences = 0u64;
    let mut oov_tokens = 0u64;
    let mut sentence_perplexities = if per_sentence { Some(Vec::new()) } else { None };

    for sentence in reader.sentences() {
        let tokens_owned: Vec<String> = tokenizer.words(&sentence).collect();
        if tokens_owned.is_empty() {
            continue;
        }

        // Create slice of references for the model methods
        let tokens: Vec<&str> = tokens_owned.iter().map(|s| s.as_str()).collect();

        // Count OOV tokens
        for token in &tokens {
            if !model.in_vocabulary(token) {
                oov_tokens += 1;
            }
        }

        // Compute sentence log probability
        let sent_log_prob = model.sentence_log_prob(&tokens);
        total_log_prob += sent_log_prob;
        total_tokens += tokens.len() as u64;
        total_sentences += 1;

        // Track per-sentence perplexity if requested
        if let Some(ref mut perps) = sentence_perplexities {
            let avg_log_prob = sent_log_prob / tokens.len() as f64;
            let sent_ppl = (-avg_log_prob).exp();
            perps.push(sent_ppl);
        }
    }

    // Compute overall perplexity: exp(-1/N * sum(log P))
    let perplexity = if total_tokens > 0 {
        let avg_log_prob = total_log_prob / total_tokens as f64;
        (-avg_log_prob).exp()
    } else {
        f64::INFINITY
    };

    PerplexityResult {
        perplexity,
        log_probability: total_log_prob,
        sentences: total_sentences,
        tokens: total_tokens,
        oov_tokens,
        per_sentence: sentence_perplexities,
        elapsed_secs: start.elapsed().as_secs_f64(),
    }
}

/// Evaluate model perplexity on test corpus.
fn eval_perplexity(args: EvalPerplexityArgs, verbose: bool, quiet: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Evaluating perplexity");
        eprintln!("  Model:  {}", args.model.display());
        eprintln!("  Corpus: {}", args.test_corpus);
    }

    // Check that model exists
    if !args.model.exists() {
        return Err(CliError::file_not_found(&args.model));
    }

    // Load model
    if !quiet {
        eprintln!("Loading model...");
    }
    let model = load_model_for_perplexity(&args.model)?;

    if verbose {
        eprintln!("  Model type: {}", model.description());
    }

    // Load test corpus
    if !quiet {
        eprintln!("Loading test corpus...");
    }
    let reader = create_corpus_reader(&args.test_corpus, args.format)?;

    // Compute perplexity
    if !quiet {
        eprintln!("Computing perplexity...");
    }
    let result = compute_perplexity(model.as_ref(), reader.as_ref(), args.per_sentence);

    // Output results
    if let Some(ref output_path) = args.output {
        // Write JSON output
        let json_output = serde_json::json!({
            "model": args.model.display().to_string(),
            "test_corpus": args.test_corpus,
            "perplexity": result.perplexity,
            "log_probability": result.log_probability,
            "sentences": result.sentences,
            "tokens": result.tokens,
            "oov_tokens": result.oov_tokens,
            "oov_rate": if result.tokens > 0 { result.oov_tokens as f64 / result.tokens as f64 * 100.0 } else { 0.0 },
            "elapsed_secs": result.elapsed_secs,
            "per_sentence": result.per_sentence,
        });

        let serialized = serde_json::to_string_pretty(&json_output)
            .expect("serde_json::Value built via json!() macro must serialize");
        std::fs::write(output_path, serialized)
            .map_err(|e| CliError::io(format!("Failed to write output: {}", e)))?;

        if !quiet {
            eprintln!("Results written to: {}", output_path.display());
        }
    }

    // Print results
    if !quiet {
        println!();
        println!(
            "Model: {}",
            style(args.model.display()).cyan()
        );
        println!(
            "Test corpus: {} ({} sentences, {} tokens)",
            args.test_corpus, result.sentences, result.tokens
        );
        println!();
        println!(
            "Perplexity:      {}",
            style(format!("{:.2}", result.perplexity)).green().bold()
        );
        println!("Log probability: {:.2}", result.log_probability);
        println!(
            "OOV rate:        {:.2}% ({} tokens)",
            if result.tokens > 0 {
                result.oov_tokens as f64 / result.tokens as f64 * 100.0
            } else {
                0.0
            },
            result.oov_tokens
        );
        println!("Avg tokens/sent: {:.2}", result.tokens as f64 / result.sentences.max(1) as f64);
        println!("Evaluation time: {:.2}s", result.elapsed_secs);

        // Per-sentence breakdown if requested
        if let Some(ref perps) = result.per_sentence {
            if !perps.is_empty() {
                let mut sorted = perps.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                println!();
                println!("Per-sentence breakdown:");
                println!("  Min perplexity:  {:.2}", sorted.first().unwrap_or(&0.0));
                println!("  Max perplexity:  {:.2}", sorted.last().unwrap_or(&0.0));
                println!(
                    "  Median:          {:.2}",
                    sorted.get(sorted.len() / 2).unwrap_or(&0.0)
                );
            }
        }
    }

    Ok(())
}

/// Compare multiple models.
fn eval_compare(args: EvalCompareArgs, verbose: bool, quiet: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Comparing {} models", args.models.len());
        eprintln!("  Corpus: {}", args.test_corpus);
        for model in &args.models {
            eprintln!("  Model:  {}", model.display());
        }
    }

    // Check that all models exist
    for model in &args.models {
        if !model.exists() {
            return Err(CliError::file_not_found(model));
        }
    }

    // Load test corpus once
    if !quiet {
        eprintln!("Loading test corpus...");
    }
    let reader = create_corpus_reader(&args.test_corpus, args.format)?;

    // Collect sentences for reuse
    let sentences: Vec<String> = reader.sentences().collect();
    let total_sentences = sentences.len();

    if !quiet {
        eprintln!("Test corpus: {} sentences", total_sentences);
    }

    // Create a simple in-memory reader for reuse
    struct MemoryReader {
        sentences: Vec<String>,
    }

    impl CorpusReader for MemoryReader {
        fn documents(&self) -> Box<dyn Iterator<Item = crate::corpus::Document> + Send + '_> {
            // Each sentence as a document
            Box::new(self.sentences.iter().map(|s| crate::corpus::Document::new(s.clone())))
        }

        fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
            Box::new(self.sentences.iter().cloned())
        }

        fn estimated_tokens(&self) -> Option<usize> {
            Some(self.sentences.iter().map(|s| s.split_whitespace().count()).sum())
        }
    }

    let memory_reader = MemoryReader { sentences };

    // Evaluate each model
    let mut results = Vec::new();

    for (i, model_path) in args.models.iter().enumerate() {
        if !quiet {
            eprintln!(
                "Evaluating model {}/{}: {}",
                i + 1,
                args.models.len(),
                model_path.display()
            );
        }

        let model = load_model_for_perplexity(model_path)?;
        let result = compute_perplexity(model.as_ref(), &memory_reader, false);

        results.push((model_path.clone(), model.description(), result));
    }

    // Output results
    if let Some(ref output_path) = args.output {
        // Write JSON output
        let json_results: Vec<_> = results
            .iter()
            .map(|(path, desc, result)| {
                serde_json::json!({
                    "model": path.display().to_string(),
                    "description": desc,
                    "perplexity": result.perplexity,
                    "log_probability": result.log_probability,
                    "oov_rate": if result.tokens > 0 { result.oov_tokens as f64 / result.tokens as f64 * 100.0 } else { 0.0 },
                    "elapsed_secs": result.elapsed_secs,
                })
            })
            .collect();

        let json_output = serde_json::json!({
            "test_corpus": args.test_corpus,
            "sentences": total_sentences,
            "models": json_results,
        });

        let serialized = serde_json::to_string_pretty(&json_output)
            .expect("serde_json::Value built via json!() macro must serialize");
        std::fs::write(output_path, serialized)
            .map_err(|e| CliError::io(format!("Failed to write output: {}", e)))?;

        if !quiet {
            eprintln!("Results written to: {}", output_path.display());
        }
    }

    // Print comparison table
    if !quiet {
        println!();

        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec!["Model", "Perplexity", "OOV Rate", "Time (s)"]);

        for (path, _desc, result) in &results {
            let oov_rate = if result.tokens > 0 {
                format!("{:.2}%", result.oov_tokens as f64 / result.tokens as f64 * 100.0)
            } else {
                "N/A".to_string()
            };

            table.add_row(vec![
                path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string()),
                format!("{:.2}", result.perplexity),
                oov_rate,
                format!("{:.2}", result.elapsed_secs),
            ]);
        }

        println!("{}", table);
    }

    // Find best model. The `!results.is_empty()` guard makes the `min_by`
    // return Some(_), but we still match explicitly to avoid a `.unwrap()`
    // and to surface any future invariant violation as `None` (skip) rather
    // than a panic.
    if !quiet {
        if let Some(best) = results.iter().min_by(|a, b| {
            a.2.perplexity
                .partial_cmp(&b.2.perplexity)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            println!();
            print_success(&format!(
                "Best model: {} (perplexity: {:.2})",
                best.0.display(),
                best.2.perplexity
            ));
        }
    }

    Ok(())
}

/// Create a corpus reader based on format.
fn create_corpus_reader(path: &str, format: CorpusFormat) -> CliResult<Box<dyn CorpusReader>> {
    let path = Path::new(path);

    match format {
        CorpusFormat::Plaintext => {
            if path.is_dir() {
                Ok(Box::new(
                    PlaintextReader::from_directory(path)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path.exists() {
                Ok(Box::new(
                    PlaintextReader::from_file(path)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path))
            }
        }
        CorpusFormat::Wikipedia => Err(CliError::unsupported(
            "Wikipedia corpus format not yet implemented for evaluation",
        )),
        CorpusFormat::Gutenberg => Err(CliError::unsupported(
            "Gutenberg corpus format not yet implemented for evaluation",
        )),
    }
}
