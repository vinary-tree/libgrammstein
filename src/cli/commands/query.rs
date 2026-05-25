//! Query command implementations.

use std::path::Path;

use console::style;

use crate::cli::args::{QueryCommands, QueryCompletionsArgs, QueryScoreArgs, QuerySimilarArgs};
use crate::cli::error::{CliError, CliResult};
use crate::cli::output;

/// Run the query command.
pub fn run(cmd: QueryCommands, verbose: bool) -> CliResult<()> {
    match cmd {
        QueryCommands::Score(args) => query_score(args, verbose),
        QueryCommands::Similar(args) => query_similar(args, verbose),
        QueryCommands::Completions(args) => query_completions(args, verbose),
    }
}

/// Trait for models that can score sequences.
trait ScoringModel {
    /// Compute log probability for a sentence (list of tokens).
    fn sentence_log_prob(&self, tokens: &[&str]) -> f64;

    /// Compute log probability of a word given context.
    fn log_prob(&self, word: &str, context: &[&str]) -> f64;

    /// Get model description.
    fn description(&self) -> String;
}

/// N-gram model wrapper for scoring.
struct NgramScoringModel {
    model: crate::ngram::NgramModel<
        liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<crate::ngram::NgramEntry>,
    >,
}

impl ScoringModel for NgramScoringModel {
    fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        self.model.sentence_log_prob(tokens)
    }

    fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        self.model.log_prob(word, context)
    }

    fn description(&self) -> String {
        format!(
            "N-gram (order={}, vocab={})",
            self.model.order(),
            self.model.vocab_size()
        )
    }
}

/// Hybrid model wrapper for scoring.
struct HybridScoringModel {
    model: crate::hybrid::HybridLanguageModel<
        liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<crate::ngram::NgramEntry>,
    >,
}

impl ScoringModel for HybridScoringModel {
    fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        self.model.sentence_log_prob(tokens)
    }

    fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        // HybridLanguageModel uses `score` instead of `log_prob`
        self.model.score(word, context)
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

/// Load a model for scoring.
fn load_model_for_scoring(path: &Path) -> CliResult<Box<dyn ScoringModel>> {
    use crate::hybrid::HybridLanguageModel;
    use crate::ngram::NgramModel;
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

    // Try to load as hybrid model first (more complex)
    if let Ok(model) = HybridLanguageModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(Box::new(HybridScoringModel { model }));
    }

    // Try to load as N-gram model
    if let Ok(model) = NgramModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(Box::new(NgramScoringModel { model }));
    }

    Err(CliError::model_load(
        path.to_path_buf(),
        "Failed to load model (unknown format or corrupted file)".to_string(),
    ))
}

/// Trait for models that can find similar words.
trait SimilarityModel {
    /// Find most similar words to a query word.
    fn most_similar(&self, word: &str, k: usize) -> Vec<(String, f32)>;

    /// Check if word is in vocabulary.
    fn contains(&self, word: &str) -> bool;

    /// Get model description.
    fn description(&self) -> String;
}

/// Embedding model wrapper for similarity.
struct EmbeddingSimilarityModel {
    model: crate::embedding::SubwordEmbedding,
}

impl SimilarityModel for EmbeddingSimilarityModel {
    fn most_similar(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        self.model.most_similar(word, k)
    }

    fn contains(&self, word: &str) -> bool {
        self.model.contains(word)
    }

    fn description(&self) -> String {
        format!(
            "Embedding (dim={}, vocab={})",
            self.model.dim(),
            self.model.vocab_size()
        )
    }
}

/// Hybrid model wrapper for similarity (uses embedding component).
struct HybridSimilarityModel {
    model: crate::hybrid::HybridLanguageModel<
        liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<crate::ngram::NgramEntry>,
    >,
}

impl SimilarityModel for HybridSimilarityModel {
    fn most_similar(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        self.model.embedding_model().most_similar(word, k)
    }

    fn contains(&self, word: &str) -> bool {
        self.model.embedding_model().contains(word)
    }

    fn description(&self) -> String {
        format!(
            "Hybrid embedding (dim={}, vocab={})",
            self.model.embedding_model().dim(),
            self.model.embedding_model().vocab_size()
        )
    }
}

/// Load a model for similarity queries.
fn load_model_for_similarity(path: &Path) -> CliResult<Box<dyn SimilarityModel>> {
    use crate::embedding::SubwordEmbedding;
    use crate::hybrid::HybridLanguageModel;
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

    // Try to load as hybrid model first
    if let Ok(model) = HybridLanguageModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(Box::new(HybridSimilarityModel { model }));
    }

    // Try to load as embedding model
    if let Ok(model) = SubwordEmbedding::load(path) {
        return Ok(Box::new(EmbeddingSimilarityModel { model }));
    }

    Err(CliError::model_load(
        path.to_path_buf(),
        "Failed to load model (must be embedding or hybrid model)".to_string(),
    ))
}

/// Trait for models that can provide completions.
trait CompletionModel {
    /// Get top completions for context.
    /// Returns (word, log_probability) pairs sorted by probability.
    fn top_completions(&self, context: &[&str], k: usize) -> Vec<(String, f64)>;

    /// Get model description.
    fn description(&self) -> String;
}

/// Hybrid model wrapper for completions (can iterate vocabulary).
struct HybridCompletionModel {
    model: crate::hybrid::HybridLanguageModel<
        liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<crate::ngram::NgramEntry>,
    >,
}

impl CompletionModel for HybridCompletionModel {
    fn top_completions(&self, context: &[&str], k: usize) -> Vec<(String, f64)> {
        // Use embedding vocabulary to get candidate words
        let embedding = self.model.embedding_model();
        let vocab_size = embedding.vocab_size();

        // Score each word in vocabulary
        let mut scored: Vec<(String, f64)> = (0..vocab_size)
            .filter_map(|idx| {
                embedding.index_to_word(idx).map(|word| {
                    // HybridLanguageModel uses `score` instead of `log_prob`
                    let log_prob = self.model.score(word, context);
                    (word.to_string(), log_prob)
                })
            })
            .collect();

        // Sort by log probability (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        scored.truncate(k);
        scored
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

/// N-gram model wrapper for completions.
/// Note: This is slower because we need to iterate the trie.
struct NgramCompletionModel {
    model: crate::ngram::NgramModel<
        liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<crate::ngram::NgramEntry>,
    >,
}

impl CompletionModel for NgramCompletionModel {
    fn top_completions(&self, context: &[&str], k: usize) -> Vec<(String, f64)> {
        // Get unique unigrams from the trie
        let mut unigrams: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (key, _) in self.model.trie().iter_entries() {
            // Extract the first/only token from unigram keys (no separator)
            if !key.contains('|') {
                unigrams.insert(key);
            }
        }

        // Score each unigram
        let mut scored: Vec<(String, f64)> = unigrams
            .into_iter()
            .map(|word| {
                let log_prob = self.model.log_prob(&word, context);
                (word, log_prob)
            })
            .collect();

        // Sort by log probability (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        scored.truncate(k);
        scored
    }

    fn description(&self) -> String {
        format!(
            "N-gram (order={}, vocab={})",
            self.model.order(),
            self.model.vocab_size()
        )
    }
}

/// Load a model for completion queries.
fn load_model_for_completions(path: &Path) -> CliResult<Box<dyn CompletionModel>> {
    use crate::hybrid::HybridLanguageModel;
    use crate::ngram::NgramModel;
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

    // Try to load as hybrid model first (preferred - has vocabulary from embedding)
    if let Ok(model) = HybridLanguageModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(Box::new(HybridCompletionModel { model }));
    }

    // Try to load as N-gram model
    if let Ok(model) = NgramModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(Box::new(NgramCompletionModel { model }));
    }

    Err(CliError::model_load(
        path.to_path_buf(),
        "Failed to load model (unknown format or corrupted file)".to_string(),
    ))
}

/// Score a sentence or continuation.
fn query_score(args: QueryScoreArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Scoring tokens");
        eprintln!("  Model: {}", args.model.display());
    }

    // Check that model exists
    if !args.model.exists() {
        return Err(CliError::file_not_found(&args.model));
    }

    // Get tokens from args or stdin
    let tokens = if args.tokens.is_empty() {
        // Read from stdin
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(CliError::Io)?;
        input.split_whitespace().map(String::from).collect()
    } else {
        args.tokens.clone()
    };

    if tokens.is_empty() {
        return Err(CliError::invalid_argument("No tokens provided"));
    }

    // Load model
    let model = load_model_for_scoring(&args.model)?;

    if verbose {
        eprintln!("  Model type: {}", model.description());
    }

    // Convert to &str slice
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();

    // Compute score based on mode
    let (log_prob, mode) = if args.continuation {
        // Score last token given preceding context
        if token_refs.len() < 2 {
            return Err(CliError::invalid_argument(
                "Continuation mode requires at least 2 tokens (context + word)",
            ));
        }
        let word = token_refs[token_refs.len() - 1];
        let context = &token_refs[..token_refs.len() - 1];
        (model.log_prob(word, context), "continuation")
    } else {
        // Score as complete sentence
        (model.sentence_log_prob(&token_refs), "sentence")
    };

    // Compute perplexity
    let perplexity = if args.sentence || !args.continuation {
        let n = token_refs.len() as f64;
        (-log_prob / n).exp()
    } else {
        // For continuation, perplexity is just exp(-log_prob)
        (-log_prob).exp()
    };

    if args.json {
        let result = serde_json::json!({
            "tokens": tokens,
            "log_probability": log_prob,
            "perplexity": perplexity,
            "mode": mode
        });
        output::print_json(&result)?;
    } else {
        println!();
        println!("Tokens: {}", style(tokens.join(" ")).cyan());
        println!("Mode:   {}", mode);
        println!();
        println!(
            "Log probability: {}",
            style(format!("{:.4}", log_prob)).green()
        );
        println!(
            "Perplexity:      {}",
            style(format!("{:.2}", perplexity)).green()
        );
    }

    Ok(())
}

/// Find similar words.
fn query_similar(args: QuerySimilarArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Finding similar words");
        eprintln!("  Model: {}", args.model.display());
        eprintln!("  Word:  {}", args.word);
    }

    // Check that model exists
    if !args.model.exists() {
        return Err(CliError::file_not_found(&args.model));
    }

    // Load model
    let model = load_model_for_similarity(&args.model)?;

    if verbose {
        eprintln!("  Model type: {}", model.description());
    }

    // Check if word is in vocabulary
    let in_vocab = model.contains(&args.word);
    if !in_vocab && verbose {
        eprintln!(
            "  {} Word '{}' not in vocabulary (using subword representation)",
            style("note:").yellow(),
            args.word
        );
    }

    // Find similar words
    let similar: Vec<(String, f64)> = model
        .most_similar(&args.word, args.top)
        .into_iter()
        .map(|(w, s)| (w, s as f64))
        .collect();

    if args.json {
        let result = serde_json::json!({
            "query": args.word,
            "in_vocabulary": in_vocab,
            "similar": similar.iter().map(|(w, s)| {
                serde_json::json!({"word": w, "similarity": s})
            }).collect::<Vec<_>>()
        });
        output::print_json(&result)?;
    } else {
        output::print_similar_words(&args.word, &similar);
    }

    Ok(())
}

/// Get top completions for context.
fn query_completions(args: QueryCompletionsArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Getting completions");
        eprintln!("  Model:   {}", args.model.display());
        eprintln!("  Context: {}", args.context.join(" "));
    }

    // Check that model exists
    if !args.model.exists() {
        return Err(CliError::file_not_found(&args.model));
    }

    // Load model
    let model = load_model_for_completions(&args.model)?;

    if verbose {
        eprintln!("  Model type: {}", model.description());
        eprintln!("  Computing top {} completions...", args.top);
    }

    // Convert context to &str slice
    let context_refs: Vec<&str> = args.context.iter().map(|s| s.as_str()).collect();

    // Get completions
    let completions = model.top_completions(&context_refs, args.top);

    if args.json {
        let result = serde_json::json!({
            "context": args.context,
            "completions": completions.iter().map(|(w, lp)| {
                serde_json::json!({
                    "word": w,
                    "log_probability": lp,
                    "probability": lp.exp()
                })
            }).collect::<Vec<_>>()
        });
        output::print_json(&result)?;
    } else {
        output::print_completions(&args.context, &completions);
    }

    Ok(())
}
