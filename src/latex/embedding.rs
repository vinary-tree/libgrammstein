//! LaTeX command and equation embeddings for semantic similarity.
//!
//! This module provides dense vector representations of LaTeX commands
//! and mathematical expressions for similarity-based scoring.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Min-heap entry for top-k selection using BinaryHeap (which is a max-heap).
/// We reverse the ordering so that the smallest similarity is at the top,
/// allowing efficient O(n log k) top-k selection instead of O(n log n) sort.
struct TopKEntry<T> {
    similarity: f32,
    item: T,
}

impl<T> PartialEq for TopKEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl<T> Eq for TopKEntry<T> {}

impl<T> PartialOrd for TopKEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for TopKEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller similarity = greater priority in heap
        // This makes BinaryHeap act as a min-heap on similarity
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

/// Configuration for LaTeX embedding generation.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Embedding dimension.
    pub dimension: usize,
    /// Whether to normalize embeddings to unit length.
    pub normalize: bool,
    /// Window size for skip-gram training.
    pub window_size: usize,
    /// Minimum count for vocabulary inclusion.
    pub min_count: usize,
    /// Number of negative samples for training.
    pub negative_samples: usize,
    /// Learning rate for training.
    pub learning_rate: f64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            normalize: true,
            window_size: 5,
            min_count: 5,
            negative_samples: 5,
            learning_rate: 0.025,
        }
    }
}

/// A single command embedding with metadata.
#[derive(Debug, Clone)]
pub struct CommandEmbedding {
    /// The LaTeX command (without backslash).
    pub command: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Frequency in training corpus.
    pub frequency: u64,
    /// Semantic category (if known).
    pub category: Option<CommandCategory>,
}

impl CommandEmbedding {
    /// Create a new command embedding.
    pub fn new(command: String, vector: Vec<f32>, frequency: u64) -> Self {
        Self {
            command,
            vector,
            frequency,
            category: None,
        }
    }

    /// Set the semantic category.
    pub fn with_category(mut self, category: CommandCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Compute cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        cosine_similarity(&self.vector, &other.vector)
    }
}

/// Semantic category for LaTeX commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandCategory {
    /// Greek letters: \alpha, \beta, etc.
    GreekLetter,
    /// Mathematical operators: \sum, \prod, \int, etc.
    Operator,
    /// Relations: \leq, \geq, \equiv, etc.
    Relation,
    /// Accents and modifiers: \hat, \bar, \tilde, etc.
    Accent,
    /// Delimiters: \left, \right, \big, etc.
    Delimiter,
    /// Functions: \sin, \cos, \log, etc.
    Function,
    /// Spacing: \quad, \qquad, \!, etc.
    Spacing,
    /// Environments: \begin, \end, etc.
    Environment,
    /// Formatting: \textbf, \textit, \emph, etc.
    Formatting,
    /// Structure: \section, \chapter, etc.
    Structure,
    /// Arrows: \rightarrow, \leftarrow, etc.
    Arrow,
    /// Other or unknown.
    Other,
}

impl CommandCategory {
    /// Categorize a command by name.
    pub fn from_command(name: &str) -> Self {
        match name {
            // Greek letters
            "alpha" | "beta" | "gamma" | "delta" | "epsilon" | "zeta" | "eta" | "theta"
            | "iota" | "kappa" | "lambda" | "mu" | "nu" | "xi" | "omicron" | "pi" | "rho"
            | "sigma" | "tau" | "upsilon" | "phi" | "chi" | "psi" | "omega" | "Gamma" | "Delta"
            | "Theta" | "Lambda" | "Xi" | "Pi" | "Sigma" | "Upsilon" | "Phi" | "Psi" | "Omega"
            | "varepsilon" | "vartheta" | "varpi" | "varrho" | "varsigma" | "varphi" => {
                CommandCategory::GreekLetter
            }

            // Operators
            "sum" | "prod" | "int" | "oint" | "iint" | "iiint" | "coprod" | "bigcup" | "bigcap"
            | "bigvee" | "bigwedge" | "bigoplus" | "bigotimes" | "biguplus" | "bigsqcup"
            | "lim" | "sup" | "inf" | "max" | "min" => CommandCategory::Operator,

            // Relations
            "leq" | "geq" | "neq" | "equiv" | "sim" | "simeq" | "approx" | "cong" | "propto"
            | "subset" | "supset" | "subseteq" | "supseteq" | "in" | "ni" | "notin" | "prec"
            | "succ" | "preceq" | "succeq" | "ll" | "gg" | "perp" | "parallel" | "vdash"
            | "dashv" | "models" => CommandCategory::Relation,

            // Accents
            "hat" | "check" | "breve" | "acute" | "grave" | "tilde" | "bar" | "vec" | "dot"
            | "ddot" | "dddot" | "ddddot" | "widehat" | "widetilde" | "overline" | "underline"
            | "overbrace" | "underbrace" => CommandCategory::Accent,

            // Delimiters
            "left" | "right" | "big" | "Big" | "bigg" | "Bigg" | "lfloor" | "rfloor" | "lceil"
            | "rceil" | "langle" | "rangle" | "vert" | "Vert" => CommandCategory::Delimiter,

            // Functions
            "sin" | "cos" | "tan" | "cot" | "sec" | "csc" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" | "coth" | "exp" | "log" | "ln" | "lg" | "det" | "dim"
            | "ker" | "hom" | "arg" | "deg" | "gcd" => CommandCategory::Function,

            // Spacing
            "quad" | "qquad" | "," | ";" | ":" | "!" | "hspace" | "vspace" | "hfill" | "vfill"
            | "smallskip" | "medskip" | "bigskip" | "thinspace" | "enspace" => {
                CommandCategory::Spacing
            }

            // Environments
            "begin" | "end" => CommandCategory::Environment,

            // Formatting
            "textbf" | "textit" | "texttt" | "textrm" | "textsf" | "textsc" | "emph" | "mathbf"
            | "mathit" | "mathrm" | "mathsf" | "mathtt" | "mathcal" | "mathfrak" | "mathbb"
            | "boldsymbol" => CommandCategory::Formatting,

            // Structure
            "section" | "subsection" | "subsubsection" | "chapter" | "part" | "paragraph"
            | "subparagraph" | "title" | "author" | "date" | "maketitle" => {
                CommandCategory::Structure
            }

            // Arrows
            "rightarrow" | "leftarrow" | "leftrightarrow" | "Rightarrow" | "Leftarrow"
            | "Leftrightarrow" | "longrightarrow" | "longleftarrow" | "longleftrightarrow"
            | "Longrightarrow" | "Longleftarrow" | "Longleftrightarrow" | "uparrow"
            | "downarrow" | "updownarrow" | "Uparrow" | "Downarrow" | "Updownarrow" | "nearrow"
            | "searrow" | "nwarrow" | "swarrow" | "mapsto" | "hookrightarrow" | "hookleftarrow" => {
                CommandCategory::Arrow
            }

            _ => CommandCategory::Other,
        }
    }
}

/// An equation embedding for similarity search.
#[derive(Debug, Clone)]
pub struct EquationEmbedding {
    /// The equation LaTeX source (normalized).
    pub source: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Document source ID (if from a corpus).
    pub source_id: Option<String>,
    /// Equation label (if present).
    pub label: Option<String>,
}

impl EquationEmbedding {
    /// Create a new equation embedding.
    pub fn new(source: String, vector: Vec<f32>) -> Self {
        Self {
            source,
            vector,
            source_id: None,
            label: None,
        }
    }

    /// Set the source document ID.
    pub fn with_source_id(mut self, id: String) -> Self {
        self.source_id = Some(id);
        self
    }

    /// Set the equation label.
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Compute cosine similarity with another equation embedding.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        cosine_similarity(&self.vector, &other.vector)
    }
}

/// LaTeX embedder for commands and equations.
pub struct LaTeXEmbedder {
    /// Command embeddings indexed by command name.
    command_embeddings: HashMap<String, CommandEmbedding>,
    /// Equation embeddings (for retrieval).
    equation_embeddings: Vec<EquationEmbedding>,
    /// Configuration.
    config: EmbeddingConfig,
    /// Unknown token embedding (for OOV handling).
    unknown_embedding: Vec<f32>,
}

impl LaTeXEmbedder {
    /// Create a new embedder with default configuration.
    pub fn new() -> Self {
        Self::with_config(EmbeddingConfig::default())
    }

    /// Create an embedder with custom configuration.
    pub fn with_config(config: EmbeddingConfig) -> Self {
        let unknown_embedding = vec![0.0; config.dimension];
        Self {
            command_embeddings: HashMap::new(),
            equation_embeddings: Vec::new(),
            config,
            unknown_embedding,
        }
    }

    /// Load command embeddings from a pre-trained file.
    #[cfg(feature = "serde-extras")]
    pub fn load_command_embeddings(&mut self, path: &std::path::Path) -> crate::Result<()> {
        use std::io::BufRead;

        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() > 1 {
                let command = parts[0].to_string();
                let vector: Vec<f32> = parts[1..].iter().filter_map(|s| s.parse().ok()).collect();

                if vector.len() == self.config.dimension {
                    let mut embedding = CommandEmbedding::new(command.clone(), vector, 0);
                    embedding.category = Some(CommandCategory::from_command(&command));
                    self.command_embeddings.insert(command, embedding);
                }
            }
        }

        Ok(())
    }

    /// Get the embedding for a command.
    pub fn command_embedding(&self, command: &str) -> Option<&CommandEmbedding> {
        self.command_embeddings.get(command)
    }

    /// Get the embedding vector for a command, or the unknown vector.
    pub fn command_vector(&self, command: &str) -> &[f32] {
        self.command_embeddings
            .get(command)
            .map(|e| e.vector.as_slice())
            .unwrap_or(&self.unknown_embedding)
    }

    /// Add a command embedding.
    pub fn add_command_embedding(&mut self, embedding: CommandEmbedding) {
        self.command_embeddings
            .insert(embedding.command.clone(), embedding);
    }

    /// Add an equation embedding.
    pub fn add_equation_embedding(&mut self, embedding: EquationEmbedding) {
        self.equation_embeddings.push(embedding);
    }

    /// Find the most similar commands to a given command.
    ///
    /// Uses a min-heap to maintain only the top-k elements during iteration,
    /// achieving O(n log k) complexity instead of O(n log n) for sort+truncate.
    pub fn most_similar_commands(&self, command: &str, k: usize) -> Vec<(&str, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let query_embedding = match self.command_embeddings.get(command) {
            Some(e) => &e.vector,
            None => return Vec::new(),
        };

        // Use min-heap to maintain top-k: O(n log k) instead of O(n log n)
        let mut heap: BinaryHeap<TopKEntry<&str>> = BinaryHeap::with_capacity(k + 1);

        for (name, embedding) in &self.command_embeddings {
            if name == command {
                continue;
            }

            let sim = cosine_similarity(query_embedding, &embedding.vector);

            if heap.len() < k {
                heap.push(TopKEntry {
                    similarity: sim,
                    item: name.as_str(),
                });
            } else if let Some(min_entry) = heap.peek() {
                // Since we use reverse ordering, peek() gives the minimum similarity
                if sim > min_entry.similarity {
                    heap.pop();
                    heap.push(TopKEntry {
                        similarity: sim,
                        item: name.as_str(),
                    });
                }
            }
        }

        // Extract results in descending order of similarity
        let mut results: Vec<(&str, f32)> = heap
            .into_iter()
            .map(|entry| (entry.item, entry.similarity))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Find the most similar equations to a given equation.
    ///
    /// Uses a min-heap to maintain only the top-k elements during iteration,
    /// achieving O(n log k) complexity instead of O(n log n) for sort+truncate.
    pub fn most_similar_equations(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Vec<(&EquationEmbedding, f32)> {
        if k == 0 {
            return Vec::new();
        }

        // Use min-heap to maintain top-k: O(n log k) instead of O(n log n)
        let mut heap: BinaryHeap<TopKEntry<usize>> = BinaryHeap::with_capacity(k + 1);

        for (idx, embedding) in self.equation_embeddings.iter().enumerate() {
            let sim = cosine_similarity(query_vector, &embedding.vector);

            if heap.len() < k {
                heap.push(TopKEntry {
                    similarity: sim,
                    item: idx,
                });
            } else if let Some(min_entry) = heap.peek() {
                // Since we use reverse ordering, peek() gives the minimum similarity
                if sim > min_entry.similarity {
                    heap.pop();
                    heap.push(TopKEntry {
                        similarity: sim,
                        item: idx,
                    });
                }
            }
        }

        // Extract results in descending order of similarity
        let mut results: Vec<(&EquationEmbedding, f32)> = heap
            .into_iter()
            .map(|entry| (&self.equation_embeddings[entry.item], entry.similarity))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Compute the centroid embedding for a sequence of commands.
    pub fn sequence_embedding(&self, commands: &[&str]) -> Vec<f32> {
        if commands.is_empty() {
            return self.unknown_embedding.clone();
        }

        let mut centroid = vec![0.0f32; self.config.dimension];
        let mut count = 0;

        for command in commands {
            if let Some(embedding) = self.command_embeddings.get(*command) {
                for (i, v) in embedding.vector.iter().enumerate() {
                    centroid[i] += v;
                }
                count += 1;
            }
        }

        if count > 0 {
            for v in &mut centroid {
                *v /= count as f32;
            }
            if self.config.normalize {
                normalize_vector(&mut centroid);
            }
        }

        centroid
    }

    /// Get vocabulary size (number of known commands).
    pub fn vocab_size(&self) -> usize {
        self.command_embeddings.len()
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Check if a command is in the vocabulary.
    pub fn contains_command(&self, command: &str) -> bool {
        self.command_embeddings.contains_key(command)
    }

    /// Get all commands in a category.
    pub fn commands_in_category(&self, category: CommandCategory) -> Vec<&str> {
        self.command_embeddings
            .iter()
            .filter(|(_, e)| e.category == Some(category))
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

impl Default for LaTeXEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Normalize a vector to unit length.
fn normalize_vector(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Embedding similarity result.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// The similar item (command or equation).
    pub item: String,
    /// Similarity score (0.0 to 1.0).
    pub score: f32,
    /// Category (if applicable).
    pub category: Option<CommandCategory>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_category() {
        assert_eq!(
            CommandCategory::from_command("alpha"),
            CommandCategory::GreekLetter
        );
        assert_eq!(
            CommandCategory::from_command("sum"),
            CommandCategory::Operator
        );
        assert_eq!(
            CommandCategory::from_command("sin"),
            CommandCategory::Function
        );
        assert_eq!(
            CommandCategory::from_command("leq"),
            CommandCategory::Relation
        );
        assert_eq!(
            CommandCategory::from_command("begin"),
            CommandCategory::Environment
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedder_basic() {
        let mut embedder = LaTeXEmbedder::new();

        let alpha = CommandEmbedding::new("alpha".to_string(), vec![1.0; 128], 100);
        embedder.add_command_embedding(alpha);

        assert!(embedder.contains_command("alpha"));
        assert!(!embedder.contains_command("beta"));
    }

    #[test]
    fn test_sequence_embedding() {
        let mut embedder = LaTeXEmbedder::new();

        embedder.add_command_embedding(CommandEmbedding::new(
            "alpha".to_string(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            100,
        ));
        embedder.add_command_embedding(CommandEmbedding::new(
            "beta".to_string(),
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            100,
        ));

        // Modify config to match dimension
        let config = EmbeddingConfig {
            dimension: 8,
            ..Default::default()
        };
        let mut embedder = LaTeXEmbedder::with_config(config);
        embedder.add_command_embedding(CommandEmbedding::new(
            "alpha".to_string(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            100,
        ));
        embedder.add_command_embedding(CommandEmbedding::new(
            "beta".to_string(),
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            100,
        ));

        let seq_emb = embedder.sequence_embedding(&["alpha", "beta"]);
        assert_eq!(seq_emb.len(), 8);
    }
}
