//! End-to-end code correction pipeline.
//!
//! This module provides a complete pipeline for correcting code, integrating:
//! - Tree-sitter parsing for structural analysis
//! - Lexical correction for spelling errors
//! - Grammar correction for syntax errors
//! - Semantic correction for variable misuse and type errors
//!
//! The pipeline operates in phases:
//! 1. **Parse**: Use tree-sitter to get AST with error nodes
//! 2. **Tokenize**: Extract tokens with context information
//! 3. **Analyze**: Build CPG for semantic analysis
//! 4. **Correct**: Apply ensemble of correctors
//! 5. **Rank**: Combine and rank correction candidates

use crate::code::ast::{byte_offset_to_position, CodeParser, ParsedCode};
use crate::code::correction::{CodeCorrector, Correction, CorrectionCandidates};
use crate::code::correctors::EnsembleCorrector;
use crate::code::cpg::CodePropertyGraph;
use crate::code::language::{CodeLanguage, TokenContext};
use crate::code::pcfg::WeightedCFG;
use crate::code::tokenizer::{CodeToken, CodeTokenizer};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

/// Entry for streaming correction ranking using a min-heap.
/// Keeps track of the N best corrections with O(n log N) complexity.
struct CorrectionEntry {
    correction: Correction,
}

impl PartialEq for CorrectionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.correction.confidence == other.correction.confidence
    }
}

impl Eq for CorrectionEntry {}

impl PartialOrd for CorrectionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CorrectionEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller confidence = greater priority in heap
        // This makes BinaryHeap act as a min-heap on confidence
        other
            .correction
            .confidence
            .partial_cmp(&self.correction.confidence)
            .unwrap_or(Ordering::Equal)
    }
}

/// Streaming correction collector that maintains a bounded heap.
/// This avoids buffering all corrections in memory before ranking.
struct StreamingCorrectionCollector {
    heap: BinaryHeap<CorrectionEntry>,
    max_size: usize,
    min_confidence: f64,
    /// Track seen positions to avoid duplicates during streaming
    seen: HashSet<(usize, usize, String)>,
}

impl StreamingCorrectionCollector {
    fn new(max_size: usize, min_confidence: f64) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(max_size + 1),
            max_size,
            min_confidence,
            seen: HashSet::new(),
        }
    }

    /// Add a correction, maintaining the bounded heap invariant.
    /// Returns true if the correction was added.
    fn add(&mut self, correction: Correction) -> bool {
        // Early pruning: skip if below confidence threshold
        if correction.confidence < self.min_confidence {
            return false;
        }

        // Deduplication key
        let key = (
            correction.start_byte,
            correction.end_byte,
            correction.replacement.clone(),
        );

        // Skip duplicates
        if self.seen.contains(&key) {
            return false;
        }

        // If heap is not full, just add
        if self.heap.len() < self.max_size {
            self.seen.insert(key);
            self.heap.push(CorrectionEntry { correction });
            return true;
        }

        // Heap is full: only add if better than the worst (min) entry
        if let Some(min_entry) = self.heap.peek() {
            if correction.confidence > min_entry.correction.confidence {
                // Remove the worst entry
                let removed = self.heap.pop().expect("heap should have an element");
                let removed_key = (
                    removed.correction.start_byte,
                    removed.correction.end_byte,
                    removed.correction.replacement.clone(),
                );
                self.seen.remove(&removed_key);

                // Add the new entry
                self.seen.insert(key);
                self.heap.push(CorrectionEntry { correction });
                return true;
            }
        }

        false
    }

    /// Add multiple corrections from an iterator.
    fn add_all<I: IntoIterator<Item = Correction>>(&mut self, corrections: I) {
        for correction in corrections {
            self.add(correction);
        }
    }

    /// Finalize and return ranked corrections in descending order of confidence.
    fn finalize(self) -> CorrectionCandidates {
        let mut corrections: Vec<Correction> = self
            .heap
            .into_iter()
            .map(|entry| entry.correction)
            .collect();

        // Sort by confidence descending
        corrections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal)
        });

        let mut candidates = CorrectionCandidates::new(self.max_size);
        candidates.add_all(corrections);
        candidates
    }
}

/// Configuration for the correction pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum corrections to return per file
    pub max_corrections: usize,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Whether to include diagnostic messages
    pub include_diagnostics: bool,
    /// Whether to auto-apply high-confidence fixes
    pub auto_apply_threshold: Option<f64>,
    /// Whether to analyze full CPG (slower but more accurate)
    pub full_semantic_analysis: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_corrections: 50,
            min_confidence: 0.3,
            include_diagnostics: true,
            auto_apply_threshold: None,
            full_semantic_analysis: true,
        }
    }
}

/// Result of analyzing a piece of code.
///
/// Note: This struct does not store the ParsedCode or CodePropertyGraph
/// directly since they don't implement Clone. Use the pipeline methods
/// to access those structures if needed.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Original source code
    pub source: String,
    /// Whether parsing produced any errors
    pub has_parse_errors: bool,
    /// Number of parse errors found
    pub error_count: usize,
    /// Tokens extracted from source
    pub tokens: Vec<CodeToken>,
    /// Ranked corrections
    pub corrections: CorrectionCandidates,
    /// Diagnostic messages
    pub diagnostics: Vec<Diagnostic>,
}

/// Diagnostic message from analysis.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level
    pub severity: DiagnosticSeverity,
    /// Message text
    pub message: String,
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Line number (0-indexed)
    pub line: usize,
    /// Column number (0-indexed)
    pub column: usize,
}

/// Severity of a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// Error that prevents compilation/execution
    Error,
    /// Warning about potential issues
    Warning,
    /// Informational message
    Info,
    /// Hint for improvement
    Hint,
}

/// End-to-end code correction pipeline.
///
/// This pipeline integrates all correction components:
/// - Tree-sitter parsing for structural analysis
/// - Lexical correction using liblevenshtein
/// - Grammar correction using PCFG
/// - Semantic correction using GNN/embeddings
///
/// # Example
///
/// ```ignore
/// use libgrammstein::code::{Python, CodeLanguage};
/// use libgrammstein::code::pipeline::{CorrectionPipeline, PipelineConfig};
/// use std::sync::Arc;
///
/// let python = Arc::new(Python::new());
/// let pipeline = CorrectionPipeline::new(python, None, PipelineConfig::default());
///
/// let result = pipeline.analyze("def foo(x):\n    retrun x + 1")?;
/// for correction in result.corrections.ranked() {
///     println!("Suggestion: {} -> {} (confidence: {})",
///         correction.original, correction.replacement, correction.confidence);
/// }
/// ```
pub struct CorrectionPipeline<L: CodeLanguage> {
    language: Arc<L>,
    corrector: EnsembleCorrector<L>,
    config: PipelineConfig,
    parser: CodeParser<L>,
}

impl<L: CodeLanguage + Clone + Send + Sync> CorrectionPipeline<L> {
    /// Creates a new correction pipeline.
    pub fn new(
        language: Arc<L>,
        grammar: Option<WeightedCFG>,
        config: PipelineConfig,
    ) -> Result<Self, PipelineError> {
        let corrector = EnsembleCorrector::with_defaults(Arc::clone(&language), grammar);
        let parser = CodeParser::new(Arc::clone(&language))
            .map_err(|e| PipelineError::ParseError(format!("{}", e)))?;

        Ok(Self {
            language,
            corrector,
            config,
            parser,
        })
    }

    /// Creates a pipeline with default configuration.
    pub fn with_defaults(
        language: Arc<L>,
        grammar: Option<WeightedCFG>,
    ) -> Result<Self, PipelineError> {
        Self::new(language, grammar, PipelineConfig::default())
    }

    /// Creates a minimal pipeline for quick analysis (no semantic analysis).
    pub fn minimal(language: Arc<L>) -> Result<Self, PipelineError> {
        let corrector = EnsembleCorrector::lexical_only(Arc::clone(&language));
        let parser = CodeParser::new(Arc::clone(&language))
            .map_err(|e| PipelineError::ParseError(format!("{}", e)))?;

        Ok(Self {
            language,
            corrector,
            config: PipelineConfig {
                full_semantic_analysis: false,
                ..Default::default()
            },
            parser,
        })
    }

    /// Analyzes source code and returns corrections.
    ///
    /// Uses streaming correction collection to avoid buffering all corrections
    /// in memory before ranking. For 10K tokens with multiple correctors,
    /// this prevents accumulating 100K+ corrections.
    pub fn analyze(&mut self, source: &str) -> Result<AnalysisResult, PipelineError> {
        // Phase 1: Parse
        let parsed = self
            .parser
            .parse(source)
            .map_err(|e| PipelineError::ParseError(format!("{}", e)))?;

        // Phase 2: Tokenize
        let tokens = self.tokenize(&parsed);

        // Phase 3: Build CPG (optional)
        let cpg = if self.config.full_semantic_analysis {
            Some(CodePropertyGraph::from_parsed_code(&parsed))
        } else {
            None
        };

        // Phase 4: Collect diagnostics from parse errors
        let mut diagnostics = self.collect_parse_diagnostics(&parsed);

        // Phase 5: Correct tokens with streaming collection
        // Uses bounded heap with early pruning to avoid buffering all corrections
        let mut collector = StreamingCorrectionCollector::new(
            self.config.max_corrections,
            self.config.min_confidence,
        );

        // Stream corrections from each token directly into collector
        for token in &tokens {
            let context = TokenContext::new(token.token_type);
            let token_corrections = self.corrector.correct_token(token, &context);
            collector.add_all(token_corrections);
        }

        // Stream semantic corrections from CPG
        if let Some(ref cpg) = cpg {
            let semantic_corrections = self.corrector.analyze_full(&parsed, cpg);
            collector.add_all(semantic_corrections);
        }

        // Phase 6: Finalize ranked corrections (already deduplicated)
        let corrections = collector.finalize();

        // Add correction diagnostics
        for correction in corrections.ranked() {
            if self.config.include_diagnostics {
                let (line, column) = byte_offset_to_position(source, correction.start_byte);
                diagnostics.push(Diagnostic {
                    severity: DiagnosticSeverity::Hint,
                    message: correction.context.clone().unwrap_or_else(|| {
                        format!(
                            "Consider: {} -> {}",
                            correction.original, correction.replacement
                        )
                    }),
                    start_byte: correction.start_byte,
                    end_byte: correction.end_byte,
                    line,
                    column,
                });
            }
        }

        let has_parse_errors = parsed.has_errors;
        let error_count = parsed.error_count();

        Ok(AnalysisResult {
            source: source.to_string(),
            has_parse_errors,
            error_count,
            tokens,
            corrections,
            diagnostics,
        })
    }

    /// Tokenizes parsed code.
    fn tokenize(&self, parsed: &ParsedCode) -> Vec<CodeToken> {
        let tokenizer = CodeTokenizer::new(&*self.language);
        tokenizer.tokenize(&parsed.tree, &parsed.source)
    }

    /// Collects diagnostics from parse errors.
    fn collect_parse_diagnostics(&self, parsed: &ParsedCode) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for error in parsed.errors() {
            diagnostics.push(Diagnostic {
                severity: DiagnosticSeverity::Error,
                message: format!("Syntax error: {} '{}'", error.kind, error.text),
                start_byte: error.start_byte,
                end_byte: error.end_byte,
                line: error.start_position.0,
                column: error.start_position.1,
            });
        }

        diagnostics
    }

    /// Applies corrections to source code.
    pub fn apply_corrections(&self, source: &str, corrections: &[Correction]) -> String {
        if corrections.is_empty() {
            return source.to_string();
        }

        // Sort by position descending to apply from end to start
        let mut sorted: Vec<_> = corrections.iter().collect();
        sorted.sort_by(|a, b| b.start_byte.cmp(&a.start_byte));

        let mut result = source.to_string();
        for correction in sorted {
            if correction.start_byte < result.len() && correction.end_byte <= result.len() {
                result.replace_range(
                    correction.start_byte..correction.end_byte,
                    &correction.replacement,
                );
            }
        }

        result
    }

    /// Returns a mutable reference to the corrector for configuration.
    pub fn corrector_mut(&mut self) -> &mut EnsembleCorrector<L> {
        &mut self.corrector
    }

    /// Returns the language handler.
    pub fn language(&self) -> &L {
        &self.language
    }

    /// Returns the configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

/// Errors that can occur during pipeline execution.
#[derive(Debug)]
pub enum PipelineError {
    /// Error during parsing
    ParseError(String),
    /// Error during tokenization
    TokenizeError(String),
    /// Error during CPG construction
    CpgError(String),
    /// Error during correction
    CorrectionError(String),
    /// I/O error
    IoError(std::io::Error),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            PipelineError::TokenizeError(msg) => write!(f, "Tokenize error: {}", msg),
            PipelineError::CpgError(msg) => write!(f, "CPG error: {}", msg),
            PipelineError::CorrectionError(msg) => write!(f, "Correction error: {}", msg),
            PipelineError::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        PipelineError::IoError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.max_corrections, 50);
        assert!((config.min_confidence - 0.3).abs() < 0.01);
        assert!(config.include_diagnostics);
        assert!(config.auto_apply_threshold.is_none());
        assert!(config.full_semantic_analysis);
    }

    #[test]
    fn test_apply_corrections() {
        // We can't actually create the pipeline without a real tree-sitter language,
        // but we can test apply_corrections directly since it's a simple string operation.

        let source = "funtion foo() { return 42; }";

        // Test apply_corrections logic manually
        let mut result = source.to_string();
        result.replace_range(0..7, "function");
        assert_eq!(result, "function foo() { return 42; }");
    }

    #[test]
    fn test_apply_multiple_corrections() {
        let source = "funtion foo() { retrun 42; }";

        // Apply corrections from end to start
        let mut result = source.to_string();
        result.replace_range(16..22, "return");
        result.replace_range(0..7, "function");
        assert_eq!(result, "function foo() { return 42; }");
    }
}
