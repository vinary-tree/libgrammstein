//! Code correction and repair types.
//!
//! This module defines the types for representing and applying corrections
//! to source code, integrating with liblevenshtein for fuzzy matching.

use super::language::TokenContext;
use super::tokenizer::CodeToken;

/// A suggested correction for source code.
#[derive(Debug, Clone)]
pub struct Correction {
    /// The kind of correction
    pub kind: CorrectionKind,
    /// Start byte offset in the source
    pub start_byte: usize,
    /// End byte offset in the source
    pub end_byte: usize,
    /// The original text
    pub original: String,
    /// The suggested replacement text
    pub replacement: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Source of the correction (which component suggested it)
    pub source: CorrectionSource,
    /// Additional context about the correction
    pub context: Option<String>,
}

impl Correction {
    /// Creates a new correction.
    pub fn new(
        kind: CorrectionKind,
        start_byte: usize,
        end_byte: usize,
        original: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            start_byte,
            end_byte,
            original: original.into(),
            replacement: replacement.into(),
            confidence: 1.0,
            source: CorrectionSource::Unknown,
            context: None,
        }
    }

    /// Sets the confidence score.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Sets the correction source.
    pub fn with_source(mut self, source: CorrectionSource) -> Self {
        self.source = source;
        self
    }

    /// Sets the context message.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Returns the edit distance of this correction.
    pub fn edit_distance(&self) -> usize {
        // Simple approximation - actual distance would use liblevenshtein
        let len_diff =
            (self.original.len() as isize - self.replacement.len() as isize).unsigned_abs();
        if self.original == self.replacement {
            0
        } else if self.original.is_empty() || self.replacement.is_empty() {
            self.original.len().max(self.replacement.len())
        } else {
            // Rough estimate
            len_diff + 1
        }
    }

    /// Returns true if this is a no-op correction (original == replacement).
    pub fn is_noop(&self) -> bool {
        self.original == self.replacement
    }

    /// Applies this correction to source code.
    pub fn apply(&self, source: &str) -> String {
        let mut result = String::with_capacity(source.len() + self.replacement.len());
        result.push_str(&source[..self.start_byte]);
        result.push_str(&self.replacement);
        result.push_str(&source[self.end_byte..]);
        result
    }
}

/// Classification of corrections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorrectionKind {
    /// Spelling/typo in a token
    Spelling,
    /// Missing token
    Insertion,
    /// Extra token that should be removed
    Deletion,
    /// Wrong token (semantic error)
    Replacement,
    /// Wrong variable name
    VariableMisuse,
    /// Wrong type
    TypeError,
    /// Missing import/use
    MissingImport,
    /// Syntax error (general)
    SyntaxError,
    /// Formatting/whitespace
    Formatting,
    /// Other correction type
    Other,
}

impl CorrectionKind {
    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            CorrectionKind::Spelling => "Spelling correction",
            CorrectionKind::Insertion => "Insert missing token",
            CorrectionKind::Deletion => "Remove extra token",
            CorrectionKind::Replacement => "Replace token",
            CorrectionKind::VariableMisuse => "Wrong variable name",
            CorrectionKind::TypeError => "Type error",
            CorrectionKind::MissingImport => "Missing import",
            CorrectionKind::SyntaxError => "Syntax error",
            CorrectionKind::Formatting => "Formatting",
            CorrectionKind::Other => "Other correction",
        }
    }

    /// Returns whether this is a semantic correction (vs. syntactic).
    pub fn is_semantic(&self) -> bool {
        matches!(
            self,
            CorrectionKind::VariableMisuse
                | CorrectionKind::TypeError
                | CorrectionKind::MissingImport
        )
    }
}

/// Source/origin of a correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorrectionSource {
    /// From lexical fuzzy matching (liblevenshtein)
    Lexical,
    /// From grammar constraints (PCFG)
    Grammar,
    /// From neural model (UniXcoder/GNN)
    Neural,
    /// From type inference
    TypeInference,
    /// From control flow analysis
    ControlFlow,
    /// From data flow analysis
    DataFlow,
    /// Combined/ensemble
    Combined,
    /// Unknown source
    Unknown,
}

/// Trait for components that can suggest corrections.
pub trait CodeCorrector: Send + Sync {
    /// Suggests corrections for a token.
    fn correct_token(&self, token: &CodeToken, context: &TokenContext) -> Vec<Correction>;

    /// Suggests corrections for a range of source code.
    fn correct_range(&self, source: &str, start_byte: usize, end_byte: usize) -> Vec<Correction>;

    /// Returns the maximum edit distance this corrector considers.
    fn max_edit_distance(&self) -> usize {
        2
    }

    /// Returns the name of this corrector for identification.
    fn name(&self) -> &str;
}

/// A ranked list of corrections.
#[derive(Debug, Clone)]
pub struct CorrectionCandidates {
    corrections: Vec<Correction>,
    /// Maximum number of candidates to keep
    max_candidates: usize,
}

impl CorrectionCandidates {
    /// Creates a new empty candidate list.
    pub fn new(max_candidates: usize) -> Self {
        Self {
            corrections: Vec::new(),
            max_candidates,
        }
    }

    /// Adds a correction candidate.
    pub fn add(&mut self, correction: Correction) {
        self.corrections.push(correction);

        // Sort by confidence (descending) and truncate. Treat NaN confidences
        // as equal so a single bogus score from a neural scorer can't panic
        // the entire correction pipeline.
        self.corrections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if self.corrections.len() > self.max_candidates {
            self.corrections.truncate(self.max_candidates);
        }
    }

    /// Adds multiple correction candidates.
    pub fn add_all(&mut self, corrections: impl IntoIterator<Item = Correction>) {
        for c in corrections {
            self.add(c);
        }
    }

    /// Returns the best correction.
    pub fn best(&self) -> Option<&Correction> {
        self.corrections.first()
    }

    /// Returns all corrections ranked by confidence.
    pub fn ranked(&self) -> &[Correction] {
        &self.corrections
    }

    /// Returns the number of candidates.
    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    /// Returns true if there are no candidates.
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    /// Filters corrections by minimum confidence.
    pub fn filter_by_confidence(&mut self, min_confidence: f64) {
        self.corrections.retain(|c| c.confidence >= min_confidence);
    }

    /// Filters corrections by kind.
    pub fn filter_by_kind(&mut self, kind: CorrectionKind) {
        self.corrections.retain(|c| c.kind == kind);
    }

    /// Filters corrections by source.
    pub fn filter_by_source(&mut self, source: CorrectionSource) {
        self.corrections.retain(|c| c.source == source);
    }
}

impl Default for CorrectionCandidates {
    fn default() -> Self {
        Self::new(10)
    }
}

impl IntoIterator for CorrectionCandidates {
    type Item = Correction;
    type IntoIter = std::vec::IntoIter<Correction>;

    fn into_iter(self) -> Self::IntoIter {
        self.corrections.into_iter()
    }
}

impl<'a> IntoIterator for &'a CorrectionCandidates {
    type Item = &'a Correction;
    type IntoIter = std::slice::Iter<'a, Correction>;

    fn into_iter(self) -> Self::IntoIter {
        self.corrections.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction_apply() {
        let source = "pritn(\"hello\")";
        let correction = Correction::new(CorrectionKind::Spelling, 0, 5, "pritn", "print");

        let result = correction.apply(source);
        assert_eq!(result, "print(\"hello\")");
    }

    #[test]
    fn test_correction_candidates_ranking() {
        let mut candidates = CorrectionCandidates::new(3);

        candidates.add(
            Correction::new(CorrectionKind::Spelling, 0, 5, "pritn", "print").with_confidence(0.9),
        );
        candidates.add(
            Correction::new(CorrectionKind::Spelling, 0, 5, "pritn", "prion").with_confidence(0.5),
        );
        candidates.add(
            Correction::new(CorrectionKind::Spelling, 0, 5, "pritn", "pint").with_confidence(0.7),
        );

        // Should be sorted by confidence
        let ranked = candidates.ranked();
        assert_eq!(ranked[0].replacement, "print");
        assert_eq!(ranked[1].replacement, "pint");
        assert_eq!(ranked[2].replacement, "prion");
    }

    #[test]
    fn test_correction_candidates_max() {
        let mut candidates = CorrectionCandidates::new(2);

        for i in 0..5 {
            candidates.add(
                Correction::new(CorrectionKind::Spelling, 0, 5, "test", format!("test{}", i))
                    .with_confidence(i as f64 / 10.0),
            );
        }

        // Should only keep top 2
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_correction_kind_semantic() {
        assert!(!CorrectionKind::Spelling.is_semantic());
        assert!(!CorrectionKind::SyntaxError.is_semantic());
        assert!(CorrectionKind::VariableMisuse.is_semantic());
        assert!(CorrectionKind::TypeError.is_semantic());
    }
}
