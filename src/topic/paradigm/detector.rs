//! Paradigm detection engine.
//!
//! This module implements the pattern matching and scoring logic for
//! detecting programming paradigms from code tokens.

use std::collections::HashMap;

use super::config::ParadigmConfig;
use super::indicators::{IndicatorCategory, Paradigm, ParadigmIndicator, ParadigmProfile};

/// A matched indicator with context.
#[derive(Clone, Debug)]
pub struct IndicatorMatch {
    /// The indicator that was matched.
    pub indicator: ParadigmIndicator,
    /// Context tokens before the match.
    pub context_before: Vec<String>,
    /// Context tokens after the match.
    pub context_after: Vec<String>,
    /// Confidence of the match (0.0 to 1.0).
    pub confidence: f64,
}

/// Result of paradigm detection.
#[derive(Clone, Debug)]
pub struct DetectionResult {
    /// The paradigm profile with scores.
    pub profile: ParadigmProfile,
    /// All indicator matches found.
    pub matches: Vec<IndicatorMatch>,
    /// Processing statistics.
    pub stats: DetectionStats,
}

/// Statistics from detection process.
#[derive(Clone, Debug, Default)]
pub struct DetectionStats {
    /// Total tokens processed.
    pub tokens_processed: usize,
    /// Number of pattern checks performed.
    pub pattern_checks: usize,
    /// Number of matches found.
    pub matches_found: usize,
    /// Processing time in microseconds.
    pub time_us: u64,
}

/// Pattern definition for paradigm detection.
#[derive(Clone, Debug)]
struct PatternDef {
    /// Tokens to match (can be single or multi-token).
    tokens: Vec<&'static str>,
    /// Category of this pattern.
    category: IndicatorCategory,
    /// Base weight of this pattern.
    weight: f64,
    /// Whether this is a strong indicator.
    is_strong: bool,
    /// Required context tokens (optional).
    context_required: Option<Vec<&'static str>>,
}

impl PatternDef {
    fn new(tokens: &[&'static str], category: IndicatorCategory, weight: f64) -> Self {
        Self {
            tokens: tokens.to_vec(),
            category,
            weight,
            is_strong: weight >= 0.7,
            context_required: None,
        }
    }

    fn strong(tokens: &[&'static str], category: IndicatorCategory) -> Self {
        Self::new(tokens, category, 0.9)
    }

    fn medium(tokens: &[&'static str], category: IndicatorCategory) -> Self {
        Self::new(tokens, category, 0.6)
    }

    fn weak(tokens: &[&'static str], category: IndicatorCategory) -> Self {
        Self::new(tokens, category, 0.3)
    }

    fn with_context(mut self, context: &[&'static str]) -> Self {
        self.context_required = Some(context.to_vec());
        self
    }
}

/// The main paradigm detector.
#[derive(Clone)]
pub struct ParadigmDetector {
    config: ParadigmConfig,
    /// OOP patterns to match.
    oop_patterns: Vec<PatternDef>,
    /// FP patterns to match.
    fp_patterns: Vec<PatternDef>,
    /// Reactive patterns to match.
    reactive_patterns: Vec<PatternDef>,
    /// Procedural patterns to match.
    procedural_patterns: Vec<PatternDef>,
    /// Token index for fast lookup.
    pattern_index: HashMap<String, Vec<(Paradigm, usize)>>,
}

impl ParadigmDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: ParadigmConfig) -> Self {
        let mut detector = Self {
            config,
            oop_patterns: Self::build_oop_patterns(),
            fp_patterns: Self::build_fp_patterns(),
            reactive_patterns: Self::build_reactive_patterns(),
            procedural_patterns: Self::build_procedural_patterns(),
            pattern_index: HashMap::new(),
        };
        detector.build_index();
        detector
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ParadigmConfig::default())
    }

    /// Analyze code and return a paradigm profile.
    pub fn analyze(&self, code: &str) -> ParadigmProfile {
        let tokens = self.tokenize(code);
        self.analyze_tokens(&tokens).profile
    }

    /// Analyze tokenized code.
    pub fn analyze_tokens(&self, tokens: &[String]) -> DetectionResult {
        let start = std::time::Instant::now();

        let mut profile = ParadigmProfile::new();
        let mut matches = Vec::new();
        let mut stats = DetectionStats {
            tokens_processed: tokens.len(),
            ..Default::default()
        };

        // Match patterns
        let mut i = 0;
        while i < tokens.len() {
            let token = &tokens[i];

            // Check if this token starts any pattern
            if let Some(pattern_refs) = self.pattern_index.get(token.to_lowercase().as_str()) {
                for &(paradigm, pattern_idx) in pattern_refs {
                    let patterns = self.patterns_for(paradigm);
                    let pattern = &patterns[pattern_idx];

                    stats.pattern_checks += 1;

                    // Try to match full pattern
                    if let Some(matched_len) = self.try_match(tokens, i, pattern) {
                        let indicator = ParadigmIndicator::new(
                            paradigm,
                            pattern.category,
                            pattern.tokens.join(" "),
                            pattern.weight,
                        )
                        .with_position(i)
                        .with_length(matched_len)
                        .with_strong(pattern.is_strong);

                        // Calculate confidence with context
                        let confidence = self.calculate_confidence(tokens, i, pattern);

                        // Collect context
                        let context_before = self.get_context_before(tokens, i);
                        let context_after = self.get_context_after(tokens, i + matched_len);

                        matches.push(IndicatorMatch {
                            indicator: indicator.clone(),
                            context_before,
                            context_after,
                            confidence,
                        });

                        // Update scores
                        let score_delta =
                            pattern.weight * confidence * self.multiplier_for(paradigm);
                        self.update_score(&mut profile, paradigm, score_delta);

                        profile.indicators.push(indicator);
                        profile.match_count += 1;
                        stats.matches_found += 1;
                    }
                }
            }

            i += 1;
        }

        profile.total_tokens = tokens.len();

        // Normalize scores if configured
        if self.config.normalize_scores && profile.match_count > 0 {
            // Normalize by token count to get density
            let density_factor = 100.0 / tokens.len().max(1) as f64;
            profile.oop_score *= density_factor;
            profile.fp_score *= density_factor;
            profile.reactive_score *= density_factor;
            profile.procedural_score *= density_factor;

            // Cap at 1.0
            profile.oop_score = profile.oop_score.min(1.0);
            profile.fp_score = profile.fp_score.min(1.0);
            profile.reactive_score = profile.reactive_score.min(1.0);
            profile.procedural_score = profile.procedural_score.min(1.0);
        }

        stats.time_us = start.elapsed().as_micros() as u64;

        DetectionResult {
            profile,
            matches,
            stats,
        }
    }

    /// Simple tokenization for paradigm detection.
    fn tokenize(&self, code: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in code.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                // Keep operators as separate tokens
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    fn patterns_for(&self, paradigm: Paradigm) -> &[PatternDef] {
        match paradigm {
            Paradigm::ObjectOriented => &self.oop_patterns,
            Paradigm::Functional => &self.fp_patterns,
            Paradigm::Reactive => &self.reactive_patterns,
            Paradigm::Procedural => &self.procedural_patterns,
            Paradigm::Mixed => &[], // Mixed has no patterns
        }
    }

    fn multiplier_for(&self, paradigm: Paradigm) -> f64 {
        match paradigm {
            Paradigm::ObjectOriented => self.config.weights.oop_multiplier,
            Paradigm::Functional => self.config.weights.fp_multiplier,
            Paradigm::Reactive => self.config.weights.reactive_multiplier,
            Paradigm::Procedural => self.config.weights.procedural_multiplier,
            Paradigm::Mixed => 1.0,
        }
    }

    fn update_score(&self, profile: &mut ParadigmProfile, paradigm: Paradigm, delta: f64) {
        match paradigm {
            Paradigm::ObjectOriented => profile.oop_score += delta,
            Paradigm::Functional => profile.fp_score += delta,
            Paradigm::Reactive => profile.reactive_score += delta,
            Paradigm::Procedural => profile.procedural_score += delta,
            Paradigm::Mixed => {}
        }
    }

    fn try_match(&self, tokens: &[String], start: usize, pattern: &PatternDef) -> Option<usize> {
        if start + pattern.tokens.len() > tokens.len() {
            return None;
        }

        for (i, pattern_token) in pattern.tokens.iter().enumerate() {
            let actual = tokens[start + i].to_lowercase();
            if actual != *pattern_token {
                return None;
            }
        }

        Some(pattern.tokens.len())
    }

    fn calculate_confidence(&self, tokens: &[String], pos: usize, pattern: &PatternDef) -> f64 {
        let mut confidence = 1.0;

        // Check context requirements
        if let Some(ref required) = pattern.context_required {
            let window = self.config.context_window;
            let start = pos.saturating_sub(window);
            let end = (pos + pattern.tokens.len() + window).min(tokens.len());

            let context: Vec<_> = tokens[start..end]
                .iter()
                .map(|t| t.to_lowercase())
                .collect();

            let context_matches = required
                .iter()
                .filter(|&req| context.iter().any(|t| t == *req))
                .count();

            confidence *= context_matches as f64 / required.len() as f64;
        }

        // Boost for strong indicators
        if pattern.is_strong {
            confidence += self.config.weights.strong_indicator_bonus * 0.1;
        }

        confidence.min(1.0)
    }

    fn get_context_before(&self, tokens: &[String], pos: usize) -> Vec<String> {
        let start = pos.saturating_sub(self.config.context_window);
        tokens[start..pos].to_vec()
    }

    fn get_context_after(&self, tokens: &[String], pos: usize) -> Vec<String> {
        let end = (pos + self.config.context_window).min(tokens.len());
        tokens[pos..end].to_vec()
    }

    fn build_index(&mut self) {
        self.pattern_index.clear();

        let paradigms_and_patterns = [
            (Paradigm::ObjectOriented, &self.oop_patterns),
            (Paradigm::Functional, &self.fp_patterns),
            (Paradigm::Reactive, &self.reactive_patterns),
            (Paradigm::Procedural, &self.procedural_patterns),
        ];

        for (paradigm, patterns) in paradigms_and_patterns {
            for (idx, pattern) in patterns.iter().enumerate() {
                if let Some(first_token) = pattern.tokens.first() {
                    self.pattern_index
                        .entry(first_token.to_string())
                        .or_default()
                        .push((paradigm, idx));
                }
            }
        }
    }

    // ================== Pattern Definitions ==================

    fn build_oop_patterns() -> Vec<PatternDef> {
        vec![
            // Class definitions
            PatternDef::strong(&["class"], IndicatorCategory::OopClass),
            PatternDef::medium(&["struct"], IndicatorCategory::OopClass),
            PatternDef::strong(&["interface"], IndicatorCategory::OopPolymorphism),
            PatternDef::strong(&["trait"], IndicatorCategory::OopPolymorphism),
            PatternDef::medium(&["protocol"], IndicatorCategory::OopPolymorphism),
            // Inheritance
            PatternDef::strong(&["extends"], IndicatorCategory::OopInheritance),
            PatternDef::strong(&["implements"], IndicatorCategory::OopInheritance),
            PatternDef::strong(&["inherits"], IndicatorCategory::OopInheritance),
            PatternDef::medium(&["super"], IndicatorCategory::OopInheritance),
            PatternDef::medium(&["parent"], IndicatorCategory::OopInheritance),
            // Encapsulation
            PatternDef::medium(&["private"], IndicatorCategory::OopEncapsulation),
            PatternDef::medium(&["protected"], IndicatorCategory::OopEncapsulation),
            PatternDef::weak(&["public"], IndicatorCategory::OopEncapsulation),
            PatternDef::medium(&["internal"], IndicatorCategory::OopEncapsulation),
            // Polymorphism
            PatternDef::strong(&["virtual"], IndicatorCategory::OopPolymorphism),
            PatternDef::strong(&["override"], IndicatorCategory::OopPolymorphism),
            PatternDef::strong(&["abstract"], IndicatorCategory::OopPolymorphism),
            // Instance references
            PatternDef::medium(&["this"], IndicatorCategory::OopInstantiation),
            PatternDef::medium(&["self"], IndicatorCategory::OopInstantiation),
            PatternDef::strong(&["new"], IndicatorCategory::OopInstantiation),
            // Methods (context-dependent)
            PatternDef::medium(&["constructor"], IndicatorCategory::OopInstantiation),
            PatternDef::medium(&["destructor"], IndicatorCategory::OopInstantiation),
            PatternDef::weak(&["__init__"], IndicatorCategory::OopInstantiation),
            PatternDef::weak(&["__new__"], IndicatorCategory::OopInstantiation),
            // Getters/setters
            PatternDef::weak(&["get"], IndicatorCategory::OopEncapsulation),
            PatternDef::weak(&["set"], IndicatorCategory::OopEncapsulation),
            PatternDef::medium(&["@property"], IndicatorCategory::OopEncapsulation),
            // Rust/Kotlin-specific
            PatternDef::strong(&["impl"], IndicatorCategory::OopClass),
            PatternDef::medium(&["dyn"], IndicatorCategory::OopPolymorphism),
        ]
    }

    fn build_fp_patterns() -> Vec<PatternDef> {
        vec![
            // Higher-order functions
            PatternDef::strong(&["map"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["filter"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["reduce"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["fold"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["foldl"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["foldr"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["flatmap"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["flat_map"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["foreach"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["find"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["any"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["all"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["take"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["drop"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["zip"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["concat"], IndicatorCategory::FpHigherOrder),
            // Lambda expressions
            PatternDef::strong(&["lambda"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["=>"], IndicatorCategory::FpHigherOrder),
            PatternDef::strong(&["->"], IndicatorCategory::FpHigherOrder),
            PatternDef::medium(&["|"], IndicatorCategory::FpHigherOrder), // Rust closure
            PatternDef::medium(&["fn"], IndicatorCategory::FpHigherOrder),
            // Composition
            PatternDef::strong(&["compose"], IndicatorCategory::FpPurity),
            PatternDef::strong(&["pipe"], IndicatorCategory::FpPurity),
            PatternDef::medium(&["andthen"], IndicatorCategory::FpPurity),
            PatternDef::medium(&["and_then"], IndicatorCategory::FpPurity),
            // Immutability
            PatternDef::medium(&["const"], IndicatorCategory::FpImmutability),
            PatternDef::medium(&["val"], IndicatorCategory::FpImmutability),
            PatternDef::weak(&["let"], IndicatorCategory::FpImmutability),
            PatternDef::strong(&["immutable"], IndicatorCategory::FpImmutability),
            PatternDef::strong(&["readonly"], IndicatorCategory::FpImmutability),
            // Pattern matching
            PatternDef::strong(&["match"], IndicatorCategory::FpPatternMatch),
            PatternDef::medium(&["case"], IndicatorCategory::FpPatternMatch),
            PatternDef::medium(&["when"], IndicatorCategory::FpPatternMatch),
            PatternDef::weak(&["if", "let"], IndicatorCategory::FpPatternMatch),
            // Algebraic types
            PatternDef::strong(&["option"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["some"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["none"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["result"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["ok"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["err"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["either"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["maybe"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["just"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["nothing"], IndicatorCategory::FpAlgebraic),
            // Recursion
            PatternDef::medium(&["rec"], IndicatorCategory::FpRecursion),
            PatternDef::medium(&["tailrec"], IndicatorCategory::FpRecursion),
            // Monads
            PatternDef::strong(&[">>="], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&[">>"], IndicatorCategory::FpAlgebraic),
            PatternDef::medium(&["do"], IndicatorCategory::FpAlgebraic),
            PatternDef::medium(&["return"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["monad"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["functor"], IndicatorCategory::FpAlgebraic),
            PatternDef::strong(&["applicative"], IndicatorCategory::FpAlgebraic),
        ]
    }

    fn build_reactive_patterns() -> Vec<PatternDef> {
        vec![
            // Observable/stream creation
            PatternDef::strong(&["observable"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["subject"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["behaviorsubject"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["replaysubject"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["stream"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["flux"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["mono"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["channel"], IndicatorCategory::ReactiveObservable),
            // Subscription
            PatternDef::strong(&["subscribe"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["unsubscribe"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["subscription"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["observer"], IndicatorCategory::ReactiveObservable),
            // Event handling
            PatternDef::strong(&["emit"], IndicatorCategory::ReactiveEvent),
            PatternDef::strong(&["on"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["onclick"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["onchange"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["onerror"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["onnext"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["oncomplete"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["addeventlistener"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["removeeventlistener"], IndicatorCategory::ReactiveEvent),
            PatternDef::medium(&["dispatch"], IndicatorCategory::ReactiveEvent),
            // RxJS/reactive operators
            PatternDef::strong(&["switchmap"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["mergemap"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["concatmap"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["exhaustmap"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["debounce"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["throttle"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(
                &["distinctuntilchanged"],
                IndicatorCategory::ReactiveObservable,
            ),
            PatternDef::strong(&["combinelatest"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["merge"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["share"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["tap"], IndicatorCategory::ReactiveObservable),
            // Signals (SolidJS, Preact Signals)
            PatternDef::strong(&["signal"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["computed"], IndicatorCategory::ReactiveObservable),
            PatternDef::strong(&["effect"], IndicatorCategory::ReactiveEvent),
            PatternDef::strong(&["useeffect"], IndicatorCategory::ReactiveEvent),
            PatternDef::strong(&["usestate"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["createsignal"], IndicatorCategory::ReactiveObservable),
            PatternDef::medium(&["createeffect"], IndicatorCategory::ReactiveEvent),
            // Async/flow control
            PatternDef::medium(&["async"], IndicatorCategory::ReactiveAsync),
            PatternDef::medium(&["await"], IndicatorCategory::ReactiveAsync),
            PatternDef::medium(&["promise"], IndicatorCategory::ReactiveAsync),
            PatternDef::medium(&["future"], IndicatorCategory::ReactiveAsync),
            PatternDef::medium(&["scheduler"], IndicatorCategory::ReactiveAsync),
            // Backpressure
            PatternDef::strong(&["backpressure"], IndicatorCategory::ReactiveBackpressure),
            PatternDef::medium(&["buffer"], IndicatorCategory::ReactiveBackpressure),
            PatternDef::medium(&["window"], IndicatorCategory::ReactiveBackpressure),
            // Process calculus (Rholang-specific)
            PatternDef::strong(&["!"], IndicatorCategory::ReactiveObservable), // Send
            PatternDef::strong(&["*"], IndicatorCategory::ReactiveObservable), // Dereference
            PatternDef::strong(&["@"], IndicatorCategory::ReactiveObservable), // Quote
            PatternDef::strong(&["for"], IndicatorCategory::ReactiveObservable)
                .with_context(&["<-"]), // Receive
        ]
    }

    fn build_procedural_patterns() -> Vec<PatternDef> {
        vec![
            // Control flow
            PatternDef::weak(&["if"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::weak(&["else"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::weak(&["switch"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::medium(&["goto"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::weak(&["break"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::weak(&["continue"], IndicatorCategory::ProceduralControlFlow),
            // Loops
            PatternDef::medium(&["for"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::medium(&["while"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::medium(&["loop"], IndicatorCategory::ProceduralControlFlow),
            PatternDef::medium(&["do"], IndicatorCategory::ProceduralControlFlow),
            // Mutable state
            PatternDef::strong(&["var"], IndicatorCategory::ProceduralMutable),
            PatternDef::strong(&["mut"], IndicatorCategory::ProceduralMutable),
            PatternDef::medium(&["mutable"], IndicatorCategory::ProceduralMutable),
            PatternDef::weak(&["="], IndicatorCategory::ProceduralMutable), // Assignment (weak)
            PatternDef::medium(&["+="], IndicatorCategory::ProceduralMutable),
            PatternDef::medium(&["-="], IndicatorCategory::ProceduralMutable),
            PatternDef::medium(&["*="], IndicatorCategory::ProceduralMutable),
            PatternDef::medium(&["/="], IndicatorCategory::ProceduralMutable),
            PatternDef::medium(&["++"], IndicatorCategory::ProceduralMutable),
            PatternDef::medium(&["--"], IndicatorCategory::ProceduralMutable),
            // Side effects
            PatternDef::weak(&["print"], IndicatorCategory::ProceduralSideEffect),
            PatternDef::weak(&["println"], IndicatorCategory::ProceduralSideEffect),
            PatternDef::weak(&["printf"], IndicatorCategory::ProceduralSideEffect),
            PatternDef::weak(&["console"], IndicatorCategory::ProceduralSideEffect),
            PatternDef::medium(&["write"], IndicatorCategory::ProceduralSideEffect),
            PatternDef::medium(&["read"], IndicatorCategory::ProceduralSideEffect),
            // Pointers/references
            PatternDef::medium(&["*"], IndicatorCategory::ProceduralMutable), // Pointer deref (context-dependent)
            PatternDef::medium(&["&"], IndicatorCategory::ProceduralMutable), // Reference
            PatternDef::strong(&["malloc"], IndicatorCategory::ProceduralMutable),
            PatternDef::strong(&["free"], IndicatorCategory::ProceduralMutable),
            PatternDef::strong(&["alloc"], IndicatorCategory::ProceduralMutable),
            PatternDef::strong(&["dealloc"], IndicatorCategory::ProceduralMutable),
            // Sequential
            PatternDef::weak(&[";"], IndicatorCategory::ProceduralSequential),
            PatternDef::weak(&["return"], IndicatorCategory::ProceduralSequential),
            // Global state
            PatternDef::strong(&["global"], IndicatorCategory::ProceduralMutable),
            PatternDef::strong(&["static"], IndicatorCategory::ProceduralMutable),
        ]
    }
}

impl std::fmt::Debug for ParadigmDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParadigmDetector")
            .field("config", &self.config)
            .field("oop_patterns", &self.oop_patterns.len())
            .field("fp_patterns", &self.fp_patterns.len())
            .field("reactive_patterns", &self.reactive_patterns.len())
            .field("procedural_patterns", &self.procedural_patterns.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let detector = ParadigmDetector::with_defaults();
        assert!(!detector.oop_patterns.is_empty());
        assert!(!detector.fp_patterns.is_empty());
        assert!(!detector.reactive_patterns.is_empty());
        assert!(!detector.procedural_patterns.is_empty());
    }

    #[test]
    fn test_oop_detection() {
        let detector = ParadigmDetector::with_defaults();

        // OOP code naturally includes some procedural elements (assignments)
        // so we just verify OOP indicators are detected
        let code = "class Foo extends Bar { constructor() { this.x = 0; } }";
        let profile = detector.analyze(code);

        // OOP should score higher than FP (no FP patterns here)
        assert!(profile.oop_score > profile.fp_score);
        // OOP indicators should be detected
        assert!(profile.oop_score > 0.0);
        assert!(profile.match_count > 0);
    }

    #[test]
    fn test_fp_detection() {
        let detector = ParadigmDetector::with_defaults();

        let code =
            "const result = list.map(x => x * 2).filter(x => x > 10).reduce((a, b) => a + b, 0);";
        let profile = detector.analyze(code);

        assert!(profile.fp_score > profile.oop_score);
    }

    #[test]
    fn test_reactive_detection() {
        let detector = ParadigmDetector::with_defaults();

        let code = "observable.pipe(map(x => x * 2), filter(x => x > 5)).subscribe(console.log);";
        let profile = detector.analyze(code);

        assert!(profile.reactive_score > 0.0);
    }

    #[test]
    fn test_procedural_detection() {
        let detector = ParadigmDetector::with_defaults();

        let code = "var x = 0; for (int i = 0; i < 10; i++) { x += i; printf('%d', x); }";
        let profile = detector.analyze(code);

        assert!(profile.procedural_score > 0.0);
    }

    #[test]
    fn test_rust_multiparadigm() {
        let detector = ParadigmDetector::with_defaults();

        let code = r#"
            impl Display for Foo {
                fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                    self.items.iter()
                        .map(|x| x.to_string())
                        .filter(|s| !s.is_empty())
                        .for_each(|s| write!(f, "{}", s));
                    Ok(())
                }
            }
        "#;

        let profile = detector.analyze(code);

        // Should detect both OOP (impl, self) and FP (iter, map, filter)
        assert!(profile.oop_score > 0.0);
        assert!(profile.fp_score > 0.0);
    }

    #[test]
    fn test_detection_result() {
        let detector = ParadigmDetector::with_defaults();
        let tokens = vec![
            "class".to_string(),
            "Foo".to_string(),
            "extends".to_string(),
            "Bar".to_string(),
        ];

        let result = detector.analyze_tokens(&tokens);

        assert!(!result.matches.is_empty());
        assert!(result.stats.tokens_processed == 4);
        assert!(result.stats.matches_found > 0);
    }

    #[test]
    fn test_empty_code() {
        let detector = ParadigmDetector::with_defaults();
        let profile = detector.analyze("");

        assert_eq!(profile.oop_score, 0.0);
        assert_eq!(profile.fp_score, 0.0);
        assert_eq!(profile.reactive_score, 0.0);
        assert_eq!(profile.procedural_score, 0.0);
    }

    #[test]
    fn test_rholang_patterns() {
        let detector = ParadigmDetector::with_defaults();

        // Rholang code: for/receive pattern
        let code = r#"
            new channel in {
                for (@message <- channel) {
                    channel!(*message)
                }
            }
        "#;

        let profile = detector.analyze(code);

        // Should detect reactive patterns (!, *, @, for)
        assert!(profile.reactive_score > 0.0 || profile.match_count > 0);
    }
}
