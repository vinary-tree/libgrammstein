//! Configuration for paradigm detection.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for paradigm detection.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParadigmConfig {
    /// Weights for different paradigm indicators.
    pub weights: ParadigmWeights,
    /// Language-specific hints to adjust detection.
    pub language_hints: LanguageHints,
    /// Minimum score threshold to report a paradigm as present.
    pub min_score_threshold: f64,
    /// Whether to normalize scores to sum to 1.0.
    pub normalize_scores: bool,
    /// Maximum number of indicators to track per paradigm.
    pub max_indicators_per_paradigm: usize,
    /// Whether to include position information in indicators.
    pub track_positions: bool,
    /// Window size for context-aware matching (tokens).
    pub context_window: usize,
    /// Enable parallel processing for large inputs.
    pub parallel: bool,
}

impl Default for ParadigmConfig {
    fn default() -> Self {
        Self {
            weights: ParadigmWeights::default(),
            language_hints: LanguageHints::default(),
            min_score_threshold: 0.1,
            normalize_scores: true,
            max_indicators_per_paradigm: 100,
            track_positions: true,
            context_window: 10,
            parallel: true,
        }
    }
}

impl ParadigmConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure for OOP-heavy detection (bias toward OOP patterns).
    pub fn for_oop_detection() -> Self {
        Self {
            weights: ParadigmWeights {
                oop_multiplier: 1.2,
                fp_multiplier: 0.8,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Configure for FP-heavy detection.
    pub fn for_fp_detection() -> Self {
        Self {
            weights: ParadigmWeights {
                fp_multiplier: 1.2,
                oop_multiplier: 0.8,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Configure for reactive pattern detection.
    pub fn for_reactive_detection() -> Self {
        Self {
            weights: ParadigmWeights {
                reactive_multiplier: 1.2,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Configure for balanced multi-paradigm detection.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Configure for quick scanning (less detailed).
    pub fn quick_scan() -> Self {
        Self {
            max_indicators_per_paradigm: 20,
            track_positions: false,
            context_window: 5,
            ..Default::default()
        }
    }

    /// Set custom weights.
    pub fn with_weights(mut self, weights: ParadigmWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set language hints.
    pub fn with_language_hints(mut self, hints: LanguageHints) -> Self {
        self.language_hints = hints;
        self
    }

    /// Set minimum score threshold.
    pub fn with_min_threshold(mut self, threshold: f64) -> Self {
        self.min_score_threshold = threshold;
        self
    }

    /// Enable/disable score normalization.
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize_scores = normalize;
        self
    }

    /// Set context window size.
    pub fn with_context_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }

    /// Enable/disable parallel processing.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Weight multipliers for paradigm scores.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParadigmWeights {
    /// Multiplier for OOP indicators.
    pub oop_multiplier: f64,
    /// Multiplier for FP indicators.
    pub fp_multiplier: f64,
    /// Multiplier for reactive indicators.
    pub reactive_multiplier: f64,
    /// Multiplier for procedural indicators.
    pub procedural_multiplier: f64,
    /// Weight for strong indicators.
    pub strong_indicator_bonus: f64,
    /// Weight for context-confirmed indicators.
    pub context_bonus: f64,
    /// Penalty for conflicting indicators.
    pub conflict_penalty: f64,
}

impl Default for ParadigmWeights {
    fn default() -> Self {
        Self {
            oop_multiplier: 1.0,
            fp_multiplier: 1.0,
            reactive_multiplier: 1.0,
            procedural_multiplier: 1.0,
            strong_indicator_bonus: 0.5,
            context_bonus: 0.3,
            conflict_penalty: 0.2,
        }
    }
}

impl ParadigmWeights {
    /// Create balanced weights.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create weights favoring OOP.
    pub fn favor_oop() -> Self {
        Self {
            oop_multiplier: 1.5,
            ..Default::default()
        }
    }

    /// Create weights favoring FP.
    pub fn favor_fp() -> Self {
        Self {
            fp_multiplier: 1.5,
            ..Default::default()
        }
    }
}

/// Language-specific hints for paradigm detection.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LanguageHints {
    /// The programming language (optional).
    pub language: Option<String>,
    /// Whether the language is primarily OOP.
    pub is_oop_language: bool,
    /// Whether the language is primarily FP.
    pub is_fp_language: bool,
    /// Whether the language supports multiple paradigms.
    pub is_multi_paradigm: bool,
    /// Custom keywords to treat as OOP indicators.
    pub custom_oop_keywords: Vec<String>,
    /// Custom keywords to treat as FP indicators.
    pub custom_fp_keywords: Vec<String>,
    /// Custom keywords to treat as reactive indicators.
    pub custom_reactive_keywords: Vec<String>,
    /// Keywords to ignore (language-specific noise).
    pub ignore_keywords: Vec<String>,
}

impl LanguageHints {
    /// Create hints for unknown language.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Create hints for Python.
    pub fn python() -> Self {
        Self {
            language: Some("python".into()),
            is_multi_paradigm: true,
            custom_oop_keywords: vec!["__init__".into(), "self".into(), "@property".into()],
            custom_fp_keywords: vec!["lambda".into(), "@functools".into()],
            ignore_keywords: vec!["import".into(), "from".into()],
            ..Default::default()
        }
    }

    /// Create hints for Rust.
    pub fn rust() -> Self {
        Self {
            language: Some("rust".into()),
            is_multi_paradigm: true,
            custom_oop_keywords: vec!["impl".into(), "self".into(), "Self".into()],
            custom_fp_keywords: vec!["iter".into(), "map".into(), "filter".into(), "fold".into(), "|".into()],
            ignore_keywords: vec!["use".into(), "mod".into(), "pub".into()],
            ..Default::default()
        }
    }

    /// Create hints for JavaScript/TypeScript.
    pub fn javascript() -> Self {
        Self {
            language: Some("javascript".into()),
            is_multi_paradigm: true,
            custom_oop_keywords: vec!["class".into(), "extends".into(), "this".into(), "new".into()],
            custom_fp_keywords: vec!["=>".into(), "map".into(), "filter".into(), "reduce".into()],
            custom_reactive_keywords: vec!["Observable".into(), "subscribe".into(), "pipe".into()],
            ignore_keywords: vec!["import".into(), "export".into(), "require".into()],
            ..Default::default()
        }
    }

    /// Create hints for Java.
    pub fn java() -> Self {
        Self {
            language: Some("java".into()),
            is_oop_language: true,
            custom_oop_keywords: vec!["class".into(), "extends".into(), "implements".into(), "new".into()],
            custom_fp_keywords: vec!["stream".into(), "->".into()],
            custom_reactive_keywords: vec!["Flux".into(), "Mono".into(), "Observable".into()],
            ignore_keywords: vec!["import".into(), "package".into()],
            ..Default::default()
        }
    }

    /// Create hints for Haskell.
    pub fn haskell() -> Self {
        Self {
            language: Some("haskell".into()),
            is_fp_language: true,
            custom_fp_keywords: vec!["where".into(), "let".into(), "in".into(), "do".into(), ">>=".into()],
            ignore_keywords: vec!["import".into(), "module".into()],
            ..Default::default()
        }
    }

    /// Create hints for Rholang.
    pub fn rholang() -> Self {
        Self {
            language: Some("rholang".into()),
            is_multi_paradigm: true,
            // Rholang is a concurrent process calculus
            custom_oop_keywords: vec!["contract".into()],
            custom_fp_keywords: vec!["match".into()],
            custom_reactive_keywords: vec!["for".into(), "!".into(), "*".into(), "@".into()],
            ignore_keywords: vec!["new".into()], // 'new' in Rholang creates channels, not objects
            ..Default::default()
        }
    }

    /// Create hints for MeTTa.
    pub fn metta() -> Self {
        Self {
            language: Some("metta".into()),
            is_fp_language: true, // MeTTa is declarative/functional
            custom_fp_keywords: vec!["match".into(), "=".into(), "->".into()],
            ignore_keywords: vec!["!".into()], // Command prefix
            ..Default::default()
        }
    }

    /// Set the language name.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Add custom OOP keywords.
    pub fn with_oop_keywords(mut self, keywords: Vec<String>) -> Self {
        self.custom_oop_keywords.extend(keywords);
        self
    }

    /// Add custom FP keywords.
    pub fn with_fp_keywords(mut self, keywords: Vec<String>) -> Self {
        self.custom_fp_keywords.extend(keywords);
        self
    }

    /// Add custom reactive keywords.
    pub fn with_reactive_keywords(mut self, keywords: Vec<String>) -> Self {
        self.custom_reactive_keywords.extend(keywords);
        self
    }

    /// Add keywords to ignore.
    pub fn with_ignore_keywords(mut self, keywords: Vec<String>) -> Self {
        self.ignore_keywords.extend(keywords);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ParadigmConfig::default();
        assert!(config.normalize_scores);
        assert!(config.parallel);
        assert_eq!(config.context_window, 10);
    }

    #[test]
    fn test_oop_detection_config() {
        let config = ParadigmConfig::for_oop_detection();
        assert!(config.weights.oop_multiplier > config.weights.fp_multiplier);
    }

    #[test]
    fn test_fp_detection_config() {
        let config = ParadigmConfig::for_fp_detection();
        assert!(config.weights.fp_multiplier > config.weights.oop_multiplier);
    }

    #[test]
    fn test_language_hints_rust() {
        let hints = LanguageHints::rust();
        assert_eq!(hints.language, Some("rust".into()));
        assert!(hints.custom_fp_keywords.contains(&"map".to_string()));
    }

    #[test]
    fn test_language_hints_rholang() {
        let hints = LanguageHints::rholang();
        assert_eq!(hints.language, Some("rholang".into()));
        assert!(hints.custom_reactive_keywords.contains(&"!".to_string()));
    }

    #[test]
    fn test_config_builder() {
        let config = ParadigmConfig::new()
            .with_min_threshold(0.2)
            .with_context_window(20)
            .with_parallel(false);

        assert_eq!(config.min_score_threshold, 0.2);
        assert_eq!(config.context_window, 20);
        assert!(!config.parallel);
    }
}
