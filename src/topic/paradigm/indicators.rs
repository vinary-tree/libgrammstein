//! Paradigm indicator definitions and profile types.
//!
//! Defines the core types for representing programming paradigms and
//! the indicators used to detect them.

use std::fmt;

#[cfg(feature = "serde-extras")]
use serde::{Deserialize, Serialize};

/// Programming paradigms that can be detected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum Paradigm {
    /// Object-Oriented Programming: classes, inheritance, encapsulation, polymorphism.
    ObjectOriented,
    /// Functional Programming: pure functions, immutability, higher-order functions.
    Functional,
    /// Reactive Programming: observables, streams, event-driven patterns.
    Reactive,
    /// Procedural Programming: sequential execution, mutable state, control flow.
    Procedural,
    /// Mixed/Hybrid: significant presence of multiple paradigms.
    Mixed,
}

impl Paradigm {
    /// Get the short name for this paradigm.
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::ObjectOriented => "OOP",
            Self::Functional => "FP",
            Self::Reactive => "Reactive",
            Self::Procedural => "Procedural",
            Self::Mixed => "Mixed",
        }
    }

    /// Get the full descriptive name.
    pub fn full_name(&self) -> &'static str {
        match self {
            Self::ObjectOriented => "Object-Oriented Programming",
            Self::Functional => "Functional Programming",
            Self::Reactive => "Reactive Programming",
            Self::Procedural => "Procedural Programming",
            Self::Mixed => "Mixed Paradigm",
        }
    }

    /// Return all paradigms (excluding Mixed).
    pub fn all_primary() -> &'static [Paradigm] {
        &[
            Paradigm::ObjectOriented,
            Paradigm::Functional,
            Paradigm::Reactive,
            Paradigm::Procedural,
        ]
    }
}

impl fmt::Display for Paradigm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

/// A profile of paradigm scores for a code sample.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct ParadigmProfile {
    /// Score for Object-Oriented patterns (0.0 to 1.0).
    pub oop_score: f64,
    /// Score for Functional patterns (0.0 to 1.0).
    pub fp_score: f64,
    /// Score for Reactive patterns (0.0 to 1.0).
    pub reactive_score: f64,
    /// Score for Procedural patterns (0.0 to 1.0).
    pub procedural_score: f64,
    /// Individual indicators that were matched.
    pub indicators: Vec<ParadigmIndicator>,
    /// Total tokens analyzed.
    pub total_tokens: usize,
    /// Number of indicator matches.
    pub match_count: usize,
}

impl ParadigmProfile {
    /// Create a new empty profile.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the dominant paradigm if one clearly dominates.
    ///
    /// Returns `None` if no paradigm has a score above the threshold,
    /// or if the top two paradigms are within 0.1 of each other (Mixed).
    pub fn dominant_paradigm(&self) -> Option<Paradigm> {
        let threshold = 0.2; // Minimum score to be considered present
        let margin = 0.1; // Required lead over second-highest

        let scores = [
            (Paradigm::ObjectOriented, self.oop_score),
            (Paradigm::Functional, self.fp_score),
            (Paradigm::Reactive, self.reactive_score),
            (Paradigm::Procedural, self.procedural_score),
        ];

        let mut sorted = scores;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (top_paradigm, top_score) = sorted[0];
        let (_, second_score) = sorted[1];

        if top_score < threshold {
            None
        } else if top_score - second_score < margin {
            Some(Paradigm::Mixed)
        } else {
            Some(top_paradigm)
        }
    }

    /// Get all paradigms with scores above the threshold.
    pub fn present_paradigms(&self, threshold: f64) -> Vec<(Paradigm, f64)> {
        let mut result = Vec::new();

        if self.oop_score >= threshold {
            result.push((Paradigm::ObjectOriented, self.oop_score));
        }
        if self.fp_score >= threshold {
            result.push((Paradigm::Functional, self.fp_score));
        }
        if self.reactive_score >= threshold {
            result.push((Paradigm::Reactive, self.reactive_score));
        }
        if self.procedural_score >= threshold {
            result.push((Paradigm::Procedural, self.procedural_score));
        }

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Get the score for a specific paradigm.
    pub fn score(&self, paradigm: Paradigm) -> f64 {
        match paradigm {
            Paradigm::ObjectOriented => self.oop_score,
            Paradigm::Functional => self.fp_score,
            Paradigm::Reactive => self.reactive_score,
            Paradigm::Procedural => self.procedural_score,
            Paradigm::Mixed => {
                // Mixed score is the variance of other scores (higher variance = less mixed)
                let scores = [
                    self.oop_score,
                    self.fp_score,
                    self.reactive_score,
                    self.procedural_score,
                ];
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                let variance: f64 =
                    scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
                1.0 - variance.sqrt() // Lower variance = more mixed
            }
        }
    }

    /// Normalize scores so they sum to 1.0.
    pub fn normalize(&mut self) {
        let total = self.oop_score + self.fp_score + self.reactive_score + self.procedural_score;
        if total > 0.0 {
            self.oop_score /= total;
            self.fp_score /= total;
            self.reactive_score /= total;
            self.procedural_score /= total;
        }
    }

    /// Create a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Merge another profile into this one (weighted combination).
    pub fn merge(&mut self, other: &ParadigmProfile, weight: f64) {
        let self_weight = 1.0 - weight;
        self.oop_score = self.oop_score * self_weight + other.oop_score * weight;
        self.fp_score = self.fp_score * self_weight + other.fp_score * weight;
        self.reactive_score = self.reactive_score * self_weight + other.reactive_score * weight;
        self.procedural_score =
            self.procedural_score * self_weight + other.procedural_score * weight;
        self.indicators.extend(other.indicators.iter().cloned());
        self.total_tokens += other.total_tokens;
        self.match_count += other.match_count;
    }
}

/// A single paradigm indicator with metadata.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct ParadigmIndicator {
    /// The paradigm this indicator belongs to.
    pub paradigm: Paradigm,
    /// The category within the paradigm.
    pub category: IndicatorCategory,
    /// The pattern or keyword that was matched.
    pub pattern: String,
    /// Weight/importance of this indicator (0.0 to 1.0).
    pub weight: f64,
    /// Position in the token stream where matched.
    pub position: Option<usize>,
    /// Length of the match in tokens.
    pub length: usize,
    /// Whether this is a strong indicator (high confidence).
    pub is_strong: bool,
}

impl ParadigmIndicator {
    /// Create a new indicator.
    pub fn new(
        paradigm: Paradigm,
        category: IndicatorCategory,
        pattern: impl Into<String>,
        weight: f64,
    ) -> Self {
        Self {
            paradigm,
            category,
            pattern: pattern.into(),
            weight,
            position: None,
            length: 1,
            is_strong: weight >= 0.7,
        }
    }

    /// Set the position.
    pub fn with_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }

    /// Set the match length.
    pub fn with_length(mut self, length: usize) -> Self {
        self.length = length;
        self
    }

    /// Mark as strong/weak indicator.
    pub fn with_strong(mut self, strong: bool) -> Self {
        self.is_strong = strong;
        self
    }
}

/// Category of indicator within a paradigm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum IndicatorCategory {
    /// OOP: Class/type definitions.
    OopClass,
    /// OOP: Inheritance patterns.
    OopInheritance,
    /// OOP: Encapsulation (private/public).
    OopEncapsulation,
    /// OOP: Polymorphism (interfaces, traits).
    OopPolymorphism,
    /// OOP: Instance creation.
    OopInstantiation,
    /// FP: Higher-order functions.
    FpHigherOrder,
    /// FP: Immutability patterns.
    FpImmutability,
    /// FP: Pure functions.
    FpPurity,
    /// FP: Pattern matching.
    FpPatternMatch,
    /// FP: Algebraic data types.
    FpAlgebraic,
    /// FP: Recursion patterns.
    FpRecursion,
    /// Reactive: Observable streams.
    ReactiveObservable,
    /// Reactive: Event handling.
    ReactiveEvent,
    /// Reactive: Backpressure/flow control.
    ReactiveBackpressure,
    /// Reactive: Schedulers/async.
    ReactiveAsync,
    /// Procedural: Control flow.
    ProceduralControlFlow,
    /// Procedural: Mutable state.
    ProceduralMutable,
    /// Procedural: Side effects.
    ProceduralSideEffect,
    /// Procedural: Sequential execution.
    ProceduralSequential,
}

impl IndicatorCategory {
    /// Get the paradigm this category belongs to.
    pub fn paradigm(&self) -> Paradigm {
        match self {
            Self::OopClass
            | Self::OopInheritance
            | Self::OopEncapsulation
            | Self::OopPolymorphism
            | Self::OopInstantiation => Paradigm::ObjectOriented,

            Self::FpHigherOrder
            | Self::FpImmutability
            | Self::FpPurity
            | Self::FpPatternMatch
            | Self::FpAlgebraic
            | Self::FpRecursion => Paradigm::Functional,

            Self::ReactiveObservable
            | Self::ReactiveEvent
            | Self::ReactiveBackpressure
            | Self::ReactiveAsync => Paradigm::Reactive,

            Self::ProceduralControlFlow
            | Self::ProceduralMutable
            | Self::ProceduralSideEffect
            | Self::ProceduralSequential => Paradigm::Procedural,
        }
    }

    /// Get the short name.
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::OopClass => "class",
            Self::OopInheritance => "inheritance",
            Self::OopEncapsulation => "encapsulation",
            Self::OopPolymorphism => "polymorphism",
            Self::OopInstantiation => "instantiation",
            Self::FpHigherOrder => "higher-order",
            Self::FpImmutability => "immutability",
            Self::FpPurity => "purity",
            Self::FpPatternMatch => "pattern-match",
            Self::FpAlgebraic => "algebraic",
            Self::FpRecursion => "recursion",
            Self::ReactiveObservable => "observable",
            Self::ReactiveEvent => "event",
            Self::ReactiveBackpressure => "backpressure",
            Self::ReactiveAsync => "async",
            Self::ProceduralControlFlow => "control-flow",
            Self::ProceduralMutable => "mutable",
            Self::ProceduralSideEffect => "side-effect",
            Self::ProceduralSequential => "sequential",
        }
    }
}

/// OOP-specific indicators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum OopIndicator {
    /// Class definition keyword.
    ClassKeyword,
    /// Struct with methods.
    StructWithMethods,
    /// Interface/protocol/trait definition.
    InterfaceKeyword,
    /// Extends/inherits keyword.
    ExtendsKeyword,
    /// Implements keyword.
    ImplementsKeyword,
    /// Super/parent reference.
    SuperReference,
    /// This/self reference.
    SelfReference,
    /// Constructor/initializer.
    Constructor,
    /// New/instantiation keyword.
    NewKeyword,
    /// Private/protected modifier.
    AccessModifier,
    /// Virtual/override keyword.
    VirtualKeyword,
    /// Abstract keyword.
    AbstractKeyword,
    /// Method definition in class.
    MethodDefinition,
    /// Property/field definition.
    PropertyDefinition,
    /// Getter/setter pattern.
    GetterSetter,
}

/// FP-specific indicators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum FpIndicator {
    /// Lambda/arrow function.
    Lambda,
    /// Map operation.
    Map,
    /// Filter operation.
    Filter,
    /// Reduce/fold operation.
    Reduce,
    /// FlatMap/bind operation.
    FlatMap,
    /// Compose/pipe function.
    Compose,
    /// Curry/partial application.
    Curry,
    /// Immutable declaration (const, val, let).
    ImmutableDecl,
    /// Pattern matching (match, case).
    PatternMatch,
    /// Option/Maybe type.
    OptionType,
    /// Result/Either type.
    ResultType,
    /// List comprehension.
    Comprehension,
    /// Pure function annotation.
    PureAnnotation,
    /// Tail recursion.
    TailRecursion,
    /// Monadic bind (>>=, >>, do).
    MonadicBind,
    /// Functor operations.
    FunctorOp,
}

/// Reactive-specific indicators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum ReactiveIndicator {
    /// Observable/Subject creation.
    ObservableCreate,
    /// Subscribe operation.
    Subscribe,
    /// Event listener attachment.
    EventListener,
    /// On-event handler (onClick, onData).
    OnHandler,
    /// Emit/publish operation.
    Emit,
    /// Stream operation.
    Stream,
    /// Flux/Mono types (Project Reactor).
    FluxMono,
    /// RxJS operators.
    RxOperator,
    /// Signal/computed (SolidJS, Preact).
    Signal,
    /// Effect/useEffect hooks.
    Effect,
    /// Async/await with streams.
    AsyncStream,
    /// Backpressure handling.
    Backpressure,
    /// Scheduler operations.
    Scheduler,
    /// Hot/cold observable patterns.
    HotCold,
}

/// Procedural-specific indicators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum ProceduralIndicator {
    /// For loop.
    ForLoop,
    /// While loop.
    WhileLoop,
    /// Mutable variable declaration.
    MutableDecl,
    /// Variable reassignment.
    Reassignment,
    /// Goto/label.
    Goto,
    /// Break/continue.
    BreakContinue,
    /// Return statement.
    ReturnStatement,
    /// Printf/print statement.
    PrintStatement,
    /// Global variable.
    GlobalVariable,
    /// Void function.
    VoidFunction,
    /// Pointer/reference manipulation.
    PointerOps,
    /// Increment/decrement operators.
    IncrementOps,
    /// Statement blocks without return value.
    StatementBlock,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paradigm_names() {
        assert_eq!(Paradigm::ObjectOriented.short_name(), "OOP");
        assert_eq!(Paradigm::Functional.short_name(), "FP");
        assert_eq!(Paradigm::Reactive.short_name(), "Reactive");
        assert_eq!(Paradigm::Procedural.short_name(), "Procedural");
    }

    #[test]
    fn test_dominant_paradigm_clear_winner() {
        let profile = ParadigmProfile {
            oop_score: 0.8,
            fp_score: 0.1,
            reactive_score: 0.05,
            procedural_score: 0.05,
            ..Default::default()
        };

        assert_eq!(profile.dominant_paradigm(), Some(Paradigm::ObjectOriented));
    }

    #[test]
    fn test_dominant_paradigm_mixed() {
        let profile = ParadigmProfile {
            oop_score: 0.45,
            fp_score: 0.40,
            reactive_score: 0.1,
            procedural_score: 0.05,
            ..Default::default()
        };

        assert_eq!(profile.dominant_paradigm(), Some(Paradigm::Mixed));
    }

    #[test]
    fn test_dominant_paradigm_none() {
        let profile = ParadigmProfile {
            oop_score: 0.1,
            fp_score: 0.05,
            reactive_score: 0.05,
            procedural_score: 0.05,
            ..Default::default()
        };

        assert_eq!(profile.dominant_paradigm(), None);
    }

    #[test]
    fn test_present_paradigms() {
        let profile = ParadigmProfile {
            oop_score: 0.6,
            fp_score: 0.3,
            reactive_score: 0.1,
            procedural_score: 0.05,
            ..Default::default()
        };

        let present = profile.present_paradigms(0.2);
        assert_eq!(present.len(), 2);
        assert_eq!(present[0].0, Paradigm::ObjectOriented);
        assert_eq!(present[1].0, Paradigm::Functional);
    }

    #[test]
    fn test_normalize() {
        let mut profile = ParadigmProfile {
            oop_score: 2.0,
            fp_score: 1.0,
            reactive_score: 0.5,
            procedural_score: 0.5,
            ..Default::default()
        };

        profile.normalize();

        assert!((profile.oop_score - 0.5).abs() < 0.001);
        assert!((profile.fp_score - 0.25).abs() < 0.001);
        assert!((profile.reactive_score - 0.125).abs() < 0.001);
        assert!((profile.procedural_score - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_indicator_category_paradigm() {
        assert_eq!(
            IndicatorCategory::OopClass.paradigm(),
            Paradigm::ObjectOriented
        );
        assert_eq!(
            IndicatorCategory::FpHigherOrder.paradigm(),
            Paradigm::Functional
        );
        assert_eq!(
            IndicatorCategory::ReactiveObservable.paradigm(),
            Paradigm::Reactive
        );
        assert_eq!(
            IndicatorCategory::ProceduralControlFlow.paradigm(),
            Paradigm::Procedural
        );
    }

    #[test]
    fn test_paradigm_indicator_creation() {
        let indicator = ParadigmIndicator::new(
            Paradigm::Functional,
            IndicatorCategory::FpHigherOrder,
            "map",
            0.8,
        )
        .with_position(10)
        .with_length(3);

        assert_eq!(indicator.paradigm, Paradigm::Functional);
        assert_eq!(indicator.pattern, "map");
        assert_eq!(indicator.position, Some(10));
        assert_eq!(indicator.length, 3);
        assert!(indicator.is_strong);
    }
}
