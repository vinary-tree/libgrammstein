//! Programming paradigm indicator extraction.
//!
//! This module provides extraction of programming paradigm indicators from code,
//! detecting patterns characteristic of:
//!
//! - **Object-Oriented Programming (OOP)**: Classes, inheritance, encapsulation, polymorphism
//! - **Functional Programming (FP)**: Pure functions, immutability, higher-order functions, closures
//! - **Reactive Programming**: Observables, streams, event-driven patterns
//! - **Procedural Programming**: Sequential execution, mutable state, control flow
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Paradigm Detection Pipeline                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  Source Code → Token Stream → Indicator Matching → Paradigm Profile    │
//! │                                                                          │
//! │  Indicators:                                                             │
//! │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────┐│
//! │  │ Class/struct  │ │ Lambda/fn     │ │ Observable    │ │ for/while    ││
//! │  │ extends/impl  │ │ map/filter    │ │ subscribe     │ │ mut state    ││
//! │  │ this/self     │ │ fold/reduce   │ │ stream/flux   │ │ goto/jump    ││
//! │  │ new/instance  │ │ compose       │ │ event/emit    │ │ side effects ││
//! │  └───────────────┘ └───────────────┘ └───────────────┘ └──────────────┘│
//! │       OOP               FP              Reactive          Procedural   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::topic::paradigm::{ParadigmDetector, ParadigmConfig, Paradigm};
//!
//! let config = ParadigmConfig::default();
//! let detector = ParadigmDetector::new(config);
//!
//! let code = "class Foo extends Bar { constructor() { this.x = 0; } }";
//! let profile = detector.analyze(code);
//!
//! assert!(profile.oop_score > profile.fp_score);
//! assert_eq!(profile.dominant_paradigm(), Some(Paradigm::ObjectOriented));
//! ```

mod api_patterns;
mod config;
mod detector;
mod domain_patterns;
mod indicators;

pub use api_patterns::{ApiPattern, ApiPatternConfig, ApiPatternMiner, MiningStats};
pub use config::{LanguageHints, ParadigmConfig, ParadigmWeights};
pub use detector::{DetectionResult, IndicatorMatch, ParadigmDetector};
pub use domain_patterns::{
    DomainPatternDetector, MettaPattern, MettaPatternCatalog, MettaPatternCategory,
    MettaPatternMatch, RholangPattern, RholangPatternCatalog, RholangPatternCategory,
    RholangPatternMatch,
};
pub use indicators::{
    FpIndicator, IndicatorCategory, OopIndicator, Paradigm, ParadigmIndicator, ParadigmProfile,
    ProceduralIndicator, ReactiveIndicator,
};
