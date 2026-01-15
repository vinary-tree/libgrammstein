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

mod config;
mod indicators;
mod detector;
mod api_patterns;
mod domain_patterns;

pub use config::{ParadigmConfig, ParadigmWeights, LanguageHints};
pub use indicators::{
    Paradigm, ParadigmProfile, ParadigmIndicator, IndicatorCategory,
    OopIndicator, FpIndicator, ReactiveIndicator, ProceduralIndicator,
};
pub use detector::{ParadigmDetector, DetectionResult, IndicatorMatch};
pub use api_patterns::{
    ApiPatternMiner, ApiPatternConfig, ApiPattern, MiningStats,
};
pub use domain_patterns::{
    RholangPatternCatalog, RholangPattern, RholangPatternCategory, RholangPatternMatch,
    MettaPatternCatalog, MettaPattern, MettaPatternCategory, MettaPatternMatch,
    DomainPatternDetector,
};
