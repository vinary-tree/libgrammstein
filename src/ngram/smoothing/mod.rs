//! Smoothing algorithms for n-gram probability estimation.
//!
//! This module provides smoothing techniques to handle unseen n-grams
//! and improve probability estimates for rare n-grams.

mod kneser_ney;

pub use kneser_ney::KneserNeySmoothing;
