//! Subtree mining for discovering frequent code patterns.
//!
//! This module implements the TreeminerD algorithm for mining frequent
//! subtree patterns from AST forests. It's useful for:
//! - Discovering common code idioms
//! - Finding design patterns
//! - Identifying code clones
//! - Building pattern-based code completion
//!
//! # Algorithm: TreeminerD
//!
//! TreeminerD uses a depth-first string encoding of trees and generates
//! candidate subtrees using equivalence class extensions. The algorithm:
//! 1. Encodes trees as depth-first sequences
//! 2. Mines frequent (k)-subtrees from frequent (k-1)-subtrees
//! 3. Uses equivalence classes to reduce search space
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::code::subtree::{TreeminerD, FlatTree, SubtreePattern};
//!
//! let trees: Vec<FlatTree> = parse_code_to_flat_trees(&sources);
//! let miner = TreeminerD::new(0.1); // 10% minimum support
//! let patterns = miner.mine(&trees);
//!
//! for pattern in patterns {
//!     println!("Pattern: {:?}, Support: {}", pattern.nodes, pattern.support);
//! }
//! ```

mod pattern;
mod treeminer;

pub use pattern::{FlatTree, FlatNode, SubtreePattern, PatternNode};
pub use treeminer::{TreeminerD, TreeminerConfig, MiningResult};
