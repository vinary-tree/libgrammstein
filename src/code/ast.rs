//! AST parsing and manipulation using tree-sitter.
//!
//! This module provides integration with tree-sitter for incremental parsing
//! with error recovery, enabling correction of partially valid code.

use super::language::CodeLanguage;
use std::collections::HashMap;
use std::sync::Arc;
use tree_sitter::{Node, Parser, Point, Tree};

const MAX_PARSE_CACHE_ENTRIES: usize = 16;

/// Converts a byte offset to (line, column) position in source text.
///
/// Line and column are 0-indexed to match tree-sitter conventions.
///
/// # Arguments
/// * `source` - The source text
/// * `byte_offset` - Byte offset into the source
///
/// # Returns
/// A tuple of (line, column) where both are 0-indexed.
///
/// # Example
/// ```
/// use libgrammstein::code::byte_offset_to_position;
///
/// let source = "hello\nworld";
/// assert_eq!(byte_offset_to_position(source, 0), (0, 0));  // 'h'
/// assert_eq!(byte_offset_to_position(source, 5), (0, 5));  // '\n'
/// assert_eq!(byte_offset_to_position(source, 6), (1, 0));  // 'w'
/// ```
pub fn byte_offset_to_position(source: &str, byte_offset: usize) -> (usize, usize) {
    let byte_offset = byte_offset.min(source.len());
    let prefix = &source[..byte_offset];

    let mut line = 0;
    let mut last_newline_pos = 0;

    for (i, c) in prefix.char_indices() {
        if c == '\n' {
            line += 1;
            last_newline_pos = i + 1; // Position after the newline
        }
    }

    // Column is the number of characters from the last newline to the offset
    // For proper UTF-8 handling, we count characters, not bytes
    let column = prefix[last_newline_pos..].chars().count();

    (line, column)
}

/// Error types for AST operations.
#[derive(Debug, Clone)]
pub enum AstError {
    /// Parser initialization failed
    ParserInit(String),
    /// Parsing failed completely (no tree produced)
    ParseFailed,
    /// Language mismatch
    LanguageMismatch {
        /// Expected language name.
        expected: String,
        /// Actual language name.
        got: String,
    },
}

impl std::fmt::Display for AstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstError::ParserInit(msg) => write!(f, "Parser initialization failed: {}", msg),
            AstError::ParseFailed => write!(f, "Parsing failed"),
            AstError::LanguageMismatch { expected, got } => {
                write!(f, "Language mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for AstError {}

/// Parsed code with its AST and metadata.
pub struct ParsedCode {
    /// The tree-sitter parse tree
    pub tree: Tree,
    /// The original source code
    pub source: String,
    /// Language used for parsing
    pub language_name: String,
    /// Whether the parse tree contains any errors
    pub has_errors: bool,
    /// Error nodes in the tree
    pub error_ranges: Vec<ErrorRange>,
}

/// A range in the source code containing an error.
#[derive(Debug, Clone)]
pub struct ErrorRange {
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Start position (line, column)
    pub start_position: (usize, usize),
    /// End position (line, column)
    pub end_position: (usize, usize),
    /// The erroneous text
    pub text: String,
    /// The node kind (usually "ERROR" or "MISSING")
    pub kind: String,
}

impl ParsedCode {
    /// Returns the root node of the AST.
    pub fn root(&self) -> Node<'_> {
        self.tree.root_node()
    }

    /// Returns an iterator over all error ranges.
    pub fn errors(&self) -> impl Iterator<Item = &ErrorRange> {
        self.error_ranges.iter()
    }

    /// Returns the number of syntax errors.
    pub fn error_count(&self) -> usize {
        self.error_ranges.len()
    }

    /// Checks if a byte offset is within an error region.
    pub fn is_in_error(&self, byte_offset: usize) -> bool {
        self.error_ranges
            .iter()
            .any(|r| byte_offset >= r.start_byte && byte_offset < r.end_byte)
    }
}

/// A simplified representation of an AST node.
#[derive(Debug, Clone)]
pub struct AstNode {
    /// The node kind (e.g., "function_definition", "identifier")
    pub kind: String,
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Start position (line, column)
    pub start_position: (usize, usize),
    /// End position (line, column)
    pub end_position: (usize, usize),
    /// Whether this is a named node
    pub is_named: bool,
    /// Whether this node is an error node
    pub is_error: bool,
    /// Whether this node is missing (expected but not present)
    pub is_missing: bool,
    /// Child nodes
    pub children: Vec<AstNode>,
    /// The text content (for leaf nodes)
    pub text: Option<String>,
}

impl AstNode {
    /// Creates an AstNode from a tree-sitter Node.
    pub fn from_ts_node(node: Node, source: &str) -> Self {
        let mut children = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            children.push(Self::from_ts_node(child, source));
        }

        let text = if children.is_empty() {
            node.utf8_text(source.as_bytes()).ok().map(String::from)
        } else {
            None
        };

        let start = node.start_position();
        let end = node.end_position();

        Self {
            kind: node.kind().to_string(),
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            start_position: (start.row, start.column),
            end_position: (end.row, end.column),
            is_named: node.is_named(),
            is_error: node.is_error(),
            is_missing: node.is_missing(),
            children,
            text,
        }
    }

    /// Returns an iterator over all descendant nodes (depth-first).
    pub fn descendants(&self) -> impl Iterator<Item = &AstNode> {
        AstNodeIterator::new(self)
    }

    /// Finds nodes by kind.
    ///
    /// Returns an iterator to avoid allocating a Vec. Call `.collect()` if
    /// a Vec is needed.
    pub fn find_by_kind<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a AstNode> {
        self.descendants().filter(move |n| n.kind == kind)
    }

    /// Finds all error nodes.
    ///
    /// Returns an iterator to avoid allocating a Vec. Call `.collect()` if
    /// a Vec is needed.
    pub fn find_errors(&self) -> impl Iterator<Item = &AstNode> {
        self.descendants().filter(|n| n.is_error || n.is_missing)
    }
}

/// Iterator for traversing AST nodes depth-first.
struct AstNodeIterator<'a> {
    stack: Vec<&'a AstNode>,
}

impl<'a> AstNodeIterator<'a> {
    fn new(root: &'a AstNode) -> Self {
        Self { stack: vec![root] }
    }
}

impl<'a> Iterator for AstNodeIterator<'a> {
    type Item = &'a AstNode;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;
        // Push children in reverse order for correct traversal
        for child in node.children.iter().rev() {
            self.stack.push(child);
        }
        Some(node)
    }
}

/// Code parser with caching and incremental parsing support.
pub struct CodeParser<L: CodeLanguage> {
    language: Arc<L>,
    parser: Parser,
    /// Cache of previously parsed trees for incremental parsing
    tree_cache: HashMap<u64, (String, Tree)>,
}

impl<L: CodeLanguage> CodeParser<L> {
    /// Creates a new parser for the given language.
    pub fn new(language: Arc<L>) -> Result<Self, AstError> {
        let mut parser = Parser::new();
        parser
            .set_language(&language.tree_sitter_language())
            .map_err(|e| AstError::ParserInit(e.to_string()))?;

        Ok(Self {
            language,
            parser,
            tree_cache: HashMap::new(),
        })
    }

    /// Parses source code into an AST.
    pub fn parse(&mut self, source: &str) -> Result<ParsedCode, AstError> {
        let cache_key = crate::util::hash::safe_hash(source.as_bytes());
        if let Some((cached_source, tree)) = self.tree_cache.get(&cache_key) {
            if cached_source == source {
                return self.parsed_code_from_tree(tree.clone(), source);
            }
        }

        let parsed = self.parse_with_old_tree(source, None)?;
        if self.tree_cache.len() >= MAX_PARSE_CACHE_ENTRIES {
            self.tree_cache.clear();
        }
        self.tree_cache
            .insert(cache_key, (source.to_string(), parsed.tree.clone()));
        Ok(parsed)
    }

    /// Parses source code with an optional old tree for incremental parsing.
    pub fn parse_with_old_tree(
        &mut self,
        source: &str,
        old_tree: Option<&Tree>,
    ) -> Result<ParsedCode, AstError> {
        let tree = self
            .parser
            .parse(source, old_tree)
            .ok_or(AstError::ParseFailed)?;

        self.parsed_code_from_tree(tree, source)
    }

    fn parsed_code_from_tree(&self, tree: Tree, source: &str) -> Result<ParsedCode, AstError> {
        let has_errors = tree.root_node().has_error();
        let error_ranges = if has_errors {
            self.collect_errors(&tree, source)
        } else {
            Vec::new()
        };

        Ok(ParsedCode {
            tree,
            source: source.to_string(),
            language_name: self.language.name().to_string(),
            has_errors,
            error_ranges,
        })
    }

    /// Incrementally parses source code after an edit.
    pub fn parse_incremental(
        &mut self,
        source: &str,
        old_tree: &mut Tree,
        edit: &EditInfo,
    ) -> Result<ParsedCode, AstError> {
        // Apply the edit to the old tree
        old_tree.edit(&edit.to_input_edit());

        self.parse_with_old_tree(source, Some(old_tree))
    }

    fn collect_errors(&self, tree: &Tree, source: &str) -> Vec<ErrorRange> {
        let mut errors = Vec::new();
        self.collect_errors_recursive(tree.root_node(), source, &mut errors);
        errors
    }

    fn collect_errors_recursive(&self, node: Node, source: &str, errors: &mut Vec<ErrorRange>) {
        if node.is_error() || node.is_missing() {
            let start = node.start_position();
            let end = node.end_position();
            let text = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();

            errors.push(ErrorRange {
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                start_position: (start.row, start.column),
                end_position: (end.row, end.column),
                text,
                kind: node.kind().to_string(),
            });
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.collect_errors_recursive(child, source, errors);
        }
    }

    /// Returns the language this parser is configured for.
    pub fn language(&self) -> &L {
        &self.language
    }
}

/// Information about an edit to source code.
#[derive(Debug, Clone)]
pub struct EditInfo {
    /// Start byte of the edit
    pub start_byte: usize,
    /// Old end byte (before edit)
    pub old_end_byte: usize,
    /// New end byte (after edit)
    pub new_end_byte: usize,
    /// Start position (row, column)
    pub start_position: (usize, usize),
    /// Old end position
    pub old_end_position: (usize, usize),
    /// New end position
    pub new_end_position: (usize, usize),
}

impl EditInfo {
    /// Converts to tree-sitter InputEdit.
    pub fn to_input_edit(&self) -> tree_sitter::InputEdit {
        tree_sitter::InputEdit {
            start_byte: self.start_byte,
            old_end_byte: self.old_end_byte,
            new_end_byte: self.new_end_byte,
            start_position: Point::new(self.start_position.0, self.start_position.1),
            old_end_position: Point::new(self.old_end_position.0, self.old_end_position.1),
            new_end_position: Point::new(self.new_end_position.0, self.new_end_position.1),
        }
    }

    /// Creates an EditInfo for an insertion at a position.
    pub fn insertion(position: usize, row: usize, column: usize, inserted_text: &str) -> Self {
        let new_lines: Vec<&str> = inserted_text.split('\n').collect();
        let new_end_row = row + new_lines.len() - 1;
        let new_end_column = if new_lines.len() == 1 {
            column + inserted_text.len()
        } else {
            new_lines.last().map(|s| s.len()).unwrap_or(0)
        };

        Self {
            start_byte: position,
            old_end_byte: position,
            new_end_byte: position + inserted_text.len(),
            start_position: (row, column),
            old_end_position: (row, column),
            new_end_position: (new_end_row, new_end_column),
        }
    }

    /// Creates an EditInfo for a deletion.
    pub fn deletion(
        start_byte: usize,
        end_byte: usize,
        start_pos: (usize, usize),
        end_pos: (usize, usize),
    ) -> Self {
        Self {
            start_byte,
            old_end_byte: end_byte,
            new_end_byte: start_byte,
            start_position: start_pos,
            old_end_position: end_pos,
            new_end_position: start_pos,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_node_descendants() {
        let root = AstNode {
            kind: "root".to_string(),
            start_byte: 0,
            end_byte: 10,
            start_position: (0, 0),
            end_position: (0, 10),
            is_named: true,
            is_error: false,
            is_missing: false,
            text: None,
            children: vec![
                AstNode {
                    kind: "child1".to_string(),
                    start_byte: 0,
                    end_byte: 5,
                    start_position: (0, 0),
                    end_position: (0, 5),
                    is_named: true,
                    is_error: false,
                    is_missing: false,
                    text: Some("hello".to_string()),
                    children: vec![],
                },
                AstNode {
                    kind: "child2".to_string(),
                    start_byte: 5,
                    end_byte: 10,
                    start_position: (0, 5),
                    end_position: (0, 10),
                    is_named: true,
                    is_error: false,
                    is_missing: false,
                    text: Some("world".to_string()),
                    children: vec![],
                },
            ],
        };

        let kinds: Vec<&str> = root.descendants().map(|n| n.kind.as_str()).collect();
        assert_eq!(kinds, vec!["root", "child1", "child2"]);
    }

    #[test]
    fn test_edit_info_insertion() {
        let edit = EditInfo::insertion(5, 0, 5, "hello");
        assert_eq!(edit.start_byte, 5);
        assert_eq!(edit.old_end_byte, 5);
        assert_eq!(edit.new_end_byte, 10);
    }
}
