//! Code-aware tokenization for programming languages.
//!
//! This module provides tokenization that preserves semantic information
//! from tree-sitter parsing, enabling type-aware correction.

use super::language::{CodeLanguage, TokenContext, TokenType};
use tree_sitter::{Node, Tree};

/// A token extracted from source code with its metadata.
#[derive(Debug, Clone)]
pub struct CodeToken {
    /// The token text
    pub text: String,
    /// Byte offset in the source
    pub byte_offset: usize,
    /// Line number (0-indexed)
    pub line: usize,
    /// Column number (0-indexed)
    pub column: usize,
    /// Token type classification
    pub token_type: TokenType,
    /// Tree-sitter node kind
    pub node_kind: String,
    /// Contextual information
    pub context: TokenContext,
}

impl CodeToken {
    /// Creates a new code token.
    pub fn new(
        text: impl Into<String>,
        byte_offset: usize,
        line: usize,
        column: usize,
        token_type: TokenType,
        node_kind: impl Into<String>,
    ) -> Self {
        let token_type_copy = token_type;
        Self {
            text: text.into(),
            byte_offset,
            line,
            column,
            token_type,
            node_kind: node_kind.into(),
            context: TokenContext::new(token_type_copy),
        }
    }

    /// Returns whether this token is inside an error region.
    pub fn is_in_error(&self) -> bool {
        self.context.in_error_region
    }

    /// Returns whether this token should be considered for correction.
    pub fn is_correctable(&self) -> bool {
        self.token_type.is_correctable()
    }
}

/// Tokenizer for extracting tokens from parsed code.
pub struct CodeTokenizer<'a, L: CodeLanguage> {
    language: &'a L,
    include_whitespace: bool,
    include_comments: bool,
}

impl<'a, L: CodeLanguage> CodeTokenizer<'a, L> {
    /// Creates a new tokenizer for the given language.
    pub fn new(language: &'a L) -> Self {
        Self {
            language,
            include_whitespace: false,
            include_comments: false,
        }
    }

    /// Configures whether to include whitespace tokens.
    pub fn with_whitespace(mut self, include: bool) -> Self {
        self.include_whitespace = include;
        self
    }

    /// Configures whether to include comment tokens.
    pub fn with_comments(mut self, include: bool) -> Self {
        self.include_comments = include;
        self
    }

    /// Extracts tokens from a parsed tree.
    pub fn tokenize(&self, tree: &Tree, source: &str) -> Vec<CodeToken> {
        let mut tokens = Vec::new();
        self.traverse_node(tree.root_node(), source, &mut tokens, 0, false);
        tokens
    }

    /// Extracts tokens only from error regions.
    pub fn tokenize_errors(&self, tree: &Tree, source: &str) -> Vec<CodeToken> {
        let mut tokens = Vec::new();
        self.collect_error_tokens(tree.root_node(), source, &mut tokens, 0);
        tokens
    }

    fn traverse_node(
        &self,
        node: Node,
        source: &str,
        tokens: &mut Vec<CodeToken>,
        depth: usize,
        in_error: bool,
    ) {
        let in_error = in_error || node.is_error();

        // Process leaf nodes (actual tokens)
        if node.child_count() == 0 {
            if let Some(token) = self.create_token(node, source, depth, in_error) {
                tokens.push(token);
            }
        } else {
            // Recurse into children
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                self.traverse_node(child, source, tokens, depth + 1, in_error);
            }
        }
    }

    fn collect_error_tokens(
        &self,
        node: Node,
        source: &str,
        tokens: &mut Vec<CodeToken>,
        depth: usize,
    ) {
        if node.is_error() || node.is_missing() {
            // Collect all tokens under this error node
            self.traverse_node(node, source, tokens, depth, true);
        } else {
            // Check children for errors
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                self.collect_error_tokens(child, source, tokens, depth + 1);
            }
        }
    }

    fn create_token(
        &self,
        node: Node,
        source: &str,
        depth: usize,
        in_error: bool,
    ) -> Option<CodeToken> {
        let text = node.utf8_text(source.as_bytes()).ok()?;
        let node_kind = node.kind();

        let token_type = self.language.classify_token(text, node_kind);

        // Filter based on configuration
        match token_type {
            TokenType::Whitespace if !self.include_whitespace => return None,
            TokenType::Comment if !self.include_comments => return None,
            _ => {}
        }

        let start = node.start_position();
        let mut token = CodeToken::new(
            text,
            node.start_byte(),
            start.row,
            start.column,
            token_type,
            node_kind,
        );

        // Enrich context
        token.context.depth = depth;
        if in_error {
            token.context.in_error_region = true;
        }

        if let Some(parent) = node.parent() {
            token.context.parent_node_type = Some(parent.kind().to_string());

            // Collect sibling types
            let mut cursor = parent.walk();
            token.context.sibling_types = parent
                .children(&mut cursor)
                .filter(|c| c.id() != node.id())
                .map(|c| c.kind().to_string())
                .collect();
        }

        Some(token)
    }
}

/// Iterator over tokens in source code.
pub struct TokenIterator<'a, L: CodeLanguage> {
    tokenizer: CodeTokenizer<'a, L>,
    tree: Tree,
    source: String,
    tokens: Vec<CodeToken>,
    position: usize,
}

impl<'a, L: CodeLanguage> TokenIterator<'a, L> {
    /// Creates a new token iterator.
    pub fn new(tokenizer: CodeTokenizer<'a, L>, tree: Tree, source: String) -> Self {
        let tokens = tokenizer.tokenize(&tree, &source);
        Self {
            tokenizer,
            tree,
            source,
            tokens,
            position: 0,
        }
    }
}

impl<L: CodeLanguage> Iterator for TokenIterator<'_, L> {
    type Item = CodeToken;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.tokens.len() {
            let token = self.tokens[self.position].clone();
            self.position += 1;
            Some(token)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_token_creation() {
        let token = CodeToken::new("test", 0, 1, 5, TokenType::Identifier, "identifier");

        assert_eq!(token.text, "test");
        assert_eq!(token.byte_offset, 0);
        assert_eq!(token.line, 1);
        assert_eq!(token.column, 5);
        assert_eq!(token.token_type, TokenType::Identifier);
        assert!(token.is_correctable());
        assert!(!token.is_in_error());
    }
}
