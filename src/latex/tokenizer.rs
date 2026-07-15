//! LaTeX-aware tokenization for language model training and scoring.
//!
//! This module provides a tokenizer that understands LaTeX syntax, distinguishing
//! between commands, environments, math mode, text, and structural elements.

use std::iter::Peekable;
use std::str::Chars;

/// Configuration for the LaTeX tokenizer.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Whether to preserve whitespace as tokens.
    pub preserve_whitespace: bool,
    /// Whether to preserve comments as tokens.
    pub preserve_comments: bool,
    /// Whether to expand simple macros (\newcommand definitions).
    pub expand_macros: bool,
    /// Whether to normalize Unicode to ASCII equivalents where possible.
    pub normalize_unicode: bool,
    /// Maximum token length before truncation.
    pub max_token_length: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            preserve_whitespace: false,
            preserve_comments: false,
            expand_macros: false,
            normalize_unicode: true,
            max_token_length: 256,
        }
    }
}

/// Math mode delimiter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathMode {
    /// Inline math: $...$
    InlineDollar,
    /// Inline math: \(...\)
    InlineParen,
    /// Display math: $$...$$
    DisplayDoubleDollar,
    /// Display math: \[...\]
    DisplayBracket,
    /// Environment-based math: equation, align, etc.
    Environment,
}

/// Brace type for grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BraceKind {
    /// Curly braces: { }
    Curly,
    /// Square brackets: [ ]
    Square,
    /// Parentheses: ( )
    Paren,
}

/// The kind of a LaTeX token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LaTeXTokenKind {
    /// A LaTeX command: \alpha, \begin, \frac, etc.
    Command(String),
    /// An environment name (appears after \begin or \end).
    Environment(String),
    /// Plain text content.
    Text(String),
    /// A number (integer or decimal).
    Number(String),
    /// A single identifier/variable.
    Identifier(String),
    /// Mathematical operator: +, -, *, /, =, <, >, etc.
    Operator(String),
    /// Opening brace.
    OpenBrace(BraceKind),
    /// Closing brace.
    CloseBrace(BraceKind),
    /// Math mode delimiter (opening).
    MathOpen(MathMode),
    /// Math mode delimiter (closing).
    MathClose(MathMode),
    /// Ampersand (table/matrix column separator).
    Ampersand,
    /// Double backslash (newline in environments).
    Newline,
    /// Comment (if preserved).
    Comment(String),
    /// Whitespace (if preserved).
    Whitespace(String),
    /// Macro parameter token: #1, #2, etc.
    Parameter(u8),
    /// Subscript operator: _
    Subscript,
    /// Superscript operator: ^
    Superscript,
    /// Tilde (non-breaking space in LaTeX).
    Tilde,
    /// Active character or special symbol.
    Special(char),
    /// Unknown or unrecognized token.
    Unknown(String),
}

/// A token from LaTeX source with position information.
#[derive(Debug, Clone, PartialEq)]
pub struct LaTeXToken {
    /// The kind of token.
    pub kind: LaTeXTokenKind,
    /// Byte offset in source where token starts.
    pub start: usize,
    /// Byte offset in source where token ends.
    pub end: usize,
    /// Whether this token is inside math mode.
    pub in_math: bool,
}

impl LaTeXToken {
    /// Create a new token.
    pub fn new(kind: LaTeXTokenKind, start: usize, end: usize, in_math: bool) -> Self {
        Self {
            kind,
            start,
            end,
            in_math,
        }
    }

    /// Get the token text representation.
    pub fn text(&self) -> String {
        match &self.kind {
            LaTeXTokenKind::Command(s) => format!("\\{}", s),
            LaTeXTokenKind::Environment(s) => s.clone(),
            LaTeXTokenKind::Text(s) => s.clone(),
            LaTeXTokenKind::Number(s) => s.clone(),
            LaTeXTokenKind::Identifier(s) => s.clone(),
            LaTeXTokenKind::Operator(s) => s.clone(),
            LaTeXTokenKind::OpenBrace(BraceKind::Curly) => "{".to_string(),
            LaTeXTokenKind::OpenBrace(BraceKind::Square) => "[".to_string(),
            LaTeXTokenKind::OpenBrace(BraceKind::Paren) => "(".to_string(),
            LaTeXTokenKind::CloseBrace(BraceKind::Curly) => "}".to_string(),
            LaTeXTokenKind::CloseBrace(BraceKind::Square) => "]".to_string(),
            LaTeXTokenKind::CloseBrace(BraceKind::Paren) => ")".to_string(),
            LaTeXTokenKind::MathOpen(MathMode::InlineDollar) => "$".to_string(),
            LaTeXTokenKind::MathOpen(MathMode::InlineParen) => "\\(".to_string(),
            LaTeXTokenKind::MathOpen(MathMode::DisplayDoubleDollar) => "$$".to_string(),
            LaTeXTokenKind::MathOpen(MathMode::DisplayBracket) => "\\[".to_string(),
            LaTeXTokenKind::MathOpen(MathMode::Environment) => String::new(),
            LaTeXTokenKind::MathClose(MathMode::InlineDollar) => "$".to_string(),
            LaTeXTokenKind::MathClose(MathMode::InlineParen) => "\\)".to_string(),
            LaTeXTokenKind::MathClose(MathMode::DisplayDoubleDollar) => "$$".to_string(),
            LaTeXTokenKind::MathClose(MathMode::DisplayBracket) => "\\]".to_string(),
            LaTeXTokenKind::MathClose(MathMode::Environment) => String::new(),
            LaTeXTokenKind::Ampersand => "&".to_string(),
            LaTeXTokenKind::Newline => "\\\\".to_string(),
            LaTeXTokenKind::Comment(s) => format!("%{}", s),
            LaTeXTokenKind::Whitespace(s) => s.clone(),
            LaTeXTokenKind::Parameter(n) => format!("#{}", n),
            LaTeXTokenKind::Subscript => "_".to_string(),
            LaTeXTokenKind::Superscript => "^".to_string(),
            LaTeXTokenKind::Tilde => "~".to_string(),
            LaTeXTokenKind::Special(c) => c.to_string(),
            LaTeXTokenKind::Unknown(s) => s.clone(),
        }
    }

    /// Check if this is a structural token (brace, delimiter, etc.).
    pub fn is_structural(&self) -> bool {
        matches!(
            self.kind,
            LaTeXTokenKind::OpenBrace(_)
                | LaTeXTokenKind::CloseBrace(_)
                | LaTeXTokenKind::MathOpen(_)
                | LaTeXTokenKind::MathClose(_)
                | LaTeXTokenKind::Ampersand
                | LaTeXTokenKind::Newline
        )
    }

    /// Check if this is a command token.
    pub fn is_command(&self) -> bool {
        matches!(self.kind, LaTeXTokenKind::Command(_))
    }

    /// Check if this is a math-related token.
    pub fn is_math(&self) -> bool {
        self.in_math
            || matches!(
                self.kind,
                LaTeXTokenKind::MathOpen(_)
                    | LaTeXTokenKind::MathClose(_)
                    | LaTeXTokenKind::Subscript
                    | LaTeXTokenKind::Superscript
            )
    }
}

/// LaTeX tokenizer with configurable behavior.
pub struct LaTeXTokenizer {
    config: TokenizerConfig,
}

impl LaTeXTokenizer {
    /// Create a new tokenizer with default configuration.
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default(),
        }
    }

    /// Create a tokenizer with custom configuration.
    pub fn with_config(config: TokenizerConfig) -> Self {
        Self { config }
    }

    /// Tokenize a LaTeX string into a vector of tokens.
    pub fn tokenize(&self, input: &str) -> Vec<LaTeXToken> {
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(input, &self.config);

        while let Some(token) = lexer.next_token() {
            let should_keep = match &token.kind {
                LaTeXTokenKind::Whitespace(_) => self.config.preserve_whitespace,
                LaTeXTokenKind::Comment(_) => self.config.preserve_comments,
                _ => true,
            };

            if should_keep {
                tokens.push(token);
            }
        }

        tokens
    }

    /// Tokenize and return a lazy iterator over tokens.
    ///
    /// Unlike [`tokenize`](Self::tokenize), this streams tokens straight from the
    /// lexer without materializing the full `Vec`, applying the same
    /// preserve-whitespace / preserve-comment filtering inline. Collecting it
    /// yields exactly the same sequence as `tokenize`.
    pub fn tokenize_iter<'a>(&'a self, input: &'a str) -> impl Iterator<Item = LaTeXToken> + 'a {
        let preserve_whitespace = self.config.preserve_whitespace;
        let preserve_comments = self.config.preserve_comments;
        let mut lexer = Lexer::new(input, &self.config);

        std::iter::from_fn(move || loop {
            let token = lexer.next_token()?;
            let keep = match &token.kind {
                LaTeXTokenKind::Whitespace(_) => preserve_whitespace,
                LaTeXTokenKind::Comment(_) => preserve_comments,
                _ => true,
            };
            if keep {
                return Some(token);
            }
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
}

impl Default for LaTeXTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal lexer state machine.
struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
    pos: usize,
    config: &'a TokenizerConfig,
    /// Stack of math mode delimiters for nesting.
    math_stack: Vec<MathMode>,
    /// Whether we just saw \begin (to capture environment name).
    after_begin: bool,
    /// Whether we just saw \end (to capture environment name).
    after_end: bool,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str, config: &'a TokenizerConfig) -> Self {
        Self {
            chars: input.chars().peekable(),
            pos: 0,
            config,
            math_stack: Vec::new(),
            after_begin: false,
            after_end: false,
        }
    }

    fn in_math(&self) -> bool {
        !self.math_stack.is_empty()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn peek(&mut self) -> Option<&char> {
        self.chars.peek()
    }

    fn next_token(&mut self) -> Option<LaTeXToken> {
        let start = self.pos;
        let c = self.advance()?;
        let in_math = self.in_math();

        match c {
            // Backslash - command or special
            '\\' => self.lex_backslash(start, in_math),

            // Dollar sign - math mode
            '$' => self.lex_dollar(start),

            // Braces
            '{' => Some(LaTeXToken::new(
                LaTeXTokenKind::OpenBrace(BraceKind::Curly),
                start,
                self.pos,
                in_math,
            )),
            '}' => Some(LaTeXToken::new(
                LaTeXTokenKind::CloseBrace(BraceKind::Curly),
                start,
                self.pos,
                in_math,
            )),
            '[' => Some(LaTeXToken::new(
                LaTeXTokenKind::OpenBrace(BraceKind::Square),
                start,
                self.pos,
                in_math,
            )),
            ']' => Some(LaTeXToken::new(
                LaTeXTokenKind::CloseBrace(BraceKind::Square),
                start,
                self.pos,
                in_math,
            )),
            '(' => Some(LaTeXToken::new(
                LaTeXTokenKind::OpenBrace(BraceKind::Paren),
                start,
                self.pos,
                in_math,
            )),
            ')' => Some(LaTeXToken::new(
                LaTeXTokenKind::CloseBrace(BraceKind::Paren),
                start,
                self.pos,
                in_math,
            )),

            // Comment
            '%' => self.lex_comment(start),

            // Ampersand
            '&' => Some(LaTeXToken::new(
                LaTeXTokenKind::Ampersand,
                start,
                self.pos,
                in_math,
            )),

            // Subscript/superscript
            '_' => Some(LaTeXToken::new(
                LaTeXTokenKind::Subscript,
                start,
                self.pos,
                in_math,
            )),
            '^' => Some(LaTeXToken::new(
                LaTeXTokenKind::Superscript,
                start,
                self.pos,
                in_math,
            )),

            // Parameter
            '#' => self.lex_parameter(start, in_math),

            // Tilde
            '~' => Some(LaTeXToken::new(
                LaTeXTokenKind::Tilde,
                start,
                self.pos,
                in_math,
            )),

            // Whitespace
            c if c.is_whitespace() => self.lex_whitespace(start, c),

            // Numbers
            c if c.is_ascii_digit() => self.lex_number(start, c, in_math),

            // Operators (in math mode)
            c if in_math && is_math_operator(c) => Some(LaTeXToken::new(
                LaTeXTokenKind::Operator(c.to_string()),
                start,
                self.pos,
                true,
            )),

            // Identifiers (letters in math mode are variables)
            c if c.is_alphabetic() => {
                if in_math {
                    Some(LaTeXToken::new(
                        LaTeXTokenKind::Identifier(c.to_string()),
                        start,
                        self.pos,
                        true,
                    ))
                } else {
                    self.lex_text(start, c)
                }
            }

            // Other characters
            _ => {
                if in_math {
                    Some(LaTeXToken::new(
                        LaTeXTokenKind::Special(c),
                        start,
                        self.pos,
                        true,
                    ))
                } else {
                    self.lex_text(start, c)
                }
            }
        }
    }

    fn lex_backslash(&mut self, start: usize, in_math: bool) -> Option<LaTeXToken> {
        match self.peek() {
            // Double backslash - newline
            Some('\\') => {
                self.advance();
                Some(LaTeXToken::new(
                    LaTeXTokenKind::Newline,
                    start,
                    self.pos,
                    in_math,
                ))
            }
            // Math delimiters
            Some('[') => {
                self.advance();
                self.math_stack.push(MathMode::DisplayBracket);
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathOpen(MathMode::DisplayBracket),
                    start,
                    self.pos,
                    false,
                ))
            }
            Some(']') => {
                self.advance();
                if self.math_stack.last() == Some(&MathMode::DisplayBracket) {
                    self.math_stack.pop();
                }
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathClose(MathMode::DisplayBracket),
                    start,
                    self.pos,
                    false,
                ))
            }
            Some('(') => {
                self.advance();
                self.math_stack.push(MathMode::InlineParen);
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathOpen(MathMode::InlineParen),
                    start,
                    self.pos,
                    false,
                ))
            }
            Some(')') => {
                self.advance();
                if self.math_stack.last() == Some(&MathMode::InlineParen) {
                    self.math_stack.pop();
                }
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathClose(MathMode::InlineParen),
                    start,
                    self.pos,
                    false,
                ))
            }
            // Command name
            Some(&c) if c.is_alphabetic() => {
                let mut name = String::new();
                while let Some(&c) = self.peek() {
                    if c.is_alphabetic() {
                        name.push(c);
                        self.advance();
                    } else {
                        break;
                    }
                }

                // Check for special commands
                let kind = match name.as_str() {
                    "begin" => {
                        self.after_begin = true;
                        LaTeXTokenKind::Command(name)
                    }
                    "end" => {
                        self.after_end = true;
                        LaTeXTokenKind::Command(name)
                    }
                    _ => LaTeXTokenKind::Command(name),
                };

                Some(LaTeXToken::new(kind, start, self.pos, in_math))
            }
            // Single special character command: \!, \,, \;, etc.
            Some(&c) if !c.is_alphanumeric() && !c.is_whitespace() => {
                self.advance();
                Some(LaTeXToken::new(
                    LaTeXTokenKind::Command(c.to_string()),
                    start,
                    self.pos,
                    in_math,
                ))
            }
            // Backslash at end of input or before whitespace
            _ => Some(LaTeXToken::new(
                LaTeXTokenKind::Unknown("\\".to_string()),
                start,
                self.pos,
                in_math,
            )),
        }
    }

    fn lex_dollar(&mut self, start: usize) -> Option<LaTeXToken> {
        // Check for $$
        if self.peek() == Some(&'$') {
            self.advance();
            if self.math_stack.last() == Some(&MathMode::DisplayDoubleDollar) {
                self.math_stack.pop();
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathClose(MathMode::DisplayDoubleDollar),
                    start,
                    self.pos,
                    false,
                ))
            } else {
                self.math_stack.push(MathMode::DisplayDoubleDollar);
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathOpen(MathMode::DisplayDoubleDollar),
                    start,
                    self.pos,
                    false,
                ))
            }
        } else {
            // Single $
            if self.math_stack.last() == Some(&MathMode::InlineDollar) {
                self.math_stack.pop();
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathClose(MathMode::InlineDollar),
                    start,
                    self.pos,
                    false,
                ))
            } else {
                self.math_stack.push(MathMode::InlineDollar);
                Some(LaTeXToken::new(
                    LaTeXTokenKind::MathOpen(MathMode::InlineDollar),
                    start,
                    self.pos,
                    false,
                ))
            }
        }
    }

    fn lex_comment(&mut self, start: usize) -> Option<LaTeXToken> {
        let mut content = String::new();
        while let Some(&c) = self.peek() {
            if c == '\n' {
                break;
            }
            content.push(c);
            self.advance();
        }
        Some(LaTeXToken::new(
            LaTeXTokenKind::Comment(content),
            start,
            self.pos,
            self.in_math(),
        ))
    }

    fn lex_parameter(&mut self, start: usize, in_math: bool) -> Option<LaTeXToken> {
        if let Some(&c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
                let n = c.to_digit(10).unwrap_or(1) as u8;
                return Some(LaTeXToken::new(
                    LaTeXTokenKind::Parameter(n),
                    start,
                    self.pos,
                    in_math,
                ));
            }
        }
        Some(LaTeXToken::new(
            LaTeXTokenKind::Special('#'),
            start,
            self.pos,
            in_math,
        ))
    }

    fn lex_whitespace(&mut self, start: usize, first: char) -> Option<LaTeXToken> {
        let mut content = String::new();
        content.push(first);
        while let Some(&c) = self.peek() {
            if c.is_whitespace() {
                content.push(c);
                self.advance();
            } else {
                break;
            }
        }
        Some(LaTeXToken::new(
            LaTeXTokenKind::Whitespace(content),
            start,
            self.pos,
            self.in_math(),
        ))
    }

    fn lex_number(&mut self, start: usize, first: char, in_math: bool) -> Option<LaTeXToken> {
        let mut content = String::new();
        content.push(first);

        // Integer part
        while let Some(&c) = self.peek() {
            if c.is_ascii_digit() {
                content.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Decimal part
        if self.peek() == Some(&'.') {
            // Look ahead to see if there's a digit after the dot
            // Avoid Vec allocation by using iterator directly
            let has_digit_after_dot = {
                let mut lookahead = self.chars.clone();
                lookahead.next(); // skip the '.'
                lookahead.next().is_some_and(|c| c.is_ascii_digit())
            };
            if has_digit_after_dot {
                content.push('.');
                self.advance();
                while let Some(&c) = self.peek() {
                    if c.is_ascii_digit() {
                        content.push(c);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        Some(LaTeXToken::new(
            LaTeXTokenKind::Number(content),
            start,
            self.pos,
            in_math,
        ))
    }

    fn lex_text(&mut self, start: usize, first: char) -> Option<LaTeXToken> {
        let mut content = String::new();
        content.push(first);

        // Accumulate text until we hit a special character
        while let Some(&c) = self.peek() {
            if is_special_char(c) || c.is_whitespace() {
                break;
            }
            content.push(c);
            self.advance();

            if content.len() >= self.config.max_token_length {
                break;
            }
        }

        // Check if this is an environment name after \begin or \end
        if self.after_begin || self.after_end {
            self.after_begin = false;
            self.after_end = false;

            // Check if this is a math environment
            if is_math_environment(&content) {
                if self.after_begin {
                    self.math_stack.push(MathMode::Environment);
                } else if self.math_stack.last() == Some(&MathMode::Environment) {
                    self.math_stack.pop();
                }
            }

            Some(LaTeXToken::new(
                LaTeXTokenKind::Environment(content),
                start,
                self.pos,
                self.in_math(),
            ))
        } else {
            Some(LaTeXToken::new(
                LaTeXTokenKind::Text(content),
                start,
                self.pos,
                self.in_math(),
            ))
        }
    }
}

/// Check if a character is a special LaTeX character.
fn is_special_char(c: char) -> bool {
    matches!(
        c,
        '\\' | '{' | '}' | '[' | ']' | '(' | ')' | '$' | '%' | '&' | '_' | '^' | '#' | '~'
    )
}

/// Check if a character is a math operator.
fn is_math_operator(c: char) -> bool {
    matches!(
        c,
        '+' | '-' | '*' | '/' | '=' | '<' | '>' | '!' | '|' | ':' | ';' | ',' | '.'
    )
}

/// Check if an environment name is a math environment.
fn is_math_environment(name: &str) -> bool {
    matches!(
        name,
        "equation"
            | "equation*"
            | "align"
            | "align*"
            | "alignat"
            | "alignat*"
            | "gather"
            | "gather*"
            | "multline"
            | "multline*"
            | "eqnarray"
            | "eqnarray*"
            | "displaymath"
            | "math"
            | "array"
            | "matrix"
            | "bmatrix"
            | "pmatrix"
            | "vmatrix"
            | "Vmatrix"
            | "cases"
            | "split"
            | "subequations"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_command() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"\alpha");

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, LaTeXTokenKind::Command("alpha".to_string()));
    }

    #[test]
    fn test_math_mode() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"$x^2$");

        // Tokens: $, x, ^, 2, $
        assert_eq!(tokens.len(), 5);
        assert!(matches!(
            tokens[0].kind,
            LaTeXTokenKind::MathOpen(MathMode::InlineDollar)
        ));
        assert!(matches!(tokens[1].kind, LaTeXTokenKind::Identifier(_)));
        assert!(tokens[1].in_math);
        assert_eq!(tokens[2].kind, LaTeXTokenKind::Superscript);
        assert!(matches!(tokens[3].kind, LaTeXTokenKind::Number(_)));
        assert!(matches!(
            tokens[4].kind,
            LaTeXTokenKind::MathClose(MathMode::InlineDollar)
        ));
    }

    #[test]
    fn test_environment() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"\begin{equation}");

        // Should have: \begin, {, equation, }
        assert!(tokens
            .iter()
            .any(|t| matches!(&t.kind, LaTeXTokenKind::Command(s) if s == "begin")));
        assert!(tokens
            .iter()
            .any(|t| matches!(&t.kind, LaTeXTokenKind::Environment(s) if s == "equation")));
    }

    #[test]
    fn test_numbers() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"$3.14$");

        assert!(tokens
            .iter()
            .any(|t| matches!(&t.kind, LaTeXTokenKind::Number(s) if s == "3.14")));
    }

    #[test]
    fn test_operators() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"$a + b = c$");

        let operators: Vec<_> = tokens
            .iter()
            .filter(|t| matches!(&t.kind, LaTeXTokenKind::Operator(_)))
            .collect();

        assert_eq!(operators.len(), 2); // + and =
    }

    #[test]
    fn test_comment() {
        let config = TokenizerConfig {
            preserve_comments: true,
            ..Default::default()
        };
        let tokenizer = LaTeXTokenizer::with_config(config);
        let tokens = tokenizer.tokenize(r"text % comment");

        assert!(tokens
            .iter()
            .any(|t| matches!(&t.kind, LaTeXTokenKind::Comment(_))));
    }

    #[test]
    fn test_display_math() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"\[x\]");

        assert!(matches!(
            tokens[0].kind,
            LaTeXTokenKind::MathOpen(MathMode::DisplayBracket)
        ));
        assert!(matches!(
            tokens.last().unwrap().kind,
            LaTeXTokenKind::MathClose(MathMode::DisplayBracket)
        ));
    }

    #[test]
    fn test_subscript_superscript() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"$x_1^2$");

        assert!(tokens.iter().any(|t| t.kind == LaTeXTokenKind::Subscript));
        assert!(tokens.iter().any(|t| t.kind == LaTeXTokenKind::Superscript));
    }

    #[test]
    fn test_parameter() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"#1");

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, LaTeXTokenKind::Parameter(1));
    }

    #[test]
    fn test_newline() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"\\");

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, LaTeXTokenKind::Newline);
    }

    #[test]
    fn test_tokenize_iter_matches_tokenize() {
        let input = "\\alpha $x^2$ % a comment\n\\beta";

        // Default config (filters whitespace + comments): streamed == eager.
        let tokenizer = LaTeXTokenizer::new();
        let eager = tokenizer.tokenize(input);
        let streamed: Vec<_> = tokenizer.tokenize_iter(input).collect();
        assert_eq!(streamed, eager);

        // Preserve-everything config exercises the keep branches and must agree too.
        let tokenizer = LaTeXTokenizer::with_config(TokenizerConfig {
            preserve_whitespace: true,
            preserve_comments: true,
            ..Default::default()
        });
        let eager = tokenizer.tokenize(input);
        let streamed: Vec<_> = tokenizer.tokenize_iter(input).collect();
        assert_eq!(streamed, eager);
        assert!(!streamed.is_empty());
    }
}
