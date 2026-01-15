//! Postprocessing for PDF extraction output.
//!
//! This module normalizes and cleans the LaTeX output from PDF extraction
//! backends, fixing common issues and ensuring consistent formatting.

use super::backend::ExtractedDocument;
use super::error::PdfResult;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Configuration for postprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessorConfig {
    /// Whether to validate LaTeX syntax.
    pub validate_latex: bool,

    /// Whether to normalize whitespace.
    pub normalize_whitespace: bool,

    /// Whether to fix common OCR errors.
    pub fix_ocr_errors: bool,

    /// Whether to normalize math delimiters.
    pub normalize_math_delimiters: bool,

    /// Whether to fix brace matching.
    pub fix_brace_matching: bool,

    /// Whether to normalize output formatting.
    pub normalize: bool,

    /// Whether to remove OCR artifacts.
    pub remove_artifacts: bool,

    /// Maximum line length for wrapping (0 = no wrapping).
    pub max_line_length: usize,
}

impl Default for PostProcessorConfig {
    fn default() -> Self {
        Self {
            validate_latex: true,
            normalize_whitespace: true,
            fix_ocr_errors: true,
            normalize_math_delimiters: true,
            fix_brace_matching: true,
            normalize: true,
            remove_artifacts: true,
            max_line_length: 0,
        }
    }
}

impl PostProcessorConfig {
    /// Create a minimal postprocessor config (fast, less correction).
    pub fn minimal() -> Self {
        Self {
            validate_latex: false,
            normalize_whitespace: true,
            fix_ocr_errors: false,
            normalize_math_delimiters: false,
            fix_brace_matching: false,
            normalize: false,
            remove_artifacts: false,
            max_line_length: 0,
        }
    }

    /// Create a thorough postprocessor config (slower, more correction).
    pub fn thorough() -> Self {
        Self {
            validate_latex: true,
            normalize_whitespace: true,
            fix_ocr_errors: true,
            normalize_math_delimiters: true,
            fix_brace_matching: true,
            normalize: true,
            remove_artifacts: true,
            max_line_length: 80,
        }
    }
}

/// Postprocessor for extracted documents.
pub struct PostProcessor {
    config: PostProcessorConfig,
}

impl PostProcessor {
    /// Create a new postprocessor with the given configuration.
    pub fn new(config: PostProcessorConfig) -> Self {
        Self { config }
    }

    /// Create a postprocessor with default configuration.
    pub fn default_processor() -> Self {
        Self::new(PostProcessorConfig::default())
    }

    /// Process an extracted document.
    pub fn process(&self, mut doc: ExtractedDocument) -> PdfResult<ExtractedDocument> {
        // Process each page
        for page in &mut doc.pages {
            page.latex = self.process_content(&page.latex)?;
            if let Some(ref mut markdown) = page.markdown {
                *markdown = self.process_content(markdown)?;
            }
        }

        // Validate if enabled
        if self.config.validate_latex {
            self.validate_document(&doc)?;
        }

        Ok(doc)
    }

    /// Process content string.
    fn process_content(&self, content: &str) -> PdfResult<String> {
        let mut result = content.to_string();

        if self.config.normalize_whitespace {
            result = self.normalize_whitespace(&result);
        }

        if self.config.fix_ocr_errors {
            result = self.fix_ocr_errors(&result);
        }

        if self.config.normalize_math_delimiters {
            result = self.normalize_math_delimiters(&result);
        }

        if self.config.fix_brace_matching {
            result = self.fix_brace_matching(&result);
        }

        if self.config.remove_artifacts {
            result = self.remove_artifacts(&result);
        }

        if self.config.max_line_length > 0 {
            result = self.wrap_lines(&result, self.config.max_line_length);
        }

        Ok(result)
    }

    /// Normalize whitespace in content.
    fn normalize_whitespace(&self, content: &str) -> String {
        let mut result = String::with_capacity(content.len());
        let mut prev_was_space = false;
        let mut prev_was_newline = false;

        for ch in content.chars() {
            match ch {
                ' ' | '\t' => {
                    if !prev_was_space && !prev_was_newline {
                        result.push(' ');
                        prev_was_space = true;
                    }
                }
                '\n' => {
                    // Allow up to 2 consecutive newlines (paragraph break)
                    if !prev_was_newline {
                        result.push('\n');
                        prev_was_newline = true;
                        prev_was_space = false;
                    } else if result.ends_with("\n\n") {
                        // Skip additional newlines
                    } else {
                        result.push('\n');
                    }
                }
                '\r' => {
                    // Skip carriage returns
                }
                _ => {
                    result.push(ch);
                    prev_was_space = false;
                    prev_was_newline = false;
                }
            }
        }

        result.trim().to_string()
    }

    /// Fix common OCR errors in LaTeX.
    fn fix_ocr_errors(&self, content: &str) -> String {
        let mut result = Cow::Borrowed(content);

        // Common OCR confusions in LaTeX commands
        let ocr_fixes = [
            // Letter confusions
            (r"\aIpha", r"\alpha"),
            (r"\AIpha", r"\Alpha"),
            (r"\Iambda", r"\lambda"),
            (r"\Iambda", r"\Lambda"),
            (r"\rho", r"\rho"), // Often confused with p
            // Number/letter confusions
            (r"\sum_", r"\sum_"), // Ensure proper spacing
            (r"\int_", r"\int_"),
            // Common misspellings
            (r"\bgin{", r"\begin{"),
            (r"\emd{", r"\end{"),
            (r"\bgegin{", r"\begin{"),
            (r"\frc{", r"\frac{"),
            (r"\sqt{", r"\sqrt{"),
            (r"\squrt{", r"\sqrt{"),
            // Environment name fixes
            ("equaton", "equation"),
            ("eqnarray", "eqnarray"),
            ("aiign", "align"),
            ("tabIe", "table"),
            ("fiqure", "figure"),
            // Greek letter fixes (l/I confusion)
            (r"\Iota", r"\iota"),
            (r"\Iambda", r"\lambda"),
            (r"\Ieft", r"\left"),
            (r"\Ieq", r"\leq"),
            (r"\Iim", r"\lim"),
            (r"\Iog", r"\log"),
            (r"\Iatex", r"\latex"),
            // 0/O confusion
            (r"\0mega", r"\Omega"),
            // Spacing fixes
            ("  ", " "),
        ];

        for (wrong, correct) in &ocr_fixes {
            if result.contains(wrong) {
                result = Cow::Owned(result.replace(wrong, correct));
            }
        }

        result.into_owned()
    }

    /// Normalize math delimiters.
    fn normalize_math_delimiters(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Normalize display math to \[ \]
        // First, handle $$ ... $$ -> \[ ... \]
        let mut in_display_math = false;
        let mut new_result = String::with_capacity(result.len());
        let chars: Vec<char> = result.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if i + 1 < chars.len() && chars[i] == '$' && chars[i + 1] == '$' {
                if in_display_math {
                    new_result.push_str("\\]");
                    in_display_math = false;
                } else {
                    new_result.push_str("\\[");
                    in_display_math = true;
                }
                i += 2;
            } else {
                new_result.push(chars[i]);
                i += 1;
            }
        }

        result = new_result;

        // Ensure proper spacing around math delimiters
        result = result
            .replace("\\[", "\n\\[\n")
            .replace("\\]", "\n\\]\n");

        // Clean up excessive newlines introduced
        while result.contains("\n\n\n") {
            result = result.replace("\n\n\n", "\n\n");
        }

        result
    }

    /// Fix brace matching issues.
    fn fix_brace_matching(&self, content: &str) -> String {
        let mut result = String::with_capacity(content.len());
        let mut brace_stack: Vec<char> = Vec::new();

        for ch in content.chars() {
            match ch {
                '{' => {
                    brace_stack.push('{');
                    result.push(ch);
                }
                '}' => {
                    if brace_stack.last() == Some(&'{') {
                        brace_stack.pop();
                        result.push(ch);
                    } else {
                        // Unmatched closing brace - skip or add opening
                        // For now, we'll skip unmatched closing braces
                        // A more sophisticated approach would try to find where to insert the opening
                    }
                }
                '[' => {
                    brace_stack.push('[');
                    result.push(ch);
                }
                ']' => {
                    if brace_stack.last() == Some(&'[') {
                        brace_stack.pop();
                        result.push(ch);
                    }
                    // Skip unmatched closing brackets
                }
                '(' => {
                    brace_stack.push('(');
                    result.push(ch);
                }
                ')' => {
                    if brace_stack.last() == Some(&'(') {
                        brace_stack.pop();
                        result.push(ch);
                    }
                    // Skip unmatched closing parens
                }
                _ => {
                    result.push(ch);
                }
            }
        }

        // Add missing closing braces at the end
        while let Some(open) = brace_stack.pop() {
            match open {
                '{' => result.push('}'),
                '[' => result.push(']'),
                '(' => result.push(')'),
                _ => {}
            }
        }

        result
    }

    /// Remove common OCR artifacts.
    fn remove_artifacts(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Remove common artifacts
        let artifacts = [
            "\u{FFFD}",  // Replacement character
            "\u{0000}",  // Null
            "\u{FEFF}",  // BOM
            "\u{200B}",  // Zero-width space
            "\u{200C}",  // Zero-width non-joiner
            "\u{200D}",  // Zero-width joiner
            "\u{2060}",  // Word joiner
            "�",         // Common replacement display
        ];

        for artifact in &artifacts {
            result = result.replace(artifact, "");
        }

        // Remove repeated punctuation artifacts
        while result.contains("..") && !result.contains("...") {
            result = result.replace("..", ".");
        }

        // Preserve ellipsis but remove longer sequences
        while result.contains("....") {
            result = result.replace("....", "...");
        }

        result
    }

    /// Wrap lines to maximum length.
    fn wrap_lines(&self, content: &str, max_length: usize) -> String {
        let mut result = String::with_capacity(content.len());

        for line in content.lines() {
            if line.len() <= max_length {
                result.push_str(line);
                result.push('\n');
            } else {
                // Don't wrap lines that are in math mode or are commands
                if line.trim().starts_with('\\') || line.contains("\\[") || line.contains("\\]") {
                    result.push_str(line);
                    result.push('\n');
                } else {
                    // Simple word wrap
                    let words: Vec<&str> = line.split_whitespace().collect();
                    let mut current_line = String::new();

                    for word in words {
                        if current_line.is_empty() {
                            current_line = word.to_string();
                        } else if current_line.len() + 1 + word.len() <= max_length {
                            current_line.push(' ');
                            current_line.push_str(word);
                        } else {
                            result.push_str(&current_line);
                            result.push('\n');
                            current_line = word.to_string();
                        }
                    }

                    if !current_line.is_empty() {
                        result.push_str(&current_line);
                        result.push('\n');
                    }
                }
            }
        }

        // Remove trailing newline if original didn't have one
        if !content.ends_with('\n') && result.ends_with('\n') {
            result.pop();
        }

        result
    }

    /// Validate the document's LaTeX syntax.
    fn validate_document(&self, doc: &ExtractedDocument) -> PdfResult<()> {
        for (page_num, page) in doc.pages.iter().enumerate() {
            if let Err(issues) = self.validate_latex(&page.latex) {
                // Log warnings but don't fail - validation is informational
                for issue in issues {
                    eprintln!(
                        "Warning: Page {}: {}",
                        page_num + 1,
                        issue
                    );
                }
            }
        }
        Ok(())
    }

    /// Validate LaTeX content and return issues found.
    fn validate_latex(&self, content: &str) -> Result<(), Vec<String>> {
        let mut issues = Vec::new();

        // Check brace balance
        let mut brace_count = 0i32;
        let mut bracket_count = 0i32;
        let mut paren_count = 0i32;

        for ch in content.chars() {
            match ch {
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                '[' => bracket_count += 1,
                ']' => bracket_count -= 1,
                '(' => paren_count += 1,
                ')' => paren_count -= 1,
                _ => {}
            }

            // Check for negative counts (closing before opening)
            if brace_count < 0 {
                issues.push("Unmatched closing brace '}'".to_string());
                brace_count = 0;
            }
            if bracket_count < 0 {
                issues.push("Unmatched closing bracket ']'".to_string());
                bracket_count = 0;
            }
            if paren_count < 0 {
                issues.push("Unmatched closing parenthesis ')'".to_string());
                paren_count = 0;
            }
        }

        if brace_count > 0 {
            issues.push(format!("{} unclosed brace(s) '{{'", brace_count));
        }
        if bracket_count > 0 {
            issues.push(format!("{} unclosed bracket(s) '['", bracket_count));
        }
        if paren_count > 0 {
            issues.push(format!("{} unclosed parenthesis '('", paren_count));
        }

        // Check for unmatched math delimiters
        let dollar_count = content.matches('$').count();
        if dollar_count % 2 != 0 {
            issues.push("Unmatched '$' delimiter".to_string());
        }

        // Check for begin/end balance
        let begin_count = content.matches("\\begin{").count();
        let end_count = content.matches("\\end{").count();
        if begin_count != end_count {
            issues.push(format!(
                "Mismatched \\begin/{} and \\end/{}",
                begin_count, end_count
            ));
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let processor = PostProcessor::default_processor();

        let input = "Hello   world\t\ttab\n\n\n\nmany newlines";
        let result = processor.normalize_whitespace(input);
        assert_eq!(result, "Hello world tab\n\nmany newlines");
    }

    #[test]
    fn test_fix_ocr_errors() {
        let processor = PostProcessor::default_processor();

        let input = r"\bgin{equation} \frc{x}{y} \emd{equation}";
        let result = processor.fix_ocr_errors(input);
        assert!(result.contains(r"\begin{"));
        assert!(result.contains(r"\frac{"));
        assert!(result.contains(r"\end{"));
    }

    #[test]
    fn test_fix_brace_matching() {
        let processor = PostProcessor::default_processor();

        // Missing closing brace
        let input = r"\frac{x}{y";
        let result = processor.fix_brace_matching(input);
        assert!(result.ends_with('}'));

        // Extra closing brace
        let input2 = r"\frac{x}}{y}";
        let result2 = processor.fix_brace_matching(input2);
        assert!(!result2.contains("}}"));
    }

    #[test]
    fn test_remove_artifacts() {
        let processor = PostProcessor::default_processor();

        let input = "Hello\u{FFFD}World\u{200B}!";
        let result = processor.remove_artifacts(input);
        assert_eq!(result, "HelloWorld!");
    }

    #[test]
    fn test_validate_latex() {
        let processor = PostProcessor::default_processor();

        // Valid LaTeX
        let valid = r"\begin{equation} x = y \end{equation}";
        assert!(processor.validate_latex(valid).is_ok());

        // Unbalanced braces
        let invalid = r"\frac{x}{y";
        assert!(processor.validate_latex(invalid).is_err());

        // Unmatched begin/end
        let invalid2 = r"\begin{equation} x = y";
        assert!(processor.validate_latex(invalid2).is_err());
    }

    #[test]
    fn test_config_presets() {
        let minimal = PostProcessorConfig::minimal();
        assert!(!minimal.validate_latex);
        assert!(!minimal.fix_ocr_errors);

        let thorough = PostProcessorConfig::thorough();
        assert!(thorough.validate_latex);
        assert!(thorough.fix_ocr_errors);
        assert_eq!(thorough.max_line_length, 80);
    }
}
