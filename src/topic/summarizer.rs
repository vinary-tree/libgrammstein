//! Topic description generation.
//!
//! This module generates natural language descriptions for topics
//! from keywords and representative documents.

use super::config::{DescriptionTemplateType, SummarizationConfig};
use super::ctfidf::format_keywords;
use super::TopicId;

/// Topic summarizer for generating descriptions.
pub struct TopicSummarizer {
    /// Configuration.
    config: SummarizationConfig,
}

impl TopicSummarizer {
    /// Create a new topic summarizer.
    pub fn new(config: SummarizationConfig) -> Self {
        Self { config }
    }

    /// Generate a description for a topic from its keywords.
    pub fn describe_from_keywords(&self, keywords: &[(String, f32)]) -> String {
        if keywords.is_empty() {
            return "Empty topic".to_string();
        }

        let keyword_str = format_keywords(keywords);

        match self.config.template {
            DescriptionTemplateType::Keywords => keyword_str,
            DescriptionTemplateType::Label => format!("Topic covering: {}", keyword_str),
            DescriptionTemplateType::Extractive => {
                // For extractive, we'd need documents - fall back to keywords
                format!("Topic: {}", keyword_str)
            }
            DescriptionTemplateType::Custom => {
                if let Some(template) = &self.config.custom_template {
                    template.replace("{keywords}", &keyword_str)
                } else {
                    keyword_str
                }
            }
        }
    }

    /// Generate a description from keywords and representative document snippets.
    pub fn describe_with_context(
        &self,
        keywords: &[(String, f32)],
        representative_docs: &[&str],
    ) -> String {
        if keywords.is_empty() && representative_docs.is_empty() {
            return "Empty topic".to_string();
        }

        let keyword_str = format_keywords(keywords);

        match self.config.template {
            DescriptionTemplateType::Extractive => {
                if let Some(first_doc) = representative_docs.first() {
                    // Use first sentence or truncate
                    let summary = extract_first_sentence(first_doc);
                    truncate_to_length(&summary, self.config.max_description_length)
                } else {
                    format!("Topic: {}", keyword_str)
                }
            }
            _ => self.describe_from_keywords(keywords),
        }
    }

    /// Generate a full topic label.
    pub fn generate_label(&self, topic_id: TopicId, keywords: &[(String, f32)]) -> String {
        if keywords.is_empty() {
            return format!("Topic {}", topic_id.as_u32());
        }

        // Use top 3 keywords for label
        let top_keywords: Vec<_> = keywords
            .iter()
            .take(3)
            .map(|(k, _)| k.as_str())
            .collect();

        format!("Topic {}: {}", topic_id.as_u32(), top_keywords.join(", "))
    }

    /// Get configuration.
    pub fn config(&self) -> &SummarizationConfig {
        &self.config
    }
}

/// Extract the first sentence from text.
fn extract_first_sentence(text: &str) -> String {
    // Find first sentence-ending punctuation
    if let Some(pos) = text.find(|c| c == '.' || c == '!' || c == '?') {
        text[..=pos].trim().to_string()
    } else {
        text.trim().to_string()
    }
}

/// Truncate text to a maximum length, respecting word boundaries.
fn truncate_to_length(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    // Find last space before max_len
    let truncated = &text[..max_len];
    if let Some(pos) = truncated.rfind(' ') {
        format!("{}...", &text[..pos])
    } else {
        format!("{}...", truncated)
    }
}

/// Compute topic coherence score.
///
/// Coherence measures how semantically related the top keywords are.
/// Uses NPMI (Normalized Pointwise Mutual Information) approximation.
pub fn compute_coherence(
    keywords: &[(String, f32)],
    co_occurrence_fn: impl Fn(&str, &str) -> f64,
) -> f64 {
    if keywords.len() < 2 {
        return 0.0;
    }

    let top_terms: Vec<_> = keywords.iter().take(10).map(|(k, _)| k.as_str()).collect();
    let n = top_terms.len();
    let mut total_npmi = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let npmi = co_occurrence_fn(top_terms[i], top_terms[j]);
            total_npmi += npmi;
            count += 1;
        }
    }

    if count > 0 {
        total_npmi / count as f64
    } else {
        0.0
    }
}

/// Simple co-occurrence counter for coherence computation.
pub struct CoOccurrenceCounter {
    /// Word pair counts.
    pair_counts: std::collections::HashMap<(String, String), usize>,
    /// Single word counts.
    word_counts: std::collections::HashMap<String, usize>,
    /// Total windows processed.
    total_windows: usize,
    /// Window size.
    window_size: usize,
}

impl CoOccurrenceCounter {
    /// Create a new counter.
    pub fn new(window_size: usize) -> Self {
        Self {
            pair_counts: std::collections::HashMap::new(),
            word_counts: std::collections::HashMap::new(),
            total_windows: 0,
            window_size,
        }
    }

    /// Process a document's tokens.
    pub fn process(&mut self, tokens: &[String]) {
        if tokens.is_empty() {
            return;
        }

        // Sliding window
        for i in 0..tokens.len() {
            let word = &tokens[i];
            *self.word_counts.entry(word.clone()).or_insert(0) += 1;

            let end = (i + self.window_size).min(tokens.len());
            for j in (i + 1)..end {
                let other = &tokens[j];
                let key = if word < other {
                    (word.clone(), other.clone())
                } else {
                    (other.clone(), word.clone())
                };
                *self.pair_counts.entry(key).or_insert(0) += 1;
            }

            self.total_windows += 1;
        }
    }

    /// Compute NPMI for a word pair.
    pub fn npmi(&self, word1: &str, word2: &str) -> f64 {
        let key = if word1 < word2 {
            (word1.to_string(), word2.to_string())
        } else {
            (word2.to_string(), word1.to_string())
        };

        let pair_count = self.pair_counts.get(&key).copied().unwrap_or(0) as f64;
        let count1 = self.word_counts.get(word1).copied().unwrap_or(0) as f64;
        let count2 = self.word_counts.get(word2).copied().unwrap_or(0) as f64;
        let total = self.total_windows as f64;

        if pair_count == 0.0 || count1 == 0.0 || count2 == 0.0 || total == 0.0 {
            return 0.0;
        }

        // PMI = log(P(w1,w2) / (P(w1) * P(w2)))
        let p_pair = pair_count / total;
        let p1 = count1 / total;
        let p2 = count2 / total;

        let pmi = (p_pair / (p1 * p2)).ln();

        // NPMI = PMI / -log(P(w1,w2))
        let normalization = -p_pair.ln();
        if normalization > 0.0 {
            pmi / normalization
        } else {
            0.0
        }
    }
}

impl Default for TopicSummarizer {
    fn default() -> Self {
        Self::new(SummarizationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_describe_from_keywords() {
        let summarizer = TopicSummarizer::default();

        let keywords = vec![
            ("machine".to_string(), 0.5),
            ("learning".to_string(), 0.4),
            ("neural".to_string(), 0.3),
        ];

        let desc = summarizer.describe_from_keywords(&keywords);
        assert!(desc.contains("machine"));
        assert!(desc.contains("learning"));
    }

    #[test]
    fn test_describe_with_label_template() {
        let config = SummarizationConfig {
            template: DescriptionTemplateType::Label,
            ..Default::default()
        };
        let summarizer = TopicSummarizer::new(config);

        let keywords = vec![("data".to_string(), 0.5), ("science".to_string(), 0.4)];

        let desc = summarizer.describe_from_keywords(&keywords);
        assert!(desc.starts_with("Topic covering:"));
        assert!(desc.contains("data"));
    }

    #[test]
    fn test_describe_with_custom_template() {
        let config = SummarizationConfig {
            template: DescriptionTemplateType::Custom,
            custom_template: Some("This topic discusses: {keywords}".to_string()),
            ..Default::default()
        };
        let summarizer = TopicSummarizer::new(config);

        let keywords = vec![("rust".to_string(), 0.5), ("programming".to_string(), 0.4)];

        let desc = summarizer.describe_from_keywords(&keywords);
        assert!(desc.starts_with("This topic discusses:"));
        assert!(desc.contains("rust"));
    }

    #[test]
    fn test_generate_label() {
        let summarizer = TopicSummarizer::default();

        let keywords = vec![
            ("ai".to_string(), 0.5),
            ("ml".to_string(), 0.4),
            ("dl".to_string(), 0.3),
            ("nn".to_string(), 0.2),
        ];

        let label = summarizer.generate_label(TopicId::new(5), &keywords);
        assert!(label.contains("Topic 5"));
        assert!(label.contains("ai"));
        assert!(label.contains("ml"));
        assert!(label.contains("dl"));
        assert!(!label.contains("nn")); // Only top 3
    }

    #[test]
    fn test_extract_first_sentence() {
        assert_eq!(
            extract_first_sentence("Hello world. This is more."),
            "Hello world."
        );
        assert_eq!(
            extract_first_sentence("No punctuation here"),
            "No punctuation here"
        );
        assert_eq!(
            extract_first_sentence("Is this a question? Yes."),
            "Is this a question?"
        );
    }

    #[test]
    fn test_truncate_to_length() {
        assert_eq!(truncate_to_length("short", 100), "short");
        assert_eq!(
            truncate_to_length("hello world test", 10),
            "hello..."
        );
        assert_eq!(truncate_to_length("nospaces", 5), "nospa...");
    }

    #[test]
    fn test_co_occurrence_counter() {
        let mut counter = CoOccurrenceCounter::new(5);

        counter.process(&[
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ]);
        counter.process(&[
            "the".to_string(),
            "lazy".to_string(),
            "brown".to_string(),
            "dog".to_string(),
        ]);

        // "the" and "brown" co-occur in both documents
        let npmi = counter.npmi("the", "brown");
        assert!(npmi > 0.0);
    }

    #[test]
    fn test_compute_coherence() {
        let keywords = vec![
            ("machine".to_string(), 0.5),
            ("learning".to_string(), 0.4),
            ("algorithm".to_string(), 0.3),
        ];

        // Mock co-occurrence function that returns high values for related words
        let coherence = compute_coherence(&keywords, |w1, w2| {
            if (w1 == "machine" && w2 == "learning") || (w1 == "learning" && w2 == "machine") {
                0.8
            } else {
                0.2
            }
        });

        // Should have some coherence
        assert!(coherence > 0.0);
    }

    #[test]
    fn test_describe_with_context() {
        let config = SummarizationConfig {
            template: DescriptionTemplateType::Extractive,
            max_description_length: 50,
            ..Default::default()
        };
        let summarizer = TopicSummarizer::new(config);

        let keywords = vec![("test".to_string(), 0.5)];
        let docs = vec!["This is the first sentence. And this is another."];

        let desc = summarizer.describe_with_context(&keywords, &docs);
        assert!(desc.contains("first sentence"));
    }

    #[test]
    fn test_empty_keywords() {
        let summarizer = TopicSummarizer::default();

        let desc = summarizer.describe_from_keywords(&[]);
        assert_eq!(desc, "Empty topic");

        let label = summarizer.generate_label(TopicId::new(0), &[]);
        assert_eq!(label, "Topic 0");
    }
}
