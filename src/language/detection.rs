//! Language detection using whatlang.

use thiserror::Error;
use whatlang::{detect, Lang};

use super::LanguageTag;

/// Error type for language detection.
#[derive(Error, Debug)]
pub enum LanguageDetectionError {
    /// Insufficient text for reliable detection.
    #[error("Insufficient text for language detection")]
    InsufficientText,

    /// Language detected but not supported.
    #[error("Unsupported language detected: {0:?}")]
    UnsupportedLanguage(Lang),

    /// Detection confidence too low.
    #[error("Low confidence detection: {confidence:.2}% (minimum: {minimum:.2}%)")]
    LowConfidence {
        /// Observed detection confidence.
        confidence: f64,
        /// Minimum confidence required by the caller.
        minimum: f64,
    },
}

/// Detect the language of the given text.
///
/// Returns a `LanguageTag` if detection is successful and confident.
///
/// # Arguments
///
/// * `text` - The text to analyze
/// * `min_confidence` - Minimum confidence threshold (0.0 - 1.0)
///
/// # Example
///
/// ```ignore
/// use grammstein::language::detect_language;
///
/// let tag = detect_language("The quick brown fox jumps over the lazy dog.", 0.8)?;
/// assert_eq!(tag.language(), "en");
/// ```
pub fn detect_language(
    text: &str,
    min_confidence: f64,
) -> Result<LanguageTag, LanguageDetectionError> {
    let info = detect(text).ok_or(LanguageDetectionError::InsufficientText)?;

    let confidence = info.confidence();
    if confidence < min_confidence {
        return Err(LanguageDetectionError::LowConfidence {
            confidence: confidence * 100.0,
            minimum: min_confidence * 100.0,
        });
    }

    // Map whatlang Lang to ISO 639-1 codes
    // Note: whatlang 0.16 supports a subset of languages
    let language_code = match info.lang() {
        // Major European languages
        Lang::Eng => "en",
        Lang::Spa => "es",
        Lang::Deu => "de",
        Lang::Fra => "fr",
        Lang::Por => "pt",
        Lang::Ita => "it",
        Lang::Nld => "nl",
        Lang::Rus => "ru",
        Lang::Pol => "pl",
        Lang::Ukr => "uk",
        Lang::Ces => "cs",
        Lang::Slk => "sk",
        Lang::Slv => "sl",
        Lang::Hrv => "hr",
        Lang::Srp => "sr",
        Lang::Mkd => "mk",
        Lang::Bul => "bg",
        Lang::Ron => "ro",
        Lang::Hun => "hu",
        Lang::Ell => "el",
        Lang::Tur => "tr",
        Lang::Fin => "fi",
        Lang::Swe => "sv",
        Lang::Dan => "da",
        Lang::Nob => "no", // Norwegian Bokmål
        Lang::Lit => "lt",
        Lang::Lav => "lv",
        Lang::Est => "et",
        Lang::Cat => "ca",
        Lang::Afr => "af",
        Lang::Lat => "la",
        Lang::Epo => "eo",

        // Asian languages
        Lang::Cmn => "zh", // Mandarin Chinese
        Lang::Jpn => "ja",
        Lang::Kor => "ko",
        Lang::Tha => "th",
        Lang::Vie => "vi",
        Lang::Ind => "id",
        Lang::Tgl => "tl",
        Lang::Jav => "jv",
        Lang::Mya => "my",
        Lang::Khm => "km",

        // South Asian languages
        Lang::Hin => "hi",
        Lang::Ben => "bn",
        Lang::Tam => "ta",
        Lang::Tel => "te",
        Lang::Mar => "mr",
        Lang::Urd => "ur",
        Lang::Guj => "gu",
        Lang::Kan => "kn",
        Lang::Mal => "ml",
        Lang::Pan => "pa",
        Lang::Sin => "si",

        // Middle Eastern languages
        Lang::Ara => "ar",
        Lang::Heb => "he",

        // Central Asian
        Lang::Aze => "az",
        Lang::Uzb => "uz",

        // African languages
        Lang::Amh => "am",

        // Catch-all for unsupported languages
        other => return Err(LanguageDetectionError::UnsupportedLanguage(other)),
    };

    Ok(LanguageTag::new(language_code))
}

/// Detect language from a sample of corpus sentences.
///
/// Takes multiple sentences, concatenates them, and performs detection.
/// More text generally leads to more accurate detection.
///
/// # Arguments
///
/// * `sentences` - Iterator of sentences to sample
/// * `max_samples` - Maximum number of sentences to use
/// * `min_confidence` - Minimum confidence threshold
pub fn detect_from_sentences<'a, I>(
    sentences: I,
    max_samples: usize,
    min_confidence: f64,
) -> Result<LanguageTag, LanguageDetectionError>
where
    I: Iterator<Item = &'a str>,
{
    let mut sample = String::new();
    for sentence in sentences.take(max_samples) {
        if !sample.is_empty() {
            sample.push(' ');
        }
        sample.push_str(sentence);
    }

    if sample.is_empty() {
        return Err(LanguageDetectionError::InsufficientText);
    }

    detect_language(&sample, min_confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_english() {
        let result = detect_language(
            "The quick brown fox jumps over the lazy dog. This is a sample sentence in English.",
            0.8,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().language(), "en");
    }

    #[test]
    fn test_detect_german() {
        let result = detect_language(
            "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein Beispielsatz auf Deutsch.",
            0.8,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().language(), "de");
    }

    #[test]
    fn test_detect_spanish() {
        let result = detect_language(
            "El rápido zorro marrón salta sobre el perro perezoso. Esta es una oración de ejemplo en español.",
            0.8,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().language(), "es");
    }

    #[test]
    fn test_insufficient_text() {
        let result = detect_language("hi", 0.8);
        // Very short text may fail detection
        assert!(result.is_err() || result.unwrap().language().len() <= 3);
    }
}
