//! Parser for Google Books N-gram TSV format.
//!
//! Google Books n-gram files use tab-separated format:
//! ```text
//! ngram\tyear\tmatch_count\tvolume_count
//! ```
//!
//! Example lines:
//! ```text
//! the     2000    5000000000      1000000
//! the cat 2000    12345678        234567
//! ```

/// A single n-gram record from Google Books.
///
/// Represents one line from a Google Books n-gram file, containing
/// the n-gram text and its frequency statistics for a specific year.
#[derive(Clone, Debug, PartialEq)]
pub struct NgramRecord {
    /// The n-gram text (space-separated words).
    ///
    /// For unigrams: "the"
    /// For bigrams: "the cat"
    /// For trigrams: "the cat sat"
    pub ngram: String,

    /// Publication year.
    ///
    /// Google Books data ranges from 1500 to present,
    /// though most data is from 1800 onwards.
    pub year: u16,

    /// Number of occurrences in that year.
    ///
    /// This is the raw frequency count across all books
    /// published in this year.
    pub match_count: u64,

    /// Number of distinct volumes containing this n-gram.
    ///
    /// Useful for filtering: n-grams appearing in very few
    /// volumes may be errors or domain-specific.
    pub volume_count: u32,
}

impl NgramRecord {
    /// Get the n-gram order (number of words).
    pub fn order(&self) -> u8 {
        self.ngram.split_whitespace().count() as u8
    }

    /// Check if this n-gram contains POS tags (e.g., "_NOUN_").
    pub fn has_pos_tag(&self) -> bool {
        self.ngram.contains("_")
    }
}

/// Parse errors for n-gram lines.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// Line has wrong number of fields.
    #[error("Expected 4 tab-separated fields, got {0}")]
    WrongFieldCount(usize),

    /// Failed to parse year.
    #[error("Invalid year: {0}")]
    InvalidYear(String),

    /// Failed to parse match count.
    #[error("Invalid match count: {0}")]
    InvalidMatchCount(String),

    /// Failed to parse volume count.
    #[error("Invalid volume count: {0}")]
    InvalidVolumeCount(String),

    /// Empty n-gram.
    #[error("Empty n-gram")]
    EmptyNgram,
}

/// A borrowed n-gram record parsed without heap allocation.
///
/// All string fields are borrowed from the input line. Use this in hot loops
/// where the line buffer outlives the record (e.g., streaming aggregation).
#[derive(Clone, Debug, PartialEq)]
pub struct NgramRecordRef<'a> {
    /// The n-gram text (borrowed from input line).
    pub ngram: &'a str,

    /// Publication year.
    pub year: u16,

    /// Number of occurrences in that year.
    pub match_count: u64,

    /// Number of distinct volumes containing this n-gram.
    pub volume_count: u32,
}

/// Find the next occurrence of `needle` in `haystack` starting at `start`.
#[inline(always)]
fn find_byte(haystack: &[u8], start: usize, needle: u8) -> Option<usize> {
    haystack[start..].iter().position(|&b| b == needle).map(|i| start + i)
}

/// Parse a single line from a Google Books n-gram file into a borrowed record.
///
/// This is a zero-allocation parser that finds tab positions via byte scanning
/// instead of `split().collect()`. The returned `NgramRecordRef` borrows the
/// ngram text directly from the input line.
///
/// # Format
///
/// Tab-separated: `ngram\tyear\tmatch_count\tvolume_count`
///
/// # Example
///
/// ```ignore
/// let rec = parse_ngram_line_ref("the cat\t2000\t12345\t678")?;
/// assert_eq!(rec.ngram, "the cat");
/// assert_eq!(rec.year, 2000);
/// assert_eq!(rec.match_count, 12345);
/// assert_eq!(rec.volume_count, 678);
/// ```
pub fn parse_ngram_line_ref(line: &str) -> Result<NgramRecordRef<'_>, ParseError> {
    let bytes = line.as_bytes();

    // Find exactly 3 tabs (4 fields)
    let tab1 = find_byte(bytes, 0, b'\t').ok_or(ParseError::WrongFieldCount(1))?;
    let tab2 = find_byte(bytes, tab1 + 1, b'\t').ok_or(ParseError::WrongFieldCount(2))?;
    let tab3 = find_byte(bytes, tab2 + 1, b'\t').ok_or(ParseError::WrongFieldCount(3))?;

    // Reject if there's a 4th tab (5+ fields)
    if find_byte(bytes, tab3 + 1, b'\t').is_some() {
        return Err(ParseError::WrongFieldCount(5));
    }

    // Extract fields as &str slices (safe: tab positions are within the UTF-8 string)
    let ngram = &line[..tab1];
    if ngram.is_empty() {
        return Err(ParseError::EmptyNgram);
    }

    let year_str = &line[tab1 + 1..tab2];
    let year = year_str
        .parse::<u16>()
        .map_err(|_| ParseError::InvalidYear(year_str.to_string()))?;

    let match_str = &line[tab2 + 1..tab3];
    let match_count = match_str
        .parse::<u64>()
        .map_err(|_| ParseError::InvalidMatchCount(match_str.to_string()))?;

    let vol_str = &line[tab3 + 1..];
    let volume_count = vol_str
        .parse::<u32>()
        .map_err(|_| ParseError::InvalidVolumeCount(vol_str.to_string()))?;

    Ok(NgramRecordRef {
        ngram,
        year,
        match_count,
        volume_count,
    })
}

/// Parse a single line from a Google Books n-gram file.
///
/// # Format
///
/// Tab-separated: `ngram\tyear\tmatch_count\tvolume_count`
///
/// # Example
///
/// ```ignore
/// let record = parse_ngram_line("the cat\t2000\t12345\t678")?;
/// assert_eq!(record.ngram, "the cat");
/// assert_eq!(record.year, 2000);
/// assert_eq!(record.match_count, 12345);
/// assert_eq!(record.volume_count, 678);
/// ```
pub fn parse_ngram_line(line: &str) -> Result<NgramRecord, ParseError> {
    let rec = parse_ngram_line_ref(line)?;
    Ok(NgramRecord {
        ngram: rec.ngram.to_string(),
        year: rec.year,
        match_count: rec.match_count,
        volume_count: rec.volume_count,
    })
}

/// Parse a batch of lines efficiently.
///
/// Skips invalid lines (logs warning) and returns all successfully parsed records.
pub fn parse_ngram_lines<'a>(
    lines: impl Iterator<Item = &'a str> + 'a,
) -> impl Iterator<Item = NgramRecord> + 'a {
    lines.filter_map(|line| {
        match parse_ngram_line(line) {
            Ok(record) => Some(record),
            Err(e) => {
                log::warn!("Skipping invalid line: {}", e);
                None
            }
        }
    })
}

/// Check if an n-gram contains POS tags.
///
/// Google Books includes syntactic annotations like:
/// - `_NOUN_`
/// - `_VERB_`
/// - `_ADJ_`
/// - `_ADV_`
/// - `_PRON_`
/// - `_DET_`
/// - `_ADP_`
/// - `_NUM_`
/// - `_CONJ_`
/// - `_PRT_`
/// - `_ROOT_`
/// - `_START_`
/// - `_END_`
pub fn contains_pos_tag(ngram: &str) -> bool {
    ngram.contains("_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_unigram() {
        let record = parse_ngram_line("the\t2000\t5000000000\t1000000").unwrap();
        assert_eq!(record.ngram, "the");
        assert_eq!(record.year, 2000);
        assert_eq!(record.match_count, 5_000_000_000);
        assert_eq!(record.volume_count, 1_000_000);
        assert_eq!(record.order(), 1);
    }

    #[test]
    fn test_parse_bigram() {
        let record = parse_ngram_line("the cat\t1950\t12345\t678").unwrap();
        assert_eq!(record.ngram, "the cat");
        assert_eq!(record.year, 1950);
        assert_eq!(record.match_count, 12345);
        assert_eq!(record.volume_count, 678);
        assert_eq!(record.order(), 2);
    }

    #[test]
    fn test_parse_fivegram() {
        let record = parse_ngram_line("the quick brown fox jumps\t2010\t999\t88").unwrap();
        assert_eq!(record.order(), 5);
    }

    #[test]
    fn test_parse_pos_tag() {
        let record = parse_ngram_line("the_DET_\t2000\t100\t10").unwrap();
        assert!(record.has_pos_tag());

        let record = parse_ngram_line("the cat_NOUN_\t2000\t100\t10").unwrap();
        assert!(record.has_pos_tag());
    }

    #[test]
    fn test_parse_unicode() {
        let record = parse_ngram_line("Müller\t2000\t100\t10").unwrap();
        assert_eq!(record.ngram, "Müller");

        let record = parse_ngram_line("日本語\t2000\t100\t10").unwrap();
        assert_eq!(record.ngram, "日本語");
    }

    #[test]
    fn test_parse_wrong_field_count() {
        let result = parse_ngram_line("the\t2000\t100");
        assert!(matches!(result, Err(ParseError::WrongFieldCount(3))));

        let result = parse_ngram_line("the\t2000\t100\t10\textra");
        assert!(matches!(result, Err(ParseError::WrongFieldCount(5))));
    }

    #[test]
    fn test_parse_invalid_year() {
        let result = parse_ngram_line("the\tabc\t100\t10");
        assert!(matches!(result, Err(ParseError::InvalidYear(_))));
    }

    #[test]
    fn test_parse_invalid_count() {
        let result = parse_ngram_line("the\t2000\tabc\t10");
        assert!(matches!(result, Err(ParseError::InvalidMatchCount(_))));

        let result = parse_ngram_line("the\t2000\t100\tabc");
        assert!(matches!(result, Err(ParseError::InvalidVolumeCount(_))));
    }

    #[test]
    fn test_parse_empty_ngram() {
        let result = parse_ngram_line("\t2000\t100\t10");
        assert!(matches!(result, Err(ParseError::EmptyNgram)));
    }

    #[test]
    fn test_contains_pos_tag() {
        assert!(contains_pos_tag("the_DET_"));
        assert!(contains_pos_tag("cat_NOUN_"));
        assert!(!contains_pos_tag("the"));
        assert!(!contains_pos_tag("the cat"));
    }

    // ---- Zero-alloc parse_ngram_line_ref tests ----

    #[test]
    fn test_parse_ref_unigram() {
        let rec = parse_ngram_line_ref("the\t2000\t5000000000\t1000000").unwrap();
        assert_eq!(rec.ngram, "the");
        assert_eq!(rec.year, 2000);
        assert_eq!(rec.match_count, 5_000_000_000);
        assert_eq!(rec.volume_count, 1_000_000);
    }

    #[test]
    fn test_parse_ref_bigram() {
        let rec = parse_ngram_line_ref("the cat\t1950\t12345\t678").unwrap();
        assert_eq!(rec.ngram, "the cat");
        assert_eq!(rec.year, 1950);
        assert_eq!(rec.match_count, 12345);
        assert_eq!(rec.volume_count, 678);
    }

    #[test]
    fn test_parse_ref_unicode() {
        let rec = parse_ngram_line_ref("Müller\t2000\t100\t10").unwrap();
        assert_eq!(rec.ngram, "Müller");
    }

    #[test]
    fn test_parse_ref_wrong_field_count() {
        // Too few fields (1 field, 0 tabs)
        let result = parse_ngram_line_ref("the");
        assert!(matches!(result, Err(ParseError::WrongFieldCount(1))));

        // Too few fields (3 fields, 2 tabs)
        let result = parse_ngram_line_ref("the\t2000\t100");
        assert!(matches!(result, Err(ParseError::WrongFieldCount(3))));

        // Too many fields (5 fields, 4 tabs)
        let result = parse_ngram_line_ref("the\t2000\t100\t10\textra");
        assert!(matches!(result, Err(ParseError::WrongFieldCount(5))));
    }

    #[test]
    fn test_parse_ref_empty_ngram() {
        let result = parse_ngram_line_ref("\t2000\t100\t10");
        assert!(matches!(result, Err(ParseError::EmptyNgram)));
    }

    #[test]
    fn test_parse_ref_invalid_fields() {
        let result = parse_ngram_line_ref("the\tabc\t100\t10");
        assert!(matches!(result, Err(ParseError::InvalidYear(_))));

        let result = parse_ngram_line_ref("the\t2000\tabc\t10");
        assert!(matches!(result, Err(ParseError::InvalidMatchCount(_))));

        let result = parse_ngram_line_ref("the\t2000\t100\tabc");
        assert!(matches!(result, Err(ParseError::InvalidVolumeCount(_))));
    }

    #[test]
    fn test_parse_ref_matches_owned() {
        // Verify ref and owned parsers produce identical results
        let lines = [
            "the\t2000\t5000000000\t1000000",
            "the cat\t1950\t12345\t678",
            "the quick brown fox jumps\t2010\t999\t88",
            "Müller\t2000\t100\t10",
        ];
        for line in lines {
            let owned = parse_ngram_line(line).unwrap();
            let borrowed = parse_ngram_line_ref(line).unwrap();
            assert_eq!(owned.ngram, borrowed.ngram);
            assert_eq!(owned.year, borrowed.year);
            assert_eq!(owned.match_count, borrowed.match_count);
            assert_eq!(owned.volume_count, borrowed.volume_count);
        }
    }
}
