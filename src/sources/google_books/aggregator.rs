//! Year aggregation for Google Books n-grams.
//!
//! Google Books data provides counts broken down by year. For language modeling,
//! we typically want the total count across all years (or a filtered year range).
//!
//! This module provides streaming aggregation that is memory-efficient:
//! it only buffers records for one n-gram at a time.

use super::parser::{NgramRecord, NgramRecordRef};

/// Aggregated n-gram statistics across years.
#[derive(Clone, Debug, Default)]
pub struct AggregatedNgram {
    /// The n-gram text.
    pub ngram: String,

    /// Total match count summed across all years.
    pub total_count: u64,

    /// Maximum volume count across any single year.
    ///
    /// This is more useful than sum because the same volume
    /// can contain the n-gram across multiple years.
    pub max_volume_count: u32,

    /// Number of distinct years with data.
    pub year_span: u16,

    /// First year with data.
    pub first_year: u16,

    /// Last year with data.
    pub last_year: u16,
}

/// Streaming year aggregator.
///
/// Since Google Books files are sorted by n-gram, we can aggregate
/// efficiently by buffering only one n-gram's worth of year data.
///
/// # Example
///
/// ```ignore
/// let mut aggregator = YearAggregator::new(None);
///
/// for record in records {
///     if let Some(aggregated) = aggregator.push(record) {
///         // Previous n-gram is complete, process it
///         process(aggregated);
///     }
/// }
///
/// // Don't forget the last n-gram
/// if let Some(aggregated) = aggregator.flush() {
///     process(aggregated);
/// }
/// ```
pub struct YearAggregator {
    /// Optional year range filter.
    year_range: Option<(u16, u16)>,

    /// Current n-gram being aggregated.
    current: Option<AggregatedNgram>,
}

impl YearAggregator {
    /// Create a new aggregator with optional year range filter.
    pub fn new(year_range: Option<(u16, u16)>) -> Self {
        Self {
            year_range,
            current: None,
        }
    }

    /// Push a record and return the previous n-gram if complete.
    ///
    /// Returns `Some(aggregated)` when we encounter a new n-gram,
    /// indicating the previous one is complete.
    pub fn push(&mut self, record: NgramRecord) -> Option<AggregatedNgram> {
        // Check year filter
        if let Some((start, end)) = self.year_range {
            if record.year < start || record.year > end {
                return None;
            }
        }

        match &mut self.current {
            Some(current) if current.ngram == record.ngram => {
                // Same n-gram, accumulate
                current.total_count += record.match_count;
                current.max_volume_count = current.max_volume_count.max(record.volume_count);
                current.year_span += 1;
                current.first_year = current.first_year.min(record.year);
                current.last_year = current.last_year.max(record.year);
                None
            }
            Some(_) => {
                // New n-gram, swap out the old one
                let previous = self.current.take();
                self.current = Some(AggregatedNgram {
                    ngram: record.ngram,
                    total_count: record.match_count,
                    max_volume_count: record.volume_count,
                    year_span: 1,
                    first_year: record.year,
                    last_year: record.year,
                });
                previous
            }
            None => {
                // First record
                self.current = Some(AggregatedNgram {
                    ngram: record.ngram,
                    total_count: record.match_count,
                    max_volume_count: record.volume_count,
                    year_span: 1,
                    first_year: record.year,
                    last_year: record.year,
                });
                None
            }
        }
    }

    /// Push a borrowed record, avoiding String allocation when the ngram matches.
    ///
    /// This is the zero-alloc hot path: when the ngram is the same as the current
    /// buffered ngram, no heap allocation occurs. A String is only allocated when
    /// a new ngram is first encountered.
    ///
    /// Returns `Some(aggregated)` when we encounter a new n-gram,
    /// indicating the previous one is complete.
    pub fn push_ref(&mut self, record: &NgramRecordRef<'_>) -> Option<AggregatedNgram> {
        // Check year filter
        if let Some((start, end)) = self.year_range {
            if record.year < start || record.year > end {
                return None;
            }
        }

        match &mut self.current {
            Some(current) if current.ngram == record.ngram => {
                // Same n-gram, accumulate — no String allocation
                current.total_count += record.match_count;
                current.max_volume_count = current.max_volume_count.max(record.volume_count);
                current.year_span += 1;
                current.first_year = current.first_year.min(record.year);
                current.last_year = current.last_year.max(record.year);
                None
            }
            Some(_) => {
                // New n-gram — allocate String only here
                let previous = self.current.take();
                self.current = Some(AggregatedNgram {
                    ngram: record.ngram.to_string(),
                    total_count: record.match_count,
                    max_volume_count: record.volume_count,
                    year_span: 1,
                    first_year: record.year,
                    last_year: record.year,
                });
                previous
            }
            None => {
                // First record — allocate String
                self.current = Some(AggregatedNgram {
                    ngram: record.ngram.to_string(),
                    total_count: record.match_count,
                    max_volume_count: record.volume_count,
                    year_span: 1,
                    first_year: record.year,
                    last_year: record.year,
                });
                None
            }
        }
    }

    /// Flush the final n-gram.
    ///
    /// Call this after processing all records to get the last n-gram.
    pub fn flush(&mut self) -> Option<AggregatedNgram> {
        self.current.take()
    }

    /// Reset the aggregator state.
    pub fn reset(&mut self) {
        self.current = None;
    }
}

/// Iterator adapter that aggregates year data from a stream of NgramRecords.
pub struct AggregatingIterator<I> {
    inner: I,
    aggregator: YearAggregator,
    flushed: bool,
}

impl<I> AggregatingIterator<I>
where
    I: Iterator<Item = NgramRecord>,
{
    /// Create a new aggregating iterator.
    pub fn new(inner: I, year_range: Option<(u16, u16)>) -> Self {
        Self {
            inner,
            aggregator: YearAggregator::new(year_range),
            flushed: false,
        }
    }
}

impl<I> Iterator for AggregatingIterator<I>
where
    I: Iterator<Item = NgramRecord>,
{
    type Item = AggregatedNgram;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to get the next aggregated n-gram from the stream
        loop {
            match self.inner.next() {
                Some(record) => {
                    if let Some(aggregated) = self.aggregator.push(record) {
                        return Some(aggregated);
                    }
                    // Continue to next record
                }
                None => {
                    // Stream exhausted, flush remaining
                    if !self.flushed {
                        self.flushed = true;
                        return self.aggregator.flush();
                    }
                    return None;
                }
            }
        }
    }
}

/// Extension trait for adding year aggregation to iterators.
pub trait AggregateExt: Iterator<Item = NgramRecord> + Sized {
    /// Aggregate year data from n-gram records.
    fn aggregate_years(self, year_range: Option<(u16, u16)>) -> AggregatingIterator<Self> {
        AggregatingIterator::new(self, year_range)
    }
}

impl<I: Iterator<Item = NgramRecord>> AggregateExt for I {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(ngram: &str, year: u16, count: u64) -> NgramRecord {
        NgramRecord {
            ngram: ngram.to_string(),
            year,
            match_count: count,
            volume_count: 100,
        }
    }

    #[test]
    fn test_single_ngram_multiple_years() {
        let mut agg = YearAggregator::new(None);

        assert!(agg.push(make_record("the", 2000, 100)).is_none());
        assert!(agg.push(make_record("the", 2001, 200)).is_none());
        assert!(agg.push(make_record("the", 2002, 150)).is_none());

        let result = agg.flush().unwrap();
        assert_eq!(result.ngram, "the");
        assert_eq!(result.total_count, 450);
        assert_eq!(result.year_span, 3);
        assert_eq!(result.first_year, 2000);
        assert_eq!(result.last_year, 2002);
    }

    #[test]
    fn test_multiple_ngrams() {
        let mut agg = YearAggregator::new(None);

        assert!(agg.push(make_record("a", 2000, 100)).is_none());
        assert!(agg.push(make_record("a", 2001, 200)).is_none());

        // New n-gram returns the previous one
        let result = agg.push(make_record("b", 2000, 50)).unwrap();
        assert_eq!(result.ngram, "a");
        assert_eq!(result.total_count, 300);

        assert!(agg.push(make_record("b", 2001, 60)).is_none());

        let result = agg.flush().unwrap();
        assert_eq!(result.ngram, "b");
        assert_eq!(result.total_count, 110);
    }

    #[test]
    fn test_year_filter() {
        let mut agg = YearAggregator::new(Some((2000, 2010)));

        // Outside range, ignored
        assert!(agg.push(make_record("the", 1999, 100)).is_none());
        assert!(agg.push(make_record("the", 2011, 100)).is_none());

        // Inside range
        assert!(agg.push(make_record("the", 2000, 100)).is_none());
        assert!(agg.push(make_record("the", 2005, 200)).is_none());

        let result = agg.flush().unwrap();
        assert_eq!(result.total_count, 300);
        assert_eq!(result.year_span, 2);
    }

    #[test]
    fn test_iterator_adapter() {
        let records = vec![
            make_record("a", 2000, 100),
            make_record("a", 2001, 200),
            make_record("b", 2000, 50),
            make_record("b", 2001, 60),
        ];

        let aggregated: Vec<_> = records.into_iter().aggregate_years(None).collect();

        assert_eq!(aggregated.len(), 2);
        assert_eq!(aggregated[0].ngram, "a");
        assert_eq!(aggregated[0].total_count, 300);
        assert_eq!(aggregated[1].ngram, "b");
        assert_eq!(aggregated[1].total_count, 110);
    }

    // ---- push_ref (zero-alloc hot path) ----

    fn make_record_ref<'a>(ngram: &'a str, year: u16, count: u64) -> NgramRecordRef<'a> {
        NgramRecordRef {
            ngram,
            year,
            match_count: count,
            volume_count: 100,
        }
    }

    #[test]
    fn test_push_ref_no_alloc_on_same_ngram() {
        // Push 5 records with the same ngram, then 1 with a different ngram.
        // The first 4 same-ngram pushes return None; the 5th same-ngram push
        // also returns None; the 6th (different) returns Some(aggregated)
        // with the summed total_count.
        let mut agg = YearAggregator::new(None);

        assert!(agg.push_ref(&make_record_ref("the", 2000, 100)).is_none());
        assert!(agg.push_ref(&make_record_ref("the", 2001, 100)).is_none());
        assert!(agg.push_ref(&make_record_ref("the", 2002, 100)).is_none());
        assert!(agg.push_ref(&make_record_ref("the", 2003, 100)).is_none());
        assert!(agg.push_ref(&make_record_ref("the", 2004, 100)).is_none());

        // Switch to a new ngram — returns the accumulated previous one
        let previous = agg
            .push_ref(&make_record_ref("a", 2000, 50))
            .expect("new ngram should yield previous");
        assert_eq!(previous.ngram, "the");
        assert_eq!(previous.total_count, 500);
        assert_eq!(previous.year_span, 5);
        assert_eq!(previous.first_year, 2000);
        assert_eq!(previous.last_year, 2004);
    }

    #[test]
    fn test_push_ref_vs_push_equivalent() {
        // Feed the same record sequence to two aggregators (one via push,
        // one via push_ref) and assert the flushed results are identical.
        let inputs = [
            ("the", 2000u16, 100u64),
            ("the", 2001, 200),
            ("the", 2002, 150),
            ("a", 2000, 50),
            ("a", 2001, 60),
            ("a", 2010, 70),
        ];

        let mut owned_agg = YearAggregator::new(None);
        let mut owned_results = Vec::new();
        for &(n, y, c) in &inputs {
            if let Some(r) = owned_agg.push(make_record(n, y, c)) {
                owned_results.push(r);
            }
        }
        if let Some(r) = owned_agg.flush() {
            owned_results.push(r);
        }

        let mut ref_agg = YearAggregator::new(None);
        let mut ref_results = Vec::new();
        for &(n, y, c) in &inputs {
            if let Some(r) = ref_agg.push_ref(&make_record_ref(n, y, c)) {
                ref_results.push(r);
            }
        }
        if let Some(r) = ref_agg.flush() {
            ref_results.push(r);
        }

        assert_eq!(owned_results.len(), ref_results.len());
        for (o, r) in owned_results.iter().zip(ref_results.iter()) {
            assert_eq!(o.ngram, r.ngram);
            assert_eq!(o.total_count, r.total_count);
            assert_eq!(o.year_span, r.year_span);
            assert_eq!(o.first_year, r.first_year);
            assert_eq!(o.last_year, r.last_year);
        }
    }
}
