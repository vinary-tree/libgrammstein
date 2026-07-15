> ⚠ **Archived — historical record, not current design.**
> A scientific ledger of finished A/B optimization experiments (each marked ACCEPTED/REJECTED with a
> commit hash). Retained for provenance. For the current formal-verification suite see
> [`../../../formal/README.md`](../../../formal/README.md).

# Scientific Ledger: Formal Proof-Based Optimizations

## Overview

This document records experiments evaluating optimizations enabled by formal verification proofs.

**Date Started**: 2026-01-20
**Hypothesis**: Formal proofs enable optimizations with measurable performance benefits.
**Success Criterion**: p < 0.05 for performance improvement (Welch's t-test)

---

## Hardware Configuration

See `/home/dylon/.claude/hardware-specifications.md` for full specs.

**Benchmark Settings**:
- CPU governor: performance (max frequency)
- 10 iterations per measurement (excluding warmup)
- Release build with `--features google-books`

---

## Experiment 1: Parallel Tree Reduction for FrequencyCounts

### Formal Basis
- **Proof**: `FrequencyCountsMerge.v` proves `merge_associative` and `merge_commutative`
- **Implication**: Any reduction order produces identical results (commutative monoid)

### Hypothesis
Replacing shared atomic counters with thread-local accumulation + parallel tree reduction will reduce cache-line contention and improve throughput by 2-5x.

### Methodology
1. Baseline: Current `AtomicFrequencyCounts` with shared atomic fetch_add
2. Treatment: Thread-local `FrequencyCounts` + `par_iter().reduce()`
3. Metric: Time to compute frequency counts across all shards

### Baseline Results
```
Benchmark: mkn_frequency_counts/compute (criterion, 100 samples)
Date: 2026-01-20

1000 n-grams:  474.05 - 477.26 µs (mean: 475.61 µs)
5000 n-grams:  524.15 - 526.76 µs (mean: 525.47 µs)
10000 n-grams: 525.41 - 528.28 µs (mean: 526.80 µs)

Note: Times are relatively flat because setup (shard iteration) dominates.
The atomic contention is visible in the close grouping at higher n-gram counts.
```

### Treatment Results
```
Benchmark: mkn_frequency_counts/compute (criterion, 100 samples)
Date: 2026-01-20
Implementation: Thread-local accumulation + parallel tree reduction

1000 n-grams:  489.75 - 491.72 µs (mean: 490.72 µs)
5000 n-grams:  537.89 - 542.07 µs (mean: 539.91 µs)
10000 n-grams: 536.31 - 539.50 µs (mean: 537.92 µs)
```

### Statistical Analysis
```
Comparison (baseline → treatment):

1000 n-grams:  475.61 µs → 490.72 µs  (+3.17% regression)
5000 n-grams:  525.47 µs → 539.91 µs  (+2.75% regression)
10000 n-grams: 526.80 µs → 537.92 µs  (+2.11% regression)

All changes show p < 0.05 (statistically significant)
Direction: REGRESSION (not improvement)
```

### Analysis

The parallel tree reduction is slower for this benchmark because:

1. **Small dataset**: The test uses 1000-10000 n-grams across few shards
2. **Low contention**: With few shards, atomic contention is minimal
3. **Overhead dominates**: Vec allocation per shard + reduce tree overhead > atomic cost
4. **Break-even point**: Tree reduction benefits when many threads compete on atomics

The formal proof (FrequencyCountsMerge.v) guarantees **correctness**, not performance.
Tree reduction would likely win with:
- Millions of n-grams
- Many more shards (hundreds)
- Higher thread counts causing significant cache-line bouncing

### Decision
- [ ] ACCEPTED (p < 0.05, performance improved)
- [x] REJECTED (p < 0.05 but performance REGRESSED)

### Action
Revert to atomic implementation. The formal proof remains valid for future use
when dataset scale justifies the overhead.

### Commit
```
5f6a87c Add MKN benchmark and document rejected tree reduction optimization
```

---

## Experiment 2: Bitmap-Based Checkpoint State

### Formal Basis
- **Proof**: `CheckpointStateMachine.tla` proves `DisjointSets` invariant
- **Implication**: Each prefix has exactly one state; 2-bit encoding sufficient

### Hypothesis
Replacing three `Vec<String>` with a 2-bit-per-prefix bitmap will reduce state transition time from O(n) to O(1).

### Methodology
1. Baseline: Current `Vec<String>` with `retain()` for state transitions
2. Treatment: HashMap<String, PrefixState> for O(1) state lookup/transition
3. Metric: Time for state transitions and needs_prefix lookups across all prefixes

### Baseline Results
```
Benchmark: checkpoint_ops (criterion, 100 samples)
Date: 2026-01-20

State Transitions (start -> complete for all prefixes):
- Order 1 (26 prefixes):  6.57 µs
- Order 2 (676 prefixes): 3.22 ms
- Order 3 (676 prefixes): 3.12 ms

Needs Prefix Lookup (all prefixes, half completed):
- Order 1 (26 prefixes):  1.71 µs
- Order 2 (676 prefixes): 729.19 µs
- Order 3 (676 prefixes): 688.79 µs

Mixed Operations (676 prefixes simulation):
- Import simulation: 2.88 ms

Sparse Remaining (95% completed, find remaining):
- 676 prefixes, 33 remaining: 978.93 µs

Note: Order 2/3 with 676 prefixes shows O(n²) behavior due to
repeated Vec::contains() and Vec::retain() operations.
```

### Treatment Results
```
Benchmark: checkpoint_ops (criterion, 100 samples)
Date: 2026-01-20
Implementation: HashMap<String, PrefixState> replacing three Vec<String>

State Transitions (start -> complete for all prefixes):
- Order 1 (26 prefixes):  6.32 µs
- Order 2 (676 prefixes): 148.06 µs (was 3.22 ms)
- Order 3 (676 prefixes): 147.80 µs (was 3.12 ms)

Needs Prefix Lookup (all prefixes, half completed):
- Order 1 (26 prefixes):  1.19 µs
- Order 2 (676 prefixes): 33.38 µs (was 729 µs)
- Order 3 (676 prefixes): 32.65 µs (was 689 µs)

Mixed Operations (676 prefixes simulation):
- Import simulation: 194.28 µs (was 2.88 ms)

Sparse Remaining (95% completed, find remaining):
- 676 prefixes, 33 remaining: 35.93 µs (was 979 µs)
```

### Statistical Analysis
```
All comparisons (criterion, Welch's t-test):

State Transitions:
- Order 2: 3.22 ms → 148 µs (-95.3%, p = 0.00)
- Order 3: 3.12 ms → 148 µs (-95.3%, p = 0.00)

Needs Prefix Lookups:
- Order 1:  1.71 µs → 1.19 µs  (-31%, p = 0.00)
- Order 2:  729 µs  → 33.4 µs  (-95.4%, p = 0.00)
- Order 3:  689 µs  → 32.7 µs  (-95.2%, p = 0.00)

Mixed Operations: 2.88 ms → 194 µs (-93.3%, p = 0.00)
Sparse Remaining: 979 µs  → 35.9 µs (-96.3%, p = 0.00)

All changes statistically significant (p < 0.05)
```

### Analysis

The HashMap optimization delivers massive performance improvements:

1. **O(1) vs O(n) complexity**: Replaced Vec::contains() and Vec::retain()
   with HashMap::get() and HashMap::insert()
2. **No allocation overhead**: State changes are in-place HashMap updates
   instead of creating/resizing Vecs
3. **Cache efficiency**: HashMap provides better locality for lookups
4. **TLA+ DisjointSets invariant**: Guarantees each prefix has exactly one
   state, making HashMap semantically correct

The 95%+ improvement for 676-prefix operations validates that the O(n²)
behavior of repeated Vec::contains() was the bottleneck. Order 1 with
only 26 prefixes shows smaller (31%) improvement as expected.

### Decision
- [x] ACCEPTED (p < 0.05, performance improved)
- [ ] REJECTED (p >= 0.05 or performance regressed)

### Commit
```
e30ccbb Replace Vec-based checkpoint state with HashMap for O(1) operations
```

---

## Experiment 3: Conditional Clamping Elision

### Formal Basis
- **Proof**: `MknStatistics.v` proves `d2_unclamped_upper_bound` and `d3_plus_unclamped_upper_bound`
- **Implication**: Under Zipf distribution (n1 >= n2 >= n3 >= n4), clamping is unnecessary

### Hypothesis
Adding a fast path that skips `.max().min()` chains when preconditions hold will marginally improve MKN computation time.

### Methodology
1. Baseline: Current always-clamp implementation
2. Treatment: Conditional skip when n1 >= n2 >= n3 >= n4
3. Metric: Time for 10,000 `DiscountParams::from_counts()` calls

### Baseline Results
```
Benchmark: mkn_discount_params/from_counts (criterion, 100 samples)
Date: 2026-01-20
Implementation: Always-clamp with .max().min() chains

Zipf distribution (n1 >= n2 >= n3 >= n4):     12.0 ns
Non-Zipf distribution (arbitrary counts):     12.6 ns

Note: Baseline is already extremely fast (~12 nanoseconds).
The .max().min() chains are highly optimized by modern CPUs.
```

### Treatment Results
```
Benchmark: mkn_discount_params/from_counts (criterion, 100 samples)
Date: 2026-01-20
Implementation: Conditional Zipf check to skip clamping

Zipf distribution:     14.7 ns (was 12.0 ns)
Non-Zipf distribution: 14.6 ns (was 12.6 ns)
```

### Statistical Analysis
```
Comparison (baseline → treatment):

Zipf distribution:     12.0 ns → 14.7 ns  (+22.2% regression, p = 0.00)
Non-Zipf distribution: 12.6 ns → 14.6 ns  (+15.9% regression, p = 0.00)

Both changes statistically significant (p < 0.05)
Direction: REGRESSION (not improvement)
```

### Analysis

The conditional clamping elision is slower because:

1. **Baseline already optimal**: At 12 nanoseconds, the always-clamp path is
   highly optimized and leaves no room for improvement
2. **Branch overhead**: The Zipf check adds 3 integer comparisons plus
   branch prediction overhead
3. **Modern CPU efficiency**: `.max().min()` chains are efficiently
   executed via conditional moves (no branch misprediction)
4. **Cache effects**: The branch creates two code paths, potentially
   reducing instruction cache efficiency

The formal proof (MknStatistics.v) correctly shows clamping is mathematically
unnecessary under Zipf distribution, but the optimization doesn't translate
to performance gains because:
- The 6 float comparisons saved (~2ns) < branch check overhead (~3ns)
- The always-clamp path is branchless and predictable

### Decision
- [ ] ACCEPTED (p < 0.05, performance improved)
- [x] REJECTED (p < 0.05 but performance REGRESSED)

### Action
Reverted to always-clamp implementation. The formal proof remains valid for
theoretical understanding but offers no practical performance benefit here.

### Commit
```
083bf7c Reject Experiment 3: Conditional clamping elision regressed +22%
```

---

## Summary Table

| Experiment | Baseline | Treatment | Change | p-value | Decision |
|------------|----------|-----------|--------|---------|----------|
| 1. Tree Reduction | 526.80 µs | 537.92 µs | +2.1% | <0.05 | REJECTED |
| 2. HashMap Checkpoint | 2.88 ms | 194 µs | -93.3% | <0.05 | **ACCEPTED** |
| 3. Clamping Elision | 12.0 ns | 14.7 ns | +22.2% | <0.05 | REJECTED |

---

## Lessons Learned

1. **Formal proofs guarantee correctness, not performance**: All three proofs
   are mathematically valid, but only one (Experiment 2) yielded performance
   gains. The proofs enable safe code transformations but don't predict
   whether those transformations will be faster.

2. **Algorithmic complexity matters more than micro-optimizations**: The 95%
   improvement in Experiment 2 came from fixing O(n²) → O(1) complexity, not
   from clever bit manipulation. The micro-optimizations in Experiments 1
   and 3 failed because their overhead exceeded savings.

3. **Measure before optimizing**: The baseline for Experiment 3 was 12ns -
   so fast that any added logic (even 3 integer comparisons) caused regression.
   Without baseline measurement, we might have assumed "skipping work" would
   be faster.

4. **Scale affects optimization viability**: Experiment 1's tree reduction
   would likely win at larger scale (millions of n-grams, many threads), but
   at benchmark scale the coordination overhead dominated. Optimizations must
   be validated at production-representative scales.

5. **Modern CPUs are efficient at simple patterns**: The `.max().min()` chains
   in Experiment 3 compile to branchless conditional moves. Modern CPUs handle
   these better than branches with unpredictable patterns.
