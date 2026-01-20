# Formal Verification for Google Books Importer

This directory contains formal specifications and proofs for the critical components
of the Google Books n-gram importer. The verification uses:

- **TLA+**: For concurrency and state machine properties
- **Rocq/Coq**: For algebraic and mathematical properties

## Directory Structure

```
formal/
├── README.md                          # This file
├── tla/
│   ├── ShardWriteToken.tla           # WriteToken concurrency spec
│   ├── MC_ShardWriteToken.cfg        # TLC model checker config
│   ├── CheckpointStateMachine.tla    # Checkpoint prefix state machine
│   └── MC_CheckpointStateMachine.cfg # TLC model checker config
└── rocq/
    ├── _CoqProject                   # Coq project file
    ├── Makefile                      # Build automation
    ├── MknStatistics.v               # MKN discount bounds proofs
    └── FrequencyCountsMerge.v        # Merge operation proofs
```

## Prerequisites

### TLA+ (TLC Model Checker)

Install the TLA+ Toolbox or standalone TLC:

```bash
# Download TLA+ Toolbox
# https://github.com/tlaplus/tlaplus/releases

# Or use standalone TLC JAR
wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
```

### Rocq/Coq

Install Coq 8.18 or later:

```bash
# Via opam
opam install coq

# Or via package manager
sudo pacman -S coq  # Arch Linux
sudo apt install coq  # Debian/Ubuntu
```

## Running Verification

### TLA+ Model Checking

```bash
cd formal/tla

# ShardWriteToken verification
java -jar /path/to/tla2tools.jar -config MC_ShardWriteToken.cfg ShardWriteToken.tla

# CheckpointStateMachine verification
java -jar /path/to/tla2tools.jar -config MC_CheckpointStateMachine.cfg CheckpointStateMachine.tla
```

Expected output: "Model checking completed. No errors."

### Rocq/Coq Proof Checking

```bash
cd formal/rocq

# Generate Makefile and build
make

# Verify no admitted lemmas
make check
```

Expected output: "All proofs complete (no Admitted lemmas found)"

## Specifications

### 1. ShardWriteToken.tla

**Target**: `src/sources/google_books/sharding/shard.rs:262-294`

Models the exclusive write access mechanism using atomic compare-and-swap
operations and generation counters.

**Key Invariants**:
- `AtMostOneWriter`: At most one worker holds a valid token per shard
- `ValidTokenGenerationMatch`: Valid tokens have matching generation numbers
- `ValidTokenImpliesHolder`: Valid token holder is recorded

**Variables**:
```tla+
write_locked       : shard -> Bool
write_holder       : shard -> worker_id | NONE
write_generation   : shard -> Nat
tokens             : (worker, shard) -> {gen, valid} | NONE
```

### 2. CheckpointStateMachine.tla

**Target**: `src/sources/google_books/checkpoint.rs`

Models the prefix lifecycle state machine for checkpoint/resume support.

**States**: `not_started` → `in_progress` → `completed` | `failed`

**Key Invariants**:
- `DisjointSets`: Completed, in_progress, failed sets are mutually exclusive
- `StateConsistent`: State variable matches set membership
- `NoDoubleProcessing`: Completed prefixes don't return to in_progress

**Crash Recovery**: In-progress prefixes move to failed on recovery.

### 3. MknStatistics.v

**Target**: `src/sources/google_books/sharding/mkn.rs:186-230`

Proves bounds on Modified Kneser-Ney discount parameters.

**Theorems**:
- `y_bounded`: 0 ≤ Y ≤ 1 when n1, n2 > 0
- `d1_bounded`: 0 ≤ D1 ≤ 1 (after clamping)
- `d2_clamped_bounded`: 0 ≤ D2 ≤ 2 (after clamping)
- `d3_plus_clamped_bounded`: 0 ≤ D3+ ≤ 3 (after clamping)

### 4. FrequencyCountsMerge.v

**Target**: `src/sources/google_books/sharding/mkn.rs:98-107`

Proves algebraic properties for parallel aggregation correctness.

**Theorems**:
- `merge_associative`: merge(merge(a,b),c) = merge(a,merge(b,c))
- `merge_commutative`: merge(a,b) = merge(b,a)
- `merge_identity_right`: merge(a,default) = a
- `merge_identity_left`: merge(default,a) = a
- `merge_is_commutative_monoid`: All properties combined

## Spec-to-Code Traceability

Each specification file contains a "Spec-to-Code Traceability" section
documenting the mapping between formal model elements and Rust implementation:

| Specification | Rust File | Lines |
|--------------|-----------|-------|
| ShardWriteToken.tla | shard.rs | 262-294 |
| CheckpointStateMachine.tla | checkpoint.rs | 314-416 |
| MknStatistics.v | mkn.rs | 186-230 |
| FrequencyCountsMerge.v | mkn.rs | 98-107 |

## Model Checking Parameters

The TLC configurations use bounded state spaces for tractable verification:

| Specification | Workers | Shards/Orders | Prefixes | MaxGeneration |
|--------------|---------|---------------|----------|---------------|
| ShardWriteToken | 3 | 2 | - | 3 |
| CheckpointStateMachine | 2 | 2 | 3 | - |

These bounds cover interesting scenarios (lock contention, state transitions)
while keeping the state space manageable.

## Verification Status

| Property | Tool | Status | Notes |
|----------|------|--------|-------|
| WriteToken single-writer | TLA+ | ✅ Verified | AtMostOneWriter invariant |
| Generation monotonic | TLA+ | ✅ Verified | Implicit in spec |
| Token validity | TLA+ | ✅ Verified | ValidTokenGenerationMatch |
| Prefix state disjoint | TLA+ | ✅ Verified | DisjointSets invariant |
| Crash recovery sound | TLA+ | ✅ Verified | RecoverInProgressAsFailed action |
| MKN Y bounded [0,1] | Rocq | ✅ Proven | y_bounded theorem |
| MKN D1 bounded [0,1] | Rocq | ✅ Proven | d1_bounded corollary |
| MKN D2 bounded [0,2] | Rocq | ✅ Proven | d2_clamped_bounded theorem |
| MKN D3+ bounded [0,3] | Rocq | ✅ Proven | d3_plus_clamped_bounded theorem |
| Merge associative | Rocq | ✅ Proven | merge_associative theorem |
| Merge commutative | Rocq | ✅ Proven | merge_commutative theorem |
| Merge identity | Rocq | ✅ Proven | merge_identity_* theorems |

## Known Limitations

1. **Floating-point vs Rational**: Rocq proofs use rational numbers (Q type)
   while Rust uses f64. Floating-point rounding errors are not modeled but
   remain within clamped bounds.

2. **u64 Overflow**: Generation counters and frequency counts use u64 in Rust.
   Overflow is theoretically possible but requires astronomical data volumes
   (2^64 operations).

3. **Memory Ordering Abstraction**: TLA+ models atomic operations as
   linearizable. The actual Rust implementation uses specific memory orderings
   (Acquire/Release/Relaxed) which provide the required synchronization.

4. **Bounded Model Checking**: TLC explores a bounded state space. While no
   violations are found within bounds, this doesn't constitute a complete proof.
   For mathematical certainty, the TLA+ specs could be verified with TLAPS
   (TLA+ Proof System).

## Extending the Verification

### Adding New Invariants

1. Add the invariant definition to the `.tla` file
2. Add `INVARIANT <name>` to the `.cfg` file
3. Run TLC to verify

### Adding New Theorems

1. Add the theorem statement and proof to the `.v` file
2. Run `make` to verify compilation
3. Run `make check` to ensure no admitted lemmas

## References

- [TLA+ Language Manual](https://lamport.azurewebsites.net/tla/tla.html)
- [TLC Model Checker](https://lamport.azurewebsites.net/tla/tools.html)
- [Coq Reference Manual](https://coq.inria.fr/refman/)
- [Modified Kneser-Ney Smoothing](https://www.speech.sri.com/projects/srilm/manpages/ngram-discount.7.html)
