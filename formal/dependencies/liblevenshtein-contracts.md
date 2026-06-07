# liblevenshtein Formal Contracts

This file records the liblevenshtein formal contracts relevant to
libgrammstein's dictionary, fuzzy-query, phonetic, and n-gram query paths. The
storage crash-recovery bridge imports only the trusted dependency assets listed
below; partial and legacy assets are useful review evidence, but they do not
support trusted libgrammstein correctness claims.

Dependency repository: `../liblevenshtein-rust`
Revision checked for this contract file:
`8b1732551e98b66bb9dc285f835947d122fe4e4c`
Dependency tree status for referenced manifest/query files: clean

## Verification Commands

Run the trusted-contract audit before relying on the trusted contracts below:

```bash
cd ../liblevenshtein-rust
scripts/verify-formal.sh trusted
```

Run the heavier dependency proof/model suite when dependency verification is
being refreshed:

```bash
cd ../liblevenshtein-rust
scripts/verify-formal.sh all
```

The trusted audit passed locally while adding this bridge. The full `all` mode
is intentionally kept as an explicit dependency refresh gate because it compiles
trusted Rocq files with memory caps and runs the dependency TLA+ suite.

## Imported Trusted Contracts

| liblevenshtein artifact | Contract imported by libgrammstein |
| --- | --- |
| `docs/verification/FORMAL_VERIFICATION_MANIFEST.tsv` | Conservative trust boundary separating trusted, partial, legacy, and debug assets. libgrammstein imports only entries marked `trusted`. |
| `scripts/verify-formal.sh trusted` | Audits trusted files for active `Admitted`, unallowlisted assumptions, unallowlisted proof contracts, and unallowlisted evidence surfaces. |
| `docs/verification/core/theories/Core/Definitions.v` | Core edit-distance definitions used by liblevenshtein transducer semantics. |
| `docs/verification/core/theories/Core/MinLemmas.v` | Minimum/substitution-cost helper lemmas used by the modular edit-distance proofs. |
| `docs/verification/core/theories/Core/LevDistance.v` | Levenshtein distance definition and basic unfolding lemmas. |
| `docs/verification/core/theories/Core/MetricProperties.v` | Trusted Levenshtein metric properties used by fuzzy matching and scoring assumptions in libgrammstein. |
| `docs/verification/core/theories/Triangle/SubstCostTriangle.v` | Substitution-cost triangle helper used by the metric proof layer. |
| `docs/verification/articulatory/theories/FeatureDistance.v` | Parameterized articulatory distance symmetry, identity, non-negativity, boundedness, and monotonicity facts. |
| `docs/verification/articulatory/theories/FeatureDistanceWeighted.v` | Weighted articulatory model matching Rust feature-distance configuration, with the trusted metric-style facts from the dependency manifest. |
| `docs/verification/wallbreaker/theories/Pigeonhole/WallBreakerPigeonhole.v` | Trusted pigeonhole proof island available to future pruning/lower-bound code paths that import liblevenshtein's WallBreaker reasoning. It is not needed by the storage bridge. |
| `docs/verification/msm/theories/Indexing/IntervalCost.v` | Interval-relaxed MSM per-element lower-bound exactness and admissibility for move/merge/split cells. |
| `docs/verification/msm/theories/Indexing/QuantizationBounds.v` | Executable uniform binning and bin-bound soundness for interval MSM indexing. |
| `docs/verification/msm/theories/Indexing/IntervalColumn.v` | Interval-column admissibility and pruning soundness for MSM trie search. |
| `docs/verification/tla/MsmTrieSearch.tla` | Trusted bounded TLA+ model for interval-pruned MSM trie search: no false positives, no missed matches, prune soundness, and termination under the manifest's bounds. |
| `tests/persistent_artrie_integration.rs` | Rust correspondence coverage that liblevenshtein's `PersistentARTrie` re-export implements the dictionary and zipper interfaces expected by its transducers. Storage crash durability still comes from libdictenstein. |

## Supporting Non-Trusted Evidence

The following dependency assets are useful for design review and regression
coverage, but their manifest status is not `trusted` at the recorded revision.
Do not import them as proof obligations until the liblevenshtein manifest
promotes them.

| liblevenshtein artifact | Supporting use |
| --- | --- |
| `docs/verification/tla/ValueYieldingQuery.tla` | Partial bounded model of value-yielding transducer queries: value correctness, no valueless yields, deduplication, finite termination, and processed-final completeness. Use as supporting model evidence alongside Rust property tests, not as a trusted contract. |
| `tests/proptest_value_yielding_query.rs` | Rust property coverage for value-yielding query behavior against concrete dictionaries. |
| `docs/verification/tla/OnlineScanner.tla` | Partial phonetic online-scanner model. Useful for phonetic embedding review, but not needed by the storage bridge. |
| `docs/verification/tla/ProductAutomaton.tla` | Partial product-automaton model with documented abstract total NFA transitions. |
| `docs/verification/tla/PriorityQuery.tla` | Partial idealized A* model; the Rust fast-first iterator uses an inadmissible heuristic, while the ordered iterator is cross-validated by tests. |
| `docs/verification/tla/Subsumption.tla` | Partial bounded subsumption model. |

## Non-Imported Assets

The dependency manifest marks several TLA+ and Rocq assets as `partial`,
`legacy`, or `debug`. Those are useful for context but are not imported as
trusted libgrammstein contracts. In particular:

- `ValueYieldingQuery.tla`, `ProductAutomaton.tla`, `PriorityQuery.tla`,
  `OnlineScanner.tla`, and `Subsumption.tla` have documented abstraction or
  manifest limits in liblevenshtein's TLA README and manifest.
- The legacy monolithic Rocq files are superseded by modular trusted files.
- Persistent storage crash semantics are owned by libdictenstein; liblevenshtein
  currently re-exports `PersistentARTrie` and cross-validates integration with
  its dictionary/transducer APIs.

## Bridge Use

`formal/tla/PersistentStorageBridge.tla` and
`formal/tla/QuerySemanticsBridge.tla` rely on liblevenshtein for the
dictionary-facing adapter boundary:

1. The persistent ARTrie re-export does not change the storage durability
   contract imported from libdictenstein.
2. Value-bearing dependency query behavior remains supporting evidence unless
   `ValueYieldingQuery.tla` is promoted to trusted scope. The local query
   bridge covers only libgrammstein wrapper obligations: root metadata hiding,
   value preservation for visible terms, OOV reads not allocating vocabulary
   indices, and aggregated lookup exactness.
3. Vocabulary-backed n-gram keys remain interpretable only when the vocabulary
   checkpoint/reopen evidence from libdictenstein is present.
4. Transducer/query correctness is treated as separate from crash recovery:
   libgrammstein's bridge proves recoverability of data and metadata claims,
   while liblevenshtein's trusted suite covers dictionary/query semantics.
