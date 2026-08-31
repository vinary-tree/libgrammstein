//! # `u64`-unit view over a byte-carrier n-gram dictionary
//!
//! liblevenshtein's `Transducer::query_units*` family performs *unit-generic* fuzzy
//! matching: the same automaton that corrects the **bytes of a word** can correct the
//! **words of a sentence** once each word is mapped to an integer term-id. To drive that
//! word-level (`Unit = u64`) search over libgrammstein's existing n-gram store, we need a
//! dictionary whose edges are *term-ids* — but the store physically keys n-grams as
//! **LEB128-varint-encoded `u64` term-id sequences stored as bytes**.
//!
//! [`U64NgramView`] is a zero-copy adapter that presents such a byte-carrier trie as a
//! `Unit = u64` dictionary: **one `u64` edge == one term-id == that term-id's complete
//! varint (1..=10 bytes) in the underlying trie.** No data is copied or rebuilt; every
//! query walks the original nodes and collapses each varint run on the fly.
//!
//! ## Two byte carriers (`u8` *and* `char`)
//!
//! libgrammstein stores the same varint bytes two ways, so the view is generic over the
//! carrier unit via the sealed [`VarintByteUnit`] trait:
//!
//! | Backing store                                   | Node `Unit` | Byte carrier                                  |
//! |-------------------------------------------------|-------------|-----------------------------------------------|
//! | importer scale store `PersistentARTrie<u64>`    | `u8`        | raw varint byte `0x00..=0xFF`                 |
//! | in-memory model `DynamicDawgChar<NgramEntry>`   | `char`      | Latin-1 lift: byte `0xNN` stored as `U+00NN`  |
//!
//! `VarintByteUnit::to_byte` recovers the raw varint byte from either carrier and
//! `VarintByteUnit::from_byte` lifts a byte back into the carrier, so all varint
//! arithmetic is done on `u8` regardless of the physical edge type.
//!
//! ## LEB128 varint scheme (must agree bit-for-bit with `vocabulary::encode_varint`)
//!
//! A term-id `v` is encoded low-7-bits-first; every non-final 7-bit group carries the
//! continuation flag `0x80`:
//!
//! ```text
//! encode:  loop { byte = v & 0x7F; v >>= 7; push(if v==0 { byte } else { byte | 0x80 }); if v==0 break }
//! decode:  v = Σ_i (b_i & 0x7F) << (7*i)   ,  terminated by the first b_i with (b_i & 0x80) == 0
//! ```
//!
//! Because every id ends in a byte `< 0x80` and every continuation byte is `>= 0x80`, the
//! code is **self-delimiting**: from any node the set of "next term-ids" is exactly the set
//! of paths matching `continuation* terminator`. This makes the collapse unambiguous.
//!
//! ## Edge collapsing (the lazy varint walk)
//!
//! ```text
//!   byte trie (Unit = u8/char)                 u64 view (Unit = u64)
//!
//!         ┌─ 0xC8 ─┬─ 0x01(term) ─▶ N1              ┌─ 200 ─▶ N1
//!   root ─┤        └─ 0x02(term) ─▶ N2   ═══▶  root ┼─ 328 ─▶ N2
//!         └─ 0x0A(term) ─────────▶ N3               └─  10 ─▶ N3
//! ```
//!
//! `0xC8` is a continuation (`>= 0x80`); the walk descends and completes a term-id only
//! when it reaches a terminator byte, yielding `(id, child_after_full_varint)`. The
//! enumeration is **lazy** — an explicit stack over `inner.edges()` that expands at most
//! one node's immediate edges at a time (bounded by the byte alphabet) and never
//! materializes the whole vocabulary; stack depth is bounded by the varint length (<= 10).
//!
//! ## Value flow
//!
//! The stored value `V` (an n-gram frequency `u64` for the importer store, an `NgramEntry`
//! for the model) flows straight through: a view node's [`value`](MappedDictionaryNode::value)
//! delegates to the underlying byte node's value, so `Transducer::query_units_values`
//! recovers the corrected term-id sequence **and** its stored value in one pass.

use libdictenstein::{
    CharUnit, Dictionary, DictionaryNode, DictionaryValue, MappedDictionary, MappedDictionaryNode,
    SyncStrategy,
};

/// Sealed-supertrait module (rust-lang API guideline C-SEALED): external crates cannot
/// name [`sealed::Sealed`], hence cannot implement [`VarintByteUnit`].
mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for char {}
}

/// A [`CharUnit`] that carries a single raw varint byte.
///
/// Implemented (sealed) for the two carriers libgrammstein actually uses:
/// - `u8` — the identity carrier (importer `PersistentARTrie<u64>`, node `Unit = u8`);
/// - `char` — the Latin-1 carrier (in-memory `DynamicDawgChar`, node `Unit = char`), where
///   each varint byte `0xNN` is stored as the scalar `U+00NN`.
///
/// # Invariant
///
/// For the `char` carrier the view is only meaningful when every edge is a Latin-1 scalar
/// (`<= U+00FF`), which the `encode_ngram_key` / `encode_indices_to_key` writers guarantee.
/// `char::to_byte` truncates to the low 8 bits, so a non-Latin-1 edge would be lossy — but
/// such an edge cannot occur in a varint carrier.
pub trait VarintByteUnit: CharUnit + sealed::Sealed {
    /// Recover the raw varint byte carried by this unit.
    fn to_byte(self) -> u8;
    /// Lift a raw varint byte into the carrier unit.
    fn from_byte(byte: u8) -> Self;
}

impl VarintByteUnit for u8 {
    #[inline]
    fn to_byte(self) -> u8 {
        self
    }
    #[inline]
    fn from_byte(byte: u8) -> Self {
        byte
    }
}

impl VarintByteUnit for char {
    #[inline]
    fn to_byte(self) -> u8 {
        // Latin-1 down-cast; exact for the `<= U+00FF` scalars the encoder emits.
        self as u32 as u8
    }
    #[inline]
    fn from_byte(byte: u8) -> Self {
        // Latin-1 lift: `0xNN` -> `U+00NN` (infallible for all `u8`).
        char::from(byte)
    }
}

/// A `Unit = u64` (term-id) dictionary view over a byte-carrier dictionary `D` whose nodes
/// key n-grams as LEB128-varint-encoded term-id sequences.
///
/// `D` must be a [`Dictionary`] whose node is a [`MappedDictionaryNode`] over a
/// [`VarintByteUnit`] carrier (`u8` or `char`). See the [module docs](self) for the full
/// design.
#[derive(Clone, Debug)]
pub struct U64NgramView<D> {
    inner: D,
}

impl<D> U64NgramView<D> {
    /// Wrap a byte-carrier dictionary in the `u64` term-id view.
    #[inline]
    pub fn new(inner: D) -> Self {
        Self { inner }
    }

    /// Borrow the underlying byte-carrier dictionary.
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Consume the view, returning the underlying byte-carrier dictionary.
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }
}

impl<D> Dictionary for U64NgramView<D>
where
    D: Dictionary,
    <D::Node as DictionaryNode>::Unit: VarintByteUnit,
{
    type Node = U64NgramNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        U64NgramNode {
            inner: self.inner.root(),
        }
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        // The term count is carrier-invariant (one final byte-node == one n-gram key ==
        // one term-id sequence), so the underlying count is the view's count too.
        self.inner.len()
    }

    #[inline]
    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
    }
}

impl<D, V> MappedDictionary for U64NgramView<D>
where
    D: Dictionary,
    D::Node: MappedDictionaryNode<Value = V>,
    <D::Node as DictionaryNode>::Unit: VarintByteUnit,
    V: DictionaryValue,
{
    type Value = V;

    /// Look up the value of a term-id sequence supplied through the `&str` surface.
    ///
    /// The `u64` fuzzy path (`Transducer::query_units_values`) never calls this — it reads
    /// each final node's value during traversal via [`MappedDictionaryNode::value`]. This
    /// method exists only to satisfy the [`MappedDictionary`] contract; it interprets
    /// `term` through the standard (lossy) `u64` string-packing so it stays consistent with
    /// the trait's `&str` convention.
    fn get_value(&self, term: &str) -> Option<V> {
        let mut node = self.root();
        for unit in <u64 as CharUnit>::from_str(term) {
            node = node.transition(unit)?;
        }
        node.value()
    }
}

/// A node in a [`U64NgramView`]: a thin wrapper over an underlying byte-carrier node `N`
/// that presents `Unit = u64` (term-ids) by collapsing each LEB128 varint run.
pub struct U64NgramNode<N> {
    inner: N,
}

// Manual `Clone` (rather than `derive`) so the bound is exactly `N: Clone` — which every
// `N: DictionaryNode` satisfies — without `derive`'s extra machinery.
impl<N: Clone> Clone for U64NgramNode<N> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<N> DictionaryNode for U64NgramNode<N>
where
    N: DictionaryNode,
    N::Unit: VarintByteUnit,
{
    type Unit = u64;
    type SnapshotCursor = N::SnapshotCursor;
    type SnapshotGraphValueHandle = N::SnapshotGraphValueHandle;

    #[inline]
    fn is_final(&self) -> bool {
        // Finality is carried unchanged: the byte node reached after a complete varint run
        // is final iff that term-id sequence is a stored n-gram key.
        self.inner.is_final()
    }

    /// Transition by one term-id: encode it as LEB128 and walk the carrier byte-by-byte.
    ///
    /// Returns `Some` iff **every** carrier-unit transition exists (the whole varint is a
    /// path from here), else `None`. The loop mirrors `vocabulary::encode_varint`
    /// bit-for-bit, so the walked bytes are exactly the stored encoding.
    fn transition(&self, label: u64) -> Option<Self> {
        let mut node = self.inner.clone();
        let mut value = label;
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            let unit = if value == 0 {
                <N::Unit as VarintByteUnit>::from_byte(byte)
            } else {
                <N::Unit as VarintByteUnit>::from_byte(byte | 0x80)
            };
            node = node.transition(unit)?;
            if value == 0 {
                break;
            }
        }
        Some(U64NgramNode { inner: node })
    }

    /// Lazily enumerate every term-id whose varint is a path from this node, paired with
    /// the child node reached after that full varint. See [`VarintEdges`].
    fn edges(&self) -> Box<dyn Iterator<Item = (u64, Self)> + '_> {
        Box::new(VarintEdges::new(self.inner.clone()))
    }
}

impl<N> MappedDictionaryNode for U64NgramNode<N>
where
    N: MappedDictionaryNode,
    N::Unit: VarintByteUnit,
{
    type Value = N::Value;

    #[inline]
    fn value(&self) -> Option<Self::Value> {
        self.inner.value()
    }
}

/// One frame of the [`VarintEdges`] depth-first walk: the (owned) remaining byte-edges of a
/// carrier node, plus the partially accumulated varint value and its next bit-shift.
struct VarintFrame<N: DictionaryNode> {
    edges: std::vec::IntoIter<(N::Unit, N)>,
    /// Accumulated value of the continuation bytes consumed so far.
    value: u64,
    /// Bit position at which the next carrier byte contributes (`7 * bytes_consumed`).
    shift: u32,
}

/// Lazy varint-collapsing edge iterator: an explicit-stack DFS over a carrier node's
/// byte-edges that yields one `(term_id, child)` per complete LEB128 run.
///
/// Laziness: only one carrier node's immediate edges are collected at a time (bounded by
/// the byte alphabet); the whole vocabulary is never materialized. A continuation byte
/// (`>= 0x80`) pushes a frame; a terminator byte (`< 0x80`) yields. Stack depth is bounded
/// by the maximal varint length (<= 10 bytes for `u64`).
struct VarintEdges<N: DictionaryNode>
where
    N::Unit: VarintByteUnit,
{
    stack: Vec<VarintFrame<N>>,
}

impl<N> VarintEdges<N>
where
    N: DictionaryNode,
    N::Unit: VarintByteUnit,
{
    fn new(root: N) -> Self {
        // Collecting `root.edges()` ends its `&root` borrow immediately (the child nodes it
        // yields are owned), so no borrow of `root` outlives construction.
        let edges: Vec<(N::Unit, N)> = root.edges().collect();
        VarintEdges {
            stack: vec![VarintFrame {
                edges: edges.into_iter(),
                value: 0,
                shift: 0,
            }],
        }
    }
}

impl<N> Iterator for VarintEdges<N>
where
    N: DictionaryNode,
    N::Unit: VarintByteUnit,
{
    type Item = (u64, U64NgramNode<N>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let top = self.stack.last_mut()?;
            let base = top.value;
            let shift = top.shift;
            // `advanced` owns its payload, so `top`'s borrow ends here (NLL) and we may
            // freely `pop`/`push` `self.stack` below.
            let advanced = top.edges.next();

            match advanced {
                None => {
                    self.stack.pop();
                }
                Some((unit, child)) => {
                    let byte = unit.to_byte();
                    // Defensive: a well-formed `u64` varint never shifts past 63.
                    if shift >= 64 {
                        continue;
                    }
                    let new_value = base | (((byte & 0x7F) as u64) << shift);

                    if byte & 0x80 == 0 {
                        // Terminator: a complete term-id varint ends at this edge.
                        return Some((new_value, U64NgramNode { inner: child }));
                    }

                    // Continuation: descend into the next 7-bit group.
                    let next_shift = shift + 7;
                    if next_shift < 64 {
                        let child_edges: Vec<(N::Unit, N)> = child.edges().collect();
                        self.stack.push(VarintFrame {
                            edges: child_edges.into_iter(),
                            value: new_value,
                            shift: next_shift,
                        });
                    }
                    // else: a continuation past 10 bytes is malformed (cannot arise from a
                    // canonical `encode_varint` key); drop the path.
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::vocabulary::encode_varint;
    use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgChar};
    use liblevenshtein::prelude::{Algorithm, Transducer};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    // ── encoding helpers (mirror `vocabulary::encode_ngram_key` / `encode_indices_to_key`) ──

    /// Concatenate the LEB128 varints of `ids` into a raw byte key (the `PersistentARTrie<u64>`
    /// / byte-`DynamicDawg` physical key).
    fn encode_seq_bytes(ids: &[u64]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            encode_varint(id, &mut buf);
        }
        buf
    }

    /// Same varints, but Latin-1-lifted into a `String` (the `DynamicDawgChar` physical key):
    /// byte `0xNN` -> scalar `U+00NN`, exactly as `vocabulary::encode_indices_to_key` does.
    fn encode_seq_latin1(ids: &[u64]) -> String {
        encode_seq_bytes(ids).into_iter().map(char::from).collect()
    }

    /// Deterministic stored value for a sequence (so duplicate insertions never create a
    /// last-write-wins mismatch between the trie and the expected map).
    fn value_for(seq: &[u64]) -> u64 {
        let mut v = 1u64;
        for &x in seq {
            v = v.wrapping_mul(1_000_003).wrapping_add(x);
        }
        v | 1
    }

    /// Recursively enumerate every stored term-id sequence reachable through the *view*,
    /// recording its value at the final node. Exercises `edges()`, `is_final()`, `value()`.
    fn collect_view<Nd>(node: &Nd, prefix: &mut Vec<u64>, out: &mut BTreeMap<Vec<u64>, Option<u64>>)
    where
        Nd: MappedDictionaryNode<Value = u64> + DictionaryNode<Unit = u64>,
    {
        if node.is_final() {
            out.insert(prefix.clone(), node.value());
        }
        for (id, child) in node.edges() {
            prefix.push(id);
            collect_view(&child, prefix, out);
            prefix.pop();
        }
    }

    fn enumerate<D>(view: &D) -> BTreeMap<Vec<u64>, Option<u64>>
    where
        D: Dictionary,
        D::Node: MappedDictionaryNode<Value = u64> + DictionaryNode<Unit = u64>,
    {
        let mut out = BTreeMap::new();
        collect_view(&view.root(), &mut Vec::new(), &mut out);
        out
    }

    fn byte_view(entries: &[(&[u64], u64)]) -> U64NgramView<DynamicDawg<u64>> {
        let dawg: DynamicDawg<u64> = DynamicDawg::new();
        for (seq, freq) in entries {
            dawg.insert_bytes_with_value(&encode_seq_bytes(seq), *freq);
        }
        U64NgramView::new(dawg)
    }

    fn char_view(entries: &[(&[u64], u64)]) -> U64NgramView<DynamicDawgChar<u64>> {
        let dawg: DynamicDawgChar<u64> = DynamicDawgChar::new();
        for (seq, freq) in entries {
            dawg.insert_with_value(&encode_seq_latin1(seq), *freq);
        }
        U64NgramView::new(dawg)
    }

    // ── Example-based tests ────────────────────────────────────────────────────────────

    /// (a) `transition` chains recover each inserted sequence and (b) `value()` at the final
    /// node equals the stored frequency — over multi-byte term-ids that exercise varint
    /// collapsing (200, 300, 16_384 each span >= 2 bytes).
    #[test]
    fn u64_view_transition_and_value_recover_multibyte_sequences() {
        let entries: &[(&[u64], u64)] = &[
            (&[10, 20, 30], 900),
            (&[200, 300], 42),
            (&[16_384, 1], 7),
            (&[10, 20], 500),
            (&[128, 129, 130], 13),
        ];
        let view = byte_view(entries);

        for (seq, freq) in entries {
            let mut node = view.root();
            for &id in *seq {
                node = node
                    .transition(id)
                    .unwrap_or_else(|| panic!("transition by {id} in {seq:?} must exist"));
            }
            assert!(
                node.is_final(),
                "sequence {seq:?} must land on a final node"
            );
            assert_eq!(node.value(), Some(*freq), "value at {seq:?}");
        }

        // A term-id that is not an edge here must not transition.
        assert!(
            view.root().transition(9_999).is_none(),
            "absent leading term-id must yield None"
        );
    }

    /// (c) `query_units(seq, 0)` returns exactly `[seq]` — lossless, no prefix-extension leak.
    #[test]
    fn u64_view_query_units_exact_is_lossless() {
        let entries: &[(&[u64], u64)] = &[
            (&[10, 20, 30], 900),
            (&[10, 20, 30, 50], 40),
            (&[10, 20], 500),
            (&[200, 300], 42),
        ];
        let view = byte_view(entries);
        let transducer = Transducer::new(view, Algorithm::Standard);

        for (seq, _) in entries {
            let hits: Vec<Vec<u64>> = transducer.query_units(seq, 0).collect();
            assert_eq!(hits, vec![seq.to_vec()], "exact query for {seq:?}");
        }
    }

    /// (d) `query_units_values(observed, 1)` returns the 1-edit neighbors with their stored
    /// values — mirrors `examples/u64_word_correction.rs`.
    #[test]
    fn u64_view_query_units_values_returns_neighbors_with_values() {
        // "the quick fox" family; the observation substitutes the third id (30 -> 40).
        let entries: &[(&[u64], u64)] = &[
            (&[10, 20, 30], 900),    // the quick fox
            (&[10, 20, 40], 150),    // the quick dog (1 substitution)
            (&[10, 20, 30, 50], 40), // the quick fox jumps (1 insertion)
            (&[10, 20], 500),        // the quick (1 deletion)
            (&[99, 98, 97], 5),      // unrelated (far away)
        ];
        let view = byte_view(entries);
        let transducer = Transducer::new(view, Algorithm::Standard);

        let observed = [10u64, 20, 40];
        let mut valued: Vec<(Vec<u64>, usize, u64)> =
            transducer.query_units_values(&observed, 1).collect();
        valued.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        // The three sequences within token-edit distance 1 of [10,20,40], each carrying its
        // own stored value:
        //   exact          [10,20,40] @0 -> 150
        //   substitution   [10,20,30] @1 -> 900  (40 -> 30)
        //   deletion       [10,20]    @1 -> 500  (- 40)
        // (No distance-1 insertion neighbor exists here: [10,20,30,50] is distance 2 from
        // the observation — a 40->30 substitution *and* a +50 insertion.)
        assert!(
            valued.contains(&(vec![10, 20, 40], 0, 150)),
            "got {valued:?}"
        );
        assert!(
            valued.contains(&(vec![10, 20, 30], 1, 900)),
            "got {valued:?}"
        );
        assert!(valued.contains(&(vec![10, 20], 1, 500)), "got {valued:?}");
        assert_eq!(
            valued.len(),
            3,
            "exactly the 3 distance-<=1 neighbors, got {valued:?}"
        );
        // Distance-2 and unrelated sequences must never appear.
        assert!(
            !valued.iter().any(|(t, _, _)| t == &vec![10, 20, 30, 50]),
            "distance-2 insertion neighbor must be pruned, got {valued:?}"
        );
        assert!(
            !valued.iter().any(|(t, _, _)| t == &vec![99, 98, 97]),
            "unrelated trigram must be pruned, got {valued:?}"
        );

        // Querying from the intended correction [10,20,30] exposes the insertion neighbor
        // with its stored value, exercising the value flow for that edit type too:
        //   insertion      [10,20,30,50] @1 -> 40   (+50)
        //   substitution   [10,20,40]    @1 -> 150  (30 -> 40)
        let valued_from_fox: Vec<(Vec<u64>, usize, u64)> =
            transducer.query_units_values(&[10, 20, 30], 1).collect();
        assert!(
            valued_from_fox.contains(&(vec![10, 20, 30, 50], 1, 40)),
            "insertion neighbor with value, got {valued_from_fox:?}"
        );
        assert!(
            valued_from_fox.contains(&(vec![10, 20, 40], 1, 150)),
            "substitution neighbor with value, got {valued_from_fox:?}"
        );
    }

    /// The `u8` byte carrier and the `char` (Latin-1) carrier present an identical `u64`
    /// view — same sequences, same values — and identical value-returning corrections.
    #[test]
    fn u8_and_char_backings_yield_identical_u64_view() {
        let entries: &[(&[u64], u64)] = &[
            (&[10, 20, 30], 900),
            (&[10, 20, 40], 150),
            (&[200, 300], 42),
            (&[16_384, 1], 7),
            (&[10, 20], 500),
            (&[128, 129, 130], 13),
        ];

        let bview = byte_view(entries);
        let cview = char_view(entries);

        // Structural equivalence: identical enumerated sequences AND values from both.
        let benum = enumerate(&bview);
        let cenum = enumerate(&cview);
        assert_eq!(
            benum, cenum,
            "u8 and char carriers must enumerate identically"
        );

        // And both match the ground-truth set of inserted (sequence -> value) pairs.
        let expected: BTreeMap<Vec<u64>, Option<u64>> = entries
            .iter()
            .map(|(seq, freq)| (seq.to_vec(), Some(*freq)))
            .collect();
        assert_eq!(benum, expected, "byte view must match inserted entries");

        // Query-level equivalence: same value-returning 1-edit corrections from both.
        let bt = Transducer::new(bview, Algorithm::Standard);
        let ct = Transducer::new(cview, Algorithm::Standard);
        let mut bvalued: Vec<(Vec<u64>, usize, u64)> =
            bt.query_units_values(&[10, 20, 30], 1).collect();
        let mut cvalued: Vec<(Vec<u64>, usize, u64)> =
            ct.query_units_values(&[10, 20, 30], 1).collect();
        bvalued.sort();
        cvalued.sort();
        assert_eq!(bvalued, cvalued, "u8 and char corrections must agree");
        assert!(
            bvalued.contains(&(vec![10, 20, 30], 0, 900)),
            "exact match present, got {bvalued:?}"
        );
    }

    // ── Property-based tests ───────────────────────────────────────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// For an arbitrary set of term-id sequences inserted into BOTH carriers:
        /// (1) `query_units(seq, 0)` returns exactly `{seq}` for every inserted `seq`
        ///     (lossless — pins the "extension of a matched prefix" leak class); and
        /// (2) the sequences enumerable through the view equal the inserted set with their
        ///     values (no drop, no dup), identically for the `u8` and `char` carriers.
        #[test]
        fn view_is_lossless_and_faithful(
            raw in prop::collection::vec(
                prop::collection::vec(1u64..=1_000_000u64, 1..6usize),
                1..12usize,
            )
        ) {
            // Ground truth: dedup to a set with deterministic per-sequence values.
            let expected: BTreeMap<Vec<u64>, Option<u64>> = raw
                .iter()
                .map(|seq| (seq.clone(), Some(value_for(seq))))
                .collect();

            let bdawg: DynamicDawg<u64> = DynamicDawg::new();
            let cdawg: DynamicDawgChar<u64> = DynamicDawgChar::new();
            for seq in &raw {
                bdawg.insert_bytes_with_value(&encode_seq_bytes(seq), value_for(seq));
                cdawg.insert_with_value(&encode_seq_latin1(seq), value_for(seq));
            }
            let bview = U64NgramView::new(bdawg);
            let cview = U64NgramView::new(cdawg);

            // (2) faithful enumeration, both carriers, matching the inserted set + values.
            let benum = enumerate(&bview);
            let cenum = enumerate(&cview);
            prop_assert_eq!(&benum, &expected);
            prop_assert_eq!(&cenum, &expected);

            // (1) lossless exact recovery for every inserted sequence, both carriers.
            let bt = Transducer::new(bview, Algorithm::Standard);
            let ct = Transducer::new(cview, Algorithm::Standard);
            for seq in expected.keys() {
                let bhits: Vec<Vec<u64>> = bt.query_units(seq, 0).collect();
                prop_assert_eq!(bhits, vec![seq.clone()]);
                let chits: Vec<Vec<u64>> = ct.query_units(seq, 0).collect();
                prop_assert_eq!(chits, vec![seq.clone()]);
            }
        }

        /// (3) `encode_varint` then the view's varint-walk decode round-trips every id: a
        /// single-key trie of `id` recovers exactly `id` through both `transition` (forward
        /// walk) and `edges` (collapsing decode), for both carriers.
        #[test]
        fn varint_walk_roundtrips_every_id(id in 1u64..=1_000_000u64) {
            let seq = [id];
            let bview = byte_view(&[(&seq, 7)]);
            let cview = char_view(&[(&seq, 7)]);

            for (label, edges_only) in [("u8", enumerate(&bview)), ("char", enumerate(&cview))] {
                // edges (collapsing decode) yields exactly the single id with its value.
                let mut expected = BTreeMap::new();
                expected.insert(vec![id], Some(7u64));
                prop_assert_eq!(&edges_only, &expected, "edges decode for {} carrier", label);
            }

            // transition (forward walk) reaches the final valued node in both carriers.
            let bnode = bview.root().transition(id).expect("u8 transition");
            prop_assert!(bnode.is_final());
            prop_assert_eq!(bnode.value(), Some(7));
            let cnode = cview.root().transition(id).expect("char transition");
            prop_assert!(cnode.is_final());
            prop_assert_eq!(cnode.value(), Some(7));
        }
    }
}
