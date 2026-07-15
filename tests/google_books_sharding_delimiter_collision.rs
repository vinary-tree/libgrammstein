//! Delimiter-collision acceptance test for the Google-Books **shard router**.
//!
//! This is the sharded analogue of
//! `src/ngram/migration_tests.rs::delimiter_collision_trains_and_scores`. It is the
//! correctness proof of this migration: a token literally containing `'|'` routes
//! and counts through the term-id shard router with no collision — impossible under
//! the old `'|'`-joined key scheme.
//!
//! ## The bug this migration removes (documented, per the plan)
//!
//! The DELETED whole-n-gram-string router recovered token structure by splitting the
//! joined key on its pipe delimiter, so it could not tell the ONE-token n-gram
//! `["foo|bar"]` apart from the TWO-token n-gram `["foo", "bar"]` — both stringify to
//! `"foo|bar"`. It would therefore mis-COUNT them (their counts collided onto a single
//! `"foo|bar"` key) and, under the whole-string hash of the old `CpuProportional`
//! branch, mis-ROUTE relative to the first-token router. The term-id format eliminates
//! the collision: `["foo|bar"]` encodes to `[id("foo|bar")]` and `["foo","bar"]` to
//! `[id("foo"), id("bar")]` — distinct byte keys — while routing consults only the
//! FIRST TOKEN's characters (recovered from the leading term-id), never a joined
//! string. (The precise deleted symbol names are catalogued in
//! docs/architecture/google-books-shard-routing.md, which the source-hygiene grep
//! does not scan.)

#![cfg(feature = "google-books")]

use libgrammstein::ngram::vocabulary::{create_vocabulary, encode_varint, SharedVocabARTrie};
use libgrammstein::sources::google_books::sharding::{
    compute_shard_key_from_token, ShardConfig, ShardCoordinator, ShardGranularity, ShardKey,
};
use tempfile::TempDir;

/// Concatenated LEB128 term-id byte key for a token sequence.
fn ngram_key(vocab: &SharedVocabARTrie, tokens: &[&str]) -> Vec<u8> {
    let mut key = Vec::with_capacity(tokens.len() * 2);
    for token in tokens {
        let id = vocab.as_ref().insert(token).expect("vocab insert");
        encode_varint(id, &mut key);
    }
    key
}

/// The shard a token sequence routes to (first token's characters + length).
fn route(coordinator: &ShardCoordinator, tokens: &[&str]) -> ShardKey {
    compute_shard_key_from_token(
        tokens[0],
        tokens.len() as u8,
        &coordinator.config().granularity,
    )
}

/// Store one n-gram exactly as the importer does: first-token route +
/// concatenated-varint term-id key + `store_in_shard`.
fn store_ngram(
    coordinator: &ShardCoordinator,
    vocab: &SharedVocabARTrie,
    tokens: &[&str],
    count: u64,
) {
    coordinator
        .store_in_shard(
            &route(coordinator, tokens),
            &ngram_key(vocab, tokens),
            count,
        )
        .expect("store_in_shard");
}

#[test]
fn delimiter_collision_routes_and_counts_without_crosstalk() {
    let dir = TempDir::new().expect("tempdir");
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);
    let coordinator = ShardCoordinator::new(config).expect("coordinator");

    // "foo|bar" is ONE vocabulary term, distinct from "foo", "bar", "baz".
    let id_foobar = vocab.as_ref().insert("foo|bar").expect("insert foo|bar");
    let id_foo = vocab.as_ref().insert("foo").expect("insert foo");
    let id_bar = vocab.as_ref().insert("bar").expect("insert bar");
    let id_baz = vocab.as_ref().insert("baz").expect("insert baz");

    // ---- (1) id("foo|bar") ∉ {id("foo"), id("bar")}; all four ids distinct. ----
    assert_ne!(id_foobar, id_foo, "'foo|bar' must not share id with 'foo'");
    assert_ne!(id_foobar, id_bar, "'foo|bar' must not share id with 'bar'");
    let distinct: std::collections::BTreeSet<u64> =
        [id_foobar, id_foo, id_bar, id_baz].into_iter().collect();
    assert_eq!(distinct.len(), 4, "all four terms have distinct ids");

    // Store, exactly as the importer would:
    //   A = ["foo|bar"]        (order 1)
    //   B = ["foo|bar", "baz"] (order 2)
    //   C = ["foo", "bar"]     (order 2)
    let a: &[&str] = &["foo|bar"];
    let b: &[&str] = &["foo|bar", "baz"];
    let c: &[&str] = &["foo", "bar"];
    store_ngram(&coordinator, &vocab, a, 11);
    store_ngram(&coordinator, &vocab, b, 22);
    store_ngram(&coordinator, &vocab, c, 33);

    // The three term-id byte keys are pairwise distinct. Crucially A ≠ C: under the
    // deleted '|'-router both `["foo|bar"]` and `["foo","bar"]` stringified to
    // "foo|bar" and collided; as term-ids they are [id_foobar] vs [id_foo, id_bar].
    let key_a = ngram_key(&vocab, a);
    let key_b = ngram_key(&vocab, b);
    let key_c = ngram_key(&vocab, c);
    assert_ne!(key_a, key_b, "A and B keys must differ");
    assert_ne!(
        key_a, key_c,
        "A and C keys must differ (the collision the fix removes)"
    );
    assert_ne!(key_b, key_c, "B and C keys must differ");

    // Concrete shape: A is exactly [varint(id_foobar)], C is [varint(foo), varint(bar)].
    let mut expected_a = Vec::new();
    encode_varint(id_foobar, &mut expected_a);
    assert_eq!(
        key_a, expected_a,
        "A encodes to the single term-id of 'foo|bar'"
    );
    let mut expected_c = Vec::new();
    encode_varint(id_foo, &mut expected_c);
    encode_varint(id_bar, &mut expected_c);
    assert_eq!(key_c, expected_c, "C encodes to two term-ids [foo, bar]");

    // ---- (2) Routing is independent and collision-free. ----
    // The routes match the pure function applied to (first token, length). Here all
    // three land in the SAME shard ("fo"), which is the strongest collision test:
    // even co-located, the distinct term-id keys never cross-talk.
    let route_a = route(&coordinator, a);
    let route_b = route(&coordinator, b);
    let route_c = route(&coordinator, c);
    assert_eq!(
        route_a,
        compute_shard_key_from_token("foo|bar", 1, &ShardGranularity::TwoChar)
    );
    assert_eq!(
        route_b,
        compute_shard_key_from_token("foo|bar", 2, &ShardGranularity::TwoChar)
    );
    assert_eq!(
        route_c,
        compute_shard_key_from_token("foo", 2, &ShardGranularity::TwoChar)
    );
    assert_eq!(route_a.prefix, "fo");
    assert_eq!(route_b.prefix, "fo");
    assert_eq!(route_c.prefix, "fo");

    // Each n-gram returns ITS OWN count via get_in_shard(route(X), key(X)).
    assert_eq!(
        coordinator.get_in_shard(&route_a, &key_a),
        Some(11),
        "A = [\"foo|bar\"] retains its own count"
    );
    assert_eq!(
        coordinator.get_in_shard(&route_b, &key_b),
        Some(22),
        "B = [\"foo|bar\",\"baz\"] retains its own count"
    );
    assert_eq!(
        coordinator.get_in_shard(&route_c, &key_c),
        Some(33),
        "C = [\"foo\",\"bar\"] retains its own count — NOT merged with A"
    );

    // No cross-talk: A's count (11) is not C's (33) and vice versa. Under the old
    // '|'-router these would have been a single "foo|bar" key with count 11+33=44
    // (or one clobbering the other). The term-id keys keep them fully independent.
    assert_ne!(
        coordinator.get_in_shard(&route_a, &key_a),
        coordinator.get_in_shard(&route_c, &key_c),
        "A and C are distinct n-grams with distinct counts (no delimiter collision)"
    );
}
