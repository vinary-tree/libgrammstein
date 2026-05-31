(*
 * Formal verification of FrequencyCounts merge operation properties.
 *
 * This specification proves algebraic properties from:
 *   src/sources/google_books/sharding/mkn.rs:98-107
 *
 * The merge operation is used for parallel aggregation of frequency counts
 * across shards. For correct parallel reduction, merge must be:
 *   1. Associative: merge (merge a b) c = merge a (merge b c)
 *   2. Commutative: merge a b = merge b a
 *   3. Have identity: merge a default = a
 *
 * These properties ensure that the order of parallel aggregation doesn't
 * affect the final result, enabling correct work distribution across threads.
 *
 * Rocq 9.1 compatible.
 *)

From Stdlib Require Import Arith.
From Stdlib Require Import PeanoNat.
From Stdlib Require Import Lia.

(* ---------------------------------------------------------------------------
 * Definitions
 * --------------------------------------------------------------------------- *)

(**
 * FrequencyCounts record corresponding to mkn.rs:76-95.
 *
 * Fields:
 *   - n1: Count of n-grams occurring exactly once
 *   - n2: Count of n-grams occurring exactly twice
 *   - n3: Count of n-grams occurring exactly 3 times
 *   - n4: Count of n-grams occurring exactly 4 times
 *   - total_unique: Total unique n-grams
 *   - total_count: Sum of all occurrence counts
 *)
Record FrequencyCounts : Type := mk_freq_counts {
  n1 : nat;
  n2 : nat;
  n3 : nat;
  n4 : nat;
  total_unique : nat;
  total_count : nat
}.

(**
 * Default (zero) frequency counts.
 * Corresponds to Default trait implementation.
 *)
Definition default_freq_counts : FrequencyCounts :=
  mk_freq_counts 0 0 0 0 0 0.

(**
 * Maximum value of Rust's u64. The executable code stores FrequencyCounts
 * fields in u64, so the algebraic nat model applies directly only while
 * these bounds are respected.
 *)
Definition u64_max : nat := 18446744073709551615.

(**
 * Merge two FrequencyCounts by field-wise addition.
 * Corresponds to mkn.rs:98-107 (merge method).
 *
 * This is the key operation for parallel aggregation.
 *)
Definition merge (a b : FrequencyCounts) : FrequencyCounts :=
  mk_freq_counts
    (n1 a + n1 b)
    (n2 a + n2 b)
    (n3 a + n3 b)
    (n4 a + n4 b)
    (total_unique a + total_unique b)
    (total_count a + total_count b).

(**
 * Equality of FrequencyCounts (field-wise).
 *)
Definition freq_counts_eq (a b : FrequencyCounts) : Prop :=
  n1 a = n1 b /\
  n2 a = n2 b /\
  n3 a = n3 b /\
  n4 a = n4 b /\
  total_unique a = total_unique b /\
  total_count a = total_count b.

(**
 * All fields fit in Rust's u64 representation.
 *)
Definition freq_counts_within_u64 (fc : FrequencyCounts) : Prop :=
  n1 fc <= u64_max /\
  n2 fc <= u64_max /\
  n3 fc <= u64_max /\
  n4 fc <= u64_max /\
  total_unique fc <= u64_max /\
  total_count fc <= u64_max.

(**
 * Field-wise merge precondition for u64 implementations: every addition must
 * fit in u64. Under this precondition, Rust u64 addition agrees with the nat
 * addition modeled by merge.
 *)
Definition merge_no_overflow (a b : FrequencyCounts) : Prop :=
  n1 a + n1 b <= u64_max /\
  n2 a + n2 b <= u64_max /\
  n3 a + n3 b <= u64_max /\
  n4 a + n4 b <= u64_max /\
  total_unique a + total_unique b <= u64_max /\
  total_count a + total_count b <= u64_max.

(**
 * Decidable equality for FrequencyCounts.
 *)
Lemma freq_counts_eq_dec : forall a b : FrequencyCounts,
  {freq_counts_eq a b} + {~ freq_counts_eq a b}.
Proof.
  intros a b.
  unfold freq_counts_eq.
  destruct (Nat.eq_dec (n1 a) (n1 b));
  destruct (Nat.eq_dec (n2 a) (n2 b));
  destruct (Nat.eq_dec (n3 a) (n3 b));
  destruct (Nat.eq_dec (n4 a) (n4 b));
  destruct (Nat.eq_dec (total_unique a) (total_unique b));
  destruct (Nat.eq_dec (total_count a) (total_count b));
  try (left; repeat split; assumption);
  right; intro H; destruct H as [? [? [? [? [? ?]]]]]; contradiction.
Qed.

(* ---------------------------------------------------------------------------
 * Main Theorems
 * --------------------------------------------------------------------------- *)

(**
 * Theorem: Merge is associative.
 *
 * merge (merge a b) c = merge a (merge b c)
 *
 * This property is critical for parallel reduction correctness.
 * It allows the merge operation to be applied in any grouping order.
 *)
Theorem merge_associative : forall a b c : FrequencyCounts,
  freq_counts_eq (merge (merge a b) c) (merge a (merge b c)).
Proof.
  intros a b c.
  unfold freq_counts_eq, merge.
  simpl.
  repeat split; lia.
Qed.

(**
 * Theorem: Merge is commutative.
 *
 * merge a b = merge b a
 *
 * This property allows parallel workers to aggregate in any order.
 *)
Theorem merge_commutative : forall a b : FrequencyCounts,
  freq_counts_eq (merge a b) (merge b a).
Proof.
  intros a b.
  unfold freq_counts_eq, merge.
  simpl.
  repeat split; lia.
Qed.

(**
 * Theorem: default_freq_counts is a right identity for merge.
 *
 * merge a default = a
 *)
Theorem merge_identity_right : forall a : FrequencyCounts,
  freq_counts_eq (merge a default_freq_counts) a.
Proof.
  intros a.
  unfold freq_counts_eq, merge, default_freq_counts.
  simpl.
  repeat split; lia.
Qed.

(**
 * Theorem: default_freq_counts is a left identity for merge.
 *
 * merge default a = a
 *)
Theorem merge_identity_left : forall a : FrequencyCounts,
  freq_counts_eq (merge default_freq_counts a) a.
Proof.
  intros a.
  unfold freq_counts_eq, merge, default_freq_counts.
  simpl.
  repeat split; lia.
Qed.

(**
 * Corollary: Merge forms a commutative monoid with default_freq_counts.
 *
 * This algebraic structure guarantees that parallel reduction produces
 * the same result regardless of partition strategy or reduction order.
 *)
Corollary merge_is_commutative_monoid : forall a b c : FrequencyCounts,
  (* Associativity *)
  freq_counts_eq (merge (merge a b) c) (merge a (merge b c)) /\
  (* Commutativity *)
  freq_counts_eq (merge a b) (merge b a) /\
  (* Identity *)
  freq_counts_eq (merge a default_freq_counts) a /\
  freq_counts_eq (merge default_freq_counts a) a.
Proof.
  intros a b c.
  split. { apply merge_associative. }
  split. { apply merge_commutative. }
  split. { apply merge_identity_right. }
  apply merge_identity_left.
Qed.

(* ---------------------------------------------------------------------------
 * Additional Properties for Parallel Aggregation
 * --------------------------------------------------------------------------- *)

(**
 * Theorem: Merging preserves field non-negativity.
 *
 * Since we use nat (natural numbers), all fields are non-negative by construction.
 * This theorem is trivial but states the property explicitly.
 *)
Theorem merge_preserves_nonneg : forall a b : FrequencyCounts,
  0 <= n1 (merge a b) /\
  0 <= n2 (merge a b) /\
  0 <= n3 (merge a b) /\
  0 <= n4 (merge a b) /\
  0 <= total_unique (merge a b) /\
  0 <= total_count (merge a b).
Proof.
  intros a b.
  unfold merge.
  simpl.
  repeat split; lia.
Qed.

(**
 * Theorem: Merging preserves u64 representability when every field-wise sum
 * fits in u64.
 *)
Theorem merge_preserves_u64_bounds : forall a b : FrequencyCounts,
  merge_no_overflow a b ->
  freq_counts_within_u64 (merge a b).
Proof.
  intros a b Hno.
  unfold merge_no_overflow in Hno.
  unfold freq_counts_within_u64, merge.
  simpl.
  assumption.
Qed.

(**
 * Theorem: Merged total_unique equals sum of inputs.
 *
 * This is important for verifying that no entries are lost during merging.
 *)
Theorem merge_total_unique_additive : forall a b : FrequencyCounts,
  total_unique (merge a b) = total_unique a + total_unique b.
Proof.
  intros a b.
  unfold merge.
  simpl.
  reflexivity.
Qed.

(**
 * Theorem: Merged total_count equals sum of inputs.
 *)
Theorem merge_total_count_additive : forall a b : FrequencyCounts,
  total_count (merge a b) = total_count a + total_count b.
Proof.
  intros a b.
  unfold merge.
  simpl.
  reflexivity.
Qed.

(**
 * Theorem: Merging multiple counts is equivalent to summing all fields.
 *
 * This shows that merge correctly accumulates statistics from multiple sources.
 *)
Theorem merge_triple_sum : forall a b c : FrequencyCounts,
  total_count (merge (merge a b) c) = total_count a + total_count b + total_count c.
Proof.
  intros a b c.
  unfold merge.
  simpl.
  lia.
Qed.

(* ---------------------------------------------------------------------------
 * Frequency Count Invariants
 * --------------------------------------------------------------------------- *)

(**
 * Invariant: The sum of counted n-grams (n1+n2+n3+n4+...) should not exceed
 * total_unique. This holds when n1-n4 are exclusive categories.
 *
 * Note: In practice, n-grams with count > 4 are not tracked in n1-n4,
 * so total_unique can exceed n1+n2+n3+n4.
 *)
Definition freq_counts_valid (fc : FrequencyCounts) : Prop :=
  n1 fc + n2 fc + n3 fc + n4 fc <= total_unique fc.

(**
 * Theorem: Merging valid counts produces valid counts.
 *)
Theorem merge_preserves_validity : forall a b : FrequencyCounts,
  freq_counts_valid a -> freq_counts_valid b ->
  freq_counts_valid (merge a b).
Proof.
  intros a b Ha Hb.
  unfold freq_counts_valid, merge in *.
  simpl.
  lia.
Qed.

(**
 * Theorem: Default counts are valid.
 *)
Theorem default_is_valid : freq_counts_valid default_freq_counts.
Proof.
  unfold freq_counts_valid, default_freq_counts.
  simpl.
  lia.
Qed.

(* ---------------------------------------------------------------------------
 * Atomic Aggregation Correctness
 * --------------------------------------------------------------------------- *)

(**
 * The observe operation from AtomicFrequencyCounts (mkn.rs:122-132).
 *
 * Increments the appropriate n_k counter based on the observed count value,
 * and always increments total_unique and adds count to total_count.
 *)
Definition observe (fc : FrequencyCounts) (count : nat) : FrequencyCounts :=
  let base := mk_freq_counts
    (n1 fc) (n2 fc) (n3 fc) (n4 fc)
    (total_unique fc + 1)
    (total_count fc + count) in
  match count with
  | 1 => mk_freq_counts (n1 base + 1) (n2 base) (n3 base) (n4 base)
                        (total_unique base) (total_count base)
  | 2 => mk_freq_counts (n1 base) (n2 base + 1) (n3 base) (n4 base)
                        (total_unique base) (total_count base)
  | 3 => mk_freq_counts (n1 base) (n2 base) (n3 base + 1) (n4 base)
                        (total_unique base) (total_count base)
  | 4 => mk_freq_counts (n1 base) (n2 base) (n3 base) (n4 base + 1)
                        (total_unique base) (total_count base)
  | _ => base  (* count = 0 or count > 4: don't increment n_k *)
  end.

(**
 * Theorem: Observing preserves validity for every count. Counts outside
 * [1,4] increase total_unique but do not affect n1..n4, matching Rust's
 * AtomicFrequencyCounts::observe behavior.
 *)
Theorem observe_preserves_validity : forall fc count,
  freq_counts_valid fc ->
  freq_counts_valid (observe fc count).
Proof.
  intros fc count Hvalid.
  unfold freq_counts_valid, observe in *.
  destruct count as [|[|[|[|[|n]]]]]; simpl; lia.
Qed.

(**
 * Field-wise observe precondition for u64 implementations. Only the bucket
 * matching count 1..4 is incremented, while total_unique and total_count are
 * always updated.
 *)
Definition observe_no_overflow (fc : FrequencyCounts) (count : nat) : Prop :=
  total_unique fc + 1 <= u64_max /\
  total_count fc + count <= u64_max /\
  match count with
  | 1 => n1 fc + 1 <= u64_max
  | 2 => n2 fc + 1 <= u64_max
  | 3 => n3 fc + 1 <= u64_max
  | 4 => n4 fc + 1 <= u64_max
  | _ => True
  end.

(**
 * Theorem: observe preserves u64 representability when the updated fields fit.
 *)
Theorem observe_preserves_u64_bounds : forall fc count,
  freq_counts_within_u64 fc ->
  observe_no_overflow fc count ->
  freq_counts_within_u64 (observe fc count).
Proof.
  intros fc count Hbounds Hno.
  unfold freq_counts_within_u64 in *.
  unfold observe_no_overflow in Hno.
  unfold observe.
  destruct Hbounds as [Hn1 [Hn2 [Hn3 [Hn4 [Hunique Htotal]]]]].
  destruct Hno as [Hunique' [Htotal' Hbucket]].
  destruct count as [|[|[|[|[|n]]]]]; simpl; repeat split; try lia.
Qed.

(**
 * Theorem: observe increases total_unique by exactly 1.
 *)
Theorem observe_increases_unique : forall fc count,
  total_unique (observe fc count) = total_unique fc + 1.
Proof.
  intros fc count.
  unfold observe.
  destruct count as [|[|[|[|[|n]]]]]; simpl; reflexivity.
Qed.

(**
 * Theorem: observe increases total_count by exactly count.
 *)
Theorem observe_increases_count : forall fc count,
  total_count (observe fc count) = total_count fc + count.
Proof.
  intros fc count.
  unfold observe.
  destruct count as [|[|[|[|[|n]]]]]; simpl; reflexivity.
Qed.

(* ---------------------------------------------------------------------------
 * Spec-to-Code Traceability
 * --------------------------------------------------------------------------- *)

(*
 * Mapping from Coq definitions to Rust implementation:
 *
 * Coq Definition           | Rust Code                      | Location
 * -------------------------|--------------------------------|----------
 * FrequencyCounts          | struct FrequencyCounts         | mkn.rs:76-95
 * mk_freq_counts           | FrequencyCounts { ... }        | mkn.rs:76-95
 * default_freq_counts      | FrequencyCounts::default()     | mkn.rs:76 (#[derive(Default)])
 * merge                    | FrequencyCounts::merge(&mut)   | mkn.rs:98-107
 * observe                  | AtomicFrequencyCounts::observe | mkn.rs:122-132
 * n1, n2, n3, n4           | FrequencyCounts.n1, .n2, ...   | mkn.rs:78-88
 * total_unique             | FrequencyCounts.total_unique   | mkn.rs:91
 * total_count              | FrequencyCounts.total_count    | mkn.rs:94
 *
 * The Rust implementation uses u64 while this proof uses nat.
 * The algebraic properties proven here hold for arbitrary-precision arithmetic.
 * For u64, overflow is possible but would require astronomical data volumes.
 *)
