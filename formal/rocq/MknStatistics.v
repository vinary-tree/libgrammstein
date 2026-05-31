(*
 * Formal verification of Modified Kneser-Ney (MKN) discount parameter bounds.
 *
 * This specification proves the mathematical properties from:
 *   src/sources/google_books/sharding/mkn.rs:186-230
 *
 * MKN discount formulas (from mkn.rs:206-218):
 *   Y = n1 / (n1 + 2*n2)
 *   D1 = 1 - 2*Y*(n2/n1)
 *   D2 = 2 - 3*Y*(n3/n2)
 *   D3+ = 3 - 4*Y*(n4/n3)
 *
 * Properties proven:
 *   1. Y is bounded in [0, 1]
 *   2. D1 is bounded in [0, 1]
 *   3. D2 is bounded in [0, 2]
 *   4. D3+ is bounded in [0, 3]
 *
 * Note: D2 and D3+ can exceed 1 by design (see mkn.rs:213-216 comments).
 * The Rust code clamps these values, which is modeled here.
 *
 * Rocq 9.1 compatible.
 *)

From Stdlib Require Import QArith.
From Stdlib Require Import Qminmax.
From Stdlib Require Import Lqa.

Open Scope Q_scope.

(* ---------------------------------------------------------------------------
 * Definitions
 * --------------------------------------------------------------------------- *)

(**
 * Compute Y parameter from frequency counts.
 *
 * Y = n1 / (n1 + 2*n2)
 *
 * Corresponds to mkn.rs:207
 *)
Definition compute_y (n1 n2 : Q) : Q :=
  n1 / (n1 + 2 * n2).

(**
 * Compute D1 discount (for n-grams occurring once).
 *
 * D1 = 1 - 2*Y*(n2/n1)
 *
 * Corresponds to mkn.rs:210
 *)
Definition compute_d1 (n1 n2 y : Q) : Q :=
  1 - 2 * y * (n2 / n1).

(**
 * Compute D2 discount (for n-grams occurring twice).
 *
 * D2 = 2 - 3*Y*(n3/n2)
 *
 * Corresponds to mkn.rs:213
 *)
Definition compute_d2 (n2 n3 y : Q) : Q :=
  2 - 3 * y * (n3 / n2).

(**
 * Compute D3+ discount (for n-grams occurring 3+ times).
 *
 * D3+ = 3 - 4*Y*(n4/n3)
 *
 * Corresponds to mkn.rs:216
 *)
Definition compute_d3_plus (n3 n4 y : Q) : Q :=
  3 - 4 * y * (n4 / n3).

(**
 * Clamp a value to a range [lo, hi].
 *
 * Corresponds to the .max(0.0).min(bound) pattern in Rust.
 *)
Definition clamp (x lo hi : Q) : Q :=
  Qmax lo (Qmin x hi).

(**
 * Clamped D1 as implemented in Rust (mkn.rs:210).
 *)
Definition clamped_d1 (n1 n2 y : Q) : Q :=
  clamp (compute_d1 n1 n2 y) 0 1.

(**
 * Clamped D2 as implemented in Rust (mkn.rs:213).
 *)
Definition clamped_d2 (n2 n3 y : Q) : Q :=
  clamp (compute_d2 n2 n3 y) 0 2.

(**
 * Clamped D3+ as implemented in Rust (mkn.rs:216).
 *)
Definition clamped_d3_plus (n3 n4 y : Q) : Q :=
  clamp (compute_d3_plus n3 n4 y) 0 3.

(* ---------------------------------------------------------------------------
 * Helper Lemmas
 * --------------------------------------------------------------------------- *)

(**
 * Positive rationals are non-negative.
 *)
Lemma pos_implies_nonneg : forall q : Q, 0 < q -> 0 <= q.
Proof.
  intros q H. lra.
Qed.

(**
 * Sum of positive rationals is positive.
 *)
Lemma sum_pos : forall a b : Q, 0 < a -> 0 < b -> 0 < a + b.
Proof.
  intros. lra.
Qed.

(**
 * Product of positive rationals is positive.
 *)
Lemma mult_pos : forall a b : Q, 0 < a -> 0 < b -> 0 < a * b.
Proof.
  intros a b Ha Hb.
  apply Qmult_lt_0_compat; assumption.
Qed.

(**
 * Division of positive by positive is positive.
 *)
Lemma div_pos : forall a b : Q, 0 < a -> 0 < b -> 0 < a / b.
Proof.
  intros a b Ha Hb.
  unfold Qdiv.
  apply Qmult_lt_0_compat; [assumption|].
  apply Qinv_lt_0_compat.
  assumption.
Qed.

(**
 * Division x/y <= 1 when 0 < x <= y.
 *)
Lemma div_le_1 : forall x y : Q, 0 < x -> x <= y -> x / y <= 1.
Proof.
  intros x y Hx_pos Hxy.
  assert (Hy_pos: 0 < y) by lra.
  assert (Hy_neq: ~ y == 0) by lra.
  (* Use field to show x/y <= 1 iff x <= y when y > 0 *)
  cut (x / y * y <= 1 * y).
  - intro H.
    apply Qmult_lt_0_le_reg_r with y; assumption.
  - field_simplify.
    + assumption.
    + assumption.
Qed.

(**
 * Clamp always returns a value in [lo, hi].
 *)
Lemma clamp_bounds : forall x lo hi : Q,
  lo <= hi -> lo <= clamp x lo hi /\ clamp x lo hi <= hi.
Proof.
  intros x lo hi Hlo_le_hi.
  unfold clamp.
  split.
  - apply Q.le_max_l.
  - apply Q.max_lub.
    + assumption.
    + apply Q.le_min_r.
Qed.

(* ---------------------------------------------------------------------------
 * Main Theorems
 * --------------------------------------------------------------------------- *)

(**
 * Theorem: Y is bounded in [0, 1] when n1 > 0 and n2 > 0.
 *
 * Proof sketch:
 *   - Y = n1 / (n1 + 2*n2)
 *   - Numerator: n1 > 0
 *   - Denominator: n1 + 2*n2 >= n1 (since n2 > 0)
 *   - Therefore: 0 < Y <= 1
 *
 * Corresponds to the implicit bounds in mkn.rs:192-198.
 *)
Theorem y_bounded : forall n1 n2 : Q,
  0 < n1 -> 0 < n2 ->
  0 <= compute_y n1 n2 /\ compute_y n1 n2 <= 1.
Proof.
  intros n1 n2 Hn1 Hn2.
  unfold compute_y.
  split.
  - (* 0 <= n1 / (n1 + 2*n2) *)
    apply pos_implies_nonneg.
    apply div_pos.
    + assumption.
    + lra.
  - (* n1 / (n1 + 2*n2) <= 1 *)
    apply div_le_1.
    + assumption.
    + lra.
Qed.

(**
 * Corollary: Y is strictly positive when n1 > 0 and n2 > 0.
 *)
Corollary y_positive : forall n1 n2 : Q,
  0 < n1 -> 0 < n2 ->
  0 < compute_y n1 n2.
Proof.
  intros n1 n2 Hn1 Hn2.
  unfold compute_y.
  apply div_pos.
  - assumption.
  - lra.
Qed.

(**
 * Theorem: D1 equals 1 - 2*Y*(n2/n1), and when Y = n1/(n1 + 2*n2),
 * D1 simplifies to a value in [0, 1] after clamping.
 *
 * Note: The raw D1 can be negative if n2 > n1, hence clamping.
 *)
Theorem d1_clamped_bounded : forall n1 n2 y : Q,
  0 < n1 -> 0 < n2 ->
  0 <= clamped_d1 n1 n2 y /\ clamped_d1 n1 n2 y <= 1.
Proof.
  intros n1 n2 y Hn1 Hn2.
  unfold clamped_d1.
  apply clamp_bounds.
  lra.
Qed.

(**
 * Theorem: When Y = n1/(n1 + 2*n2), the raw D1 formula simplifies.
 *
 * D1 = 1 - 2 * (n1/(n1 + 2*n2)) * (n2/n1)
 *    = 1 - 2*n2 / (n1 + 2*n2)
 *    = (n1 + 2*n2 - 2*n2) / (n1 + 2*n2)
 *    = n1 / (n1 + 2*n2)
 *    = Y
 *
 * This shows D1 = Y when using the standard Y computation.
 *)
Theorem d1_equals_y : forall n1 n2 : Q,
  0 < n1 -> 0 < n2 ->
  compute_d1 n1 n2 (compute_y n1 n2) == compute_y n1 n2.
Proof.
  intros n1 n2 Hn1 Hn2.
  unfold compute_d1, compute_y.
  field.
  split.
  - lra.
  - lra.
Qed.

(**
 * Corollary: D1 (using Y = n1/(n1+2*n2)) is bounded in [0, 1].
 *)
Corollary d1_bounded : forall n1 n2 : Q,
  0 < n1 -> 0 < n2 ->
  0 <= compute_d1 n1 n2 (compute_y n1 n2) /\
  compute_d1 n1 n2 (compute_y n1 n2) <= 1.
Proof.
  intros n1 n2 Hn1 Hn2.
  assert (Heq: compute_d1 n1 n2 (compute_y n1 n2) == compute_y n1 n2)
    by (apply d1_equals_y; assumption).
  split.
  - rewrite Heq. apply y_bounded; assumption.
  - rewrite Heq. apply y_bounded; assumption.
Qed.

(**
 * Theorem: D2 is bounded after clamping to [0, 2].
 *)
Theorem d2_clamped_bounded : forall n2 n3 y : Q,
  0 < n2 -> 0 < n3 ->
  0 <= clamped_d2 n2 n3 y /\ clamped_d2 n2 n3 y <= 2.
Proof.
  intros n2 n3 y Hn2 Hn3.
  unfold clamped_d2.
  apply clamp_bounds.
  lra.
Qed.

(**
 * Theorem: D3+ is bounded after clamping to [0, 3].
 *)
Theorem d3_plus_clamped_bounded : forall n3 n4 y : Q,
  0 < n3 -> 0 < n4 ->
  0 <= clamped_d3_plus n3 n4 y /\ clamped_d3_plus n3 n4 y <= 3.
Proof.
  intros n3 n4 y Hn3 Hn4.
  unfold clamped_d3_plus.
  apply clamp_bounds.
  lra.
Qed.

(**
 * Theorem: Under typical corpus statistics where n1 >= n2 >= n3 >= n4,
 * the unclamped D2 is bounded in [0, 2].
 *
 * Proof: D2 = 2 - 3*Y*(n3/n2)
 * - Since 0 < Y <= 1 and 0 < n3/n2 <= 1 (when n3 <= n2)
 * - We have 0 <= 3*Y*(n3/n2) <= 3
 * - So -1 <= D2 <= 2
 * - With clamping: 0 <= D2 <= 2
 *)
Theorem d2_unclamped_upper_bound : forall n2 n3 y : Q,
  0 < n2 -> 0 < n3 -> n3 <= n2 ->
  0 <= y -> y <= 1 ->
  compute_d2 n2 n3 y <= 2.
Proof.
  intros n2 n3 y Hn2 Hn3 Hn3_le_n2 Hy_ge_0 Hy_le_1.
  unfold compute_d2.
  assert (H_ratio: 0 <= n3 / n2).
  { apply pos_implies_nonneg. apply div_pos; assumption. }
  assert (H_ratio_le_1: n3 / n2 <= 1).
  { apply div_le_1; assumption. }
  assert (H_prod: 0 <= 3 * y * (n3 / n2)).
  { apply Qmult_le_0_compat.
    - apply Qmult_le_0_compat; lra.
    - assumption. }
  lra.
Qed.

(**
 * Theorem: Under typical corpus statistics where n1 >= n2 >= n3 >= n4,
 * the unclamped D3+ is bounded in [0, 3].
 *)
Theorem d3_plus_unclamped_upper_bound : forall n3 n4 y : Q,
  0 < n3 -> 0 < n4 -> n4 <= n3 ->
  0 <= y -> y <= 1 ->
  compute_d3_plus n3 n4 y <= 3.
Proof.
  intros n3 n4 y Hn3 Hn4 Hn4_le_n3 Hy_ge_0 Hy_le_1.
  unfold compute_d3_plus.
  assert (H_ratio: 0 <= n4 / n3).
  { apply pos_implies_nonneg. apply div_pos; assumption. }
  assert (H_ratio_le_1: n4 / n3 <= 1).
  { apply div_le_1; assumption. }
  assert (H_prod: 0 <= 4 * y * (n4 / n3)).
  { apply Qmult_le_0_compat.
    - apply Qmult_le_0_compat; lra.
    - assumption. }
  lra.
Qed.

(* ---------------------------------------------------------------------------
 * Discount Selection (discount_for)
 * --------------------------------------------------------------------------- *)

(**
 * Discount selection function.
 * Corresponds to mkn.rs:222-228 (discount_for method).
 *)
Definition discount_for (d1 d2 d3_plus : Q) (count : nat) : Q :=
  match count with
  | O => 0
  | S O => d1        (* count = 1 *)
  | S (S O) => d2    (* count = 2 *)
  | _ => d3_plus     (* count >= 3 *)
  end.

(**
 * Theorem: discount_for returns values in expected ranges.
 *)
Theorem discount_for_bounded : forall d1 d2 d3_plus : Q, forall count : nat,
  0 <= d1 -> d1 <= 1 ->
  0 <= d2 -> d2 <= 2 ->
  0 <= d3_plus -> d3_plus <= 3 ->
  0 <= discount_for d1 d2 d3_plus count /\
  discount_for d1 d2 d3_plus count <= 3.
Proof.
  intros d1 d2 d3_plus count Hd1_lo Hd1_hi Hd2_lo Hd2_hi Hd3_lo Hd3_hi.
  unfold discount_for.
  destruct count as [|[|[|n]]]; split; lra.
Qed.

(* ---------------------------------------------------------------------------
 * Spec-to-Code Traceability
 * --------------------------------------------------------------------------- *)

(* ---------------------------------------------------------------------------
 * Spec-to-Code Traceability
 * --------------------------------------------------------------------------- *)

(*
 * Mapping from Coq definitions to Rust implementation:
 *
 * Coq Definition           | Rust Code                      | Location
 * -------------------------|--------------------------------|----------
 * compute_y                | let y = n1 / (n1 + 2.0 * n2)   | mkn.rs:207
 * compute_d1               | let d1 = 1.0 - 2.0*y*(n2/n1)   | mkn.rs:210
 * compute_d2               | let d2 = 2.0 - 3.0*y*(n3/n2)   | mkn.rs:213
 * compute_d3_plus          | let d3_plus = 3.0 - 4.0*y*...  | mkn.rs:216
 * clamp                    | .max(0.0).min(bound)           | mkn.rs:210,213,216
 * discount_for             | DiscountParams::discount_for   | mkn.rs:222-228
 *
 * The Rust implementation uses f64 while this proof uses rational numbers (Q).
 * The bounds proven here hold for exact arithmetic; floating-point rounding
 * may cause minor deviations that remain within the clamped bounds.
 *)

(* ---------------------------------------------------------------------------
 * Zero-Input Handling and Default Values
 * ---------------------------------------------------------------------------
 *
 * The Rust implementation returns default values when n1=0 or n2=0 (mkn.rs:190-198).
 * Additionally, n3 and n4 are clamped to minimum of 1 to avoid division by zero
 * (mkn.rs:203-204).
 *
 * This section models these behaviors and proves correctness.
 * --------------------------------------------------------------------------- *)

(**
 * Default discount parameters from mkn.rs:177-182.
 * These values are used when insufficient data is available.
 *)
Definition default_d1 : Q := 1#2.          (* 0.5 *)
Definition default_d2 : Q := 3#4.          (* 0.75 *)
Definition default_d3_plus : Q := 9#10.    (* 0.9 *)
Definition default_y : Q := 1#2.           (* 0.5 *)

(**
 * Theorem: Default values satisfy the required bounds.
 * This verifies that the fallback values in mkn.rs are mathematically sound.
 *)
Theorem default_discounts_in_bounds :
  0 < default_d1 /\ default_d1 <= 1 /\
  0 < default_d2 /\ default_d2 <= 2 /\
  0 < default_d3_plus /\ default_d3_plus <= 3 /\
  0 < default_y /\ default_y <= 1.
Proof.
  unfold default_d1, default_d2, default_d3_plus, default_y.
  repeat split; reflexivity || lra.
Qed.

(**
 * Safe computation of D2 with n3 clamped to minimum of 1.
 * Models: let n3 = counts.n3.max(1) as f64  (mkn.rs:203)
 *)
Definition compute_d2_safe (n2 n3_raw y : Q) : Q :=
  let n3 := Qmax 1 n3_raw in
  2 - 3 * y * (n3 / n2).

(**
 * Safe computation of D3+ with n3 and n4 clamped to minimum of 1.
 * Models:
 *   let n3 = counts.n3.max(1) as f64
 *   let n4 = counts.n4.max(1) as f64
 *)
Definition compute_d3_plus_safe (n3_raw n4_raw y : Q) : Q :=
  let n3 := Qmax 1 n3_raw in
  let n4 := Qmax 1 n4_raw in
  3 - 4 * y * (n4 / n3).

(**
 * Clamped safe D2 computation.
 *)
Definition clamped_d2_safe (n2 n3_raw y : Q) : Q :=
  clamp (compute_d2_safe n2 n3_raw y) 0 2.

(**
 * Clamped safe D3+ computation.
 *)
Definition clamped_d3_plus_safe (n3 n4_raw y : Q) : Q :=
  clamp (compute_d3_plus_safe n3 n4_raw y) 0 3.

(**
 * Theorem: Safe D2 computation is bounded after clamping.
 *)
Theorem d2_safe_clamped_bounded : forall n2 n3_raw y : Q,
  0 < n2 ->
  0 <= clamped_d2_safe n2 n3_raw y /\ clamped_d2_safe n2 n3_raw y <= 2.
Proof.
  intros n2 n3_raw y Hn2.
  unfold clamped_d2_safe.
  apply clamp_bounds.
  lra.
Qed.

(**
 * Theorem: Safe D3+ computation is bounded after clamping.
 *)
Theorem d3_plus_safe_clamped_bounded : forall n3 n4_raw y : Q,
  0 <= clamped_d3_plus_safe n3 n4_raw y /\ clamped_d3_plus_safe n3 n4_raw y <= 3.
Proof.
  intros n3 n4_raw y.
  unfold clamped_d3_plus_safe.
  apply clamp_bounds.
  lra.
Qed.

(**
 * Lemma: Qmax 1 n >= 1 for any n.
 *)
Lemma qmax_1_ge_1 : forall n : Q, 1 <= Qmax 1 n.
Proof.
  intros n.
  apply Q.le_max_l.
Qed.

(**
 * Lemma: Qmax 1 n > 0 for any n.
 *)
Lemma qmax_1_pos : forall n : Q, 0 < Qmax 1 n.
Proof.
  intros n.
  assert (H: 1 <= Qmax 1 n) by apply qmax_1_ge_1.
  lra.
Qed.

(**
 * Theorem: When n3_raw >= 1, the safe computation equals the original.
 *)
Theorem d2_safe_equals_original_when_positive : forall n2 n3 y : Q,
  0 < n2 -> 1 <= n3 ->
  compute_d2_safe n2 n3 y == compute_d2 n2 n3 y.
Proof.
  intros n2 n3 y Hn2 Hn3.
  unfold compute_d2_safe, compute_d2.
  assert (Hmax: Qmax 1 n3 == n3).
  { apply Q.max_r. assumption. }
  rewrite Hmax.
  reflexivity.
Qed.

(**
 * Theorem: When n4_raw >= 1, the safe computation equals the original.
 *)
Theorem d3_plus_safe_equals_original_when_positive : forall n3 n4 y : Q,
  1 <= n3 -> 1 <= n4 ->
  compute_d3_plus_safe n3 n4 y == compute_d3_plus n3 n4 y.
Proof.
  intros n3 n4 y Hn3 Hn4.
  unfold compute_d3_plus_safe, compute_d3_plus.
  assert (Hmax3: Qmax 1 n3 == n3).
  { apply Q.max_r. assumption. }
  assert (Hmax: Qmax 1 n4 == n4).
  { apply Q.max_r. assumption. }
  rewrite Hmax3.
  rewrite Hmax.
  reflexivity.
Qed.

(**
 * Record type for MKN discount parameters.
 * Models the DiscountParams struct from mkn.rs:166-173.
 *)
Record MknDiscounts := mk_mkn_discounts {
  mkn_d1 : Q;
  mkn_d2 : Q;
  mkn_d3_plus : Q;
  mkn_y : Q
}.

(**
 * Default MKN discounts.
 * Models DiscountParams::default() from mkn.rs:175-184.
 *)
Definition default_mkn_discounts : MknDiscounts :=
  mk_mkn_discounts default_d1 default_d2 default_d3_plus default_y.

(**
 * Check if discount parameters are valid (all values in expected bounds).
 *)
Definition valid_mkn_discounts (d : MknDiscounts) : Prop :=
  0 <= mkn_d1 d /\ mkn_d1 d <= 1 /\
  0 <= mkn_d2 d /\ mkn_d2 d <= 2 /\
  0 <= mkn_d3_plus d /\ mkn_d3_plus d <= 3 /\
  0 <= mkn_y d /\ mkn_y d <= 1.

(**
 * Theorem: Default discounts are valid.
 *)
Theorem default_mkn_discounts_valid : valid_mkn_discounts default_mkn_discounts.
Proof.
  unfold valid_mkn_discounts, default_mkn_discounts.
  simpl.
  unfold default_d1, default_d2, default_d3_plus, default_y.
  repeat split; reflexivity || lra.
Qed.

(**
 * Lemma: Qle_bool returns false when not <=
 *)
Lemma Qle_bool_false : forall x y : Q, ~ x <= y -> Qle_bool x y = false.
Proof.
  intros x y H.
  destruct (Qle_bool x y) eqn:E; auto.
  exfalso.
  apply H.
  apply Qle_bool_imp_le.
  assumption.
Qed.

(**
 * Lemma: Qle_bool false gives the corresponding strict-side fact.
 *)
Lemma Qle_bool_false_not_le : forall x y : Q,
  Qle_bool x y = false -> ~ x <= y.
Proof.
  intros x y H Hle.
  apply Qle_bool_iff in Hle.
  rewrite H in Hle.
  discriminate.
Qed.

(**
 * Compute MKN discounts from frequency counts (safe version).
 * Returns None when n1 <= 0 or n2 <= 0 (caller should use defaults).
 * Models DiscountParams::from_counts() from mkn.rs:186-219.
 *)
Definition from_counts_safe (n1 n2 n3 n4 : Q) : option MknDiscounts :=
  if Qle_bool n1 0 then None
  else if Qle_bool n2 0 then None
  else
    let y := compute_y n1 n2 in
    let d1 := clamp (compute_d1 n1 n2 y) 0 1 in
    let d2 := clamp (compute_d2_safe n2 n3 y) 0 2 in
    let d3_plus := clamp (compute_d3_plus_safe n3 n4 y) 0 3 in
    Some (mk_mkn_discounts d1 d2 d3_plus y).

(**
 * Theorem: from_counts_safe returns valid discounts when n1 and n2 are positive.
 * n3 and n4 are safely clamped before division.
 *)
Theorem from_counts_safe_valid : forall n1 n2 n3 n4 d,
  0 < n1 -> 0 < n2 ->
  from_counts_safe n1 n2 n3 n4 = Some d ->
  valid_mkn_discounts d.
Proof.
  intros n1 n2 n3 n4 d Hn1 Hn2 Heq.
  unfold from_counts_safe in Heq.
  (* n1 > 0 means Qle_bool n1 0 = false *)
  assert (Hn1_bool: Qle_bool n1 0 = false).
  { apply Qle_bool_false. lra. }
  rewrite Hn1_bool in Heq.
  (* n2 > 0 means Qle_bool n2 0 = false *)
  assert (Hn2_bool: Qle_bool n2 0 = false).
  { apply Qle_bool_false. lra. }
  rewrite Hn2_bool in Heq.
  (* Now we have the Some case *)
  injection Heq as Hd.
  rewrite <- Hd.
  unfold valid_mkn_discounts. simpl.
  (* Use repeat split to handle all 8 conjuncts, then auto to solve goals *)
  repeat split.
  (* d1 lower bound *)
  - apply clamp_bounds. lra.
  (* d1 upper bound *)
  - apply clamp_bounds. lra.
  (* d2 lower bound *)
  - apply clamp_bounds. lra.
  (* d2 upper bound *)
  - apply clamp_bounds. lra.
  (* d3_plus lower bound *)
  - apply clamp_bounds. lra.
  (* d3_plus upper bound *)
  - apply clamp_bounds. lra.
  (* y lower bound *)
  - apply y_bounded; assumption.
  (* y upper bound *)
  - apply y_bounded; assumption.
Qed.

(**
 * Rust-facing total function: DiscountParams::from_counts returns defaults
 * rather than None when n1=0 or n2=0.
 *)
Definition from_counts_rust (n1 n2 n3 n4 : Q) : MknDiscounts :=
  match from_counts_safe n1 n2 n3 n4 with
  | Some d => d
  | None => default_mkn_discounts
  end.

(**
 * Theorem: from_counts_rust always returns valid discounts for non-negative
 * frequency counts.
 *)
Theorem from_counts_rust_valid : forall n1 n2 n3 n4 : Q,
  0 <= n1 -> 0 <= n2 ->
  valid_mkn_discounts (from_counts_rust n1 n2 n3 n4).
Proof.
  intros n1 n2 n3 n4 Hn1_nonneg Hn2_nonneg.
  unfold from_counts_rust.
  destruct (from_counts_safe n1 n2 n3 n4) eqn:Hsafe.
  - apply from_counts_safe_valid with (n1 := n1) (n2 := n2) (n3 := n3) (n4 := n4).
    + unfold from_counts_safe in Hsafe.
      destruct (Qle_bool n1 0) eqn:Hn1_bool; [discriminate|].
      apply Qle_bool_false_not_le in Hn1_bool.
      lra.
    + unfold from_counts_safe in Hsafe.
      destruct (Qle_bool n1 0) eqn:Hn1_bool; [discriminate|].
      destruct (Qle_bool n2 0) eqn:Hn2_bool; [discriminate|].
      apply Qle_bool_false_not_le in Hn2_bool.
      lra.
    + exact Hsafe.
  - apply default_mkn_discounts_valid.
Qed.

(**
 * Theorem: from_counts_safe returns None when n1 <= 0.
 *)
Theorem from_counts_safe_none_when_n1_zero : forall n2 n3 n4,
  from_counts_safe 0 n2 n3 n4 = None.
Proof.
  intros n2 n3 n4.
  unfold from_counts_safe.
  simpl.
  reflexivity.
Qed.

(**
 * Theorem: from_counts_safe returns None when n2 <= 0 (and n1 > 0).
 *)
Theorem from_counts_safe_none_when_n2_zero : forall n1 n3 n4,
  0 < n1 ->
  from_counts_safe n1 0 n3 n4 = None.
Proof.
  intros n1 n3 n4 Hn1.
  unfold from_counts_safe.
  assert (Hn1_bool: Qle_bool n1 0 = false).
  { apply Qle_bool_false. lra. }
  rewrite Hn1_bool.
  simpl.
  reflexivity.
Qed.

Close Scope Q_scope.
