(**
 * Floating-point envelope model for Rust MKN discount computation.
 *
 * The exact arithmetic proofs in MknStatistics.v use rationals. This file
 * models the real-valued shape of the Rust f64 calculation and proves that
 * the raw expressions stay many orders of magnitude below the binary64
 * overflow exponent for u64-derived inputs. The final clamped discounts are
 * then proven to stay in the public MKN ranges.
 *)

From Stdlib Require Import Reals Lra Psatz.
Require Import Flocq.Core.Core.
Require Import Flocq.IEEE754.Binary.
Require Import Interval.Tactic.

Open Scope R_scope.

Definition binary64 := binary_float 53 1024.

Definition u64_max_R : R := 18446744073709551615.

(**
 * A conservative envelope for every raw MKN expression below. This is far
 * below binary64's normal overflow threshold, proved with Flocq's bpow.
 *)
Definition binary64_no_overflow_margin : R :=
  10000000000000000000000000000000000000000.

Definition clamp (lo hi x : R) : R :=
  if Rlt_dec x lo then lo
  else if Rlt_dec hi x then hi
  else x.

Definition at_least_one (x : R) : R :=
  if Rlt_dec x 1 then 1 else x.

Definition y_model (n1 n2 : R) : R :=
  n1 / (n1 + 2 * n2).

Definition d1_raw (n1 n2 : R) : R :=
  1 - 2 * y_model n1 n2 * (n2 / n1).

Definition d2_raw (n1 n2 n3_raw : R) : R :=
  2 - 3 * y_model n1 n2 * (at_least_one n3_raw / n2).

Definition d3_plus_raw (n1 n2 n3_raw n4_raw : R) : R :=
  3 - 4 * y_model n1 n2 * (at_least_one n4_raw / at_least_one n3_raw).

Definition d1_model (n1 n2 : R) : R :=
  clamp 0 1 (d1_raw n1 n2).

Definition d2_model (n1 n2 n3_raw : R) : R :=
  clamp 0 2 (d2_raw n1 n2 n3_raw).

Definition d3_plus_model (n1 n2 n3_raw n4_raw : R) : R :=
  clamp 0 3 (d3_plus_raw n1 n2 n3_raw n4_raw).

Definition valid_u64_real (x : R) : Prop :=
  0 <= x <= u64_max_R.

Definition positive_u64_real (x : R) : Prop :=
  1 <= x <= u64_max_R.

Lemma positive_is_valid : forall x : R,
  positive_u64_real x ->
  valid_u64_real x.
Proof.
  intros x [Hlow Hhigh].
  split; lra.
Qed.

Lemma u64_max_at_least_one : 1 <= u64_max_R.
Proof.
  unfold u64_max_R.
  lra.
Qed.

Lemma clamp_bounds : forall lo hi x : R,
  lo <= hi ->
  lo <= clamp lo hi x <= hi.
Proof.
  intros lo hi x Hle.
  unfold clamp.
  destruct (Rlt_dec x lo) as [Hlt_lo | Hnot_lo].
  - lra.
  - destruct (Rlt_dec hi x) as [Hlt_hi | Hnot_hi]; lra.
Qed.

Lemma at_least_one_bounds : forall x : R,
  valid_u64_real x ->
  1 <= at_least_one x <= u64_max_R.
Proof.
  intros x [Hnonneg Hupper].
  unfold at_least_one.
  destruct (Rlt_dec x 1) as [Hlt | Hnot]; split; try lra.
  apply u64_max_at_least_one.
Qed.

Lemma ratio_nonnegative : forall numerator denominator : R,
  0 <= numerator ->
  1 <= denominator ->
  0 <= numerator / denominator.
Proof.
  intros numerator denominator Hnum Hden.
  apply Rmult_le_reg_r with (r := denominator); [lra|].
  field_simplify; lra.
Qed.

Lemma ratio_u64_upper : forall numerator denominator : R,
  0 <= numerator <= u64_max_R ->
  1 <= denominator ->
  numerator / denominator <= u64_max_R.
Proof.
  intros numerator denominator [Hnum_nonneg Hnum_upper] Hden.
  apply Rmult_le_reg_r with (r := denominator); [lra|].
  field_simplify; nra.
Qed.

Lemma y_model_bounds : forall n1 n2 : R,
  positive_u64_real n1 ->
  valid_u64_real n2 ->
  0 <= y_model n1 n2 <= 1.
Proof.
  intros n1 n2 [Hn1_low _] [Hn2_nonneg _].
  unfold y_model.
  split.
  - apply Rmult_le_reg_r with (r := n1 + 2 * n2); [nra|].
    field_simplify; nra.
  - apply Rmult_le_reg_r with (r := n1 + 2 * n2); [nra|].
    field_simplify; nra.
Qed.

Lemma d1_raw_envelope : forall n1 n2 : R,
  positive_u64_real n1 ->
  valid_u64_real n2 ->
  1 - 2 * u64_max_R <= d1_raw n1 n2 <= 1.
Proof.
  intros n1 n2 Hn1 Hn2.
  pose proof (y_model_bounds n1 n2 Hn1 Hn2) as [Hy_low Hy_high].
  assert (0 <= n2 / n1 <= u64_max_R) as [Hr_low Hr_high].
  {
    split.
    - apply ratio_nonnegative; [apply Hn2 | apply Hn1].
    - apply ratio_u64_upper; [apply Hn2 | apply Hn1].
  }
  unfold d1_raw.
  nra.
Qed.

Lemma d2_raw_envelope : forall n1 n2 n3 : R,
  positive_u64_real n1 ->
  positive_u64_real n2 ->
  valid_u64_real n3 ->
  2 - 3 * u64_max_R <= d2_raw n1 n2 n3 <= 2.
Proof.
  intros n1 n2 n3 Hn1 Hn2 Hn3.
  pose proof (y_model_bounds n1 n2 Hn1 (positive_is_valid n2 Hn2)) as [Hy_low Hy_high].
  pose proof (at_least_one_bounds n3 Hn3) as [Hn3_low Hn3_high].
  assert (0 <= at_least_one n3 / n2 <= u64_max_R) as [Hr_low Hr_high].
  {
    split.
    - apply ratio_nonnegative; [lra | apply Hn2].
    - apply ratio_u64_upper; [split; lra | apply Hn2].
  }
  unfold d2_raw.
  nra.
Qed.

Lemma d3_plus_raw_envelope : forall n1 n2 n3 n4 : R,
  positive_u64_real n1 ->
  valid_u64_real n2 ->
  valid_u64_real n3 ->
  valid_u64_real n4 ->
  3 - 4 * u64_max_R <= d3_plus_raw n1 n2 n3 n4 <= 3.
Proof.
  intros n1 n2 n3 n4 Hn1 Hn2 Hn3 Hn4.
  pose proof (y_model_bounds n1 n2 Hn1 Hn2) as [Hy_low Hy_high].
  pose proof (at_least_one_bounds n3 Hn3) as [Hn3_low _].
  pose proof (at_least_one_bounds n4 Hn4) as [Hn4_low Hn4_high].
  assert (0 <= at_least_one n4 / at_least_one n3 <= u64_max_R) as [Hr_low Hr_high].
  {
    split.
    - apply ratio_nonnegative; lra.
    - apply ratio_u64_upper; [split; lra | lra].
  }
  unfold d3_plus_raw.
  nra.
Qed.

Theorem mkn_discount_models_bounded : forall n1 n2 n3 n4 : R,
  positive_u64_real n1 ->
  positive_u64_real n2 ->
  valid_u64_real n3 ->
  valid_u64_real n4 ->
  (0 <= y_model n1 n2 <= 1)
  /\ 0 <= d1_model n1 n2 <= 1
  /\ 0 <= d2_model n1 n2 n3 <= 2
  /\ 0 <= d3_plus_model n1 n2 n3 n4 <= 3.
Proof.
  intros n1 n2 n3 n4 Hn1 Hn2 Hn3 Hn4.
  repeat split.
  - apply y_model_bounds; [assumption | apply positive_is_valid; assumption].
  - apply y_model_bounds; [assumption | apply positive_is_valid; assumption].
  - apply clamp_bounds. lra.
  - apply clamp_bounds. lra.
  - apply clamp_bounds. lra.
  - apply clamp_bounds. lra.
  - apply clamp_bounds. lra.
  - apply clamp_bounds. lra.
Qed.

Lemma u64_envelope_under_margin :
  4 * u64_max_R + 3 < binary64_no_overflow_margin.
Proof.
  unfold u64_max_R, binary64_no_overflow_margin.
  lra.
Qed.

Theorem raw_terms_within_binary64_margin : forall n1 n2 n3 n4 : R,
  positive_u64_real n1 ->
  positive_u64_real n2 ->
  valid_u64_real n3 ->
  valid_u64_real n4 ->
  (Rabs (y_model n1 n2) < binary64_no_overflow_margin)
  /\ Rabs (d1_raw n1 n2) < binary64_no_overflow_margin
  /\ Rabs (d2_raw n1 n2 n3) < binary64_no_overflow_margin
  /\ Rabs (d3_plus_raw n1 n2 n3 n4) < binary64_no_overflow_margin.
Proof.
  intros n1 n2 n3 n4 Hn1 Hn2 Hn3 Hn4.
  pose proof (u64_envelope_under_margin) as Hmargin.
  pose proof (y_model_bounds n1 n2 Hn1 (positive_is_valid n2 Hn2)) as Hy.
  pose proof (d1_raw_envelope n1 n2 Hn1 (positive_is_valid n2 Hn2)) as Hd1.
  pose proof (d2_raw_envelope n1 n2 n3 Hn1 Hn2 Hn3) as Hd2.
  pose proof (d3_plus_raw_envelope n1 n2 n3 n4 Hn1 (positive_is_valid n2 Hn2) Hn3 Hn4) as Hd3.
  repeat split; apply Rabs_lt; lra.
Qed.

Theorem binary64_margin_below_overflow_threshold :
  binary64_no_overflow_margin < bpow radix2 1023.
Proof.
  unfold binary64_no_overflow_margin.
  interval.
Qed.
