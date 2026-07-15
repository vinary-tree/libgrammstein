(**
 * The minimum-cost decode is the MAP correction (noisy channel, log semiring).
 *
 * The `GrammarCorrector` (`src/integration/grammar_corrector.rs`) decodes by
 * MINIMIZING an additive cost in the negative-log ("tropical" / log-semiring)
 * domain:
 *
 *     cost(w) = c_channel(w) + lm_weight * (- ln P(w)),
 *
 * where `c_channel(w) = c_lex(x|w) + c_gram(x|w)` is the edit (channel) cost and
 * `P(w)` is the source score. This module proves that this minimization is exactly
 * the noisy-channel MAXIMUM A POSTERIORI decision: the hypothesis of least cost is
 * the hypothesis of greatest multiplicative score
 *
 *     score(w) = exp(- c_channel(w)) * P(w) ^ lm_weight,
 *
 * because `cost(w) = - ln(score(w))` and `- ln` is strictly antitone on positives.
 * Specializing to `c_channel = - ln P(x|w)` and `lm_weight = 1` gives
 * `score(w) = P(x|w) * P(w)`, so — since the evidence `P(x)` is constant in `w` —
 * the least-cost `w` maximizes `P(x|w) P(w) ∝ P(w|x)`, the Bayes-optimal correction
 * (Kernighan, Church & Gale 1990). No axioms or admitted goals are used.
 *)

From Stdlib Require Import Reals Lra.

Open Scope R_scope.

(** ** [- ln] is strictly antitone on the positives *)

(** From `- ln a <= - ln b` (i.e. a has the smaller cost) recover `b <= a`
    (i.e. a has the larger score). This is the reflection that turns a cost
    inequality back into a score inequality. *)
Lemma neg_ln_reflect : forall a b : R,
  0 < a -> 0 < b -> - ln a <= - ln b -> b <= a.
Proof.
  intros a b Ha Hb H.
  apply Ropp_le_cancel in H.               (* ln b <= ln a *)
  destruct (Rle_lt_dec b a) as [Hle | Hgt]; [ exact Hle | exfalso ].
  assert (ln a < ln b) by (apply ln_increasing; lra).
  lra.
Qed.

(** ** Cost and score, and their logarithmic duality *)

(** The additive negative-log-domain cost of a correction. *)
Definition decode_cost (c_channel lm_weight logP : R) : R :=
  c_channel + lm_weight * (- logP).

(** The equivalent multiplicative score whose negative log IS the cost. *)
Definition decode_score (c_channel lm_weight P : R) : R :=
  exp (- c_channel) * Rpower P lm_weight.

Lemma decode_score_pos : forall c_channel lm_weight P,
  0 < P -> 0 < decode_score c_channel lm_weight P.
Proof.
  intros c lw P HP. unfold decode_score.
  apply Rmult_lt_0_compat.
  - apply exp_pos.
  - unfold Rpower. apply exp_pos.
Qed.

(** The bridge: the additive cost equals the negative log of the multiplicative
    score, so minimizing one maximizes the other. *)
Lemma cost_is_neg_log_score : forall c_channel lm_weight P,
  0 < P ->
  decode_cost c_channel lm_weight (ln P)
  = - ln (decode_score c_channel lm_weight P).
Proof.
  intros c lw P HP. unfold decode_cost, decode_score, Rpower.
  rewrite ln_mult; [| apply exp_pos | apply exp_pos ].
  rewrite !ln_exp.
  lra.
Qed.

(** ** The MAP theorem *)

(** The minimum-cost hypothesis has the maximum score: if candidate 1 costs no
    more than candidate 2, then candidate 1 scores at least as high. Over a finite
    candidate set this makes the argmin-cost element an argmax-score element. *)
Theorem min_cost_maximizes_score : forall c1 c2 lw P1 P2 : R,
  0 < P1 -> 0 < P2 ->
  decode_cost c1 lw (ln P1) <= decode_cost c2 lw (ln P2) ->
  decode_score c2 lw P2 <= decode_score c1 lw P1.
Proof.
  intros c1 c2 lw P1 P2 HP1 HP2 Hcost.
  rewrite (cost_is_neg_log_score c1 lw P1 HP1) in Hcost.
  rewrite (cost_is_neg_log_score c2 lw P2 HP2) in Hcost.
  apply neg_ln_reflect.
  - apply decode_score_pos; exact HP1.
  - apply decode_score_pos; exact HP2.
  - exact Hcost.
Qed.

(** With the channel cost set to the channel likelihood `- ln P(x|w)` and unit
    language-model weight, the score is exactly the joint `P(x|w) * P(w)`. *)
Lemma map_score_identity : forall px_w pw : R,
  0 < px_w -> 0 < pw ->
  decode_score (- ln px_w) 1 pw = px_w * pw.
Proof.
  intros px_w pw Hpx Hpw. unfold decode_score.
  rewrite Ropp_involutive.
  rewrite exp_ln by exact Hpx.
  rewrite Rpower_1 by exact Hpw.
  reflexivity.
Qed.

(** The noisy-channel MAP decision: the least-cost correction maximizes the joint
    likelihood `P(x|w) P(w)`, which — the evidence `P(x)` being constant in `w` — is
    proportional to the posterior `P(w|x)`. Hence the cascade's min-cost decode is
    the Bayes-optimal (MAP) correction. *)
Theorem min_cost_is_map : forall px1 pw1 px2 pw2 : R,
  0 < px1 -> 0 < pw1 -> 0 < px2 -> 0 < pw2 ->
  decode_cost (- ln px1) 1 (ln pw1) <= decode_cost (- ln px2) 1 (ln pw2) ->
  px2 * pw2 <= px1 * pw1.
Proof.
  intros px1 pw1 px2 pw2 Hpx1 Hpw1 Hpx2 Hpw2 Hcost.
  rewrite <- (map_score_identity px1 pw1 Hpx1 Hpw1).
  rewrite <- (map_score_identity px2 pw2 Hpx2 Hpw2).
  apply (min_cost_maximizes_score (- ln px1) (- ln px2) 1 pw1 pw2); assumption.
Qed.
