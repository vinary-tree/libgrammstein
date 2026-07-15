(**
 * Correctness of the LEB128 varint term-id encoding and the u64-unit n-gram view.
 *
 * libgrammstein keys its n-gram store by concatenating the LEB128 varints of a
 * sequence of vocabulary term-ids (`src/ngram/vocabulary.rs`: [encode_varint] /
 * [decode_varint]). The word-level correction automaton `T_gram` walks this store
 * through [U64NgramView] (`src/ngram/u64_view.rs`), which presents one term-id per
 * traversal step by collapsing each varint run. For that view to be a faithful
 * `Unit = u64` dictionary — enumerating EXACTLY the stored term-id sequences, with
 * no term-id dropped, duplicated, or invented — the varint codec must be a
 * self-delimiting bijection.
 *
 * This module models [encode_varint] / [decode_varint] over [nat] and proves:
 *
 *   1. [decode_encode] — round-trip / self-delimiting: decoding the encoding of
 *      [v] followed by any suffix recovers exactly [v] and the untouched suffix.
 *      (This is both the codec's injectivity and the fact that a concatenation of
 *      varints parses unambiguously, one term-id at a time.)
 *   2. [encode_seq_decode_seq] — VIEW FAITHFULNESS: the concatenated-varint key of
 *      a term-id sequence decodes back to exactly that sequence (soundness AND
 *      completeness of the view's enumeration).
 *   3. [encode_seq_injective] — distinct term-id sequences produce distinct keys.
 *
 * The Rust code masks with `& 0x7F` (the low 7 bits, [_ mod 128]), tests the
 * continuation bit with `& 0x80 == 0` (a byte value `< 128`), and shifts by
 * `<< shift` (multiplication by `2 ^ shift`); each is modelled exactly over [nat].
 * The bijection holds for ALL naturals; the store additionally reserves term-id 0
 * (`FIRST_VALID_INDEX = 1`, so the `\x00` metadata prefix never collides with a
 * key), which is an orthogonal namespace-disjointness invariant, not needed here.
 *)

From Stdlib Require Import List Arith Lia.
Import ListNotations.

(** ** The codec, modelled over [nat] *)

(** Encode [v] as LEB128 varint bytes (each a [nat] in [0,255]). [fuel] bounds the
    recursion; [encode_varint] supplies [S v], which always suffices (proved via
    the [v < fuel] premise below). Low 7 bits first; non-final groups carry the
    continuation flag [+ 128] (the `| 0x80` of the Rust encoder). *)
Fixpoint encode (fuel v : nat) : list nat :=
  match fuel with
  | O => []
  | S f =>
      let byte := v mod 128 in
      let hi := v / 128 in
      if Nat.eqb hi 0 then [byte] else (byte + 128) :: encode f hi
  end.

Definition encode_varint (v : nat) : list nat := encode (S v) v.

(** Decode one varint from a byte list, threading the accumulated value and its
    bit-shift. `byte & 0x7F` is [b mod 128]; `byte & 0x80 == 0` is [b <? 128];
    `<< shift` is [* 2 ^ shift]. Returns the value and the unconsumed suffix. *)
Fixpoint decode_aux (bytes : list nat) (acc shift : nat) : option (nat * list nat) :=
  match bytes with
  | [] => None
  | b :: bs =>
      let acc' := acc + (b mod 128) * 2 ^ shift in
      if Nat.ltb b 128 then Some (acc', bs) else decode_aux bs acc' (shift + 7)
  end.

Definition decode_varint (bytes : list nat) : option (nat * list nat) :=
  decode_aux bytes 0 0.

(** ** Round-trip / self-delimiting bijection *)

(** The generalized round-trip, with an arbitrary accumulator and shift so the
    induction goes through the continuation-byte recursive call. *)
Lemma decode_aux_encode :
  forall fuel v acc shift rest,
    v < fuel ->
    decode_aux (encode fuel v ++ rest) acc shift
    = Some (acc + v * 2 ^ shift, rest).
Proof.
  induction fuel as [| f IH]; intros v acc shift rest Hfuel.
  - lia.
  - cbn [encode].
    destruct (Nat.eqb (v / 128) 0) eqn:Hhi.
    + (* terminator: v / 128 = 0, i.e. v < 128 *)
      assert (Hzero : v / 128 = 0) by (apply Nat.eqb_eq; exact Hhi).
      assert (Hub : v mod 128 < 128) by (apply Nat.mod_upper_bound; lia).
      assert (Hlt : v < 128).
      { assert (Hdm := Nat.div_mod_eq v 128).
        rewrite Hzero in Hdm. lia. }
      assert (Hmod : v mod 128 = v) by (apply Nat.mod_small; exact Hlt).
      cbn [app decode_aux].
      rewrite !Hmod.
      rewrite (proj2 (Nat.ltb_lt v 128) Hlt).
      reflexivity.
    + (* continuation: v / 128 <> 0, i.e. v >= 128 *)
      assert (Hnz : v / 128 <> 0) by (apply Nat.eqb_neq; exact Hhi).
      assert (Hge : 128 <= v).
      { destruct (Nat.lt_ge_cases v 128) as [Hsmall | Hge']; [ exfalso | exact Hge' ].
        apply Hnz. apply Nat.div_small; exact Hsmall. }
      assert (Hbyte_lt : v mod 128 < 128) by (apply Nat.mod_upper_bound; lia).
      cbn [app decode_aux].
      (* the emitted continuation byte carries the flag, so it is >= 128 *)
      assert (Hcont : (v mod 128 + 128) <? 128 = false) by (apply Nat.ltb_ge; lia).
      rewrite Hcont.
      (* low 7 bits of the continuation byte are just [v mod 128] again *)
      assert (Hmod : (v mod 128 + 128) mod 128 = v mod 128).
      { replace (v mod 128 + 128) with (v mod 128 + 1 * 128) by lia.
        rewrite Nat.Div0.mod_add. apply Nat.mod_small; exact Hbyte_lt. }
      rewrite Hmod.
      (* recurse on [v / 128] with [shift + 7]; the fuel bound gives [v / 128 < f] *)
      assert (Hhi_lt : v / 128 < f).
      { assert (v / 128 < v) by (apply Nat.div_lt; lia). lia. }
      rewrite (IH (v / 128) (acc + v mod 128 * 2 ^ shift) (shift + 7) rest Hhi_lt).
      (* reassemble: (v mod 128) + 128 * (v / 128) = v and 2^(shift+7) = 2^shift * 128 *)
      f_equal. f_equal.
      assert (Hpow : 2 ^ (shift + 7) = 2 ^ shift * 128) by (rewrite Nat.pow_add_r; reflexivity).
      rewrite Hpow.
      assert (Hv : v mod 128 + 128 * (v / 128) = v).
      { assert (Hdm := Nat.div_mod_eq v 128). lia. }
      nia.
Qed.

(** Round-trip on a single varint: decoding [encode_varint v ++ rest] recovers
    [v] and leaves [rest] untouched. This is the codec's injectivity together with
    its self-delimiting property (a concatenation of varints parses one at a time). *)
Theorem decode_encode :
  forall v rest, decode_varint (encode_varint v ++ rest) = Some (v, rest).
Proof.
  intros v rest. unfold decode_varint, encode_varint.
  rewrite (decode_aux_encode (S v) v 0 0 rest (Nat.lt_succ_diag_r v)).
  f_equal. f_equal. simpl. lia.
Qed.

(** An encoded varint is never empty (it always emits at least one byte). *)
Lemma encode_varint_nonempty : forall v, encode_varint v <> [].
Proof.
  intros v Heq. unfold encode_varint in Heq. cbn [encode] in Heq.
  destruct (Nat.eqb (v / 128) 0); discriminate Heq.
Qed.

(** ** View faithfulness over a term-id SEQUENCE (the n-gram key) *)

(** The store key of a term-id sequence: concatenate each id's varint. *)
Fixpoint encode_seq (ids : list nat) : list nat :=
  match ids with
  | [] => []
  | id :: rest => encode_varint id ++ encode_seq rest
  end.

(** The view's enumeration: peel off one varint at a time. [fuel] counts the ids. *)
Fixpoint decode_seq (fuel : nat) (bytes : list nat) : list nat :=
  match fuel with
  | O => []
  | S f =>
      match decode_varint bytes with
      | Some (v, rest) => v :: decode_seq f rest
      | None => []
      end
  end.

(** VIEW FAITHFULNESS: the key of [ids] decodes back to exactly [ids] — soundness
    (no id invented) and completeness (no id dropped) of the u64-view enumeration. *)
Theorem encode_seq_decode_seq :
  forall ids, decode_seq (length ids) (encode_seq ids) = ids.
Proof.
  induction ids as [| id rest IH]; simpl.
  - reflexivity.
  - rewrite decode_encode. rewrite IH. reflexivity.
Qed.

(** Distinct term-id sequences produce distinct keys: the view is a bijection onto
    the stored keys, so no two sequences alias. *)
Theorem encode_seq_injective :
  forall a b, encode_seq a = encode_seq b -> a = b.
Proof.
  induction a as [| x a' IH]; intros [| y b'] Heq; cbn [encode_seq] in Heq.
  - reflexivity.
  - exfalso. symmetry in Heq.
    apply app_eq_nil in Heq. destruct Heq as [Hy _].
    exact (encode_varint_nonempty y Hy).
  - exfalso.
    apply app_eq_nil in Heq. destruct Heq as [Hx _].
    exact (encode_varint_nonempty x Hx).
  - (* peel the leading varint off both sides with the round-trip lemma *)
    assert (Hx := decode_encode x (encode_seq a')).
    rewrite Heq in Hx.
    rewrite (decode_encode y (encode_seq b')) in Hx.
    injection Hx as Hxy Hrest.
    rewrite Hxy. f_equal. apply IH. symmetry. exact Hrest.
Qed.
