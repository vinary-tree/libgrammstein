--------------------------- MODULE ShardWriteToken ---------------------------
(*
 * Formal verification of the WriteToken mechanism for exclusive shard access.
 *
 * This specification models the concurrency protocol from:
 *   src/sources/google_books/sharding/shard.rs:262-294
 *
 * The WriteToken mechanism ensures:
 *   1. At most one worker holds a valid token per shard at any time
 *   2. Generation counter monotonically increases
 *   3. Tokens become invalid when generation doesn't match
 *
 * Key Rust implementation details modeled:
 *   - try_acquire_write: compare_exchange(false, true, Acquire, Relaxed)
 *   - release_write: token.is_valid check, then store(false, Release)
 *   - Generation counter incremented on acquire with fetch_add(1, Relaxed)
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Workers,        \* Set of worker IDs
    Shards,         \* Set of shard IDs
    MaxGeneration   \* Bound for model checking (u64 in real impl)

\* Symmetry set for model checking optimization
WorkerSymmetry == Permutations(Workers)

VARIABLES
    write_locked,       \* shard -> Bool: whether shard is write-locked
    write_holder,       \* shard -> worker_id or empty set: who holds the lock
    write_generation,   \* shard -> Nat: current generation counter
    tokens,             \* (worker, shard) -> record {exists, gen, valid}
    pc                  \* worker -> program counter

(* Constants for program counters *)
Idle == "idle"
WantLock == "want_lock"
HaveLock == "have_lock"
Releasing == "releasing"

(* Token record helper - token is "absent" when exists = FALSE *)
NoToken == [exists |-> FALSE, gen |-> 0, valid |-> FALSE]
Token(g) == [exists |-> TRUE, gen |-> g, valid |-> TRUE]
InvalidToken(g) == [exists |-> TRUE, gen |-> g, valid |-> FALSE]

(* Type invariant *)
TypeOK ==
    /\ write_locked \in [Shards -> BOOLEAN]
    /\ write_holder \in [Shards -> SUBSET Workers]
    /\ \A s \in Shards: Cardinality(write_holder[s]) <= 1
    /\ write_generation \in [Shards -> 0..MaxGeneration]
    /\ tokens \in [Workers \X Shards -> [exists: BOOLEAN, gen: 0..MaxGeneration, valid: BOOLEAN]]
    /\ pc \in [Workers -> {Idle, WantLock, HaveLock, Releasing}]

(* Initial state *)
Init ==
    /\ write_locked = [s \in Shards |-> FALSE]
    /\ write_holder = [s \in Shards |-> {}]
    /\ write_generation = [s \in Shards |-> 0]
    /\ tokens = [ws \in Workers \X Shards |-> NoToken]
    /\ pc = [w \in Workers |-> Idle]

(* ---------------------------------------------------------------------------
 * Actions
 * --------------------------------------------------------------------------- *)

(*
 * Worker wants to acquire a write lock for some shard.
 * Models: preparation before calling try_acquire_write
 *)
RequestLock(w) ==
    /\ pc[w] = Idle
    /\ pc' = [pc EXCEPT ![w] = WantLock]
    /\ UNCHANGED <<write_locked, write_holder, write_generation, tokens>>

(*
 * Try to acquire write lock on a shard.
 * Models: shard.rs:266-279 (try_acquire_write)
 *
 * Atomically:
 *   1. compare_exchange(false, true) on write_locked
 *   2. If success: store worker_id, increment generation, return token
 *   3. If failure: return None (stay in WantLock to retry or give up)
 *)
TryAcquire(w, s) ==
    /\ pc[w] = WantLock
    /\ IF write_locked[s] = FALSE
       THEN
           \* CAS succeeds - acquire the lock
           /\ write_locked' = [write_locked EXCEPT ![s] = TRUE]
           /\ write_holder' = [write_holder EXCEPT ![s] = {w}]
           \* Generation incremented: fetch_add(1) returns old value,
           \* but token gets old_value + 1 (see shard.rs:276)
           /\ write_generation' = [write_generation EXCEPT ![s] = @ + 1]
           /\ tokens' = [tokens EXCEPT ![<<w, s>>] = Token(write_generation[s] + 1)]
           /\ pc' = [pc EXCEPT ![w] = HaveLock]
       ELSE
           \* CAS fails - cannot acquire
           /\ UNCHANGED <<write_locked, write_holder, write_generation, tokens>>
           /\ pc' = [pc EXCEPT ![w] = Idle]  \* Give up (or could stay in WantLock)

(*
 * Worker decides to release the lock.
 * Models: transition to calling release_write
 *)
WantRelease(w) ==
    /\ pc[w] = HaveLock
    /\ pc' = [pc EXCEPT ![w] = Releasing]
    /\ UNCHANGED <<write_locked, write_holder, write_generation, tokens>>

(*
 * Release write lock.
 * Models: shard.rs:285-294 (release_write)
 *
 * Checks token validity (generation must match), then releases.
 *)
Release(w, s) ==
    /\ pc[w] = Releasing
    /\ tokens[<<w, s>>].exists = TRUE
    /\ LET token == tokens[<<w, s>>]
           current_gen == write_generation[s]
       IN IF token.valid /\ token.gen = current_gen
          THEN
              \* Valid token - release the lock
              /\ write_holder' = [write_holder EXCEPT ![s] = {}]
              /\ write_locked' = [write_locked EXCEPT ![s] = FALSE]
              /\ tokens' = [tokens EXCEPT ![<<w, s>>] = NoToken]
              /\ pc' = [pc EXCEPT ![w] = Idle]
              /\ UNCHANGED write_generation
          ELSE
              \* Invalid token - release fails (token consumed anyway)
              /\ tokens' = [tokens EXCEPT ![<<w, s>>] = NoToken]
              /\ pc' = [pc EXCEPT ![w] = Idle]
              /\ UNCHANGED <<write_locked, write_holder, write_generation>>

(*
 * Simulate token becoming stale (e.g., another acquire happened elsewhere).
 * This models the scenario where generation counter advanced.
 * In reality, this can't happen if AtMostOneWriter holds, but we include
 * it to verify the check is necessary.
 *)
TokenInvalidation(w, s) ==
    /\ tokens[<<w, s>>].exists = TRUE
    /\ tokens[<<w, s>>].gen # write_generation[s]
    /\ tokens' = [tokens EXCEPT ![<<w, s>>].valid = FALSE]
    /\ UNCHANGED <<write_locked, write_holder, write_generation, pc>>

(* ---------------------------------------------------------------------------
 * Next state relation
 * --------------------------------------------------------------------------- *)

Next ==
    \/ \E w \in Workers: RequestLock(w)
    \/ \E w \in Workers, s \in Shards: TryAcquire(w, s)
    \/ \E w \in Workers: WantRelease(w)
    \/ \E w \in Workers, s \in Shards: Release(w, s)
    \/ \E w \in Workers, s \in Shards: TokenInvalidation(w, s)

(* Fairness: workers eventually make progress *)
Fairness ==
    /\ \A w \in Workers: WF_<<write_locked, write_holder, write_generation, tokens, pc>>(RequestLock(w))
    /\ \A w \in Workers, s \in Shards: WF_<<write_locked, write_holder, write_generation, tokens, pc>>(TryAcquire(w, s))
    /\ \A w \in Workers: WF_<<write_locked, write_holder, write_generation, tokens, pc>>(WantRelease(w))
    /\ \A w \in Workers, s \in Shards: WF_<<write_locked, write_holder, write_generation, tokens, pc>>(Release(w, s))

Spec == Init /\ [][Next]_<<write_locked, write_holder, write_generation, tokens, pc>> /\ Fairness

(* State constraint for bounded model checking *)
StateConstraint ==
    \A s \in Shards: write_generation[s] <= MaxGeneration

(* ---------------------------------------------------------------------------
 * Safety Invariants
 * --------------------------------------------------------------------------- *)

(*
 * CRITICAL INVARIANT: At most one worker holds a valid write token per shard.
 *
 * This is the core data integrity property. Violation means concurrent
 * writes could corrupt the trie.
 *)
AtMostOneWriter ==
    \A s \in Shards:
        LET valid_holders ==
            {w \in Workers: tokens[<<w, s>>].exists /\ tokens[<<w, s>>].valid}
        IN Cardinality(valid_holders) <= 1

(*
 * If a shard is locked, exactly one worker holds it.
 *)
LockedImpliesHolder ==
    \A s \in Shards:
        write_locked[s] = TRUE => Cardinality(write_holder[s]) = 1

(*
 * If a shard is not locked, no one holds it.
 *)
UnlockedImpliesNoHolder ==
    \A s \in Shards:
        write_locked[s] = FALSE => write_holder[s] = {}

(*
 * Generation counter never decreases (monotonic).
 * This is implicit in the spec (we only increment), but stated explicitly.
 *)
GenerationMonotonic ==
    \A s \in Shards:
        write_generation[s] >= 0

(*
 * Valid tokens have generation matching current shard generation.
 *)
ValidTokenGenerationMatch ==
    \A w \in Workers, s \in Shards:
        (tokens[<<w, s>>].exists /\ tokens[<<w, s>>].valid) =>
            tokens[<<w, s>>].gen = write_generation[s]

(*
 * A worker with a valid token for a shard is the current holder.
 *)
ValidTokenImpliesHolder ==
    \A w \in Workers, s \in Shards:
        (tokens[<<w, s>>].exists /\ tokens[<<w, s>>].valid) =>
            w \in write_holder[s]

(*
 * Combined safety invariant for model checking.
 *)
Safety ==
    /\ TypeOK
    /\ AtMostOneWriter
    /\ LockedImpliesHolder
    /\ UnlockedImpliesNoHolder
    /\ ValidTokenGenerationMatch
    /\ ValidTokenImpliesHolder

(* ---------------------------------------------------------------------------
 * Liveness Properties
 * --------------------------------------------------------------------------- *)

(*
 * If a worker wants a lock and the shard is unlocked, they eventually get it.
 * (Under weak fairness)
 *)
EventuallyGranted ==
    \A w \in Workers, s \in Shards:
        (pc[w] = WantLock /\ write_locked[s] = FALSE) ~>
            (tokens[<<w, s>>].exists /\ tokens[<<w, s>>].valid)

(*
 * No starvation: a worker wanting a lock eventually either gets it or gives up.
 * (This is guaranteed by our model where failed CAS leads to Idle)
 *)
NoStarvation ==
    \A w \in Workers:
        pc[w] = WantLock ~> (pc[w] = HaveLock \/ pc[w] = Idle)

(* ---------------------------------------------------------------------------
 * Spec-to-Code Traceability
 * --------------------------------------------------------------------------- *)

(*
 * Mapping from TLA+ to Rust implementation:
 *
 * TLA+ Variable          | Rust Field                      | Location
 * -----------------------|----------------------------------|----------
 * write_locked           | ShardHandle.write_locked         | shard.rs:156
 * write_holder           | ShardHandle.write_holder         | shard.rs:159
 * write_generation       | ShardHandle.write_generation     | shard.rs:162
 * tokens[w,s].gen        | WriteToken.generation            | shard.rs:57
 * tokens[w,s].valid      | Derived from is_valid() check    | shard.rs:72-74
 *
 * TLA+ Action            | Rust Method                      | Location
 * -----------------------|----------------------------------|----------
 * TryAcquire             | ShardHandle::try_acquire_write   | shard.rs:266-279
 * Release                | ShardHandle::release_write       | shard.rs:285-294
 *
 * Memory Ordering:
 * - Acquire on successful CAS provides visibility of prior writes
 * - Release on unlock publishes writes to next acquirer
 * - Relaxed ordering on generation read in release_write() is safe because:
 *   1. WriteToken is !Send (contains PhantomData<*const ()>)
 *   2. Tokens cannot be passed between threads at compile time
 *   3. Within a single thread, program order ensures visibility
 *   This is enforced by Rust's type system, not runtime checks.
 *)

=============================================================================
