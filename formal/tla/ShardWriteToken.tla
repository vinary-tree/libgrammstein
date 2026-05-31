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
 *   - Generation counter incremented on acquire only when it is below u64::MAX
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    \* @type: Set(Str);
    Workers,        \* Set of worker IDs
    \* @type: Set(Str);
    Shards,         \* Set of shard IDs
    \* @type: Int;
    MaxGeneration   \* Bound for model checking (u64 in real impl)

\* Symmetry set for model checking optimization
WorkerSymmetry == Permutations(Workers)

VARIABLES
    \* @type: Str -> Bool;
    write_locked,       \* shard -> Bool: whether shard is write-locked
    \* @type: Str -> Set(Str);
    write_holder,       \* shard -> worker_id or empty set: who holds the lock
    \* @type: Str -> Int;
    write_generation,   \* shard -> Nat: current generation counter
    \* @type: Str -> Int;
    max_generation_seen,\* shard -> Nat: highest generation observed so far
    \* @type: <<Str, Str>> -> [exists: Bool, gen: Int, valid: Bool];
    tokens,             \* (worker, shard) -> record {exists, gen, valid}
    \* @type: Str -> Str;
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

(* Type invariant, decomposed so TLAPS can prove preservation clause-by-clause. *)
WriteLockedTypeOK == write_locked \in [Shards -> BOOLEAN]
WriteHolderTypeOK == write_holder \in [Shards -> SUBSET Workers]
WriteHolderUnique == \A s \in Shards: \A w1, w2 \in write_holder[s]: w1 = w2
WriteGenerationTypeOK == write_generation \in [Shards -> 0..MaxGeneration]
MaxGenerationSeenTypeOK == max_generation_seen \in [Shards -> 0..MaxGeneration]
TokensTypeOK == tokens \in [Workers \X Shards -> [exists: BOOLEAN, gen: 0..MaxGeneration, valid: BOOLEAN]]
PcTypeOK == pc \in [Workers -> {Idle, WantLock, HaveLock, Releasing}]

TypeOK ==
    /\ WriteLockedTypeOK
    /\ WriteHolderTypeOK
    /\ WriteHolderUnique
    /\ WriteGenerationTypeOK
    /\ MaxGenerationSeenTypeOK
    /\ TokensTypeOK
    /\ PcTypeOK

(* Initial state *)
Init ==
    /\ write_locked = [s \in Shards |-> FALSE]
    /\ write_holder = [s \in Shards |-> {}]
    /\ write_generation = [s \in Shards |-> 0]
    /\ max_generation_seen = [s \in Shards |-> 0]
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
    /\ UNCHANGED <<write_locked, write_holder, write_generation, max_generation_seen, tokens>>

(*
 * Try to acquire write lock on a shard.
 * Models: shard.rs:266-279 (try_acquire_write)
 *
 * Atomically:
 *   1. compare_exchange(false, true) on write_locked
 *   2. If success: store worker_id, increment generation, return token
 *   3. If failure: return None (stay in WantLock to retry or give up)
 *)
TryAcquireSuccess(w, s) ==
    /\ pc[w] = WantLock
    /\ write_locked[s] = FALSE
    /\ write_generation[s] < MaxGeneration
    \* CAS succeeds - acquire the lock
    /\ write_locked' = [write_locked EXCEPT ![s] = TRUE]
    /\ write_holder' = [write_holder EXCEPT ![s] = {w}]
    \* Checked generation increment; token receives the incremented value.
    /\ write_generation' = [write_generation EXCEPT ![s] = @ + 1]
    /\ max_generation_seen' = [max_generation_seen EXCEPT ![s] = @ + 1]
    /\ tokens' = [tokens EXCEPT ![<<w, s>>] = Token(write_generation[s] + 1)]
    /\ pc' = [pc EXCEPT ![w] = HaveLock]

TryAcquireFailure(w, s) ==
    /\ pc[w] = WantLock
    /\ ~(write_locked[s] = FALSE /\ write_generation[s] < MaxGeneration)
    \* CAS fails or generation is exhausted - cannot acquire.
    /\ UNCHANGED <<write_locked, write_holder, write_generation, max_generation_seen, tokens>>
    /\ pc' = [pc EXCEPT ![w] = Idle]  \* Give up (or could stay in WantLock)

TryAcquire(w, s) ==
    \/ TryAcquireSuccess(w, s)
    \/ TryAcquireFailure(w, s)

(*
 * Worker decides to release the lock.
 * Models: transition to calling release_write
 *)
WantRelease(w) ==
    /\ pc[w] = HaveLock
    /\ pc' = [pc EXCEPT ![w] = Releasing]
    /\ UNCHANGED <<write_locked, write_holder, write_generation, max_generation_seen, tokens>>

(*
 * Release write lock.
 * Models: shard.rs:285-294 (release_write)
 *
 * Checks token validity (generation must match), then releases.
 *)
ReleaseValid(w, s) ==
    /\ pc[w] = Releasing
    /\ tokens[<<w, s>>].exists = TRUE
    /\ tokens[<<w, s>>].valid
    /\ tokens[<<w, s>>].gen = write_generation[s]
    \* Valid token - release the lock.
    /\ write_holder' = [write_holder EXCEPT ![s] = {}]
    /\ write_locked' = [write_locked EXCEPT ![s] = FALSE]
    /\ tokens' = [tokens EXCEPT ![<<w, s>>] = NoToken]
    /\ pc' = [pc EXCEPT ![w] = Idle]
    /\ UNCHANGED <<write_generation, max_generation_seen>>

ReleaseInvalid(w, s) ==
    /\ pc[w] = Releasing
    /\ tokens[<<w, s>>].exists = TRUE
    /\ ~(tokens[<<w, s>>].valid /\ tokens[<<w, s>>].gen = write_generation[s])
    \* Invalid token - release fails (token consumed anyway).
    /\ tokens' = [tokens EXCEPT ![<<w, s>>] = NoToken]
    /\ pc' = [pc EXCEPT ![w] = Idle]
    /\ UNCHANGED <<write_locked, write_holder, write_generation, max_generation_seen>>

Release(w, s) ==
    \/ ReleaseValid(w, s)
    \/ ReleaseInvalid(w, s)

(*
 * Simulate token becoming stale (e.g., another acquire happened elsewhere).
 * This models the scenario where generation counter advanced.
 * In reality, this can't happen if AtMostOneWriter holds, but we include
 * it to verify the check is necessary.
 *)
TokenInvalidation(w, s) ==
    /\ tokens[<<w, s>>].exists = TRUE
    /\ tokens[<<w, s>>].gen # write_generation[s]
    /\ tokens' = [tokens EXCEPT ![<<w, s>>] = InvalidToken(tokens[<<w, s>>].gen)]
    /\ UNCHANGED <<write_locked, write_holder, write_generation, max_generation_seen, pc>>

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
vars == <<write_locked, write_holder, write_generation, max_generation_seen, tokens, pc>>

Fairness ==
    /\ \A w \in Workers: WF_vars(RequestLock(w))
    /\ \A w \in Workers, s \in Shards: WF_vars(TryAcquire(w, s))
    /\ \A w \in Workers: WF_vars(WantRelease(w))
    /\ \A w \in Workers, s \in Shards: WF_vars(Release(w, s))

Spec == Init /\ [][Next]_vars

FairSpec == Spec /\ Fairness

(* State constraint for bounded model checking *)
StateConstraint ==
    \A s \in Shards: write_generation[s] <= MaxGeneration

(* ---------------------------------------------------------------------------
 * Safety Invariants
 * --------------------------------------------------------------------------- *)

ValidTokenHolders(s) ==
    {w \in Workers: tokens[<<w, s>>].exists /\ tokens[<<w, s>>].valid}

(*
 * CRITICAL INVARIANT: At most one worker holds a valid write token per shard.
 *
 * This is the core data integrity property. Violation means concurrent
 * writes could corrupt the trie.
 *)
AtMostOneWriter ==
    \A s \in Shards:
        \A w1, w2 \in ValidTokenHolders(s): w1 = w2

(*
 * If a shard is locked, exactly one worker holds it.
 *)
LockedImpliesHolder ==
    \A s \in Shards:
        write_locked[s] = TRUE => \E w \in Workers: write_holder[s] = {w}

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
        write_generation[s] = max_generation_seen[s]

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
    /\ GenerationMonotonic
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

(*
 * A worker that has acquired a lock eventually releases it.
 *)
HeldLockEventuallyReleases ==
    \A w \in Workers:
        pc[w] = HaveLock ~> pc[w] = Idle

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
