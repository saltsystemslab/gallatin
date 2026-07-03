# Gallatin static-counter correctness invariants

The static-counter fast path moves the per-block malloc counter *off* the block into a
per-slot padded counter (`g_ctr64`), with a cached base (`g_sbase` / per-thread `cbase`)
and a `Block*` (`g_block`). That decoupling is fast but introduces coupling invariants the
on-block counter used to enforce implicitly. Below are the invariants that must hold for
correct (crash-free, no-double-alloc, no-leak) behavior, organized by layer, with what
enforces each and the failure mode if it breaks.

Notation: `chunk_ids[seg]` = a segment's tree id (`~0` = free/unowned); `active_counts[seg]`
= per-segment block reservation gate; `Block.free_counter`, `Block.malloc_counter`,
`Block.home`; static slot state `g_ctr64[slot]=(gen<<32)|count`, `g_block[slot]`,
`g_sbase[slot]`; per-thread context `(cidx, cbase, cgen)`.

---

## Layer 0 — Segment ownership

**S0. A segment's tree id is constant for its lifetime.**
`chunk_ids[seg]` goes `~0 -> tree` at `setup_segment` (release-CAS) and `tree -> ~0` only at
deregister (`reset_tree_id`). It never changes while the segment is owned.
*Why it matters:* `allocation_to_offset(addr, tree)` derives a slice's `block_id` by dividing
by that tree's slice size. If the tree is constant for the whole life of every slice in the
segment, every free resolves to the correct block. **This is the linchpin** — most Layer 1/2
corruption is downstream of a segment re-typing while a slice/block is still live.

---

## Layer 1 — Segment ↔ block accounting

**S1. A segment is deregistered only when *all* its blocks are free.**
"Free" = no outstanding allocations (`free_counter == 0`) and no static slot references it
(`home == 0`). Enforced by `active_counts` (`finish_freeing_block` deregisters when it returns
to `num_blocks-1`).
*Failure:* premature deregister -> segment re-types under a live block -> S0 broken -> frees of
the old incarnation mis-resolve. (Observed as `FREE-UNOWNED` / `__match_any_sync` collapse.)

**S2. `active_counts` is balanced: one decrement per acquire, one increment per return.**
`get_block` decrements exactly once on success (failed acquires decrement-then-increment, net
0); `return_block`/`finish_freeing_block` increments exactly once per real block return.
*Failure:* a **phantom return** (a `return_block` with no matching acquire) over-increments
`active_counts` -> S1 threshold reached with a live block -> premature deregister.

**S3. A block a segment hands out is free and uniquely owned.**
`get_block` must return a block that is not currently owned by any consumer (`home == 0`,
`free_counter == 0`). A block is in the reuse queue only after a full drain + reset.
*Failure:* handing a still-owned block -> two consumers dispense the same memory (double-alloc).

---

## Layer 2 — Block ↔ slice accounting

**B1. A block dispenses exactly 4096 unique slices per incarnation** (indices `0..4095`);
index `>= 4096` dispenses nothing.

**B2. Each dispensed slice is handed to exactly one consumer** (no double-allocation).

**B3. Each dispensed slice is freed exactly once** (no double-free), and a free resolves to
the block that dispensed it (guaranteed by S0: constant tree ⇒ correct `block_id`).

**B4. A block is returned/recycled exactly once per incarnation, and only when all 4096 of
its slices are freed.** `free_counter` reaches 4096 **iff** every current slice is freed; the
thread that observes 4096 does `reset_free` (→0) then `free_block` → `return_block` → enqueue.
*Failure:* over-free (`free_counter` pushed past 4096 by a double-free from a B2 violation)
makes `free_block` fire while slices are live (`ENQUEUE-LIVE`) and can fire twice
(double-return) → S2 broken.

**B5. `free_counter` counts only the current incarnation.** It is 0 at acquire, climbs to 4096
as the incarnation's slices free, and is reset before re-acquire. A stale/late free from a
prior incarnation must not count toward the current one (which is impossible if S1/B4 hold —
the block only recycles when genuinely drained, so no prior-incarnation free is outstanding).

---

## Layer 3 — Static-counter slot

**C1. `g_ctr64[slot]` dispenses a unique `count` per generation.** `atomicAdd` gives unique
`count` values `0..4095` for a given `gen`; `count >= 4096` returns nothing (the past-full
increments are wiped by the swap's `atomicExch`).

**C2. The slot's base is valid for its generation.** `g_sbase[slot]` (and a thread's cached
`cbase`) always addresses the slot's *current* block for the *current* `gen`. The base changes
only at a swap, which bumps `gen`. Therefore the warm-path check `gen == cgen` is sufficient to
trust `cbase` — **provided C3 holds.**

**C3. A slot dispenses only into memory its tree currently owns.** The slot's block/segment
must be owned by the slot's tree the entire time the slot references it. If the underlying
block is recycled or its segment re-types, the slot **must be invalidated** (its `gen` bumped
so the warm-path `gen == cgen` fails and the thread re-resolves).
*This is the invariant the static counter originally dropped* (the on-block counter re-checked
ownership on every malloc via `check_valid`; the off-block warm path does not). Its violation
is the root cause fixed this session: a slot kept dispensing off a stale `cbase` into memory
another slot/tree now owned → double-alloc → over-free → phantom return → premature deregister →
crash.

**C4. A block lives in at most one slot at a time (`Block.home`).** `home` records the single
slot a block currently lives in (`home == 0` ⇔ free). Set at assign via `atomicCAS(home,0,slot)`
(a CAS *failure* means the block is being assigned while still owned — a violation), wiped at
`free_block`. The `home` back-ref lets the free/deregister paths find and invalidate the owning
slot in O(1).

**C5. A slot swaps single-writer.** Exactly one thread per generation triggers the swap (the
lane landing on `count == 4095`); it atomically retires the old block (ring-push for rollback),
installs a fresh block, publishes base/block, and bumps `gen` — or marks the slot dead
(`count = 4096`, `g_block = null`) on OOM so reservers miss to the slow path instead of
dispensing a stale base.

---

## How the fix maintains C3 (the one that was broken)

C3 requires a slot to be invalidated whenever its block/segment stops being its tree's. That
happens at exactly two cold-path events, and the fix seals the slot at both (atomic gen-CAS:
detach `g_block`, zero `g_sbase`, bump `gen`):

- **`GALLATIN_SEAL_ON_FREE`** — at `free_block` (a block becomes reassignable), seal any slot
  still referencing that block.
- **`GALLATIN_WIPE_ON_RETYPE`** — at segment deregister (`return_block`, before
  `reset_tree_id`), seal every slot pointing into the segment, found O(1) via each block's
  `Block.home`.

Plus `GALLATIN_SWAP_DEADMARK` (don't leave `g_block` on the old block during a parked
`replace_block`) and `GALLATIN_STATIC_FILL_MC` (stamp `malloc_counter` full at adoption so
on-block metadata never lies). All are cold-path (block-free / swap / deregister — never the
allocation hot path): measured zero perf cost, `doubles 148 -> 0`, cross-tree reuse preserved.
All default-on with `GALLATIN_STATIC_COUNTER`; `tree-private` (never deregister) is the blunt
opt-in alternative (`GALLATIN_STATIC_TREE_PRIVATE_ENABLE`) that satisfies S1 by construction but
strands capacity.

---

## Non-invariant: over-subscription is a liveness/OOM concern, not a safety bug

When the live working set exceeds the pool (e.g. 256 MB pool, 2M threads × ~1 KB × 5 sizes ≈ 8×
over capacity), the allocator exhausts and collapses regardless of the fixes — this is a
graceful-OOM matter, not a violation of S0–C5. Validated: at realistic pool sizes with the
working set **under** capacity, all safety invariants hold (even the fully-unpatched baseline is
crash-free); the crash only appears at/over pool capacity.

---

## C6 — the resolve read of {base, gen} must be atomic (live descriptor)

C2 says a slot's base is valid for its generation. The warm hot path enforces this with the
register-resident `cbase`/`cgen` and the monotonic-`gen` check, and never re-loads the base.
But the **cold resolve paths** (`malloc_static`, `gstatic_fast` stale branch, `gstatic_slow`,
grouped stale branch) recover the base after the reserving `atomicAdd`. Originally they read
base (`g_sbase`) and gen (`g_ctr64`) as **two separate words**. Because a swap publishes the
new base (release) *before* bumping the gen (release), an acquire reader could observe
`{new base, OLD gen}`, pass the recheck, and dispense an old-gen count on the fresh block →
temporally-separated double-alloc → over-free → premature return → `FREE-UNOWNED`.

**Fix (`GALLATIN_LIVE_DESCRIPTOR`, default-on):** `g_live64[slot] = valid|gen|block_id`, one
64-bit word. A resolve reads it once (atomic), checks `gen`, and derives the base from the
`block_id` in the *same* word — so base and gen are a single linearizable unit. The swap/seal
writers publish/invalidate `g_live64` alongside the gen bump. Warm path never reads it (zero
hot-path cost). Also hardened `SEAL_ON_FREE` to a gen-conditional CAS (matching
`WIPE_ON_RETYPE`) so it can't stomp a concurrent swap's fresh generation.

## M1 — multi-slice large allocs on a static-managed tree must go through the static counter

A request larger than `biggest` but smaller than a block is served as a contiguous run of
N slices of the top tree (`num_trees-1`) via `block_tree < 0` in `malloc()`. That tree is
static-managed, so its slices are dispensed OFF-block (`g_ctr64`) with the on-block counter
forced "full" (`claim_all_static`). Routing the run to the on-block `malloc_slice_allocation`
therefore dispenses the same slices twice (off-block **and** on-block) → double-alloc
(`churn_diagnostics_test` V3, sizes 16..4111, `back!=marker`).

**Fix (`malloc_static_multi`, always on under the static counter):** one `atomicAdd(N)` on
`g_ctr64` yields a contiguous run `[start, start+N)`; resolve the base via the live descriptor;
handle the edges — gen-raced → ring rollback of the real prefix; run straddles the block end
(`start+N>4096`) → swap + free the real prefix, retry; run fills the block (`start+N==4096`) →
swap. Free accounting: the app frees ONCE but N slices were consumed, so pre-add `(N-1)` to the
backing block's `free_counter` at alloc (mirrors `block_correct_frees`); provably `< 4096`
before the app's own free, so no early recycle.
