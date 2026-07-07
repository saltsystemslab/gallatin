# Using the static-counter fast path (cached context)

This is the how-to for Gallatin's fastest allocation path — the **off-block static
counter** driven through a **per-thread cached context**. It is the path IndexinGPU uses
for fixed-size slab allocation and is the recommended path for any workload that
repeatedly allocates same-sized objects in a persistent kernel.

For the internal correctness rules see [`STATIC_COUNTER_INVARIANTS.md`](STATIC_COUNTER_INVARIANTS.md);
for every build flag see [`CONFIGURATION.md`](CONFIGURATION.md).

---

## 1. The model in one paragraph

Each managed size class (a *tree*) pins a set of blocks. Every block holds **4096
slices**. Instead of the reservation counter living on the block (a dependent load
before the atomic), the static counter keeps a **padded off-block counter** per
`(tree, slot)` — `g_ctr64[slot] = (gen<<32) | count` — plus a **live descriptor**
`g_live64[slot] = valid|gen|block_id`. A thread that has cached its slot does the whole
reservation with **one `atomicAdd` and a register compare**, no dependent load. When a
slot's block fills (count reaches 4096) the lane that crosses the boundary **swaps** in a
fresh block and bumps `gen`; a thread whose cached `gen` no longer matches simply
re-resolves. That generation check is the single hot-path safety gate.

## 2. The two allocation paths

Both are **warp-coalesced** off-block static-counter paths — the warp's active same-tree
lanes reserve a contiguous run with **one `atomicAdd(n)`** (Gallatin's native coalescing),
not one atomic per thread. They differ only in whether a **cached context** is carried
across calls. It is safe to **mix** the two on the same tree (identical off-block counter
and free accounting).

### A. Static allocation with coalescing — stateless

No per-thread state. `alloc->malloc(size)` / `alloc->free(ptr)`. For a static-managed tree,
`malloc` routes to `malloc_static` (single slice) or `malloc_static_multi` (multi-slice, see
§5). `malloc_static` coalesces the warp (`coalesced_threads()` + `labeled_partition(tree)`;
leader reserves `team.size()`), so it is *not* a per-thread atomic — but it re-resolves the
slot (`slot_hash` + `get_tree_alloc_size` + partition setup) on **every** call.

```cpp
void* p = alloc->malloc(size);   // coalesced; resolves a slot each call
... use p ...
alloc->free(p);
```

### B. Static allocation with coalescing **+ context** (recommended — fastest)

Carry an `allocator_context` (constructed once per thread / per tile leader, reused across
the loop). Its per-thread `malloc()` uses the **same coalescing** as path A
(`gstatic_fast_grouped`) *plus* the cached slot, so it skips the per-call slot re-resolve.

```cpp
allocator_context<my_gallatin> ctx(alloc, size);   // size -> tree resolved once
for (...) {
  void* p = ctx.malloc();        // coalesced (1 atomic/warp) on the CACHED slot
  ... use p ...
  ctx.free(p);
}
void* q = ctx.malloc(tile);      // coalesced at cg-tile granularity: one shared slice/tile
```

Key properties:
- **Coalesced.** `ctx.malloc()` calls `gstatic_fast_grouped`: active same-tree lanes reserve
  one run with one `atomicAdd(n)`. The `n==1` bypass drops lone/divergent callers to the
  plain single-atomic `gstatic_fast` — no shfl/sync overhead when there's nothing to merge.
- **Warm cached slot.** It reuses `{cidx,cbase,cgen}` (see §3) instead of re-resolving, which
  is the edge over path A.
- **Tile granularity.** `ctx.malloc(tile)` (cg tile of size 16/32) gives the tile leader one
  shared slice broadcast to the tile — a pattern the per-thread API can't express.
- **Self-healing.** On a miss (slot swapped/full) it falls back to `gstatic_slow`, which
  re-resolves and refreshes the cache so the next call is warm again.

### Measured (H200, 1M threads × 64 malloc+free roundtrips, `Gallatin<16MB,16,128>`, miss=0)

| size | A stateless coalesced | B context coalesced | B vs A |
|------|-----------------------|---------------------|--------|
| 16B  | 5972 Mops | **6455 Mops** | +8% |
| 64B  | 6047 Mops | **6781 Mops** | +12% |
| 128B | 6003 Mops | **6660 Mops** | +11% |

Both coalesce, so both are ~6 GMops; the context adds the warm cached slot for +8–12%. (Use
harness `tests/src/ctx_perf_test.cu` to re-measure.) IndexinGPU uses a compile-time-size
variant of path B, `device_allocator_context`, with `allocate(tile)`/`address`/
`deallocate_*` — same machinery, constexpr `tree_id`/slice size for leaner registers.

## 3. The cached context parameters

The context caches three values per tile leader (see `device_allocator_context`):

| field   | meaning | why it's cached |
|---------|---------|-----------------|
| `cidx_` | the `(tree,slot)` index currently reserved from (`-1` = none) | avoids re-probing a slot each call |
| `cbase_`| slice-0 **address** of that slot's current block | lets the warm path compute the result with no load: `cbase_ + count*size` |
| `cgen_` | the slot **generation** `cbase_` belongs to | the hot-path safety gate: the reserving `atomicAdd` returns the live `gen`; if it equals `cgen_`, nothing has moved the slot since we cached it, so `cbase_` is still valid |

Invariant that makes this safe: **`gen` is monotonic per slot** (every swap/seal only ever
does `gen+1`, 32-bit, no wrap). So `gen == cgen_` can only hold when no swap, no block-free,
and no segment re-type has happened since the thread cached the slot — i.e. `cbase_` still
addresses a block this tree owns. The warm path therefore never re-loads the base or the
block pointer. (Cold re-resolves read the base from `g_live64` as one atomic word so they
can't tear a new base against an old gen — see the invariants doc.)

Lifecycle notes:
- Treat the context as **thread-local scratch**: don't share one context across threads,
  and don't persist it past the kernel.
- `tree_id_` and the slice size are **compile-time constants** derived from `SLAB` (asserted
  in the constructor), so they cost no register and no construct-time load.
- The heap base is **not** held in a register; with `GALLATIN_CONST_BASE` it lives in
  `__constant__` memory and `address()`/`pointer_to_handle()` read it per-op (cheap, no
  loop-carried register), which frees registers for occupancy.

## 4. Freeing

- Path A (stateless) / Path B (`allocator_context`): `alloc->free(ptr)` / `ctx.free(ptr)` —
  both land in `free_offset`. For a tile allocation from `ctx.malloc(tile)`, free the shared
  slice **exactly once** (the pointer is the same on every lane).
- IndexinGPU `device_allocator_context`: `deallocate_coop(handle, tile)` (cooperative, one
  leader frees, coalesced) / `deallocate_perlane(handle)` + `deallocate_perlane_finish(sum, tile)`.

All of these land in `free_offset`, which credits the slice's block (`block_free`); when a
block's `free_counter` reaches 4096 it is reset and recycled. Frees are coalesced per block
(`GALLATIN_TEAM_FREE`).

## 5. Large / multi-slice allocations

A request larger than the largest slice (`biggest`) but smaller than a block is served as a
**contiguous run of N slices** of the top tree. On a static-managed tree this goes through
`malloc_static_multi` (one `atomicAdd(N)` for a contiguous run) — **not** the on-block
cooperative path — because the static counter owns those slices off-block; routing it
elsewhere would double-dispense. You don't call this directly; `malloc(size)` routes to it.
It is transparent to `free` (the run is freed with a single `free(ptr)`; the allocator
pre-accounts the extra slices).

## 6. Quick checklist

- [ ] Build with the static counter enabled (default; see CONFIGURATION.md). IndexinGPU:
      `-DGX_STATIC_COUNTER=ON` (default).
- [ ] Call the allocator's boot/publish step so the per-slot cache is filled before any
      device allocation (the block-cache fill kernel). Without it, `malloc` safely falls
      through to the cooperative path.
- [ ] Pick a path: **A** stateless `alloc->malloc(size)` (coalesced, no state) or **B**
      `allocator_context ctx(alloc,size); ctx.malloc()` (coalesced + cached slot, ~+10%).
      Both are safe to mix on the same tree.
- [ ] For path B, construct one context per thread / tile leader and reuse it across the loop.
- [ ] For a shared per-tile allocation use `ctx.malloc(tile)` (cg tile size 16 or 32); free the
      returned pointer exactly once.
- [ ] Free with `alloc->free` / `ctx.free` (or IndexinGPU's `deallocate_*`).
