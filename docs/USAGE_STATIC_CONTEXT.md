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

## 2. Two ways to call it

### A. Cached context (recommended — the fast path)

Use `device_allocator_context`. Construct it **once per thread** (really per tile
leader) and reuse it across the allocation loop; it holds the cached slot so repeated
allocations stay on the one-atomic warm path.

```cpp
// tile is a cooperative_groups tile of size 16 or 32 (static_assert enforces this).
device_allocator_context<gallatin_allocator<SLAB>> ctx(alloc_device_instance, tile);

for (...) {
  auto handle = ctx.allocate(tile);          // fast path: 1 atomic on the cached slot
  void* p     = ctx.address(handle);         // handle -> pointer
  ... use p ...
  ctx.deallocate_coop(handle, tile);         // or deallocate_perlane(handle)
}
// ctx destructor returns any outstanding prefetched reservation (GX_PREFETCH only).
```

Key properties:
- **Tile-leader only.** `allocate` reserves on `tile.thread_rank()==0` and `shfl`-broadcasts
  the address to the tile. Tile-16 warps have two leaders; tile-32 warps have one.
- **Warp coalescing (grouped).** For tile size < 32, `allocate` calls
  `gstatic_fast_grouped`: the active same-tree leaders in a warp reserve a *contiguous
  run with one `atomicAdd(n)`* (Gallatin's native coalescing), cutting the per-warp atomic
  count. Tile-32 (one leader/warp) skips the coalescing machinery and takes the plain
  single-atomic `gstatic_fast` — coalescing would be pure overhead with nothing to merge.
- **Self-healing cache.** On a miss (slot swapped/full) `allocate` falls back to
  `gstatic_slow`, which re-resolves and refreshes the cached `{cidx,cbase,cgen}` so the
  next call is warm again.

### B. Stateless `malloc` / `free`

`alloc->malloc(size)` / `alloc->free(ptr)` work without a context. For a static-managed
tree, `malloc` routes to `malloc_static` (single slice) or `malloc_static_multi`
(multi-slice, see §5) on the same off-block counter, so it is safe to **mix** context and
stateless calls on the same tree. This is slower than the cached context (it re-resolves a
slot every call) but requires no per-thread state.

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

- `deallocate_coop(handle, tile)` — cooperative free (one leader frees, coalesced).
- `deallocate_perlane(handle)` / `deallocate_perlane_finish(sum, tile)` — per-lane free.
- Stateless: `alloc->free(ptr)`.

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
- [ ] Construct one `device_allocator_context` per tile leader; reuse it across the loop.
- [ ] Use tile size 16 or 32 (enforced by `static_assert`). Tile-16 gets warp coalescing.
- [ ] Free with the matching `deallocate_*` or stateless `free`.
