# Gallatin configuration reference

Build-time `-D` flags and tuning knobs. The **canonical / shipping** configuration is the
static-counter fast path with all correctness fixes on; everything below is grouped by role.

See [`USAGE_STATIC_CONTEXT.md`](USAGE_STATIC_CONTEXT.md) for how to use the path and
[`STATIC_COUNTER_INVARIANTS.md`](STATIC_COUNTER_INVARIANTS.md) for why the fixes are needed.

---

## Canonical path

| flag | default | meaning |
|------|---------|---------|
| `GALLATIN_STATIC_COUNTER` | on in the shipping build (via `-D`; IndexinGPU: `GX_STATIC_COUNTER=ON`) | the off-block one-atomic reservation counter + cached context. The fast path. |
| `GALLATIN_BLOCK_CACHE` | **on** (opt out: `GALLATIN_NO_BLOCK_CACHE`) | block-pointer cache; the fallback path when the static counter is off. Also hosts the per-slot arrays the static counter reuses. |

When `GALLATIN_STATIC_COUNTER` is defined it takes precedence over the block cache for
managed trees.

## Correctness fixes (default-ON — do not disable in production)

Each is auto-enabled with the static counter; the `NO_` form is for A/B regression testing
only. Disabling any of these reintroduces a validated double-alloc / crash class.

| fix | opt-out | closes |
|-----|---------|--------|
| `GALLATIN_LIVE_DESCRIPTOR` | `GALLATIN_NO_LIVE_DESCRIPTOR` | the resolve-path `{new base, old gen}` tear (reads gen+block-id as one atomic word). |
| `GALLATIN_SEAL_ON_FREE` | `GALLATIN_NO_SEAL_ON_FREE` | a freed block still referenced by a slot (seals the slot, gen-conditional CAS). |
| `GALLATIN_WIPE_ON_RETYPE` | `GALLATIN_NO_WIPE_ON_RETYPE` | a slot pointing into a segment that is re-typing (seals via the `home` back-ref). |
| `GALLATIN_SWAP_DEADMARK` | `GALLATIN_NO_SWAP_DEADMARK` | a parked swap leaving the old block dispensable. |
| `GALLATIN_STATIC_FILL_MC` | `GALLATIN_NO_STATIC_FILL_MC` | on-block metadata lying while the static counter owns the block (`claim_all_static`). |
| `GALLATIN_BLOCK_HOME` | `GALLATIN_NO_BLOCK_HOME` | the O(1) slot back-ref used by seal/wipe. |
| multi-slice via static | *(none — always on under static)* | multi-slice large allocs double-dispensing vs the on-block path (`malloc_static_multi`). |

## Opt-in alternatives

| flag | default | meaning |
|------|---------|---------|
| `GALLATIN_STATIC_TREE_PRIVATE_ENABLE` | off | blunt safety mode: static-managed segments never deregister. Satisfies the segment-ownership invariant by construction but strands capacity (~20% miss). The seal fixes make this unnecessary; use only if you need the guarantee by construction. |
| `GALLATIN_ATOMIC_RING` | off | pack the rollback ring entry `(block, gen)` into one atomic word (no torn ring read). Recommended when heavily exercising the resolve/rollback path. |

## Tuning knobs

| knob | default | meaning |
|------|---------|---------|
| `GALLATIN_PINNED_WAVEFRONT` | 256 | pinned blocks per size class (slot count). More slots = less atomic contention, more pinned memory. |
| `GALLATIN_CACHE_MAX_N` | 4096 | max slots per tree in the cache arrays; must cover per-tree `num_blocks`. |
| `GALLATIN_MAX_ATTEMPTS` | 500 | slot-probe attempts before reporting exhaustion. |
| `GALLATIN_MALLOC_LOOP_ATTEMPTS` | 10 | slice-alloc retry multiplier. |
| `GALLATIN_MALLOC_BLOCK_ATTEMPTS` / `_SEGMENT_ATTEMPTS` | 500 | block / segment alloc retries. |
| `GALLATIN_TEAM_FREE` | 1 | coalesce frees per block. |
| `GALLATIN_TRAP_ON_ERR` | 1 | `trap` on an unrecoverable error (e.g. free into a deregistered segment). Set 0 to print-and-continue for diagnosis. |
| `GALLATIN_BLOCK_CHECK` | 0 | extra tree-id re-check in the cooperative slice path. |

## IndexinGPU (GX) CMake mapping

`IndexinGPU_gx/CMakeLists.txt` exposes the knobs as CMake options → `-D` flags:

| CMake option | default | maps to |
|--------------|---------|---------|
| `GX_STATIC_COUNTER` | ON | `-DGALLATIN_STATIC_COUNTER` |
| `GX_CONST_BASE` | ON | `-DGALLATIN_CONST_BASE` (heap base in `__constant__`) |
| `GX_GROUPED` | ON | `-DGALLATIN_GROUPED` (warp-coalesced fast path) |
| `GX_PREFETCH` | OFF | `-DGALLATIN_PREFETCH` (pipelined reservation) |
| `GX_RDC` | ON | separable compilation for the lean hot path |
| `GX_MAXRREG` | 32 | `-maxrregcount` |

## Diagnostics (debug-only)

- `GALLATIN_BLOCK_DEBUG` — per-slice free/alloc stamping; the `run_cachesafe` census and
  `churn_diagnostics_test` rely on it. **Heavy** (device `printf`); never ship with it on.
- Reusable safety harnesses: `tests/src/churn_diagnostics_test.cu` (V1–V4 size/marker
  variants, catches double-alloc via `back!=marker`) and `run_cachesafe.sh` (80-iter
  FREE-UNOWNED census, static vs cache). Use these to re-validate any change to the path.

> The inline observe-only probes (`GALLATIN_DETECT_*`, `GALLATIN_INVARIANTS`,
> `GALLATIN_INV_HOT`, `GALLATIN_HOME_VERIFY`, `GALLATIN_REPORT_UNOWNED`) and the rejected /
> superseded experiments (`SWAP_NOREUSE`, `SEAL_PRIOR`, `STATIC_VALIDATE*`, `SWAP_HOLD`,
> `FENCE_CTR`, `BLOCK_CACHE_REF`, `DEREG_GUARD`, `STOCK_NO_VALIDATE`) have been **removed**
> from the canonical source (they were all default-off; recover from git history if a
> future bug hunt needs them). Re-validate via the two harnesses above, not inline probes.

## Misc knobs (kept)

| flag | default | meaning |
|------|---------|---------|
| `GALLATIN_NO_N1_BYPASS` | off (bypass on) | disable the tile-leader n==1 fast bypass in the grouped path (A/B). |
| `GALLATIN_NO_OUTLINE_COLD` | off (outline on) | inline the cold paths instead of `__noinline__` (register/perf A/B). |
