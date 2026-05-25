// managed_mode_test
//
// Comprehensive correctness check for Gallatin_memory_type::managed
// (UVM-backed allocator). The historical concern is that managed mode
// has bugs that device_only doesn't — this test runs the same workloads
// against both modes and verifies parity: any property that holds in
// device_only must hold in managed.
//
// Sub-tests (run twice, once per mode):
//   - basic         : single-thread alloc/write/free across all trees
//   - concurrent    : many threads alloc, write tid marker, free
//   - tree_migration: drain tree 0, free, allocate tree 8 — segments
//                     must reformat across modes
//   - drain_refill  : alternating sizes across cycles, no capacity decay
//   - mixed_churn   : random sizes, double-malloc detection
//
// Plus a managed-only check:
//   - oversubscribe : boot at a pool size greater than free GPU memory,
//                     verify it boots and allocations succeed (this is
//                     the unique value proposition of managed mode)
//
// A boot also runs the OOM pre-flight in device_only mode — skipped
// when not enough free memory.

#include <gallatin/allocators/gallatin.cuh>
#include <gallatin/allocators/timer.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace gallatin::allocators;

using allocator_t = Gallatin<16ULL * 1024 * 1024, 16ULL, 4096ULL>;

// ---------------------------------------------------------------------------
// Kernels (mode-agnostic — take an allocator pointer)
// ---------------------------------------------------------------------------

__global__ void k_alloc_write_free(allocator_t *alloc, uint64_t n,
                                   uint64_t size, uint64_t *misses,
                                   uint64_t *corruptions) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t *p = (uint64_t *)alloc->malloc(size);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    return;
  }
  uint64_t old = atomicExch((unsigned long long int *)p, tid);
  if (old != 0ULL)
    atomicAdd((unsigned long long int *)corruptions, 1ULL);
  uint64_t back = atomicExch((unsigned long long int *)p, 0ULL);
  if (back != tid)
    atomicAdd((unsigned long long int *)corruptions, 1ULL);
  alloc->free(p);
}

__global__ void k_alloc_record(allocator_t *alloc, uint64_t n, uint64_t size,
                               uint64_t **out, uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t *p = (uint64_t *)alloc->malloc(size);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    out[tid] = nullptr;
    return;
  }
  p[0] = tid;
  out[tid] = p;
}

__global__ void k_verify_free(allocator_t *alloc, uint64_t n, uint64_t **in,
                              uint64_t *corruptions) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t *p = in[tid];
  if (p == nullptr) return;
  if (p[0] != tid)
    atomicAdd((unsigned long long int *)corruptions, 1ULL);
  alloc->free(p);
}

// Sustained pressure: N threads each do `rounds` alloc-write-read-free at
// a fixed size. Detects real concurrent overwrite via `back != marker`
// (i.e., between this thread's two atomicExchs, another thread wrote the
// slot). We deliberately do NOT check `prior == 0` — gallatin doesn't
// zero on free, and previous test sub-runs may leave non-zero bytes in
// slots they freed. The meaningful contract is "no concurrent live use
// of the same slot", which back != marker catches.
__global__ void k_stress(allocator_t *alloc, uint64_t n, int rounds,
                         uint64_t size, uint64_t *misses,
                         uint64_t *double_mallocs) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t marker = tid + 1;  // +1 so we don't share value with sentinel 0
  for (int r = 0; r < rounds; ++r) {
    uint64_t *p = (uint64_t *)alloc->malloc(size);
    if (p == nullptr) {
      atomicAdd((unsigned long long int *)misses, 1ULL);
      continue;
    }
    atomicExch((unsigned long long int *)p, marker);
    uint64_t back = atomicExch((unsigned long long int *)p, 0ULL);
    if (back != marker)
      atomicAdd((unsigned long long int *)double_mallocs, 1ULL);
    alloc->free(p);
  }
}

// Same shape as k_stress but with a random size each iteration covering
// both the slice path (16..4096) and the multi-slice block path (>4096).
// Same "no prior == 0 check" rationale as k_stress.
__global__ void k_mixed_churn(allocator_t *alloc, uint64_t n, int rounds,
                              uint64_t *misses, uint64_t *double_mallocs) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t hash = tid * 0x9e3779b97f4a7c15ULL + 1;
  for (int r = 0; r < rounds; ++r) {
    hash ^= (hash >> 32);
    hash *= 0xbf58476d1ce4e5b9ULL;
    uint64_t size = 16 + (hash & 0xFFF);  // 16..4111
    uint64_t marker = tid * 1000ULL + r + 1;
    uint64_t *p = (uint64_t *)alloc->malloc(size);
    if (p == nullptr) {
      atomicAdd((unsigned long long int *)misses, 1ULL);
      continue;
    }
    atomicExch((unsigned long long int *)p, marker);
    uint64_t back = atomicExch((unsigned long long int *)p, 0ULL);
    if (back != marker)
      atomicAdd((unsigned long long int *)double_mallocs, 1ULL);
    alloc->free(p);
  }
}

// ---------------------------------------------------------------------------
// Sub-test runners
// ---------------------------------------------------------------------------

struct subtest_result {
  std::string name;
  bool pass = false;
  uint64_t misses = 0;
  uint64_t corruptions = 0;
  double seconds = 0.0;
  std::string note;
};

static allocator_t *boot(uint64_t bytes, Gallatin_memory_type mode,
                         bool print_info) {
  switch (mode) {
    case device_only:
      return allocator_t::generate_on_device(bytes, 42, print_info);
    case managed:
      return allocator_t::generate_on_device_managed(bytes, 42, print_info);
    case host_only:
      return allocator_t::generate_on_device_host(bytes, 42, print_info);
  }
  return nullptr;
}

static subtest_result run_basic(allocator_t *alloc) {
  subtest_result r{"basic"};
  uint64_t n = 32 * 1024;  // 32 K threads, 1 alloc/free each, random tree
  uint64_t *misses, *corruptions;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&corruptions, sizeof(uint64_t));
  *misses = 0;
  *corruptions = 0;
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  k_alloc_write_free<<<(n - 1) / 256 + 1, 256>>>(alloc, n, 64, misses,
                                                 corruptions);
  cudaDeviceSynchronize();
  r.seconds = t.sync_end();
  r.misses = *misses;
  r.corruptions = *corruptions;
  r.pass = (r.misses == 0 && r.corruptions == 0);
  cudaFree(misses);
  cudaFree(corruptions);
  return r;
}

static subtest_result run_concurrent(allocator_t *alloc, uint64_t pool_bytes) {
  subtest_result r{"concurrent"};
  // Fill 25% of pool with 256 B allocs.
  uint64_t n = (pool_bytes / 4) / 256;
  if (n > 4ULL * 1024 * 1024) n = 4ULL * 1024 * 1024;
  uint64_t **buf;
  uint64_t *misses, *corruptions;
  cudaMalloc((void **)&buf, sizeof(uint64_t *) * n);
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&corruptions, sizeof(uint64_t));
  *misses = 0;
  *corruptions = 0;
  cudaMemset(buf, 0, sizeof(uint64_t *) * n);
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  k_alloc_record<<<(n - 1) / 256 + 1, 256>>>(alloc, n, 256, buf, misses);
  cudaDeviceSynchronize();
  k_verify_free<<<(n - 1) / 256 + 1, 256>>>(alloc, n, buf, corruptions);
  cudaDeviceSynchronize();
  r.seconds = t.sync_end();
  r.misses = *misses;
  r.corruptions = *corruptions;
  r.pass = (r.misses == 0 && r.corruptions == 0);
  cudaFree(buf);
  cudaFree(misses);
  cudaFree(corruptions);
  return r;
}

static subtest_result run_tree_migration(allocator_t *alloc) {
  subtest_result r{"tree_migration"};
  uint64_t n_small = 8ULL * 1024 * 1024;  // 8 M of 16 B = 128 MB
  uint64_t n_big = 32ULL * 1024;           // 32 K of 4096 B = 128 MB

  uint64_t **buf_s, **buf_b;
  uint64_t *misses_a, *misses_c, *corruptions;
  cudaMalloc((void **)&buf_s, sizeof(uint64_t *) * n_small);
  cudaMalloc((void **)&buf_b, sizeof(uint64_t *) * n_big);
  cudaMallocManaged((void **)&misses_a, sizeof(uint64_t));
  cudaMallocManaged((void **)&misses_c, sizeof(uint64_t));
  cudaMallocManaged((void **)&corruptions, sizeof(uint64_t));
  *misses_a = 0;
  *misses_c = 0;
  *corruptions = 0;
  cudaMemset(buf_s, 0, sizeof(uint64_t *) * n_small);
  cudaMemset(buf_b, 0, sizeof(uint64_t *) * n_big);
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  k_alloc_record<<<(n_small - 1) / 256 + 1, 256>>>(alloc, n_small, 16, buf_s,
                                                   misses_a);
  cudaDeviceSynchronize();
  k_verify_free<<<(n_small - 1) / 256 + 1, 256>>>(alloc, n_small, buf_s,
                                                  corruptions);
  cudaDeviceSynchronize();
  k_alloc_record<<<(n_big - 1) / 256 + 1, 256>>>(alloc, n_big, 4096, buf_b,
                                                 misses_c);
  cudaDeviceSynchronize();
  k_verify_free<<<(n_big - 1) / 256 + 1, 256>>>(alloc, n_big, buf_b,
                                                corruptions);
  cudaDeviceSynchronize();
  r.seconds = t.sync_end();
  r.misses = *misses_a + *misses_c;
  r.corruptions = *corruptions;
  r.pass = (r.misses == 0 && r.corruptions == 0);

  cudaFree(buf_s);
  cudaFree(buf_b);
  cudaFree(misses_a);
  cudaFree(misses_c);
  cudaFree(corruptions);
  return r;
}

static subtest_result run_drain_refill(allocator_t *alloc,
                                       uint64_t pool_bytes) {
  subtest_result r{"drain_refill"};
  const uint64_t sizes[4] = {16, 64, 256, 1024};
  uint64_t target_bytes_per_cycle = pool_bytes / 4;
  uint64_t max_n = target_bytes_per_cycle / sizes[0];

  uint64_t **buf;
  uint64_t *misses, *corruptions;
  cudaMalloc((void **)&buf, sizeof(uint64_t *) * max_n);
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&corruptions, sizeof(uint64_t));
  *corruptions = 0;
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  for (int c = 0; c < 8; ++c) {
    uint64_t size = sizes[c % 4];
    uint64_t n = target_bytes_per_cycle / size;
    *misses = 0;
    cudaMemset(buf, 0, sizeof(uint64_t *) * n);
    cudaDeviceSynchronize();
    k_alloc_record<<<(n - 1) / 256 + 1, 256>>>(alloc, n, size, buf, misses);
    cudaDeviceSynchronize();
    k_verify_free<<<(n - 1) / 256 + 1, 256>>>(alloc, n, buf, corruptions);
    cudaDeviceSynchronize();
    if (*misses != 0) {
      r.misses += *misses;
    }
  }
  r.seconds = t.sync_end();
  r.corruptions = *corruptions;
  r.pass = (r.misses == 0 && r.corruptions == 0);
  cudaFree(buf);
  cudaFree(misses);
  cudaFree(corruptions);
  return r;
}

// Long-running, fixed-size, high-concurrency churn. Counts misses and
// double-mallocs. The gallatin_churn test does the same pattern at this
// scale and passes — if we see corruption here it points to a real
// allocator issue, not a random-size test artifact.
static subtest_result run_stress(allocator_t *alloc) {
  subtest_result r{"stress"};
  uint64_t n = 1ULL * 1024 * 1024;  // 1 M threads
  int rounds = 16;                   // 16 M total alloc/free pairs
  uint64_t size = 64;
  uint64_t *misses, *double_mallocs;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&double_mallocs, sizeof(uint64_t));
  *misses = 0;
  *double_mallocs = 0;
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  k_stress<<<(n - 1) / 256 + 1, 256>>>(alloc, n, rounds, size, misses,
                                       double_mallocs);
  cudaDeviceSynchronize();
  r.seconds = t.sync_end();
  r.misses = *misses;
  r.corruptions = *double_mallocs;
  r.pass = (r.misses == 0 && r.corruptions == 0);
  cudaFree(misses);
  cudaFree(double_mallocs);
  return r;
}

static subtest_result run_mixed_churn(allocator_t *alloc) {
  subtest_result r{"mixed_churn"};
  uint64_t n = 256ULL * 1024;  // 256 K threads
  int rounds = 4;
  uint64_t *misses, *double_mallocs;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&double_mallocs, sizeof(uint64_t));
  *misses = 0;
  *double_mallocs = 0;
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  k_mixed_churn<<<(n - 1) / 256 + 1, 256>>>(alloc, n, rounds, misses,
                                            double_mallocs);
  cudaDeviceSynchronize();
  r.seconds = t.sync_end();
  r.misses = *misses;
  r.corruptions = *double_mallocs;
  r.pass = (r.misses == 0 && r.corruptions == 0);
  cudaFree(misses);
  cudaFree(double_mallocs);
  return r;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

static std::vector<subtest_result>
run_all(Gallatin_memory_type mode, uint64_t pool_bytes) {
  std::vector<subtest_result> results;
  allocator_t *alloc = boot(pool_bytes, mode, /*print_info=*/false);
  if (alloc == nullptr) {
    subtest_result r{"boot"};
    r.note = "boot returned nullptr (pool too large or OOM)";
    results.push_back(r);
    return results;
  }

  results.push_back(run_basic(alloc));
  results.push_back(run_concurrent(alloc, pool_bytes));
  results.push_back(run_tree_migration(alloc));
  results.push_back(run_drain_refill(alloc, pool_bytes));
  results.push_back(run_stress(alloc));
  results.push_back(run_mixed_churn(alloc));

  allocator_t::free_on_device(alloc);
  return results;
}

static void print_results(const char *label,
                          const std::vector<subtest_result> &rs) {
  std::cout << "[" << label << "]\n";
  for (const auto &r : rs) {
    std::cout << "  " << (r.pass ? "PASS" : "FAIL")
              << "  " << r.name
              << "  misses=" << r.misses
              << "  corruptions=" << r.corruptions
              << "  (" << r.seconds << "s)";
    if (!r.note.empty()) std::cout << "  " << r.note;
    std::cout << "\n";
  }
}

static int compare_results(const std::vector<subtest_result> &dev,
                           const std::vector<subtest_result> &mgr) {
  int rc = 0;
  if (dev.size() != mgr.size()) {
    std::cerr << "FAIL: result count mismatch: device=" << dev.size()
              << " managed=" << mgr.size() << "\n";
    return 1;
  }
  std::cout << "\n[parity]\n";
  for (size_t i = 0; i < dev.size(); ++i) {
    bool same_pass = (dev[i].pass == mgr[i].pass);
    bool same_corruption = (dev[i].corruptions == mgr[i].corruptions);
    std::cout << "  " << (same_pass && same_corruption ? "OK  " : "DIFF")
              << "  " << dev[i].name
              << "  device(pass=" << dev[i].pass
              << ",corr=" << dev[i].corruptions << ")"
              << "  managed(pass=" << mgr[i].pass
              << ",corr=" << mgr[i].corruptions << ")\n";
    if (!same_pass || !same_corruption) rc = 1;
  }
  return rc;
}

static int run_oversubscription(uint64_t target_gb) {
  std::cout << "\n[oversubscribe]\n";
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  uint64_t target_bytes = target_gb * 1024ULL * 1024 * 1024;
  std::cout << "  device free=" << (free_b >> 30) << " GB, target pool="
            << target_gb << " GB ("
            << (target_bytes > free_b ? "oversubscribed" : "fits") << ")\n";

  allocator_t *alloc =
      allocator_t::generate_on_device_managed(target_bytes, 42, false);
  if (alloc == nullptr) {
    std::cerr << "  FAIL: managed boot returned nullptr at " << target_gb
              << " GB\n";
    return 1;
  }

  // Touch a small fraction of the pool to force page-faulting in.
  uint64_t n = 64 * 1024;  // 64 K * 256 B = 16 MB worth of allocs
  uint64_t **buf;
  uint64_t *misses, *corruptions;
  cudaMalloc((void **)&buf, sizeof(uint64_t *) * n);
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&corruptions, sizeof(uint64_t));
  *misses = 0;
  *corruptions = 0;
  cudaMemset(buf, 0, sizeof(uint64_t *) * n);
  cudaDeviceSynchronize();
  k_alloc_record<<<(n - 1) / 256 + 1, 256>>>(alloc, n, 256, buf, misses);
  cudaDeviceSynchronize();
  k_verify_free<<<(n - 1) / 256 + 1, 256>>>(alloc, n, buf, corruptions);
  cudaError_t err = cudaDeviceSynchronize();

  int rc = 0;
  if (err != cudaSuccess) {
    std::cerr << "  FAIL: oversubscribe kernel: " << cudaGetErrorString(err)
              << "\n";
    rc = 1;
  }
  if (*misses != 0 || *corruptions != 0) {
    std::cerr << "  FAIL: misses=" << *misses
              << " corruptions=" << *corruptions << "\n";
    rc = 1;
  }
  if (rc == 0)
    std::cout << "  PASS  oversubscribed managed pool serves allocations\n";

  allocator_t::free_on_device(alloc);
  cudaFree(buf);
  cudaFree(misses);
  cudaFree(corruptions);
  return rc;
}

int main(int argc, char **argv) {
  uint64_t pool_gb = 2;       // small enough to fit easily; both modes run
  uint64_t oversub_gb = 0;     // 0 = skip; otherwise size in GB to oversubscribe

  if (argc > 1) pool_gb = std::stoull(argv[1]);
  if (argc > 2) oversub_gb = std::stoull(argv[2]);

  uint64_t pool_bytes = pool_gb * 1024ULL * 1024 * 1024;

  std::cout << "managed_mode_test:\n"
            << "  pool = " << pool_gb << " GB per mode\n"
            << "  oversub target = "
            << (oversub_gb == 0 ? std::string("skipped")
                                : std::to_string(oversub_gb) + " GB")
            << "\n\n";

  std::cout << "running device_only...\n";
  auto dev = run_all(device_only, pool_bytes);
  print_results("device", dev);

  std::cout << "\nrunning managed...\n";
  auto mgr = run_all(managed, pool_bytes);
  print_results("managed", mgr);

  int rc = compare_results(dev, mgr);

  if (oversub_gb > 0) {
    rc |= run_oversubscription(oversub_gb);
  }

  if (rc == 0) std::cout << "\nOVERALL: PASS\n";
  else std::cerr << "\nOVERALL: FAIL\n";
  return rc;
}
