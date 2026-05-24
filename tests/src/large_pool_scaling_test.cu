// large_pool_scaling_test
//
// Tries to reproduce the andes report: 64 GB gallatin pool works, 96 GB
// pool breaks even moderate workloads. Walks several pool sizes
// {16, 32, 64, 80, 96} GB and at each one runs:
//   1. boot
//   2. concurrent small-alloc stress (mimics ForEach1M)
//   3. report miss rate and a quick free pass
//
// If gallatin has a scaling bug at higher segment counts the miss rate
// climbs (or boot fails). If the issue is just total-GPU-memory
// exhaustion, the failure is in boot itself (cudaMalloc returns OOM).
//
// Pass criterion: every size that boots successfully also completes the
// alloc kernel with miss rate < 1%.

#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/timer.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace gallatin::allocators;

__global__ void alloc_kernel(uint64_t n, uint64_t size, uint64_t **out,
                             uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t *p = (uint64_t *)global_malloc(size);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    out[tid] = nullptr;
    return;
  }
  p[0] = tid;
  out[tid] = p;
}

__global__ void free_kernel(uint64_t n, uint64_t **in) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  if (in[tid] != nullptr) global_free(in[tid]);
}

struct trial_result {
  uint64_t pool_gb;
  bool boot_ok;
  uint64_t n_threads;
  uint64_t misses;
  double alloc_seconds;
  cudaError_t last_err;
};

static trial_result run_trial(uint64_t pool_gb, uint64_t n_threads,
                              uint64_t alloc_size) {
  trial_result r{};
  r.pool_gb = pool_gb;
  r.n_threads = n_threads;
  r.last_err = cudaSuccess;

  uint64_t pool_b = pool_gb * 1024ULL * 1024 * 1024;
  uint64_t **buf = nullptr;
  uint64_t *misses = nullptr;

  // Don't pre-check free memory here — let gallatin's pre-flight OOM
  // warning fire so we can verify it works.
  bool ok = init_global_allocator(pool_b, 42, /*print_info=*/false);
  if (!ok) {
    std::cout << "  boot rejected by gallatin pre-flight (warning above)\n";
    return r;
  }
  r.boot_ok = true;

  cudaMalloc((void **)&buf, sizeof(uint64_t *) * n_threads);
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  *misses = 0;
  cudaMemset(buf, 0, sizeof(uint64_t *) * n_threads);
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  alloc_kernel<<<(n_threads - 1) / 256 + 1, 256>>>(n_threads, alloc_size, buf,
                                                   misses);
  cudaError_t err = cudaDeviceSynchronize();
  r.alloc_seconds = t.sync_end();
  if (err != cudaSuccess) {
    std::cerr << "  alloc kernel FAILED: " << cudaGetErrorString(err) << "\n";
    r.last_err = err;
  }
  r.misses = *misses;

  free_kernel<<<(n_threads - 1) / 256 + 1, 256>>>(n_threads, buf);
  cudaDeviceSynchronize();

  free_global_allocator();
  cudaFree(buf);
  cudaFree(misses);
  return r;
}

int main(int argc, char **argv) {
  // Defaults match andes ForEach1M shape: 1M threads each allocating
  // a small task-path-sized buffer.
  uint64_t n_threads = 1ULL * 1024 * 1024;
  uint64_t alloc_size = 256;

  if (argc > 1) n_threads = std::stoull(argv[1]);
  if (argc > 2) alloc_size = std::stoull(argv[2]);

  // Pool sizes to probe. The interesting transition is around the
  // 64->80->96 GB range mentioned in the bug report.
  const uint64_t pools[] = {16, 32, 64, 80, 96};

  std::cout << "large_pool_scaling_test:\n"
            << "  n_threads = " << n_threads << "\n"
            << "  alloc_size = " << alloc_size << " B\n\n";

  int rc = 0;
  for (uint64_t gb : pools) {
    std::cout << "--- pool = " << gb << " GB ("
              << (gb * 1024 / 16) << " segments) ---\n";
    trial_result r = run_trial(gb, n_threads, alloc_size);
    if (!r.boot_ok) {
      // Boot failure is informational, not a test failure — likely just
      // not enough free GPU memory.
      continue;
    }
    double miss_pct = 100.0 * r.misses / r.n_threads;
    std::cout << "  alloc kernel: " << r.misses << "/" << r.n_threads
              << " misses (" << miss_pct << "%) in " << r.alloc_seconds
              << "s\n";
    if (r.last_err != cudaSuccess) {
      rc = 1;
    } else if (miss_pct > 1.0) {
      std::cerr << "  FAIL: miss rate >1% at pool=" << gb << " GB\n";
      rc = 1;
    } else {
      std::cout << "  PASS\n";
    }
    std::cout << "\n";
  }

  if (rc == 0) std::cout << "OVERALL: PASS\n";
  return rc;
}
