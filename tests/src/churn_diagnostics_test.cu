// churn_diagnostics_test
//
// Diagnostic harness that isolates which variable affects the "double
// malloc" counter under sustained churn (size: fixed/random; marker:
// scheme). Originally written to investigate a ~26% counter in
// managed_mode_test's mixed_churn — root cause turned out to be earlier
// sub-tests leaving non-zero bytes in slots when they freed (gallatin
// doesn't zero on free), and the kernel's `prior == 0` check assumed
// otherwise. Kept around as a probe for future "is the allocator
// double-allocating?" questions.
//
//   V1: FIXED size 64,           marker = tid+1
//   V2: RANDOM size 16..4096,    marker = tid+1   (slice-only path)
//   V3: RANDOM size 16..4111,    marker = tid+1   (mixes block-path)
//   V4: RANDOM size 16..4096,    marker = tid*1000+r+1

#include <gallatin/allocators/gallatin.cuh>
#include <gallatin/allocators/timer.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace gallatin::allocators;

using allocator_t = Gallatin<16ULL * 1024 * 1024, 16ULL, 4096ULL>;

template <int VARIANT>
__global__ void k_churn(allocator_t *alloc, uint64_t n, int rounds,
                        uint64_t *misses, uint64_t *prior_nonzero,
                        uint64_t *back_mismatch) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t hash = tid * 0x9e3779b97f4a7c15ULL + 1;

  for (int r = 0; r < rounds; ++r) {
    hash ^= (hash >> 32);
    hash *= 0xbf58476d1ce4e5b9ULL;

    uint64_t size;
    uint64_t marker;
    if constexpr (VARIANT == 1) {
      size = 64;
      marker = tid + 1;  // +1 so marker != 0 (sentinel)
    } else if constexpr (VARIANT == 2) {
      size = 16 + (hash % 4081);  // 16..4096
      marker = tid + 1;
    } else if constexpr (VARIANT == 3) {
      size = 16 + (hash & 0xFFF);  // 16..4111 (some land on block-path)
      marker = tid + 1;
    } else {  // VARIANT == 4
      size = 16 + (hash % 4081);  // 16..4096
      marker = tid * 1000ULL + r + 1;
    }

    uint64_t *p = (uint64_t *)alloc->malloc(size);
    if (p == nullptr) {
      atomicAdd((unsigned long long int *)misses, 1ULL);
      continue;
    }
    uint64_t prior = atomicExch((unsigned long long int *)p, marker);
    if (prior != 0ULL)
      atomicAdd((unsigned long long int *)prior_nonzero, 1ULL);
    uint64_t back = atomicExch((unsigned long long int *)p, 0ULL);
    if (back != marker)
      atomicAdd((unsigned long long int *)back_mismatch, 1ULL);
    alloc->free(p);
  }
}

struct variant_result {
  std::string name;
  uint64_t misses;
  uint64_t prior_nonzero;
  uint64_t back_mismatch;
  double seconds;
};

template <int VARIANT>
static variant_result run_variant(allocator_t *alloc, const char *name,
                                  uint64_t n, int rounds) {
  variant_result r;
  r.name = name;
  uint64_t *misses, *prior_nonzero, *back_mismatch;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&prior_nonzero, sizeof(uint64_t));
  cudaMallocManaged((void **)&back_mismatch, sizeof(uint64_t));
  *misses = 0;
  *prior_nonzero = 0;
  *back_mismatch = 0;
  cudaDeviceSynchronize();

  gallatin::utils::timer t;
  k_churn<VARIANT><<<(n - 1) / 256 + 1, 256>>>(alloc, n, rounds, misses,
                                               prior_nonzero, back_mismatch);
  cudaDeviceSynchronize();
  r.seconds = t.sync_end();
  r.misses = *misses;
  r.prior_nonzero = *prior_nonzero;
  r.back_mismatch = *back_mismatch;
  cudaFree(misses);
  cudaFree(prior_nonzero);
  cudaFree(back_mismatch);
  return r;
}

int main(int argc, char **argv) {
  uint64_t pool_bytes = 2ULL * 1024 * 1024 * 1024;
  uint64_t n_threads = 256ULL * 1024;
  int rounds = 4;

  if (argc > 1) n_threads = std::stoull(argv[1]);
  if (argc > 2) rounds = std::stoi(argv[2]);

  uint64_t total_ops = n_threads * rounds;

  std::cout << "churn_diagnostics_test:\n"
            << "  pool = " << (pool_bytes >> 20) << " MB\n"
            << "  threads = " << n_threads << ", rounds = " << rounds
            << "  (total ops = " << total_ops << ")\n\n";

  allocator_t *alloc =
      allocator_t::generate_on_device(pool_bytes, 42, /*print_info=*/false);
  if (alloc == nullptr) {
    std::cerr << "FAIL: boot failed\n";
    return 1;
  }

  std::vector<variant_result> results;
  results.push_back(run_variant<1>(alloc, "V1 fixed-64,     marker=tid",
                                    n_threads, rounds));
  results.push_back(run_variant<2>(alloc, "V2 random 16-4096, marker=tid",
                                    n_threads, rounds));
  results.push_back(run_variant<3>(alloc, "V3 random 16-4111, marker=tid",
                                    n_threads, rounds));
  results.push_back(run_variant<4>(alloc, "V4 random 16-4096, marker=tid*1000+r+1",
                                    n_threads, rounds));

  allocator_t::free_on_device(alloc);

  std::cout << "results (counts / " << total_ops << " ops):\n";
  std::cout << "                                          "
            << "misses  prior!=0  back!=marker  seconds\n";
  for (auto &r : results) {
    char buf[256];
    snprintf(buf, sizeof(buf), "  %-42s  %6llu  %8llu  %12llu  %7.4fs\n",
             r.name.c_str(), (unsigned long long)r.misses,
             (unsigned long long)r.prior_nonzero,
             (unsigned long long)r.back_mismatch, r.seconds);
    std::cout << buf;
  }

  std::cout << "\ninterpretation:\n";
  bool v1_clean = (results[0].prior_nonzero + results[0].back_mismatch) == 0;
  bool v2_clean = (results[1].prior_nonzero + results[1].back_mismatch) == 0;
  bool v3_clean = (results[2].prior_nonzero + results[2].back_mismatch) == 0;
  bool v4_clean = (results[3].prior_nonzero + results[3].back_mismatch) == 0;
  if (v1_clean && v2_clean && v3_clean && v4_clean) {
    std::cout << "  All variants clean — original mixed_churn bug not "
                 "reproduced; needs more variations.\n";
  } else if (v1_clean && v2_clean && !v3_clean) {
    std::cout << "  V3 dirty (16-4111 size range mixes block-path) — block "
                 "or multi-slice malloc/free path has a bug.\n";
  } else if (v1_clean && !v2_clean) {
    std::cout << "  V2 dirty (random size on pure slice path) — slice malloc "
                 "has a size-variation issue.\n";
  } else if (!v1_clean) {
    std::cout << "  V1 dirty (fixed size, simplest pattern) — a real "
                 "allocator bug under sustained pressure.\n";
  } else if (v1_clean && v2_clean && v3_clean && !v4_clean) {
    std::cout << "  V4 dirty but V2 clean — marker scheme matters; test "
                 "pattern artifact.\n";
  }

  return 0;
}
