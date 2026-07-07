// Config-parametrized static-path probe: boot Gallatin<16MB, 16, BIGGEST>, dump the
// static-counter per-tree slot count (g_nblk) to confirm malloc_static actually engages,
// then allocate a range of sizes and report success. Build with -DGALLATIN_STATIC_COUNTER
// and -DGT_BIGGEST=128 to try the lean small-alloc config (4 trees 16/32/64/128).
#include <gallatin/allocators/gallatin.cuh>
#include <cstdio>
#include <cstdlib>

using namespace gallatin::allocators;
#ifndef GT_BIGGEST
#define GT_BIGGEST 4096ULL
#endif
using alloc_t = Gallatin<16ULL * 1024 * 1024, 16ULL, (uint64_t)GT_BIGGEST>;

__global__ void dump_gnblk(alloc_t *a) {
  if (threadIdx.x || blockIdx.x) return;
  int nt = a->num_trees;
  printf("num_trees=%d (config: smallest=16, biggest=%llu)\n", nt, (unsigned long long)GT_BIGGEST);
#ifdef GALLATIN_STATIC_COUNTER
  for (int t = 0; t < nt; t++)
    printf("  [GNBLK] tree=%d slice=%lluB g_nblk=%d\n", t,
           (unsigned long long)a->table->get_tree_alloc_size(t),
           block_cache::S().g_nblk[t]);
#else
  printf("  (static counter disabled)\n");
#endif
}

__global__ void probe_one(alloc_t *a, uint64_t size, uint64_t *ok) {
  if (threadIdx.x || blockIdx.x) return;
  void *p = a->malloc(size);
  if (p == nullptr) { *ok = 0; printf("  malloc(%llu) -> NULL\n", (unsigned long long)size); return; }
  ((uint64_t *)p)[0] = 0xABCD;
  *ok = (((uint64_t *)p)[0] == 0xABCD) ? 1 : 2;
  printf("  malloc(%llu) -> %p ok=%llu (tree_alloc=%lluB)\n", (unsigned long long)size, p,
         (unsigned long long)*ok, (unsigned long long)size);
  a->free(p);
}

int main(int argc, char **argv) {
  uint64_t pool = (argc > 1) ? strtoull(argv[1], nullptr, 10) : (16ULL * 1024 * 1024 * 1024);
  printf("=== static-path probe: pool=%.1f GB, Gallatin<16MB,16,%llu> ===\n",
         pool / 1e9, (unsigned long long)GT_BIGGEST);
  alloc_t *a = alloc_t::generate_on_device(pool, 42);
  cudaDeviceSynchronize();
  dump_gnblk<<<1, 1>>>(a); cudaDeviceSynchronize();
  uint64_t *ok; cudaMallocManaged(&ok, sizeof(uint64_t));
  for (uint64_t sz : {16ULL, 32ULL, 64ULL, 128ULL, 256ULL, 4096ULL}) {
    *ok = 99;
    probe_one<<<1, 1>>>(a, sz, ok);
    cudaError_t e = cudaDeviceSynchronize();
    printf("RESULT size=%llu : %s (cuda=%s)\n", (unsigned long long)sz,
           (*ok == 1) ? "SUCCESS" : "FAIL", cudaGetErrorString(e));
  }
  return 0;
}
