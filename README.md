
![Gallatin Header](logo_big.png)

## Gallatin
Gallatin is a generic GPU allocator that allows for threads to dynamically malloc and free device-side allocations without relying on host calls or synchronization with host. The system scales efficiently to GPUs of any size and supports allocations of any size, up to the entire GPU DRAM.


Gallatin is built around the use of [van Emde Boas (vEB) trees](https://ieeexplore.ieee.org/abstract/document/4567861) to control a small universe of memory regions. GPU memory is paritioned into memory regions called **segments**. These segments can be formatted into smaller allocations called **blocks**, which are further subdivided into small allocations called **slices**. They can also be combined into larger allocations, with one or more contigous segments being treated as a single allocation. vEB trees provide fast successor search over the space, allowing us to always pick the segment with the smallest ID. This helps to minimize fragmentation and maximize memory reuse.


Gallatin is currently under revision.


# Usage
----------------------

To get started with Gallatin, we recommend using the global version of the allocator. This is a generic variant of the allocator that is well suited to any application type. This variant has been modified to be accessible through a series of static device functions, meaning that it can be easily added to existing projects and kernels.

To use the global variant, `#include <gallatin/allocators/global_allocator.cuh>` after including Gallatin in your project via CMAKE (details below). This will expose the following types and functions in the global namespace `gallatin::allocators`:

- `__device__ global_allocator_type * global_gallatin`: The pointer to the allocator in device memory. This is a global value that exists whenever `global_allocator.cuh` is included.
- `__host__ void init_global_allocator(uint64_t num_bytes, uint64_t seed)`: Initialize the global allocator `global_gallatin` to control `num_bytes` of memory - `seed` sets randomness for the vEB trees used internally.
- `__host__ void free_global_allocator()`: Release the memory held by the allocator. This will release all memory that has been allocated from the allocator as well[^1]. 
- `__device__ void * global_malloc(uint64_t num_bytes)`: Request an allocation of size at least `num_bytes` from the allocator. returns `nullptr` if the request can't be satisfied.
- `__device__ void global_free(void * ptr)`: free a pointer that has been previously allocated by Gallatin back to the global allocator[^2].
- `__host__ void print_global_stats()`: Print information about the current allocator status.


# Advanced tuning
--------------------

You may be able to further increase performance by hand-tuning the template parameters to better suit your application. An example of this would be a data structure like [slabHash](https://github.com/owensgroup/SlabHash) that only ever requires 128 byte allocations. Setting the minimum allocation size to 128 bytes allows Gallatin to provide more space in the block storage for this allocation size, which can increase peak throughput by over 2x for this use case.

To tune Gallatin, you can modify the 3 template parameters `segment_size`, `min_size`, and `max_size`.

- `segment_size` is the number of bytes per segment. By default, this is 16 megabytes. Increasing segment size increases the throughput of slice and block allocations of all sizes (as less vEB calls are needed), but reduces the granularity of reuse between allocation sizes, as all allocations in a segment must be released before the segment is returned to the segment tree.
- `min_size` and `max_size` are the minimum and maximum slice size, respectively. Gallatin generates one tree for each power of two between `min_size` and `max_size` (inclusive). Block sizes are always `4096*slice_size`, so the number of trees, intermediate slice sizes and block sizes are based on these values.

Gallatin must be constructed and destructed by host. To do so, call `Gallatin<template_args>::generate_on_device()`
and supply the # of bytes to be made allocable, along with a random seed. This function returns a handle to the allocator that can be used in device kernels.

To free device memory at the end of execution, call `Gallatin<template_args>::free_on_device(your_pointer)`

This will free the associated device memory, including all memory that has been handed out.[^1].

Inside of a kernel, you must pass a pointer to the allocator.
You can then allocate new memory with the malloc method: `void * allocator->malloc(uint64_t num_bytes)`

This returns a `void *` type of at least `num_bytes`, or `nullptr` if no allocation is available.

Once the memory is no longer needed, it can be returned via `void allocator->free(void * memory_ptr);`


# Including Gallatin
---------------------

Gallatin is a header-only library and only depends on CUDA and the CUDA Cooperative Groups library, which is included with all new versions of CUDA.

To include Gallatin in your project, you can use the [CMake Package Manager](https://github.com/cpm-cmake/CPM.cmake)

To add CPM, download `CPM.cmake` and add 

```include(cmake/CPM.cmake)``` 

to your cmake file.

To add Gallatin, include the following snippet and select a version.

If you remove the version tag, CPM will pull the most up-to-date build.

```
CPMAddPackage(
  NAME gallatin
  GITHUB_REPOSITORY saltsystemslab/gallatin
  GIT_TAG origin/main
)
```

Once Gallatin has been added to the project, you need to link the internal library `gallatin` to use the allocator. This can be done with `target_link_libraries(${EXE_NAME} PRIVATE gallatin)`. From there, all that's left is to include `#include <gallatin/allocators/global_allocator.cuh>
` and start doing dynamic allocations!



# Building Tests
---------------

Gallatin comes with some tests that showcase basic usage and performance of the allocator. To reduce build time/complexity when Gallatin is used in other projects, these tests are not enabled by default. To build the tests, pass the flag `-DGAL_TESTS=ON` to cmake when building the project. The tests available are as follows:

- `gallatin_test`: Test the allocator by repeatedly mallocing/freeing the majority of device memory.
- `global_test`: Same test suite as `gallatin_test`, but implemented with the global allocator.
- `global_churn`: Perform a series of random allocations/frees of varying sizes.

[^1]: This does not release associated pointers. Memory distributed by Gallatin should not be accessed once the allocator has been destroyed.
[^2]: Freeing a pointer that was not allocated by Gallatin or double freeing a pointer will result in undefined behavior.