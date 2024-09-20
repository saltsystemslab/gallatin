#ifndef GALLATIN_SINGLETPN
#define GALLATIN_SINGLETPN

/******* ABOUT
 * The error log is a device-side global variable that records logs.
 * if the macro GALLATIN_DEBUG is set to 1 this will be initialized
 * stores a fixed number of logs - This is only intended to diagnose issues in the allocator itself.
 * If the allocator is running smoothly than it can power a dynamic logging system.
 * *******/

namespace gallatin {

namespace internals {


template<typename singleton, typename T>
__global__ void write_singleton_kernel(T * device_version){

  singleton::instance() = device_version[0];

}

template <typename singleton, typename T>
__global__ void read_singleton_kernel(T * device_version){

  device_version[0] = singleton::instance();

}

template <typename T>
struct singleton {


  using my_type = singleton<T>;


  __device__ static T & instance()
  {
    static T s;
    return s;
  } // instance

 static __host__ T read_instance(){

    T * device_version;

    cudaMallocManaged((void **)&device_version, sizeof(T));

    read_singleton_kernel<my_type, T><<<1,1>>>(device_version);

    cudaDeviceSynchronize();

    T output = device_version[0];

    cudaFree(device_version);

    return output;

  }

  static __host__ void write_instance(T write){

    T * device_version;

    cudaMallocManaged((void **)&device_version, sizeof(T));

    device_version[0] = write;

    write_singleton_kernel<my_type, T><<<1,1>>>(device_version);

    cudaDeviceSynchronize();

    cudaFree(device_version);

  }


  singleton(const singleton &) = delete;
  singleton & operator = (const singleton &) = delete;

private:

  singleton() {}
  ~singleton() {}

}; // struct singleton



}  // namespace internals

}  // namespace gallatin

#endif  // End of error log