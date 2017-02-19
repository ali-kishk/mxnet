/*!
 * Copyright (c) 2015 by Contributors
 * \file gpu_device_storage.h
 * \brief GPU storage implementation.
 */
#ifndef MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_GPU_DEVICE_STORAGE_H_

#include <mxnet/base.h>
#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#endif  // MXNET_USE_CUDA
#include <new>

namespace mxnet {
namespace storage {

/*!
 * \brief GPU storage implementation.
 */
class GPUDeviceStorage {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  inline static void* Alloc(size_t size);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(void* ptr);
};  // class GPUDeviceStorage

inline void* GPUDeviceStorage::Alloc(size_t size) {
  void* ret = nullptr;
#if MXNET_USE_CUDA
  cudaError_t e = cudaMalloc(&ret, size);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading)
    throw std::bad_alloc();
#elif MXNET_USE_OPENCL
  ret = new cl::Buffer(vex::current_context().context(0), CL_MEM_READ_WRITE, size);
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA or OpenCL enabled";
#endif  // MXNET_USE_CUDA
  return ret;
}

inline void GPUDeviceStorage::Free(void* ptr) {
#if MXNET_USE_CUDA
  // throw special exception for caller to catch.
  cudaError_t err = cudaFree(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
    LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
  }
#elif MXNET_USE_OPENCL
  delete static_cast<cl::Buffer*>(ptr);
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA or OpenCL enabled";
#endif  // MXNET_USE_CUDA
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
