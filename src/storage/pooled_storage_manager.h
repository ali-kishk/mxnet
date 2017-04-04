/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#include <mxnet/base.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <new>
#include "./storage_manager.h"
#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA || MXNET_USE_OPENCL
/*!
 * \brief Storage manager with a memory pool on gpu.
 */
class GPUPooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() {
    ReleaseAll();
  }
  void* Alloc(size_t raw_size) override;
  void Free(void* ptr, size_t raw_size) override;
  void DirectFree(void* ptr, size_t raw_size) override;

 private:
  void ReleaseAll();
  // internal mutex
  std::mutex mutex_;
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // number of devices
  const int NDEV = 32;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

void* GPUPooledStorageManager::Alloc(size_t raw_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  void* ret;
  size_t size = raw_size + NDEV;
  auto&& reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
#if MXNET_USE_CUDA
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free <= total * reserve_ / 100 || size > free - total * reserve_ / 100)
      ReleaseAll();
    cudaError_t e = cudaMalloc(&ret, size);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
#elif MXNET_USE_OPENCL
    try {
      try {
        ret = new cl::Buffer(cl::Context::getDefault(), CL_MEM_READ_WRITE, size);
      } catch (cl::Error& e) {
        if (e.err() == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
          ReleaseAll();
          ret = new cl::Buffer(cl::Context::getDefault(), CL_MEM_READ_WRITE, size);
        } else {
          LOG(FATAL) << "clCreateBuffer failed: " << e.what();
        }
      }
    } catch (cl::Error& e) {
      LOG(FATAL) << "clCreateBuffer failed: " << e.what();
    }
#endif
    used_memory_ += size;
  } else {
    auto&& reuse_pool = reuse_it->second;
    ret = reuse_pool.back();
    reuse_pool.pop_back();
  }
  return ret;
}

void GPUPooledStorageManager::Free(void* ptr, size_t raw_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t size = raw_size + NDEV;
  auto&& reuse_pool = memory_pool_[size];
  reuse_pool.push_back(ptr);
}

void GPUPooledStorageManager::DirectFree(void* ptr, size_t raw_size) {
  size_t size = raw_size + NDEV;
#if MXNET_USE_CUDA
  cudaError_t err = cudaFree(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
    LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
  }
#elif MXNET_USE_OPENCL
  delete static_cast<cl::Buffer*>(ptr);
#endif
  used_memory_ -= size;
}

void GPUPooledStorageManager::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      DirectFree(j, i.first - NDEV);
    }
  }
  memory_pool_.clear();
}
#endif  // MXNET_USE_CUDA || MXNET_USE_OPENCL

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
