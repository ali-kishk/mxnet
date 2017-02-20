/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_ENGINE_STREAM_MANAGER_H_
#define MXNET_ENGINE_STREAM_MANAGER_H_

#include <dmlc/base.h>
#include <mxnet/base.h>
#include <cstddef>
#include <array>
#include <string>
#include <mutex>
#include "../common/cuda_utils.h"

namespace mxnet {
namespace engine {

/*!
 * \brief Stream manager.
 *
 * Uses a basic round-robin algorithm to dispatch GPU streams. Returns default
 * context on CPU.
 */
template <std::size_t kNumGpus, std::size_t kStreams>
class StreamManager {
 public:
  StreamManager();
  ~StreamManager() {
    Finalize();
  }
  RunContext GetRunContext(Context const& ctx);
  RunContext GetIORunContext(Context const& ctx);
  void Finalize();
 private:
  std::mutex m_;
#if MXNET_USE_CUDA
  typedef mshadow::Stream<gpu>* stream_type;
#elif MXNET_USE_OPENCL
  typedef vex::backend::command_queue* stream_type;
#endif
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
  std::array<std::array<stream_type, kStreams>, kNumGpus> gpu_streams_;
  std::array<stream_type, kNumGpus> gpu_io_streams_;
  std::array<int, kNumGpus> gpu_cnt_;
#endif  // MXNET_USE_CUDA || MXNET_USE_OPENCL
  DISALLOW_COPY_AND_ASSIGN(StreamManager);
};  // class StreamManager

template <std::size_t kNumGpus, std::size_t kStreams>
RunContext StreamManager<kNumGpus, kStreams>::GetRunContext(
    Context const& ctx) {
  RunContext ret;
  ret.stream = nullptr;
  switch (ctx.dev_mask()) {
    case cpu::kDevMask: break;
    case gpu::kDevMask: {
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
      std::size_t use_counter;
#if MXNET_USE_CUDA
      CUDA_CALL(cudaSetDevice(ctx.dev_id));
#endif
      {
        std::lock_guard<std::mutex> lock{m_};
        auto&& counter = gpu_cnt_.at(ctx.dev_id);
        if (counter == -1) {
          for (auto&& i : gpu_streams_.at(ctx.dev_id)) {
#if MXNET_USE_CUDA
            i = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0);
#elif MXNET_USE_OPENCL
            i = new vex::backend::command_queue(vex::Context(vex::Filter::GPU && vex::Filter::Position(ctx.dev_id)).queue(0));
#endif
          }
          counter = 0;
        }
        use_counter = counter;
        counter = (counter + 1) % kStreams;
      }
      ret.stream = gpu_streams_.at(ctx.dev_id).at(use_counter);
      break;
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif  // MXNET_USE_CUDA || MXNET_USE_OPENCL
    }
  }
  return ret;
}

template <std::size_t kNumGpus, std::size_t kStreams>
RunContext StreamManager<kNumGpus, kStreams>::GetIORunContext(
    Context const& ctx) {
  RunContext ret;
  ret.stream = nullptr;
  switch (ctx.dev_mask()) {
    case cpu::kDevMask: break;
    case gpu::kDevMask: {
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
#if MXNET_USE_CUDA
      CUDA_CALL(cudaSetDevice(ctx.dev_id));
#endif
      {
        std::lock_guard<std::mutex> lock{m_};
        if (gpu_io_streams_.at(ctx.dev_id) == nullptr) {
#if MXNET_USE_CUDA
          gpu_io_streams_.at(ctx.dev_id) = mshadow::NewStream<gpu>(false, false);
#elif MXNET_USE_OPENCL
          gpu_io_streams_.at(ctx.dev_id) = new vex::backend::command_queue(vex::Context(vex::Filter::GPU && vex::Filter::Position(ctx.dev_id)).queue(0));
#endif
        }
      }
      ret.stream = gpu_io_streams_.at(ctx.dev_id);
      break;
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif  // MXNET_USE_CUDA || MXNET_USE_OPENCL
    }
  }
  return ret;
}

template <std::size_t kNumGpus, std::size_t kStreams>
StreamManager<kNumGpus, kStreams>::StreamManager() {
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
  for (std::size_t i = 0; i < kNumGpus; ++i) {
    gpu_cnt_.at(i) = -1;
  }
  for (auto&& i : gpu_io_streams_) {
    i = nullptr;
  }
#endif  // MXNET_USE_CUDA || MXNET_USE_OPENCL
}

template <std::size_t kNumGpus, std::size_t kStreams>
void StreamManager<kNumGpus, kStreams>::Finalize() {
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
  for (std::size_t i = 0; i < kNumGpus; ++i) {
    if (gpu_cnt_.at(i) != -1) {
      for (auto&& j : gpu_streams_.at(i)) {
#if MXNET_USE_CUDA
        // Catch exception for CUDA driver shutdown
        MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(j));
#elif MXNET_USE_OPENCL
        delete j;
#endif
      }
      gpu_cnt_.at(i) = -1;
    }
  }
#endif  // MXNET_USE_CUDA || MXNET_USE_OPENCL
}

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_ENGINE_STREAM_MANAGER_H_
