/*!
 *  Copyright (c) 2015 by Contributors
 * \file resource.cc
 * \brief Implementation of resource manager.
 */
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/resource.h>
#include <mxnet/storage.h>
#include <limits>
#include <atomic>
#include "./common/lazy_alloc_array.h"

namespace mxnet {
namespace resource {

// A workaround to get a unified Random interface
template <typename Device, typename DType=float>
struct RandomType {
  typedef mshadow::Random<Device, DType> type;
};
#if MXNET_USE_OPENCL
template<typename DType>
struct RandomType<gpu, DType> {
  typedef class RandomCL {
  public:
  /*!
  * \brief constructor of random engine
  * \param seed random number seed
  */
  explicit RandomCL(int seed) {
    this->Seed(seed);
  }
  ~RandomCL(void) {
  }
  /*!
  * \brief seed random number generator using this seed
  * \param seed seed of prng
  */
  inline void Seed(int seed) {
    this->rseed_ = static_cast<cl_ulong>(seed);
  }
  /*!
  * \brief get random seed used in random generator
  * \return seed in unsigned
  */
  inline unsigned GetSeed() const {
    return rseed_;
  }
  /*!
  * \brief generate data from uniform [a,b)
  * \param dst destination
  * \param a lower bound of uniform
  * \param b upper bound of uniform
  * \tparam dim dimension of tensor
  */
  inline void SampleUniform(vex::vector<DType> *dst,
                            DType a = 0.0f, DType b = 1.0f) {
    *dst = rnd_uniform_(vex::element_index(), rseed_) * (b-a) + a;
  }
  /*!
  * \brief generate data from standard gaussian
  * \param dst destination
  * \param mu mean variable
  * \param sigma standard deviation
  * \tparam dim dimension of tensor
  */
  inline void SampleGaussian(vex::vector<DType> *dst,
                             DType mu = 0.0f, DType sigma = 1.0f) {
    if (sigma <= 0.0f) {
      *dst = mu;
    } else {
      *dst = rnd_normal_(vex::element_index(), rseed_) * sigma + mu;
    }
  }
  private:
    cl_ulong rseed_;
    vex::Random<DType> rnd_uniform_;
    vex::RandomNormal<DType> rnd_normal_;
  } type;
};
#endif
template<typename Device, typename DType=float>
using Random = typename RandomType<Device, DType>::type;

// internal structure for space allocator
struct SpaceAllocator {
  // internal context
  Context ctx;
  // internal handle
  Storage::Handle handle;
  // internal CPU handle
  Storage::Handle host_handle;

  SpaceAllocator() {
    handle.dptr = nullptr;
    handle.size = 0;
    host_handle.dptr = nullptr;
    host_handle.size = 0;
  }
  inline void ReleaseAll() {
    if (handle.size != 0) {
      Storage::Get()->DirectFree(handle);
      handle.size = 0;
    }
    if (host_handle.size != 0) {
      Storage::Get()->DirectFree(host_handle);
      host_handle.size = 0;
    }
  }
  inline void* GetSpace(size_t size) {
    if (handle.size >= size) return handle.dptr;
    if (handle.size != 0) {
      Storage::Get()->DirectFree(handle);
    }
    handle = Storage::Get()->Alloc(size, ctx);
    return handle.dptr;
  }

  inline void* GetHostSpace(size_t size) {
    if (host_handle.size >= size) return host_handle.dptr;
    if (handle.size != 0) {
      Storage::Get()->DirectFree(host_handle);
    }
    host_handle = Storage::Get()->Alloc(size, Context());
    return host_handle.dptr;
  }
};


// Implements resource manager
class ResourceManagerImpl : public ResourceManager {
 public:
  ResourceManagerImpl() noexcept(false)
      : global_seed_(0) {
    cpu_temp_space_copy_ = dmlc::GetEnv("MXNET_CPU_TEMP_COPY", 4);
    gpu_temp_space_copy_ = dmlc::GetEnv("MXNET_GPU_TEMP_COPY", 1);
    engine_ref_ = Engine::_GetSharedRef();
    storage_ref_ = Storage::_GetSharedRef();
    cpu_rand_.reset(new ResourceRandom<cpu>(
        Context::CPU(), global_seed_));
    cpu_space_.reset(new ResourceTempSpace(
        Context::CPU(), cpu_temp_space_copy_));
  }
  ~ResourceManagerImpl() {
    // need explicit delete, before type get killed
    cpu_rand_.reset(nullptr);
    cpu_space_.reset(nullptr);
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
    gpu_rand_.Clear();
    gpu_space_.Clear();
#endif
    if (engine_ref_ != nullptr) {
      engine_ref_ = nullptr;
    }
    if (storage_ref_ != nullptr) {
      storage_ref_ = nullptr;
    }
  }

  // request resources
  Resource Request(Context ctx, const ResourceRequest &req) override {
    if (ctx.dev_mask() == cpu::kDevMask) {
      switch (req.type) {
        case ResourceRequest::kRandom: return cpu_rand_->resource;
        case ResourceRequest::kTempSpace: return cpu_space_->GetNext();
        default: LOG(FATAL) << "Unknown supported type " << req.type;
      }
    } else {
      CHECK_EQ(ctx.dev_mask(), gpu::kDevMask);
#if MSHADOW_USE_CUDA || MXNET_USE_OPENCL
      switch (req.type) {
        case ResourceRequest::kRandom: {
          return gpu_rand_.Get(ctx.dev_id, [ctx, this]() {
              return new ResourceRandom<gpu>(ctx, global_seed_);
            })->resource;
        }
        case ResourceRequest::kTempSpace: {
          return gpu_space_.Get(ctx.dev_id, [ctx, this]() {
              return new ResourceTempSpace(ctx, gpu_temp_space_copy_);
            })->GetNext();
        }
        default: LOG(FATAL) << "Unknown supported type " << req.type;
      }
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
    Resource ret;
    return ret;
  }

  void SeedRandom(uint32_t seed) override {
    global_seed_ = seed;
    cpu_rand_->Seed(global_seed_);
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
    gpu_rand_.ForEach([seed](size_t i, ResourceRandom<gpu> *p) {
        p->Seed(seed);
      });
#endif
  }

 private:
  /*! \brief Maximum number of GPUs */
  static constexpr std::size_t kMaxNumGPUs = 16;
  /*! \brief Random number magic number to seed different random numbers */
  static constexpr uint32_t kRandMagic = 127UL;
  // the random number resources
  template<typename xpu>
  struct ResourceRandom {
    /*! \brief the context of the PRNG */
    Context ctx;
    /*! \brief pointer to PRNG */
    Random<xpu> *prnd;
    /*! \brief resource representation */
    Resource resource;
    /*! \brief constructor */
    explicit ResourceRandom(Context ctx, uint32_t global_seed)
        : ctx(ctx) {
#if MXNET_USE_CUDA
      mshadow::SetDevice<xpu>(ctx.dev_id);
#endif
      resource.var = Engine::Get()->NewVariable();
      prnd = new Random<xpu>(ctx.dev_id + global_seed * kRandMagic);
      resource.ptr_ = prnd;
      resource.req = ResourceRequest(ResourceRequest::kRandom);
    }
    ~ResourceRandom() {
      Random<xpu> *r = prnd;
      Engine::Get()->DeleteVariable(
          [r](RunContext rctx) {
            MSHADOW_CATCH_ERROR(delete r);
          }, ctx, resource.var);
    }
    // set seed to a PRNG
    inline void Seed(uint32_t global_seed) {
      uint32_t seed = ctx.dev_id + global_seed * kRandMagic;
      Random<xpu> *r = prnd;
      Engine::Get()->PushSync([r, seed](RunContext rctx) {
#if MXNET_USE_CUDA
          r->set_stream(rctx.get_stream<xpu>());
#endif
          r->Seed(seed);
        }, ctx, {}, {resource.var},
        FnProperty::kNormal, 0, PROFILER_MESSAGE("ResourceRandomSetSeed"));
    }
  };

  // temporal space resource.
  struct ResourceTempSpace {
    /*! \brief the context of the device */
    Context ctx;
    /*! \brief the underlying space */
    std::vector<SpaceAllocator> space;
    /*! \brief resource representation */
    std::vector<Resource> resource;
    /*! \brief current pointer to the round roubin alloator */
    std::atomic<size_t> curr_ptr;
    /*! \brief constructor */
    explicit ResourceTempSpace(Context ctx, size_t ncopy)
        : ctx(ctx), space(ncopy), resource(ncopy), curr_ptr(0) {
      for (size_t i = 0; i < space.size(); ++i) {
        resource[i].var = Engine::Get()->NewVariable();
        resource[i].id = static_cast<int32_t>(i);
        resource[i].ptr_ = &space[i];
        resource[i].req = ResourceRequest(ResourceRequest::kTempSpace);
        space[i].ctx = ctx;
        CHECK_EQ(space[i].handle.size, 0);
      }
    }
    ~ResourceTempSpace() {
      for (size_t i = 0; i < space.size(); ++i) {
        SpaceAllocator r = space[i];
        Engine::Get()->DeleteVariable(
            [r](RunContext rctx){
              SpaceAllocator rcpy = r;
              MSHADOW_CATCH_ERROR(rcpy.ReleaseAll());
            }, ctx, resource[i].var);
      }
    }
    // get next resource in round roubin matter
    inline Resource GetNext() {
      const size_t kMaxDigit = std::numeric_limits<size_t>::max() / 2;
      size_t ptr = ++curr_ptr;
      // reset ptr to avoid undefined behavior during overflow
      // usually this won't happen
      if (ptr > kMaxDigit) {
        curr_ptr.store((ptr + 1) % space.size());
      }
      return resource[ptr % space.size()];
    }
  };
  /*! \brief number of copies in CPU temp space */
  int cpu_temp_space_copy_;
  /*! \brief number of copies in GPU temp space */
  int gpu_temp_space_copy_;
  /*! \brief Reference to the type */
  std::shared_ptr<Engine> engine_ref_;
  /*! \brief Reference to the storage */
  std::shared_ptr<Storage> storage_ref_;
  /*! \brief internal seed to the random number generator */
  uint32_t global_seed_;
  /*! \brief CPU random number resources */
  std::unique_ptr<ResourceRandom<cpu> > cpu_rand_;
  /*! \brief CPU temp space resources */
  std::unique_ptr<ResourceTempSpace> cpu_space_;
#if MXNET_USE_CUDA || MXNET_USE_OPENCL
  /*! \brief random number generator for GPU */
  common::LazyAllocArray<ResourceRandom<gpu> > gpu_rand_;
  /*! \brief temp space for GPU */
  common::LazyAllocArray<ResourceTempSpace> gpu_space_;
#endif
};
}  // namespace resource

void* Resource::get_space_internal(size_t size) const {
  return static_cast<resource::SpaceAllocator*>(ptr_)->GetSpace(size);
}

void* Resource::get_host_space_internal(size_t size) const {
  return static_cast<resource::SpaceAllocator*>(ptr_)->GetHostSpace(size);
}

ResourceManager* ResourceManager::Get() {
  typedef dmlc::ThreadLocalStore<resource::ResourceManagerImpl> inst;
  return inst::Get();
}
}  // namespace mxnet
