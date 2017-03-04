/*!
 *  Copyright (c) 2015 by Contributors
 * \file resource.h
 * \brief Global resource allocation handling.
 */
#ifndef MXNET_RESOURCE_H_
#define MXNET_RESOURCE_H_

#include <dmlc/logging.h>
#include "./base.h"
#include "./engine.h"

namespace mxnet {

/*!
 * \brief The resources that can be requested by Operator
 */
struct ResourceRequest {
  /*! \brief Resource type, indicating what the pointer type is */
  enum Type {
    /*! \brief mshadow::Random<xpu> object */
    kRandom,
    /*! \brief A dynamic temp space that can be arbitrary size */
    kTempSpace
  };
  /*! \brief type of resources */
  Type type;
  /*! \brief default constructor */
  ResourceRequest() {}
  /*!
   * \brief constructor, allow implicit conversion
   * \param type type of resources
   */
  ResourceRequest(Type type)  // NOLINT(*)
      : type(type) {}
};

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
    DType r = b-a;
    *dst = rnd_uniform_(vex::element_index(), rseed_) * r + a;
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

/*!
 * \brief Resources used by mxnet operations.
 *  A resource is something special other than NDArray,
 *  but will still participate
 */
struct Resource {
  /*! \brief The original request */
  ResourceRequest req;
  /*! \brief engine variable */
  engine::VarHandle var;
  /*! \brief identifier of id information, used for debug purpose */
  int32_t id;
  /*!
   * \brief pointer to the resource, do not use directly,
   *  access using member functions
   */
  void *ptr_;
  /*! \brief default constructor */
  Resource() : id(0) {}
  /*!
   * \brief Get random number generator.
   * \param stream The stream to use in the random number generator.
   * \return the mshadow random number generator requested.
   * \tparam xpu the device type of random number generator.
   */
  template<typename xpu, typename DType>
  inline Random<xpu, DType>* get_random(
      mshadow::Stream<xpu> *stream) const {
    CHECK_EQ(req.type, ResourceRequest::kRandom);
    Random<xpu, DType> *ret = static_cast<Random<xpu, DType>*>(ptr_);
    ret->set_stream(stream);
    return ret;
  }
  /*!
   * \brief Get space requested as mshadow Tensor.
   *  The caller can request arbitrary size.
   *
   *  This space can be shared with other calls to this->get_space.
   *  So the caller need to serialize the calls when using the conflicted space.
   *  The old space can get freed, however, this will incur a synchronization,
   *  when running on device, so the launched kernels that depend on the temp space
   *  can finish correctly.
   *
   * \param shape the Shape of returning tensor.
   * \param stream the stream of retruning tensor.
   * \return the mshadow tensor requested.
   * \tparam xpu the device type of random number generator.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<typename xpu, int ndim>
  inline mshadow::Tensor<xpu, ndim, real_t> get_space(
      mshadow::Shape<ndim> shape, mshadow::Stream<xpu> *stream) const {
    return get_space_typed<xpu, ndim, real_t>(shape, stream);
  }
#if MXNET_USE_OPENCL
  /*!
   * \brief Get space requested as VexCL vector.
   *  The caller can request arbitrary size.
   *
   *  This space can be shared with other calls to this->get_space.
   *  So the caller need to serialize the calls when using the conflicted space.
   *  The old space can get freed, however, this will incur a synchronization,
   *  when running on device, so the launched kernels that depend on the temp space
   *  can finish correctly.
   *
   * \param q the command_queue of returning vector.
   * \return the vector requested.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<typename xpu, int ndim>
  inline vex::vector<real_t> get_space(
      mshadow::Shape<ndim> shape, vex::backend::command_queue &q) const {
    return get_space_typed<real_t>(shape, q);
  }
#endif
  /*!
   * \brief Get cpu space requested as mshadow Tensor.
   *  The caller can request arbitrary size.
   *
   * \param shape the Shape of returning tensor.
   * \return the mshadow tensor requested.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<int ndim>
  inline mshadow::Tensor<cpu, ndim, real_t> get_host_space(
      mshadow::Shape<ndim> shape) const {
    return get_host_space_typed<cpu, ndim, real_t>(shape);
  }
  /*!
   * \brief Get space requested as mshadow Tensor in specified type.
   *  The caller can request arbitrary size.
   *
   * \param shape the Shape of returning tensor.
   * \param stream the stream of retruning tensor.
   * \return the mshadow tensor requested.
   * \tparam xpu the device type of random number generator.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<typename xpu, int ndim, typename DType>
  inline mshadow::Tensor<xpu, ndim, DType> get_space_typed(
      mshadow::Shape<ndim> shape, mshadow::Stream<xpu> *stream) const {
    CHECK_EQ(req.type, ResourceRequest::kTempSpace);
    return mshadow::Tensor<xpu, ndim, DType>(
        reinterpret_cast<DType*>(get_space_internal(shape.Size() * sizeof(DType))),
        shape, shape[ndim - 1], stream);
  }
#if MXNET_USE_OPENCL
  /*!
   * \brief Get space requested as VexCL vector in specified type.
   *  The caller can request arbitrary size.
   *
   * \param q the command_queue of returning vector.
   * \return the vector requested.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<int ndim, typename DType>
  inline vex::vector<DType> get_space_typed(
      mshadow::Shape<ndim> shape, vex::backend::command_queue &q) const {
    CHECK_EQ(req.type, ResourceRequest::kTempSpace);
    return vex::vector<DType>(
        q, vex::backend::device_vector<DType>(*reinterpret_cast<cl::Buffer*>(get_space_internal(shape.Size() * sizeof(DType)))));
  }
#endif
  /*!
   * \brief Get CPU space as mshadow Tensor in specified type.
   * The caller can request arbitrary size.
   *
   * \param shape the Shape of returning tensor
   * \return the mshadow tensor requested
   * \tparam ndim the number of dimnesion of tensor requested
   * \tparam DType request data type
   */
  template<int ndim, typename DType>
  inline mshadow::Tensor<cpu, ndim, DType> get_host_space_typed(
    mshadow::Shape<ndim> shape) const {
      return mshadow::Tensor<cpu, ndim, DType>(
        reinterpret_cast<DType*>(get_host_space_internal(shape.Size() * sizeof(DType))),
        shape, shape[ndim - 1], NULL);
  }
  /*!
   * \brief internal function to get space from resources.
   * \param size The size of the space.
   * \return The allocated space.
   */
  void* get_space_internal(size_t size) const;
  /*!
   * \brief internal function to get cpu space from resources.
   * \param size The size of space.
   * \return The allocated space
   */
  void *get_host_space_internal(size_t size) const;
};

/*! \brief Global resource manager */
class ResourceManager {
 public:
  /*!
   * \brief Get resource of requested type.
   * \param ctx the context of the request.
   * \param req the resource request.
   * \return the requested resource.
   * \note The returned resource's ownership is
   *       still hold by the manager singleton.
   */
  virtual Resource Request(Context ctx, const ResourceRequest &req) = 0;
  /*!
   * \brief Seed all the allocated random numbers.
   * \param seed the seed to the random number generators on all devices.
   */
  virtual void SeedRandom(uint32_t seed) = 0;
  /*! \brief virtual destructor */
  virtual ~ResourceManager() DMLC_THROW_EXCEPTION {}
  /*!
   * \return Resource manager singleton.
   */
  static ResourceManager *Get();
};
}  // namespace mxnet
#endif  // MXNET_RESOURCE_H_
