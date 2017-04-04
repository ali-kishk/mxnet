/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_function-inl.h
 * \brief The real implementation of NDArray functions.
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_

#include <vector>
#include "./ndarray_function.h"
// this file will be included twice by CPU and GPU
// macro to help specialize evaluation function

#ifndef DECL_TERNARY
#define DECL_TERNARY(XPU, OP, FUN)                                       \
  template<>                                                            \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &mhs, \
                                       const TBlob &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, mhs, rhs, ret, ctx);                                   \
  }
#endif

#ifndef DECL_BINARY
#define DECL_BINARY(XPU, OP, FUN)                                       \
  template<>                                                            \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                   \
  }
#endif

#ifndef DECL_SCALAR
#define DECL_SCALAR(XPU, OP, FUN, REVERSE)                              \
  template<>                                                            \
  void Eval<XPU, OP, REVERSE>(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP, REVERSE>(lhs, rhs, ret, ctx);                          \
  }
#endif

#if defined(__CUDACC__)
#define DEVICE gpu
#else
#define DEVICE cpu
#endif

namespace mxnet {
namespace ndarray {
// true implementation
template<typename xpu, typename OP>
struct EvalBinaryBase_ {
  static inline void Eval(const TBlob &lhs, const TBlob &rhs,
                          TBlob *ret, RunContext ctx) {
    using namespace mshadow::expr;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Only support input/output with the same data type";
    CHECK_EQ(ret->type_flag_, rhs.type_flag_)
      << "Only support input/output with the same data type";
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s),
                                     rhs.FlatTo2D<xpu, DType>(s));
    });
  }
};
#if MXNET_USE_OPENCL
template<typename OP>
struct EvalBinaryBase_<gpu, OP> {
  static inline void Eval(const TBlob &lhs, const TBlob &rhs,
                          TBlob *ret, RunContext ctx) {
    CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Only support input/output with the same data type";
    CHECK_EQ(ret->type_flag_, rhs.type_flag_)
      << "Only support input/output with the same data type";
    vex::backend::command_queue *q = ctx.get_queue();
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      vex::vector<DType> result(ret->get<DType>(*q));
      OP::vexcl_op(lhs.get<DType>(*q), rhs.get<DType>(*q), result);
    });
  }
};
#endif

template<typename xpu, typename OP>
inline void EvalBinary_(const TBlob &lhs, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  EvalBinaryBase_<xpu, OP>::Eval(lhs, rhs, ret, ctx);
}

template<typename xpu, typename OP>
inline void EvalOneHot_(const TBlob &index, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  LOG(INFO) << "The operator onehot_encode is deprecated; use one_hot instead.";
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type encoding, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(index.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  ret->get<xpu, 2, real_t>(s) =
    one_hot_encode(index.get<xpu, 1, real_t>(s),
                   rhs.shape_[1]);
}

#if MXNET_USE_OPENCL
template<>
inline void EvalOneHot_<gpu, OneHotEncode>(const TBlob &index, const TBlob &rhs,
                                           TBlob *ret, RunContext ctx) {
  VEX_FUNCTION(float, one_hot_encode, (float, choice) (size_t, index) (size_t, num_choice) (float*, one_hot_matrix),
    float *one_hot_vector = one_hot_matrix + index * num_choice;
    for (size_t i = 0; i < num_choice; ++i)
      one_hot_vector = 0.0f;
    one_hot_vector[size_t(choice+1e-4f)] = 1.0f;
    return choice;
  );
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(index.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  vex::backend::command_queue &q = *ctx.get_queue();
  vex::vector<float> one_hot_matrix(ret->get<float>(q)),
                     index_vector(index.get<float>(q));
  index_vector = one_hot_encode(index_vector, vex::element_index(), rhs.shape_[1], vex::raw_pointer(one_hot_matrix));
}
#endif

template<typename xpu, typename OP>
inline void EvalMatChooseRowElem_(const TBlob &lhs, const TBlob &rhs,
                                  TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type choose, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(lhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  ret->get<xpu, 1, real_t>(s)
      = mat_choose_row_element(lhs.get<xpu, 2, real_t>(s),
                               rhs.get<xpu, 1, real_t>(s));
}

#if MXNET_USE_OPENCL
template<>
inline void EvalMatChooseRowElem_<gpu, MatChooseRowElem>(const TBlob &lhs, const TBlob &rhs,
                                                         TBlob *ret, RunContext ctx) {
  VEX_FUNCTION(float, mat_choose_row_element, (float, choice) (size_t, index) (size_t, num_choice) (float*, matrix),
    return matrix[index * num_choice + size_t(choice+1e-4)];
  );
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(lhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  vex::backend::command_queue &q = *ctx.get_queue();
  vex::vector<float> chosen_vector(ret->get<float>(q)),
                     index_vector(rhs.get<float>(q)),
                     matrix(lhs.get<float>(q));
  chosen_vector = mat_choose_row_element(index_vector, vex::element_index(), rhs.shape_[1], vex::raw_pointer(matrix));
}
#endif

template<typename xpu, typename OP>
inline void EvalMatFillRowElem_(const TBlob &lhs, const TBlob &mhs, const TBlob &rhs,
                                  TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  ret->get<xpu, 2, real_t>(s)
          = mat_fill_row_element(lhs.get<xpu, 2, real_t>(s),
                                 mhs.get<xpu, 1, real_t>(s),
                                 rhs.get<xpu, 1, real_t>(s));
}

#if MXNET_USE_OPENCL
template<>
inline void EvalMatFillRowElem_<gpu, MatFillRowElem>(const TBlob &lhs, const TBlob &mhs, const TBlob &rhs,
                                                     TBlob *ret, RunContext ctx) {
  VEX_FUNCTION(float, mat_fill_row_element, (float, choice) (float, value) (size_t, index) (size_t, num_choice) (float*, fill_matrix),
    fill_matrix[index * num_choice + size_t(choice+1e-4)] = value;
    return choice;
  );
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "mat_fill_row_element only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "mat_fill_row_element only support float32 as input/output";
  CHECK_EQ(lhs.type_flag_, mshadow::default_type_flag)
    << "mat_fill_row_element only support float32 as input/output";
  CHECK_EQ(mhs.type_flag_, mshadow::default_type_flag)
    << "mat_fill_row_element only support float32 as input/output";
  vex::backend::command_queue &q = *ctx.get_queue();
  vex::vector<float> fill_matrix(ret->get<float>(q)),
                     index_vector(rhs.get<float>(q)),
                     value_vector(mhs.get<float>(q));
  index_vector = mat_fill_row_element(index_vector, value_vector, vex::element_index(), rhs.shape_[1], vex::raw_pointer(fill_matrix));
}
#endif

template<typename xpu, typename OP, bool reverse>
struct EvalScalarBase_;
template<typename xpu, typename OP>
struct EvalScalarBase_<xpu, OP, false> {
  static inline void Eval(const TBlob &lhs, const real_t &rhs,
                          TBlob *ret, RunContext ctx) {
    using namespace mshadow::expr;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Only support input/output with the same data type";
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s), scalar(DType(rhs)));
    });
  }
};
template<typename xpu, typename OP>
struct EvalScalarBase_<xpu, OP, true> {
  static inline void Eval(const TBlob &lhs, const real_t &rhs,
                          TBlob *ret, RunContext ctx) {
    using namespace mshadow::expr;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Only support input/output with the same data type";
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(scalar(DType(rhs)), lhs.FlatTo2D<xpu, DType>(s));
    });
  }
};
#if MXNET_USE_OPENCL
template<typename OP>
struct EvalScalarBase_<gpu, OP, false> {
  static inline void Eval(const TBlob &lhs, const real_t &rhs,
                          TBlob *ret, RunContext ctx) {
    using namespace mshadow::expr;
    CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Only support input/output with the same data type";
    vex::backend::command_queue *q = ctx.get_queue();
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      vex::vector<DType> result(ret->get<DType>(*q));
      OP::vexcl_op(lhs.get<DType>(*q), DType(rhs), result);
    });
  }
};
template<typename OP>
struct EvalScalarBase_<gpu, OP, true> {
  static inline void Eval(const TBlob &lhs, const real_t &rhs,
                          TBlob *ret, RunContext ctx) {
    using namespace mshadow::expr;
    CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Only support input/output with the same data type";
    vex::backend::command_queue *q = ctx.get_queue();
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      vex::vector<DType> result(ret->get<DType>(*q));
      OP::vexcl_op(DType(rhs), lhs.get<DType>(*q), result);
    });
  }
};
#endif

template<typename xpu, typename OP, bool reverse>
inline void EvalScalar_(const TBlob &lhs, const real_t &rhs,
                        TBlob *ret, RunContext ctx) {
  EvalScalarBase_<xpu, OP, reverse>::Eval(lhs, rhs, ret, ctx);
}

template<>
void EvalClip<DEVICE>(const TBlob &src, const real_t &a_min, const real_t &a_max,
                      TBlob *ret, RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s)
      = F<ClipMax::mshadow_op>(
          F<ClipMin::mshadow_op>(src.FlatTo2D<xpu, DType>(s), scalar(DType(a_min))),
          scalar(DType(a_max)));
  });
}

template<>
void EvalRandom<DEVICE, UniformDistribution>(
    const real_t &a,
    const real_t &b,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleUniform(&tmp, float(a), float(b));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleUniform(&tmp, double(a), double(b));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GaussianDistribution>(
    const real_t &mu,
    const real_t &sigma,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGaussian(&tmp, float(mu), float(sigma));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGaussian(&tmp, double(mu), double(sigma));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

#if MXNET_USE_OPENCL
template<>
void EvalRandom<gpu, UniformDistribution>(
    const real_t &a,
    const real_t &b,
    const Resource &resource,
    TBlob *ret,  RunContext ctx) {
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      Random<gpu, float> *prnd = static_cast<Random<gpu, float>*>(resource.ptr_);
      vex::vector<float> tmp(ret->get<float>(*ctx.get_queue()));
      prnd->SampleUniform(&tmp, a, b);
      break;
    }
  case mshadow::kFloat64:
    {
      Random<gpu, double> *prnd = static_cast<Random<gpu, double>*>(resource.ptr_);
      vex::vector<double> tmp(ret->get<double>(*ctx.get_queue()));
      prnd->SampleUniform(&tmp, a, b);
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<gpu, GaussianDistribution>(
    const real_t &mu,
    const real_t &sigma,
    const Resource &resource,
    TBlob *ret, RunContext ctx) {
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      Random<gpu, float> *prnd = static_cast<Random<gpu, float>*>(resource.ptr_);
      vex::vector<float> tmp(ret->get<float>(*ctx.get_queue()));
      prnd->SampleGaussian(&tmp, mu, sigma);
      break;
    }
  case mshadow::kFloat64:
    {
      Random<gpu, double> *prnd = static_cast<Random<gpu, double>*>(resource.ptr_);
      vex::vector<double> tmp(ret->get<double>(*ctx.get_queue()));
      prnd->SampleGaussian(&tmp, mu, sigma);
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}
#endif

template<>
void Eval<DEVICE>(const real_t &rhs, TBlob *ret, RunContext ctx) {
  mshadow::Stream<DEVICE> *s = ctx.get_stream<DEVICE>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<DEVICE, DType>(s) = DType(rhs);
  });
}
#if MXNET_USE_OPENCL
template<>
void Eval<gpu>(const real_t &rhs, TBlob *ret, RunContext ctx) {
  vex::backend::command_queue *q = ctx.get_queue();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->get<DType>(*q) = DType(rhs);
  });
}
#endif

template<>
void ElementwiseSum<DEVICE>(const std::vector<TBlob> source,
                            TBlob *dst,
                            RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  for (size_t i = 1; i < source.size(); ++i) {
    CHECK_EQ(source[i].type_flag_, dst->type_flag_)
      << "Only support input/output with the same data type";
  }
  MSHADOW_TYPE_SWITCH(dst->type_flag_, DType, {
    Tensor<xpu, 2, DType> out = dst->FlatTo2D<xpu, DType>(s);

    switch (source.size()) {
      case 2: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1;
        break;
      }
      case 3: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2;
        break;
      }
      case 4: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_3 = source[3].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2 + in_3;
        break;
      }
      default: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        out = F<mshadow::op::identity>(in_0);
        for (size_t i = 1; i < source.size(); ++i) {
          out += source[i].FlatTo2D<xpu, DType>(s);
        }
        break;
      }
    }
  });
}

#if MXNET_USE_OPENCL
template<>
void ElementwiseSum<gpu>(const std::vector<TBlob> source,
                         TBlob *dst,
                         RunContext ctx) {
  vex::backend::command_queue &q = *ctx.get_queue();
  for (size_t i = 1; i < source.size(); ++i) {
    CHECK_EQ(source[i].type_flag_, dst->type_flag_)
      << "Only support input/output with the same data type";
  }
  MSHADOW_TYPE_SWITCH(dst->type_flag_, DType, {
    vex::vector<DType> out(dst->get<DType>(q));

    switch (source.size()) {
      case 2: {
        vex::vector<DType> in_0 = source[0].get<DType>(q);
        vex::vector<DType> in_1 = source[1].get<DType>(q);
        out = in_0 + in_1;
        break;
      }
      case 3: {
        vex::vector<DType> in_0 = source[0].get<DType>(q);
        vex::vector<DType> in_1 = source[1].get<DType>(q);
        vex::vector<DType> in_2 = source[2].get<DType>(q);
        out = in_0 + in_1 + in_2;
        break;
      }
      case 4: {
        vex::vector<DType> in_0 = source[0].get<DType>(q);
        vex::vector<DType> in_1 = source[1].get<DType>(q);
        vex::vector<DType> in_2 = source[2].get<DType>(q);
        vex::vector<DType> in_3 = source[3].get<DType>(q);
        out = in_0 + in_1 + in_2 + in_3;
        break;
      }
      default: {
        out = source[0].get<DType>(q);
        for (size_t i = 1; i < source.size(); ++i) {
          out += source[i].get<DType>(q);
        }
        break;
      }
    }
  });
}
#endif

template <>
void EvalBroadcast<DEVICE>(TBlob const& src, TBlob* ret, int size, RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 3> out = ret->get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 2> in = src.get<xpu, 2, real_t>(s);
  out = mshadow::expr::broadcast_with_axis(in, 0, size);
}

#if MXNET_USE_OPENCL
template <>
void EvalBroadcast<gpu>(TBlob const& src, TBlob* ret, int size, RunContext ctx) {
  vex::backend::command_queue &q = *ctx.get_queue();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    vex::vector<DType> out(ret->get<DType>(q)),
                       in(src.get<DType>(q));
    size_t M = src.shape_[0], N = src.shape_[1];
    out = vex::reshape(in, vex::extents[M][size][N], vex::extents[0][2]);
  });
}
#endif

// declarations
DECL_BINARY(DEVICE, MatChooseRowElem, EvalMatChooseRowElem_)
DECL_TERNARY(DEVICE, MatFillRowElem, EvalMatFillRowElem_)
DECL_BINARY(DEVICE, OneHotEncode, EvalOneHot_)
DECL_BINARY(DEVICE, Plus, EvalBinary_)
DECL_BINARY(DEVICE, Minus, EvalBinary_)
DECL_BINARY(DEVICE, Mul, EvalBinary_)
DECL_BINARY(DEVICE, Div, EvalBinary_)
DECL_SCALAR(DEVICE, Plus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, true)
DECL_SCALAR(DEVICE, Div, EvalScalar_, true)
// for reverse seq
DECL_SCALAR(DEVICE, Plus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, false)
DECL_SCALAR(DEVICE, Div, EvalScalar_, false)
#if MXNET_USE_OPENCL
DECL_BINARY(gpu, MatChooseRowElem, EvalMatChooseRowElem_)
DECL_TERNARY(gpu, MatFillRowElem, EvalMatFillRowElem_)
DECL_BINARY(gpu, OneHotEncode, EvalOneHot_)
DECL_BINARY(gpu, Plus, EvalBinary_)
DECL_BINARY(gpu, Minus, EvalBinary_)
DECL_BINARY(gpu, Mul, EvalBinary_)
DECL_BINARY(gpu, Div, EvalBinary_)
DECL_SCALAR(gpu, Plus, EvalScalar_, true)
DECL_SCALAR(gpu, Minus, EvalScalar_, true)
DECL_SCALAR(gpu, Mul, EvalScalar_, true)
DECL_SCALAR(gpu, Div, EvalScalar_, true)
// for reverse seq
DECL_SCALAR(gpu, Plus, EvalScalar_, false)
DECL_SCALAR(gpu, Minus, EvalScalar_, false)
DECL_SCALAR(gpu, Mul, EvalScalar_, false)
DECL_SCALAR(gpu, Div, EvalScalar_, false)
#endif
}  // namespace ndarray
}  // namespace mxnet

#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
