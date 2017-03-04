/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_function_cpu.cc
 * \brief CPU Implementation of ndarray function.
 */

// this will be invoked by gcc and compile CPU version
#include "./ndarray_function.h"
#include "./ndarray_function-inl.h"

namespace mxnet {
namespace ndarray {
template<>
void Copy<cpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    if (to->type_flag_ == from.type_flag_) {
        mshadow::Copy(to->FlatTo1D<cpu, DType>(),
                      from.FlatTo1D<cpu, DType>());
    } else {
        MSHADOW_TYPE_SWITCH(from.type_flag_, SrcDType, {
            to->FlatTo1D<cpu, DType>() =
                mshadow::expr::tcast<DType>(from.FlatTo1D<cpu, SrcDType>());
        })
    }
  })
}

#if MXNET_USE_OPENCL
template<>
void Copy<cpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    vex::vector<DType> to_vec(to->get<DType>(*ctx.get_queue()));
    vex::copy(from.dptr<DType>(), to_vec, false);
  });
}

template<>
void Copy<gpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    vex::copy(from.get<DType>(*ctx.get_queue()), to->dptr<DType>(), false);
  });
}

template<>
void Copy<gpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  CHECK_EQ(from_ctx.dev_id, to_ctx.dev_id)
    << "Cross device copy has not been implemented yet.";
  vex::backend::command_queue *q = ctx.get_queue();
  q->enqueueCopyBuffer(from.cl_buffer(), to->cl_buffer(), 0, 0, from.Size());
}
#endif
}  // namespace ndarray
}  // namespace mxnet
