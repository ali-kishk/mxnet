#include <cstdio>
#include <gtest/gtest.h>
#include <dmlc/logging.h>
#include <mxnet/ndarray.h>

#if MXNET_USE_CUDA || MXNET_USE_OPENCL
TEST(NDArray, Basic_GPU) {
  mxnet::Context ctx = mxnet::Context::GPU(0);
  mxnet::NDArray a(mxnet::TShape({10, 10}), ctx);
  mxnet::NDArray b(mxnet::TShape({10, 10}), ctx);
  mxnet::NDArray c(mxnet::TShape({10, 10}), ctx);
  a = 0.5;
  b = 1.0;
  c = 2.0;
  a = a + b * c;
  float r[100];
  a.SyncCopyToCPU(r, 100);
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(r[i], 2.5);
}
#endif  // MXNET_USE_CUDA

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
