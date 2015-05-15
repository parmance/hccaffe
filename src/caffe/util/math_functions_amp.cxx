#include "caffe/util/math_functions.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;


namespace caffe {

#ifdef USE_CPPAMP
template <typename Dtype>
void caffe_amp_abs(const int N, Dtype* a, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.extent,
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = aView[idx] >= 0 ? aView[idx] : -1 * aView[idx];
  }
  );
}                                                                                                                                                

template <typename Dtype>
void caffe_amp_sign(const int N, Dtype* a, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.extent,
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = aView[idx] == 0 ? 0 : (aView[idx] < 0 ? -1 : 1);
  }
  );
}

template <typename Dtype>
void caffe_amp_mul(const int N, Dtype* a, Dtype* b, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> bView(N, b);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.extent,
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = (aView[idx] * bView[idx]);
  }
  );
}
template <typename Dtype>
void set_kernel(const int N, const Dtype alpha, Dtype* y) {
  array_view<Dtype, 1> outView(N, y);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = alpha;
    }
  );
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel(N, alpha, Y);
}

//template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
//template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
//template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

#endif //USE_CPPAMP
}  // namespace caffe
