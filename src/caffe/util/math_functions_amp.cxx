#include "caffe/util/math_functions.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;


namespace caffe {

#ifdef USE_CPPAMP
template <>
void caffe_amp_abs<float>(const int n, float* a, float* y) {
  array_view<float, 1> aView(n, a);
  array_view<float, 1> yView(n, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = Concurrency::fast_math::fabs(aView[idx]);
  }
  );
}
template <>
void caffe_amp_mul<float>(const int n, const float* a, const float* b, float* y){
  /*array_view<float, 1> aView(N, a);
  array_view<float, 1> bView(N, b);
  array_view<float, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = (aView[idx] * bView[idx]);
  }
  );*/
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
