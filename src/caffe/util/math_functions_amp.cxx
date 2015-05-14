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
#endif //USE_CPPAMP
}  // namespace caffe
