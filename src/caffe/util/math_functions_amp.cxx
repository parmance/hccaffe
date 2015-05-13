#include "caffe/util/math_functions.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;

#define TILESIZE 1

void caffe_amp_abs(const int N, float* a, float* y);
void caffe_amp_abs(const int N, float* a, float* y) {
  array_view<float, 1> aView(N, a);
  array_view<float, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = Concurrency::fast_math::fabs(aView[idx]);
  }
  );
}

