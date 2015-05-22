#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "amp.h"
#include "amp_math.h"
using namespace concurrency;


template <typename Dtype>
void CLLForward(const int N, const int channels, const Dtype margin, const Dtype alpha,
    Dtype* y, Dtype* diff, Dtype* dist_sq, Dtype* bottom_diff);


template <>
void CLLForward<float>(const int N, const int channels, const float margin, const float alpha,
    float* y, float* diff, float* dist_sq, float* bottom_diff) {
  array_view<float, 1> yView(N, y);
  array_view<float, 1> diffView(N, diff);
  array_view<float, 1> dist_sqView(N, dist_sq);
  array_view<float, 1> bottom_diffView(N, bottom_diff);
  parallel_for_each(
    bottom_diffView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels;  // the num index, to access y and dist_sq
      if (static_cast<int>(yView[n])) {  // similar pairsS
        bottom_diffView[idx] = alpha * diffView[idx];
      }
      else {  // dissimilar pairs
        if ((margin - dist_sqView[n]) > 0) {
          bottom_diffView[idx] = -alpha * diffView[idx];
        }
        else {
          bottom_diffView[idx] = 0;
        }
      }
    }
  );
  bottom_diffView.synchronize();
}

template <>
void CLLForward<double>(const int N, const int channels, const double margin, const double alpha,
    double* y, double* diff, double* dist_sq, double* bottom_diff) {
  array_view<double, 1> yView(N, y);
  array_view<double, 1> diffView(N, diff);
  array_view<double, 1> dist_sqView(N, dist_sq);
  array_view<double, 1> bottom_diffView(N, bottom_diff);
  parallel_for_each(
    bottom_diffView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels;  // the num index, to access y and dist_sq
      if (static_cast<int>(yView[n])) {  // similar pairsS
        bottom_diffView[idx] = alpha * diffView[idx];
      }
      else {  // dissimilar pairs
        if ((margin - dist_sqView[n]) > 0) {
          bottom_diffView[idx] = -alpha * diffView[idx];
        }
        else {
          bottom_diffView[idx] = 0;
        }
      }
    }
  );
  bottom_diffView.synchronize();
}

