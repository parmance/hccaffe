#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;


template <typename Dtype>
void MaxForward(const int N, Dtype* a, Dtype* b, const int blob_idx, Dtype* y, int* mask);
  
template <typename Dtype>
void MaxBackward(const int N, Dtype* top_diff, int blob_idx, int* mask, Dtype* bottom_diff);

template <>
void MaxForward<float>(const int N, float* a, float* b,
  const int blob_idx, float* y, int* mask) {
  array_view<float, 1> aView(N, a);
  array_view<float, 1> bView(N, b);
  array_view<float, 1> yView(N, y);
  array_view<int, 1> maskView(N, mask);
  parallel_for_each(
    yView.get_extent(),
    [=] (concurrency::index<1> idx) restrict(amp)
    {
      float maxval = -FLT_MAX;
      int maxidx = -1;
      if (aView[idx] > bView[idx]) {
        if (blob_idx == 0) {
          maxval = aView[idx];
          yView[idx] = maxval;
          maxidx = blob_idx;
          maskView[idx] = maxidx;
        }
      }
      else {
        maxval = bView[idx];
        yView[idx] = maxval;
        maxidx = blob_idx + 1;
        maskView[idx] = maxidx;
      }
    }
  );
  yView.synchronize();
}

template <>
void MaxForward<double>(const int N, double* a, double* b,
  const int blob_idx, double* y, int* mask) {
  array_view<double, 1> aView(N, a);
  array_view<double, 1> bView(N, b);
  array_view<double, 1> yView(N, y);
  array_view<int, 1> maskView(N, mask);
  parallel_for_each(
    yView.get_extent(),
    [=] (concurrency::index<1> idx) restrict(amp)
    {
      double maxval = -FLT_MAX;
      int maxidx = -1;
      if (aView[idx] > bView[idx]) {
        if (blob_idx == 0) {
          maxval = aView[idx];
          yView[idx] = maxval;
          maxidx = blob_idx;
          maskView[idx] = maxidx;
        }
      }
      else {
        maxval = bView[idx];
        yView[idx] = maxval;
        maxidx = blob_idx + 1;
        maskView[idx] = maxidx;
      }
    }
  );
  yView.synchronize();
}

template <>
void MaxBackward<float>(const int N, float* top_diff,
  int blob_idx, int* mask, float* bottom_diff) {
  array_view<float, 1> top_diffView(N, top_diff);
  array_view<float, 1> bottom_diffView(N, bottom_diff);
  array_view<int, 1> maskView(N, mask);
  parallel_for_each(
    bottom_diffView.get_extent(),
    [=] (concurrency::index<1> idx) restrict(amp)
    {
      float gradient = 0;
      if (maskView[idx] == blob_idx) {
        gradient += top_diffView[idx];
      }
      bottom_diffView[idx] = gradient;
    }
  );
  bottom_diffView.synchronize();
}

template <>
void MaxBackward<double>(const int N, double* top_diff,
  int blob_idx, int* mask, double* bottom_diff) {
  array_view<double, 1> top_diffView(N, top_diff);
  array_view<double, 1> bottom_diffView(N, bottom_diff);
  array_view<int, 1> maskView(N, mask);
  parallel_for_each(
    bottom_diffView.get_extent(),
    [=] (concurrency::index<1> idx) restrict(amp)
    {
      double gradient = 0;
      if (maskView[idx] == blob_idx) {
        gradient += top_diffView[idx];
      }
      bottom_diffView[idx] = gradient;
    }
  );
  bottom_diffView.synchronize();
}

