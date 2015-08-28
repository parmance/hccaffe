#include <cfloat>
#include <vector>

#include "amp.h"
#include "amp_math.h"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

template <typename Dtype>
void MaxForward(const int N, Dtype* a, Dtype* b,
    const int blob_idx, Dtype* y, int* mask);

template <typename Dtype>
void MaxBackward(const int N, Dtype* top_diff,
    int blob_idx, int* mask, Dtype* bottom_diff);

template <>
void MaxForward<float>(const int N, float* a, float* b,
  const int blob_idx, float* y, int* mask) {
  Concurrency::array_view<float, 1> aView =
    *((Concurrency::array_view<float, 1>*)(a));
  Concurrency::array_view<float, 1> bView =
    *((Concurrency::array_view<float, 1>*)(b));
  Concurrency::array_view<float, 1> yView =
    *((Concurrency::array_view<float, 1>*)(y));
  Concurrency::array_view<int, 1> maskView =
    *((Concurrency::array_view<int, 1>*)(mask));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (Concurrency::index<1> idx) restrict(amp){
      float maxval = -FLT_MAX;
      int maxidx = -1;
      if (aView[idx] > bView[idx]) {
        if (blob_idx == 0) {
          maxval = aView[idx];
          yView[idx] = maxval;
          maxidx = blob_idx;
          maskView[idx] = maxidx;
        }
      } else{
        maxval = bView[idx];
        yView[idx] = maxval;
        maxidx = blob_idx + 1;
        maskView[idx] = maxidx;
      }
    });
}

template <>
void MaxForward<double>(const int N, double* a, double* b,
  const int blob_idx, double* y, int* mask) {
  Concurrency::array_view<double, 1> aView =
    *((Concurrency::array_view<double, 1>*)(a));
  Concurrency::array_view<double, 1> bView =
    *((Concurrency::array_view<double, 1>*)(b));
  Concurrency::array_view<double, 1> yView =
    *((Concurrency::array_view<double, 1>*)(y));
  Concurrency::array_view<int, 1> maskView =
    *((Concurrency::array_view<int, 1>*)(mask));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (Concurrency::index<1> idx) restrict(amp){
      double maxval = -FLT_MAX;
      int maxidx = -1;
      if (aView[idx] > bView[idx]) {
        if (blob_idx == 0) {
          maxval = aView[idx];
          yView[idx] = maxval;
          maxidx = blob_idx;
          maskView[idx] = maxidx;
        }
      } else{
        maxval = bView[idx];
        yView[idx] = maxval;
        maxidx = blob_idx + 1;
        maskView[idx] = maxidx;
      }
    });
}

template <>
void MaxBackward<float>(const int N, float* top_diff,
  int blob_idx, int* mask, float* bottom_diff) {
  Concurrency::array_view<float, 1> top_diffView =
    *((Concurrency::array_view<float, 1>*)(top_diff));
  Concurrency::array_view<float, 1> bottom_diffView =
    *((Concurrency::array_view<float, 1>*)(bottom_diff));
  Concurrency::array_view<int, 1> maskView =
    *((Concurrency::array_view<int, 1>*)(mask));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (Concurrency::index<1> idx) restrict(amp){
      float gradient = 0;
      if (maskView[idx] == blob_idx) {
        gradient += top_diffView[idx];
      }
      bottom_diffView[idx] = gradient;
    });
}

template <>
void MaxBackward<double>(const int N, double* top_diff,
  int blob_idx, int* mask, double* bottom_diff) {
  Concurrency::array_view<double, 1> top_diffView =
    *((Concurrency::array_view<double, 1>*)(top_diff));
  Concurrency::array_view<double, 1> bottom_diffView =
    *((Concurrency::array_view<double, 1>*)(bottom_diff));
  Concurrency::array_view<int, 1> maskView =
    *((Concurrency::array_view<int, 1>*)(mask));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (Concurrency::index<1> idx) restrict(amp){
      double gradient = 0;
      if (maskView[idx] == blob_idx) {
        gradient += top_diffView[idx];
      }
      bottom_diffView[idx] = gradient;
    });
}

