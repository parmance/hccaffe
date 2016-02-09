#include <cfloat>
#include <vector>
#include "hc.hpp"
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
  hc::array_view<float, 1> aView =
    *((hc::array_view<float, 1>*)(a));
  hc::array_view<float, 1> bView =
    *((hc::array_view<float, 1>*)(b));
  hc::array_view<float, 1> yView =
    *((hc::array_view<float, 1>*)(y));
  hc::array_view<int, 1> maskView =
    *((hc::array_view<int, 1>*)(mask));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
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
  hc::array_view<double, 1> aView =
    *((hc::array_view<double, 1>*)(a));
  hc::array_view<double, 1> bView =
    *((hc::array_view<double, 1>*)(b));
  hc::array_view<double, 1> yView =
    *((hc::array_view<double, 1>*)(y));
  hc::array_view<int, 1> maskView =
    *((hc::array_view<int, 1>*)(mask));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
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
  hc::array_view<float, 1> top_diffView =
    *((hc::array_view<float, 1>*)(top_diff));
  hc::array_view<float, 1> bottom_diffView =
    *((hc::array_view<float, 1>*)(bottom_diff));
  hc::array_view<int, 1> maskView =
    *((hc::array_view<int, 1>*)(mask));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
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
  hc::array_view<double, 1> top_diffView =
    *((hc::array_view<double, 1>*)(top_diff));
  hc::array_view<double, 1> bottom_diffView =
    *((hc::array_view<double, 1>*)(bottom_diff));
  hc::array_view<int, 1> maskView =
    *((hc::array_view<int, 1>*)(mask));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
      double gradient = 0;
      if (maskView[idx] == blob_idx) {
        gradient += top_diffView[idx];
      }
      bottom_diffView[idx] = gradient;
    });
}

