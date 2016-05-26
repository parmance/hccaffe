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
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
      float maxval = -FLT_MAX;
      int maxidx = -1;
      if (a[idx[0]] > b[idx[0]]) {
        if (blob_idx == 0) {
          maxval = a[idx[0]];
          y[idx[0]] = maxval;
          maxidx = blob_idx;
          mask[idx[0]] = maxidx;
        }
      } else{
        maxval = b[idx[0]];
        y[idx[0]] = maxval;
        maxidx = blob_idx + 1;
        mask[idx[0]] = maxidx;
      }
    }).wait();
}

template <>
void MaxForward<double>(const int N, double* a, double* b,
  const int blob_idx, double* y, int* mask) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
      double maxval = -FLT_MAX;
      int maxidx = -1;
      if (a[idx[0]] > b[idx[0]]) {
        if (blob_idx == 0) {
          maxval = a[idx[0]];
          y[idx[0]] = maxval;
          maxidx = blob_idx;
          mask[idx[0]] = maxidx;
        }
      } else{
        maxval = b[idx[0]];
        y[idx[0]] = maxval;
        maxidx = blob_idx + 1;
        mask[idx[0]] = maxidx;
      }
    }).wait();
}

template <>
void MaxBackward<float>(const int N, float* top_diff,
  int blob_idx, int* mask, float* bottom_diff) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
      float gradient = 0;
      if (mask[idx[0]] == blob_idx) {
        gradient += top_diff[idx[0]];
      }
      bottom_diff[idx[0]] = gradient;
    }).wait();
}

template <>
void MaxBackward<double>(const int N, double* top_diff,
  int blob_idx, int* mask, double* bottom_diff) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=] (hc::index<1> idx) __attribute__((hc, cpu)){
      double gradient = 0;
      if (mask[idx[0]] == blob_idx) {
        gradient += top_diff[idx[0]];
      }
      bottom_diff[idx[0]] = gradient;
    }).wait();
}

