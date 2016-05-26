#include <algorithm>
#include <vector>
#include "hc.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

float kBNLL_THRESHOLD = 50.;
template <typename Dtype>
void BNLLForward(const int n, Dtype* in, Dtype* out);
template <typename Dtype>
void BNLLBackward(const int n, Dtype* in_diff,
                  Dtype* in_data, Dtype* out_diff);

template <>
void BNLLForward(const int n, float* in, float* out) {
  hc::extent<1> e(n);
    parallel_for_each(e,
      [=](hc::index<1> idx) __attribute__((hc, cpu)){
      out[idx[0]] = in[idx[0]] > 0 ?
        in[idx[0]] +
        hc::fast_math::log(1. +
          hc::fast_math::exp(-in[idx[0]])) :
        hc::fast_math::log(1. +
          hc::fast_math::exp(in[idx[0]]));
    }).wait();
}
template <>
void BNLLForward(const int n, double* in, double* out) {
  hc::extent<1> e(n);
  parallel_for_each(e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out[idx[0]] = in[idx[0]] > 0 ?
    in[idx[0]] +
    hc::fast_math::log(1. +
      hc::fast_math::exp(-in[idx[0]])) :
      hc::fast_math::log(1. +
      hc::fast_math::exp(in[idx[0]]));
  }).wait();
}
template <>
void BNLLBackward(const int n,  float* in_diff,
  float* in_data, float* out_diff) {
  hc::extent<1> e(n);
  parallel_for_each(e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    float expval = hc::fast_math::
      exp(hc::fast_math::fmin(in_data[idx[0]], kBNLL_THRESHOLD));
    out_diff[idx[0]] = in_diff[idx[0]] * expval / (expval + 1.);
    }).wait();
}
template <>
void BNLLBackward(const int n, double* in_diff,
  double* in_data, double* out_diff) {
  hc::extent<1> e(n);
  parallel_for_each(e,
  [=](hc::index<1> idx) __attribute__((hc, cpu)){
  double expval = hc::fast_math::exp
    (hc::fast_math::fmin(in_data[idx[0]], kBNLL_THRESHOLD));
  out_diff[idx[0]] = in_diff[idx[0]] * expval / (expval + 1.);
  }).wait();
}

