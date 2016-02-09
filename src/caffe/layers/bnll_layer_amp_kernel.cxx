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
  hc::array_view<float, 1> inView =
    *((hc::array_view<float, 1>*)(in));
  hc::array_view<float, 1> outView =
    *((hc::array_view<float, 1>*)(out));
  hc::extent<1> e(n);
    parallel_for_each(e,
      [=](hc::index<1> idx) __attribute__((hc, cpu)){
      outView[idx] = inView[idx] > 0 ?
        inView[idx] +
        hc::fast_math::log(1. +
          hc::fast_math::exp(-inView[idx])) :
        hc::fast_math::log(1. +
          hc::fast_math::exp(inView[idx]));
    });
}
template <>
void BNLLForward(const int n, double* in, double* out) {
  hc::array_view<double, 1> inView =
    *((hc::array_view<double, 1>*)(in));
  hc::array_view<double, 1> outView =
    *((hc::array_view<double, 1>*)(out));
  hc::extent<1> e(n);
  parallel_for_each(e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    outView[idx] = inView[idx] > 0 ?
    inView[idx] +
    hc::fast_math::log(1. +
      hc::fast_math::exp(-inView[idx])) :
      hc::fast_math::log(1. +
      hc::fast_math::exp(inView[idx]));
  });
}
template <>
void BNLLBackward(const int n,  float* in_diff,
  float* in_data, float* out_diff) {
  hc::array_view<float, 1> inDiffView =
    *((hc::array_view<float, 1>*)(in_diff));
  hc::array_view<float, 1> inDataView =
    *((hc::array_view<float, 1>*)(in_data));
  hc::array_view<float, 1> outDiffView =
    *((hc::array_view<float, 1>*)(out_diff));
  hc::extent<1> e(n);
  parallel_for_each(e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    float expval = hc::fast_math::
      exp(hc::fast_math::fmin(inDataView[idx], kBNLL_THRESHOLD));
    outDiffView[idx] = inDiffView[idx] * expval / (expval + 1.);
    });
}
template <>
void BNLLBackward(const int n, double* in_diff,
  double* in_data, double* out_diff) {
  hc::array_view<double, 1> inDiffView =
    *((hc::array_view<double, 1>*)(in_diff));
  hc::array_view<double, 1> inDataView =
    *((hc::array_view<double, 1>*)(in_data));
  hc::array_view<double, 1> outDiffView =
    *((hc::array_view<double, 1>*)(out_diff));
  hc::extent<1> e(n);
  parallel_for_each(e,
  [=](hc::index<1> idx) __attribute__((hc, cpu)){
  double expval = hc::fast_math::exp
    (hc::fast_math::fmin(inDataView[idx], kBNLL_THRESHOLD));
  outDiffView[idx] = inDiffView[idx] * expval / (expval + 1.);
  });
}

