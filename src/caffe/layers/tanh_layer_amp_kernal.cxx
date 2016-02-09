// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>
#include "hc.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

template <typename Dtype>
void TanHForward(const int N, Dtype* in, Dtype* out);

template <typename Dtype>
void TanHBackward(const int N, Dtype* in_diff,
  Dtype* out_data, Dtype* out_diff);


template <>
void TanHForward(const int N, float* in, float* out) {
  hc::array_view<float, 1> inView =
    *((hc::array_view<float, 1>*)(in));
  hc::array_view<float, 1> outView =
    *((hc::array_view<float, 1>*)(out));
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    outView[idx] = hc::fast_math::tanh(inView[idx]);
  });
}

template <>
void TanHBackward(const int N, float* in_diff,
  float* out_data, float* out_diff) {
  hc::array_view<float, 1> in_diffView =
    *((hc::array_view<float, 1>*)(in_diff));
  hc::array_view<float, 1> out_diffView =
    *((hc::array_view<float, 1>*)(out_diff));
  hc::array_view<float, 1> out_dataView =
    *((hc::array_view<float, 1>*)(out_data));
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    float tanhx = out_dataView[idx];
    out_diffView[idx] = in_diffView[idx] * (1 - tanhx * tanhx);
  });
}

template <>
void TanHForward(const int N, double* in, double* out) {
  hc::array_view<double, 1> inView =
    *((hc::array_view<double, 1>*)(in));
  hc::array_view<double, 1> outView =
    *((hc::array_view<double, 1>*)(out));
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    outView[idx] = hc::fast_math::tanh(inView[idx]);
  });
}

template <>
void TanHBackward(const int N, double* in_diff,
  double* out_data, double* out_diff) {
  hc::array_view<double, 1> in_diffView =
    *((hc::array_view<double, 1>*)(in_diff));
  hc::array_view<double, 1> out_diffView =
    *((hc::array_view<double, 1>*)(out_diff));
  hc::array_view<double, 1> out_dataView =
    *((hc::array_view<double, 1>*)(out_data));
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    double tanhx = out_dataView[idx];
    out_diffView[idx] = in_diffView[idx] * (1 - tanhx * tanhx);
  });
}




