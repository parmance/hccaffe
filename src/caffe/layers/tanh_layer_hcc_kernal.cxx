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
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out[idx[0]] = hc::fast_math::tanh(in[idx[0]]);
  }).wait();
}

template <>
void TanHBackward(const int N, float* in_diff,
  float* out_data, float* out_diff) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    float tanhx = out_data[idx[0]];
    out_diff[idx[0]] = in_diff[idx[0]] * (1 - tanhx * tanhx);
  }).wait();
}

template <>
void TanHForward(const int N, double* in, double* out) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out[idx[0]] = hc::fast_math::tanh(in[idx[0]]);
  }).wait();
}

template <>
void TanHBackward(const int N, double* in_diff,
  double* out_data, double* out_diff) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    double tanhx = out_data[idx[0]];
    out_diff[idx[0]] = in_diff[idx[0]] * (1 - tanhx * tanhx);
  }).wait();
}




