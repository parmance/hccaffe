// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "amp.h"
#include "amp_math.h"

using namespace Concurrency;

template <typename Dtype>
void TanHForward(const int N, Dtype* in, Dtype* out);

template <typename Dtype>
void TanHBackward(const int N, Dtype* in_diff,
  Dtype* out_data, Dtype* out_diff);


template <>
void TanHForward(const int N, float* in, float* out) {
  array_view<float, 1> inView(N, in);
  array_view<float, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    outView[idx] = Concurrency::fast_math::tanh(inView[idx]);
  }
  );
  outView.synchronize();
}

template <>
void TanHBackward(const int N, float* in_diff,
  float* out_data, float* out_diff) {
  array_view<float, 1> in_diffView(N, in_diff);
  array_view<float, 1> out_diffView(N, out_diff);
  array_view<float, 1> out_dataView(N, out_data);
  parallel_for_each(
    out_dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    float tanhx = out_dataView[idx];
    out_diffView[idx] = in_diffView[idx] * (1 - tanhx * tanhx);
  }
  );
  out_diffView.synchronize();
}

template <>
void TanHForward(const int N, double* in, double* out) {
  array_view<double, 1> inView(N, in);
  array_view<double, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    outView[idx] = Concurrency::fast_math::tanh(inView[idx]);
  }
  );
  outView.synchronize();
}

template <>
void TanHBackward(const int N, double* in_diff,
  double* out_data, double* out_diff) {
  array_view<double, 1> in_diffView(N, in_diff);
  array_view<double, 1> out_diffView(N, out_diff);
  array_view<double, 1> out_dataView(N, out_data);
  parallel_for_each(
    out_dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    double tanhx = out_dataView[idx];
    out_diffView[idx] = in_diffView[idx] * (1 - tanhx * tanhx);
  }
  );
  out_diffView.synchronize();
}




