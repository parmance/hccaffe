#include <amp.h>
#include <amp_math.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

template <typename Dtype>
void SigmoidForward(const int N, Dtype* in, Dtype* out);

template <typename Dtype>
void SigmoidBackward(const int N, Dtype* in_diff,
                     Dtype* out_data, Dtype* out_diff);



template <>
void SigmoidForward<float>(const int N, float* in, float* out) {
  Concurrency::array_view<float, 1> inView =
    *((Concurrency::array_view<float, 1>*)(in));
  Concurrency::array_view<float, 1> outView =
    *((Concurrency::array_view<float, 1>*)(out));
  parallel_for_each(
     outView.get_extent(),
     [=](Concurrency::index<1> idx) restrict(amp) {
       outView[idx] = 1. / (1. + Concurrency::fast_math::exp(-inView[idx]));
     });
}

template <>
void SigmoidForward<double>(const int N, double* in, double* out) {
  Concurrency::array_view<double, 1> inView =
    *((Concurrency::array_view<double, 1>*)(in));
  Concurrency::array_view<double, 1> outView =
    *((Concurrency::array_view<double, 1>*)(out));
  parallel_for_each(
     outView.get_extent(),
     [=](Concurrency::index<1> idx) restrict(amp) {
       outView[idx] =
         1. / (1. + Concurrency::fast_math::exp(-inView[idx]));
     });
}


template <>
void SigmoidBackward<float>(const int N, float* in_diff,
                            float* out_data, float* out_diff) {
  Concurrency::array_view<float, 1> in_diffView =
    *((Concurrency::array_view<float, 1>*)(in_diff));
  Concurrency::array_view<float, 1> out_dataView =
    *((Concurrency::array_view<float, 1>*)(out_data));
  Concurrency::array_view<float, 1> out_diffView =
    *((Concurrency::array_view<float, 1>*)(out_diff));
  parallel_for_each(
    out_diffView.get_extent(),
    [=](Concurrency::index<1> idx) restrict(amp) {
      const float sigmoid_x = out_dataView[idx];
      out_diffView[idx] =
        in_diffView[idx] * sigmoid_x * (1 - sigmoid_x);
    });
}

template <>
void SigmoidBackward<double>(const int N, double* in_diff,
                             double* out_data, double* out_diff) {
  Concurrency::array_view<double, 1> in_diffView =
    *((Concurrency::array_view<double, 1>*)(in_diff));
  Concurrency::array_view<double, 1> out_dataView =
    *((Concurrency::array_view<double, 1>*)(out_data));
  Concurrency::array_view<double, 1> out_diffView =
    *((Concurrency::array_view<double, 1>*)(out_diff));
  parallel_for_each(
    out_diffView.get_extent(),
    [=](Concurrency::index<1> idx) restrict(amp) {
      const double sigmoid_x = out_dataView[idx];
      out_diffView[idx] = in_diffView[idx] * sigmoid_x * (1 - sigmoid_x);
    });
}

