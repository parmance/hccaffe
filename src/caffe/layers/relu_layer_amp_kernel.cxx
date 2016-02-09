#include <algorithm>
#include <vector>
#include "hc.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void ReLUForward(const int N, Dtype* in, Dtype* out,
  Dtype negative_slope);

template <typename Dtype>
void ReLUBackward(const int N, Dtype* in_diff,
  Dtype* in_data, Dtype* out_diff, Dtype negative_slope);


template <>
void ReLUForward(const int N, float* in, float* out,
  float negative_slope) {
  hc::array_view<float, 1> inView =
    *((hc::array_view<float, 1>*)(in));
  hc::array_view<float, 1> outView =
    *((hc::array_view<float, 1>*)(out));
  parallel_for_each(
    outView.get_extent(),
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    outView[idx] = inView[idx] > 0 ? inView[idx] : inView[idx] * negative_slope;
  });
}

template <>
void ReLUBackward(const int N, float* in_diff,
  float* in_data, float* out_diff, float negative_slope) {
  hc::array_view<float, 1> in_diffView =
    *((hc::array_view<float, 1>*)(in_diff));
  hc::array_view<float, 1> out_diffView =
    *((hc::array_view<float, 1>*)(out_diff));
  hc::array_view<float, 1> int_dataView =
    *((hc::array_view<float, 1>*)(in_data));
  parallel_for_each(
    out_diffView.get_extent(),
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out_diffView[idx] = in_diffView[idx] * ((int_dataView[idx] > 0)
      + (int_dataView[idx] <= 0) * negative_slope);
  });
}

template <>
void ReLUForward(const int N, double* in, double* out,
  double negative_slope) {
  hc::array_view<double, 1> inView =
    *((hc::array_view<double, 1>*)(in));
  hc::array_view<double, 1> outView =
    *((hc::array_view<double, 1>*)(out));

  parallel_for_each(
    outView.get_extent(),
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    outView[idx] = inView[idx] > 0 ? inView[idx] : inView[idx] * negative_slope;
  });
}

template <>
void ReLUBackward(const int N, double* in_diff,
  double* in_data, double* out_diff, double negative_slope) {
  hc::array_view<double, 1> in_diffView =
    *((hc::array_view<double, 1>*)(in_diff));
  hc::array_view<double, 1> out_diffView =
    *((hc::array_view<double, 1>*)(out_diff));
  hc::array_view<double, 1> int_dataView =
    *((hc::array_view<double, 1>*)(in_data));

  parallel_for_each(
    out_diffView.get_extent(),
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out_diffView[idx] = in_diffView[idx] * ((int_dataView[idx] > 0) +
      (int_dataView[idx] <= 0) * negative_slope);
  });
}






