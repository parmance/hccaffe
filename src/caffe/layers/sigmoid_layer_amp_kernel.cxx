#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;



template <typename Dtype>
void SigmoidForward(const int N, Dtype* in, Dtype* out);

template <typename Dtype>
void SigmoidBackward(const int N, Dtype* in_diff, Dtype* out_data, Dtype* out_diff);



template <>
void SigmoidForward<float>(const int N, float* in, float* out) {
  //array_view<float, 1> inView(N, in);
  array_view<float, 1> inView = *((Concurrency::array_view<float, 1>*)(in));
  //array_view<float, 1> outView(N, out);
  array_view<float, 1> outView = *((Concurrency::array_view<float, 1>*)(out));
  parallel_for_each(
     outView.get_extent(),
     [=](index<1> idx) restrict(amp)
     {
       outView[idx] = 1. / (1. + Concurrency::fast_math::exp(-inView[idx]));
     }
  );
  //outView.synchronize();
}

template <>
void SigmoidForward<double>(const int N, double* in, double* out) {
  //array_view<double, 1> inView(N, in);
  array_view<double, 1> inView = *((Concurrency::array_view<double, 1>*)(in));
  //array_view<double, 1> outView(N, out);
  array_view<double, 1> outView = *((Concurrency::array_view<double, 1>*)(out));
  parallel_for_each(
     outView.get_extent(),
     [=](index<1> idx) restrict(amp)
     {
       outView[idx] = 1. / (1. + Concurrency::fast_math::exp(-inView[idx]));
     }
  );
  //outView.synchronize();
}


template <>
void SigmoidBackward<float>(const int N, float* in_diff, float* out_data, float* out_diff) {
  //array_view<float, 1> in_diffView(N, in_diff);
  array_view<float, 1> in_diffView = *((Concurrency::array_view<float, 1>*)(in_diff));
  //array_view<float, 1> out_dataView(N, out_data);
  array_view<float, 1> out_dataView = *((Concurrency::array_view<float, 1>*)(out_data));
  //array_view<float, 1> out_diffView(N, out_diff);
  array_view<float, 1> out_diffView = *((Concurrency::array_view<float, 1>*)(out_diff));
  parallel_for_each(
    out_diffView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      const float sigmoid_x = out_dataView[idx];
      out_diffView[idx] = in_diffView[idx] * sigmoid_x * (1 - sigmoid_x);
    }
  );
  //out_diffView.synchronize();
}

template <>
void SigmoidBackward<double>(const int N, double* in_diff, double* out_data, double* out_diff) {
  //array_view<double, 1> in_diffView(N, in_diff);
  array_view<double, 1> in_diffView = *((Concurrency::array_view<double, 1>*)(in_diff));
  //array_view<double, 1> out_dataView(N, out_data);
  array_view<double, 1> out_dataView = *((Concurrency::array_view<double, 1>*)(out_data));
  //array_view<double, 1> out_diffView(N, out_diff);
  array_view<double, 1> out_diffView = *((Concurrency::array_view<double, 1>*)(out_diff));
  parallel_for_each(
    out_diffView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      const double sigmoid_x = out_dataView[idx];
      out_diffView[idx] = in_diffView[idx] * sigmoid_x * (1 - sigmoid_x);
    }
  );
  //out_diffView.synchronize();
}

