#include <amp.h>
#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void ThresholdForwardKernel(const int N, Dtype threshold,
                            Dtype* in, Dtype* out);
template <>
void ThresholdForwardKernel(const int N, float threshold,
                            float* in, float* out) {
  Concurrency::array_view<float, 1> inView =
               *((Concurrency::array_view<float, 1>*)(in));
  Concurrency::array_view<float, 1> outView =
               *((Concurrency::array_view<float, 1>*)(out));
  parallel_for_each(
        outView.get_extent(),
        [=](Concurrency::index<1> idx) restrict(amp){
        outView[idx] = inView[idx] > threshold ? 1 : 0;
    });
}
template <>
void ThresholdForwardKernel(const int N, double threshold,
                            double* in, double* out) {
  Concurrency::array_view<double, 1> inView =
               *((Concurrency::array_view<double, 1>*)(in));
  Concurrency::array_view<double, 1> outView =
               *((Concurrency::array_view<double, 1>*)(out));
  parallel_for_each(
        outView.get_extent(),
        [=](Concurrency::index<1> idx) restrict(amp){
       outView[idx] = inView[idx] > threshold ? 1 : 0;
  });
}
