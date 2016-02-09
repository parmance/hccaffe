#include <hc.hpp>
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
  hc::array_view<float, 1> inView =
    *((hc::array_view<float, 1>*)(in));
  hc::array_view<float, 1> outView =
    *((hc::array_view<float, 1>*)(out));
  parallel_for_each(
        outView.get_extent(),
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        outView[idx] = inView[idx] > threshold ? 1 : 0;
    });
}
template <>
void ThresholdForwardKernel(const int N, double threshold,
                            double* in, double* out) {
  hc::array_view<double, 1> inView =
    *((hc::array_view<double, 1>*)(in));
  hc::array_view<double, 1> outView =
    *((hc::array_view<double, 1>*)(out));
  parallel_for_each(
        outView.get_extent(),
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
       outView[idx] = inView[idx] > threshold ? 1 : 0;
  });
}
