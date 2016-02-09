#include "hc.hpp"
#include <algorithm>
#include <limits>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void DropoutForward(int n, Dtype* in,
                    unsigned int* mask, int threshold, float scale,
                    Dtype* out);
template <typename Dtype>
void DropoutBackward(int n, Dtype* in_diff,
                     unsigned int* mask, int threshold, float scale,
                     Dtype* out_diff);
template <>
void DropoutForward(int n,  float* in,
                    unsigned int* mask, int threshold, float scale,
                    float* out) {
    hc::array_view<float, 1> inView =
      *((hc::array_view<float, 1>*)(in));
    hc::array_view<unsigned int, 1> maskView =
      *((hc::array_view<unsigned int, 1>*)(mask));
    hc::array_view<float, 1> outView =
      *((hc::array_view<float, 1>*)(out));
    parallel_for_each(
        outView.get_extent(),
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        outView[idx] =
          inView[idx] * (maskView[idx] > threshold) * scale;
    });
}
template <>
void DropoutForward(int n, double* in,
                    unsigned int* mask, int threshold, float scale,
                    double* out) {
    hc::array_view<double, 1> inView =
      *((hc::array_view<double, 1>*)(in));
    hc::array_view<unsigned int, 1> maskView =
      *((hc::array_view<unsigned int, 1>*)(mask));
    hc::array_view<double, 1> outView =
      *((hc::array_view<double, 1>*)(out));

    parallel_for_each(
        outView.get_extent(),
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        outView[idx] = inView[idx] * (maskView[idx] > threshold) * scale;
    });
}
template <>
void DropoutBackward(int n,  float* in_diff,
                     unsigned int* mask,   int threshold,  float scale,
                     float* out_diff) {
    hc::array_view<float, 1> inDiffView =
      *((hc::array_view<float, 1>*)(in_diff));
    hc::array_view<unsigned int, 1> maskView =
      *((hc::array_view<unsigned int, 1>*)(mask));
    hc::array_view<float, 1> outDiffView =
      *((hc::array_view<float, 1>*)(out_diff));
    parallel_for_each(
        outDiffView.get_extent(),
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        outDiffView[idx] =
          inDiffView[idx] * scale * (maskView[idx] > threshold);
    });
}
template <>
void DropoutBackward(int n, double* in_diff,
                     unsigned int* mask, int threshold, float scale,
                     double* out_diff) {
    hc::array_view<double, 1> inDiffView =
      *((hc::array_view<double, 1>*)(in_diff));
    hc::array_view<unsigned int, 1> maskView =
      *((hc::array_view<unsigned int, 1>*)(mask));
    hc::array_view<double, 1> outDiffView =
      *((hc::array_view<double, 1>*)(out_diff));
    parallel_for_each(
        outDiffView.get_extent(),
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        outDiffView[idx] =
          inDiffView[idx] * scale * (maskView[idx] > threshold);
    });
}

