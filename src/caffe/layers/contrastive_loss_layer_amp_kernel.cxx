#include <algorithm>
#include <vector>
#include "hc.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

template <typename Dtype>
void CLLForward(const int N,
                const int channels,
                const Dtype margin,
                const Dtype alpha,
                const int y_count,
                Dtype* y,
                Dtype* diff,
                Dtype* dist_sq,
                Dtype* bottom_diff);


template <>
void CLLForward<float>(const int N,
    const int channels,
    const float margin,
    const float alpha,
    const int y_count,
    float* y,
    float* diff,
    float* dist_sq,
    float* bottom_diff) {

  hc::array_view<float, 1> yView =
    *((hc::array_view<float, 1>*)(y));
  hc::array_view<float, 1> diffView =
    *((hc::array_view<float, 1>*)(diff));
  hc::array_view<float, 1> dist_sqView =
    *((hc::array_view<float, 1>*)(dist_sq));
  hc::array_view<float, 1> bottom_diffView =
    *((hc::array_view<float, 1>*)(bottom_diff));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / channels;  // the num index, to access y and dist_sq
      if (static_cast<int>(yView[n])) {  // similar pairsS
        bottom_diffView[idx] = alpha * diffView[idx];
      } else {  // dissimilar pairs
        if ((margin - dist_sqView[n]) > 0.0) {
          bottom_diffView[idx] = -alpha * diffView[idx];
        } else {
          bottom_diffView[idx] = 0;
        }
      }
    });
}

template <>
void CLLForward<double>(const int N,
    const int channels,
    const double margin,
    const double alpha,
    const int y_count,
    double* y,
    double* diff,
    double* dist_sq,
    double* bottom_diff) {

  hc::array_view<double, 1> yView =
    *((hc::array_view<double, 1>*)(y));
  hc::array_view<double, 1> diffView =
    *((hc::array_view<double, 1>*)(diff));
  hc::array_view<double, 1> dist_sqView =
    *((hc::array_view<double, 1>*)(dist_sq));
  hc::array_view<double, 1> bottom_diffView =
    *((hc::array_view<double, 1>*)(bottom_diff));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / channels;  // the num index, to access y and dist_sq
      if (static_cast<int>(yView[n])) {  // similar pairsS
        bottom_diffView[idx] = alpha * diffView[idx];
      } else {  // dissimilar pairs
        if ((margin - dist_sqView[n]) > 0.0) {
          bottom_diffView[idx] = -alpha * diffView[idx];
        } else {
          bottom_diffView[idx] = 0;
        }
      }
    });
}

