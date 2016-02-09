#include <hc.hpp>
#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void SoftmaxLossForwardGPU(int N, const int nthreads,
                           Dtype* prob_data, Dtype* label, Dtype* loss,
                           const int num, const int dim, const int spatial_dim,
                           const bool has_ignore_label_,
                           const int ignore_label_, Dtype* counts);
template <typename Dtype>
void SoftmaxLossBackwardGPU(int N, const int nthreads, Dtype* top,
                            Dtype* label, Dtype* bottom_diff, const int num,
                            const int dim, const int spatial_dim,
                             const bool has_ignore_label_,
                            const int ignore_label_, Dtype* counts);
template <>
void SoftmaxLossForwardGPU(int N, const int nthreads,
                           float* prob_data, float* label, float* loss,
                           const int num, const int dim,
                           const int spatial_dim,
                           const bool has_ignore_label_,
                           const int ignore_label_,
                           float* counts) {
    hc::array_view<float, 1> probDataView =
      *((hc::array_view<float, 1>*)(prob_data));
    hc::array_view<float, 1> labelView =
      *((hc::array_view<float, 1>*)(label));
    hc::array_view<float, 1> countsView =
      *((hc::array_view<float, 1>*)(counts));
    hc::array_view<float, 1> lossView =
      *((hc::array_view<float, 1>*)(loss));

     hc::extent<1> e(nthreads);

    parallel_for_each(
        e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        float data_temp;
        int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && (label_value == ignore_label_)) {
            lossView[idx] = 0;
            countsView[idx] = 0;
        } else {
            data_temp = hc::fast_math::fmax(
            probDataView[n * dim + label_value * spatial_dim + s],
            static_cast<float>(FLT_MIN));
            lossView[idx] = -hc::fast_math::log(data_temp);
            countsView[idx] = 1;
        }
    });
}
template <>
void SoftmaxLossForwardGPU(int N, const int nthreads,
                           double* prob_data, double* label, double* loss,
                           const int num, const int dim, const int spatial_dim,
                           const bool has_ignore_label_,
                           const int ignore_label_,
                           double* counts) {
    hc::array_view<double, 1> probDataView =
      *((hc::array_view<double, 1>*)(prob_data));
    hc::array_view<double, 1> labelView =
      *((hc::array_view<double, 1>*)(label));
    hc::array_view<double, 1> countsView =
      *((hc::array_view<double, 1>*)(counts));
    hc::array_view<double, 1> lossView =
      *((hc::array_view<double, 1>*)(loss));

     hc::extent<1> e(nthreads);

    parallel_for_each(
        e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        double data_temp;
        int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && (label_value == ignore_label_)) {
            lossView[idx] = 0;
            countsView[idx] = 0;
        } else {
            data_temp = hc::fast_math::fmax(
              probDataView[n * dim + label_value * spatial_dim + s],
              static_cast<double>(FLT_MIN));
            lossView[idx] = -hc::fast_math::log(data_temp);
            lossView[idx] = -hc::fast_math::log(data_temp);
            countsView[idx] = 1;
        }
    });
}
template <>
void SoftmaxLossBackwardGPU(int N, const int nthreads,  float* top,
                            float* label, float* bottom_diff,
                            const int num, const int dim,
                            const int spatial_dim, const bool has_ignore_label_,
                            const int ignore_label_, float* counts) {
    const int channels = dim / spatial_dim;

    hc::array_view<float, 1> bottomDiffView =
      *((hc::array_view<float, 1>*)(bottom_diff));
    hc::array_view<float, 1> labelView =
      *((hc::array_view<float, 1>*)(label));
    hc::array_view<float, 1> countsView =
      *((hc::array_view<float, 1>*)(counts));

     hc::extent<1> e(nthreads);

    parallel_for_each(e,
                      [=](hc::index<1> idx) __attribute__((hc, cpu)){
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        const int label_value =
            static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < channels; ++c) {
                bottomDiffView[n * dim + c * spatial_dim + s] = 0;
            }
            countsView[idx] = 0;
        } else {
            bottomDiffView[n * dim + label_value * spatial_dim + s] -= 1;
            countsView[idx] = 1;
        }
    });
}
template <>
void SoftmaxLossBackwardGPU(int N, const int nthreads, double* top,
                            double* label, double* bottom_diff,
                            const int num, const int dim,
                            const int spatial_dim, const bool has_ignore_label_,
                            const int ignore_label_, double* counts) {
    const int channels = dim / spatial_dim;

    hc::array_view<double, 1> bottomDiffView =
      *((hc::array_view<double, 1>*)(bottom_diff));
    hc::array_view<double, 1> labelView =
      *((hc::array_view<double, 1>*)(label));
    hc::array_view<double, 1> countsView =
      *((hc::array_view<double, 1>*)(counts));

     hc::extent<1> e(nthreads);

    parallel_for_each(e,
                      [=](hc::index<1> idx) __attribute__((hc, cpu)){
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        const int label_value =
            static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < channels; ++c) {
                bottomDiffView[n * dim + c * spatial_dim + s] = 0;
            }
            countsView[idx] = 0;
        } else {
            bottomDiffView[n * dim + label_value * spatial_dim + s] -= 1;
            countsView[idx] = 1;
        }
    });
}
