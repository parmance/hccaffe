#include <algorithm>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <amp.h>
#include <amp_math.h>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using namespace concurrency;
template <typename Dtype>
void SoftmaxLossForwardGPU(const int nthreads,
                           Dtype* prob_data, Dtype* label, Dtype* loss,
                           const int num, const int dim, const int spatial_dim,
                           const bool has_ignore_label_, const int ignore_label_,
                           Dtype* counts);
template <typename Dtype>
void SoftmaxLossBackwardGPU(const int nthreads, Dtype* top,
                            Dtype* label, Dtype* bottom_diff, const int num, const int dim,
                            const int spatial_dim, const bool has_ignore_label_,
                            const int ignore_label_, Dtype* counts);
template <>
void SoftmaxLossForwardGPU(const int nthreads,
                           float* prob_data, float* label, float* loss,
                           const int num, const int dim, const int spatial_dim,
                           const bool has_ignore_label_, const int ignore_label_,
                           float* counts) {
    array_view<float, 1> probDataView(nthreads, prob_data);
    array_view<float, 1> labelView(nthreads, label);
    array_view<float, 1> countsView(nthreads, counts);
    array_view<float, 1> lossView(nthreads, loss);
    parallel_for_each(
        lossView.get_extent(),
    [=](index<1> idx) restrict(amp) {
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        float data_temp;
        int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && (label_value == ignore_label_)) {
            lossView[idx] = 0;
            countsView[idx] = 0;
        } else {
            data_temp = max(probDataView[n * dim + label_value * spatial_dim + s], float(FLT_MIN));
            lossView[idx] = -concurrency::fast_math::log(data_temp);
            countsView[idx] = 1;
        }
    }
    );
    lossView.synchronize();
    countsView.synchronize();
}
template <>
void SoftmaxLossForwardGPU(const int nthreads,
                           double* prob_data, double* label, double* loss,
                           const int num, const int dim, const int spatial_dim,
                           const bool has_ignore_label_, const int ignore_label_,
                           double* counts) {
    array_view<double, 1> probDataView(nthreads, prob_data);
    array_view<double, 1> labelView(nthreads, label);
    array_view<double, 1> countsView(nthreads, counts);
    array_view<double, 1> lossView(nthreads, loss);
    parallel_for_each(
        lossView.get_extent(),
    [=](index<1> idx) restrict(amp) {
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        double data_temp;
        int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && (label_value == ignore_label_)) {
            lossView[idx] = 0;
            countsView[idx] = 0;
        }
        else {
            data_temp = max(probDataView[n * dim + label_value * spatial_dim + s], double(FLT_MIN));
            lossView[idx] = -concurrency::fast_math::log(data_temp);
            countsView[idx] = 1;
        }
    }
    );
    lossView.synchronize();
    countsView.synchronize();
}
template <>
void SoftmaxLossBackwardGPU(const int nthreads,  float* top,
                            float* label, float* bottom_diff, const int num, const int dim,
                            const int spatial_dim, const bool has_ignore_label_,
                            const int ignore_label_, float* counts) {
    const int channels = dim / spatial_dim;
    array_view<float, 1> labelView(nthreads, label);
    array_view<float, 1> bottomDiffView(nthreads, bottom_diff);
    array_view<float, 1> countsView(nthreads, counts);
    parallel_for_each(bottomDiffView.get_extent(),
                      [=](index<1> idx) restrict(amp)
    {
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        const int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < channels; ++c) {
                bottomDiffView[n * dim + c * spatial_dim + s] = 0;
            }
            countsView[idx] = 0;
        } else {
            bottomDiffView[n * dim + label_value * spatial_dim + s] -= 1;
            countsView[idx] = 1;
        }
    }
                     );
    bottomDiffView.synchronize();
    countsView.synchronize();
}
template <>
void SoftmaxLossBackwardGPU(const int nthreads, double* top,
                            double* label, double* bottom_diff, const int num, const int dim,
                            const int spatial_dim, const bool has_ignore_label_,
                            const int ignore_label_, double* counts) {
    const int channels = dim / spatial_dim;
    array_view<double, 1> labelView(nthreads, label);
    array_view<double, 1> bottomDiffView(nthreads, bottom_diff);
    array_view<double, 1> countsView(nthreads, counts);
    parallel_for_each(bottomDiffView.get_extent(),
                      [=](index<1> idx) restrict(amp)
    {
        const int n = idx[0] / spatial_dim;
        const int s = idx[0] % spatial_dim;
        const int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
        if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < channels; ++c) {
                bottomDiffView[n * dim + c * spatial_dim + s] = 0;
            }
            countsView[idx] = 0;
        }
        else {
            bottomDiffView[n * dim + label_value * spatial_dim + s] -= 1;
            countsView[idx] = 1;
        }
    }
                     );
    bottomDiffView.synchronize();
    countsView.synchronize();
}
