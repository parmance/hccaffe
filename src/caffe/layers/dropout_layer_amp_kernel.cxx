#include <algorithm>
#include <limits>
#include <vector>
#include <amp.h>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;
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
                    unsigned int* mask,int threshold, float scale,
                    float* out) {
    array_view<float, 1> inView(n, in);
    array_view<unsigned int, 1> maskView(n, mask);
    array_view<float, 1> outView(n, out);
    parallel_for_each(
        outView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        outView[idx] = inView[idx] * (maskView[idx] > threshold) * scale;
    }
    );
    outView.synchronize();
}
template <>
void DropoutForward(int n, double* in,
                    unsigned int* mask, int threshold, float scale,
                    double* out) {
    array_view<double, 1> inView(n, in);
    array_view<unsigned int, 1> maskView(n, mask);
    array_view<double, 1> outView(n, out);
    parallel_for_each(
        outView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        outView[idx] = inView[idx] * (maskView[idx] > threshold) * scale;
    }
    );
    outView.synchronize();
}
template <>
void DropoutBackward( int n,  float* in_diff,
                      unsigned int* mask,   int threshold,  float scale,
                      float* out_diff) {
    array_view<float, 1> inDiffView(n, in_diff);
    array_view<unsigned int, 1> maskView(n, mask);
    array_view<float, 1> outDiffView(n, out_diff);
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        outDiffView[idx] = inDiffView[idx] * scale * (maskView[idx] > threshold);
    }
    );
    outDiffView.synchronize();
}
template <>
void DropoutBackward(int n, double* in_diff,
                     unsigned int* mask, int threshold, float scale,
                     double* out_diff) {
    array_view<double, 1> inDiffView(n, in_diff);
    array_view<unsigned int, 1> maskView(n, mask);
    array_view<double, 1> outDiffView(n, out_diff);
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        outDiffView[idx] = inDiffView[idx] * scale * (maskView[idx] > threshold);
    }
    );
    outDiffView.synchronize();
}

