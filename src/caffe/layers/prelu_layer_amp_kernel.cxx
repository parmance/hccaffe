#include <algorithm>
#include <vector>
#include <amp.h>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;
template <typename Dtype>
void PReLUForward(const int n, const int channels, const int dim,
                  Dtype* in, Dtype* out, Dtype* slope_data,
                  const int div_factor);
template <typename Dtype>
void PReLUBackward(const int n, const int channels, const int dim,
                   Dtype* in_diff, Dtype* in_data, Dtype* out_diff,
                   Dtype* slope_data, const int div_factor);
template <typename Dtype>
void PReLUParamBackward(const int n, Dtype* in_diff,
                        Dtype* in_data, Dtype* out_diff, int in_diff_offset, int in_data_offset);

template <>
void PReLUForward(const int n, const int channels, const int dim,
                  float* in, float* out, float* slope_data,
                  const int div_factor) {
    //array_view<float, 1> inView(n, in);
    array_view<float, 1> inView = *((Concurrency::array_view<float, 1>*)(in));
    //array_view<float, 1> outView(n, out);
    array_view<float, 1> outView = *((Concurrency::array_view<float, 1>*)(out));
    //array_view<float, 1> slopeDataView(n, slope_data);
    array_view<float, 1> slopeDataView = *((Concurrency::array_view<float, 1>*)(slope_data));
    parallel_for_each(
        outView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = (int)idx[0];//.global;
        int c = (index / dim) % channels / div_factor;
        outView[idx] = inView[idx] > 0 ? inView[idx] : inView[idx] * slopeDataView[c];
    }
    );
    //outView.synchronize();
}
template <>
void PReLUForward(const int n, const int channels, const int dim,
                  double* in, double* out, double* slope_data,
                  const int div_factor) {
   // array_view<double, 1> inView(n, in);
   array_view<double, 1> inView = *((Concurrency::array_view<double, 1>*)(in));
   // array_view<double, 1> outView(n, out);
   array_view<double, 1> outView = *((Concurrency::array_view<double, 1>*)(out));
   // array_view<double, 1> slopeDataView(n, slope_data);
   array_view<double, 1> slopeDataView = *((Concurrency::array_view<double, 1>*)(slope_data));
    parallel_for_each(
        outView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = (int)idx[0];//.global;
        int c = (index / dim) % channels / div_factor;
        outView[idx] = inView[idx] > 0 ? inView[idx] : inView[idx] * slopeDataView[c];
    }
    );
   // outView.synchronize();
}

template <>
void PReLUBackward(const int n, const int channels, const int dim,
                   float* in_diff, float* in_data, float* out_diff,
                   float* slope_data, const int div_factor) {
    //array_view<float, 1> inDataView(n, in_data);
    array_view<float, 1> inDataView = *((Concurrency::array_view<float, 1>*)(in_data));
    //array_view<float, 1> inDiffView(n, in_diff);
    array_view<float, 1> inDiffView = *((Concurrency::array_view<float, 1>*)(in_diff));
    //array_view<float, 1> outDiffView(n, out_diff);
    array_view<float, 1> outDiffView = *((Concurrency::array_view<float, 1>*)(out_diff));
    //array_view<float, 1> slopeDataView(n, slope_data);
    array_view<float, 1> slopeDataView = *((Concurrency::array_view<float, 1>*)(slope_data));
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {

        int c = (idx[0] / dim) % channels / div_factor;
        outDiffView[idx] = inDiffView[idx] * ((inDataView[idx] > 0)
                                              + (inDataView[idx] <= 0) * slopeDataView[c]);
    }
    );
    //outDiffView.synchronize();
}
template <>
void PReLUBackward(const int n, const int channels, const int dim,
                   double* in_diff, double* in_data, double* out_diff,
                   double* slope_data, const int div_factor) {
    //array_view<double, 1> inDataView(n, in_data);
    array_view<double, 1> inDataView = *((Concurrency::array_view<double, 1>*)(in_data));
    //array_view<double, 1> inDiffView(n, in_diff);
    array_view<double, 1> inDiffView = *((Concurrency::array_view<double, 1>*)(in_diff));
    //array_view<double, 1> outDiffView(n, out_diff);
    array_view<double, 1> outDiffView = *((Concurrency::array_view<double, 1>*)(out_diff));
    //array_view<double, 1> slopeDataView(n, slope_data);
    array_view<double, 1> slopeDataView = *((Concurrency::array_view<double, 1>*)(slope_data));
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {

        int c = (idx[0] / dim) % channels / div_factor;
        outDiffView[idx] = inDiffView[idx] * ((inDataView[idx] > 0)
                                              + (inDataView[idx] <= 0) * slopeDataView[c]);
    }
    );
    //outDiffView.synchronize();
}
template <>
void PReLUParamBackward(const int n, float* in_diff,
                        float* in_data, float* out_diff, int in_diff_offset, int in_data_offset) {
    //array_view<float, 1> inDataView(n, in_data);
    array_view<float, 1> inDataView = *((Concurrency::array_view<float, 1>*)(in_data));
    //array_view<float, 1> inDiffView(n, in_diff);
    array_view<float, 1> inDiffView = *((Concurrency::array_view<float, 1>*)(in_diff));
    //array_view<float, 1> outDiffView(n, out_diff);
    array_view<float, 1> outDiffView = *((Concurrency::array_view<float, 1>*)(out_diff));
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        outDiffView[idx] = inDiffView[idx + in_diff_offset] * inDataView[idx + in_data_offset] * (inDataView[idx + in_data_offset] <= 0);
    }
    );
    //outDiffView.synchronize();
}
template <>
void PReLUParamBackward(const int n, double* in_diff,
                        double* in_data, double* out_diff, int in_diff_offset, int in_data_offset) {
    //array_view<double, 1> inDataView(n, in_data);
    array_view<double, 1> inDataView = *((Concurrency::array_view<double, 1>*)(in_data));
    //array_view<double, 1> inDiffView(n, in_diff);
    array_view<double, 1> inDiffView = *((Concurrency::array_view<double, 1>*)(in_diff));
    //array_view<double, 1> outDiffView(n, out_diff);
    array_view<double, 1> outDiffView = *((Concurrency::array_view<double, 1>*)(out_diff));
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp) {
        outDiffView[idx] = inDiffView[idx + in_diff_offset] * inDataView[idx + in_data_offset] * (inDataView[idx + in_data_offset] <= 0);
    } );
   // outDiffView.synchronize();
}
