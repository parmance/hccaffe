#include <amp.h>
#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void PReLUForward(const int n, const int channels, const int dim,
                  Dtype *in, Dtype *out, Dtype *slope_data,
                  const int div_factor);
template <typename Dtype>
void PReLUBackward(const int n, const int channels, const int dim,
                   Dtype *in_diff, Dtype *in_data, Dtype *out_diff,
                   Dtype *slope_data, const int div_factor);
template <typename Dtype>
void PReLUParamBackward(const int n, Dtype *in_diff,
                        Dtype *in_data, Dtype *out_diff,
                        int in_diff_offset, int in_data_offset);

template <>
void PReLUForward(const int n, const int channels, const int dim,
                  float *in, float *out, float *slope_data,
                  const int div_factor) {
  Concurrency::array_view<float, 1> inView =
    *((Concurrency::array_view<float, 1> *)(in));
  Concurrency::array_view<float, 1> outView =
    *((Concurrency::array_view<float, 1> *)(out));
  Concurrency::array_view<float, 1> slopeDataView =
    *((Concurrency::array_view<float, 1> *)(slope_data));
  parallel_for_each(
    outView.get_extent(),
  [ = ](Concurrency::index<1> idx) restrict(amp) {
    int index = static_cast<int>(idx[0]);
    int c = (index / dim) % channels / div_factor;
    outView[idx] = inView[idx] > 0 ? inView[idx]
                   : inView[idx] * slopeDataView[c];
  });
}
template <>
void PReLUForward(const int n, const int channels, const int dim,
                  double *in, double *out, double *slope_data,
                  const int div_factor) {
  Concurrency::array_view<double, 1> inView =
    *((Concurrency::array_view<double, 1> *)(in));
  Concurrency::array_view<double, 1> outView =
    *((Concurrency::array_view<double, 1> *)(out));
  Concurrency::array_view<double, 1> slopeDataView =
    *((Concurrency::array_view<double, 1> *)(slope_data));
  parallel_for_each(
    outView.get_extent(),
  [ = ](Concurrency::index<1> idx) restrict(amp) {
    int index = static_cast<int>(idx[0]);
    int c = (index / dim) % channels / div_factor;
    outView[idx] = inView[idx] > 0 ? inView[idx]
                   : inView[idx] * slopeDataView[c];
  });
}

template <>
void PReLUBackward(const int n, const int channels, const int dim,
                   float *in_diff, float *in_data, float *out_diff,
                   float *slope_data, const int div_factor) {
  Concurrency::array_view<float, 1> inDataView =
    *((Concurrency::array_view<float, 1> *)(in_data));
  Concurrency::array_view<float, 1> inDiffView =
    *((Concurrency::array_view<float, 1> *)(in_diff));
  Concurrency::array_view<float, 1> outDiffView =
    *((Concurrency::array_view<float, 1> *)(out_diff));
  Concurrency::array_view<float, 1> slopeDataView =
    *((Concurrency::array_view<float, 1> *)(slope_data));
  parallel_for_each(
    outDiffView.get_extent(),
  [ = ](Concurrency::index<1> idx) restrict(amp) {
    int c = (idx[0] / dim) % channels / div_factor;
    outDiffView[idx] = inDiffView[idx] * ((inDataView[idx] > 0)
                       + (inDataView[idx] <= 0) * slopeDataView[c]);
  });
}
template <>
void PReLUBackward(const int n, const int channels, const int dim,
                   double *in_diff, double *in_data, double *out_diff,
                   double *slope_data, const int div_factor) {
  Concurrency::array_view<double, 1> inDataView =
    *((Concurrency::array_view<double, 1> *)(in_data));
  Concurrency::array_view<double, 1> inDiffView =
    *((Concurrency::array_view<double, 1> *)(in_diff));
  Concurrency::array_view<double, 1> outDiffView =
    *((Concurrency::array_view<double, 1> *)(out_diff));
  Concurrency::array_view<double, 1> slopeDataView =
    *((Concurrency::array_view<double, 1> *)(slope_data));
  parallel_for_each(
    outDiffView.get_extent(),
  [ = ](Concurrency::index<1> idx) restrict(amp) {
    int c = (idx[0] / dim) % channels / div_factor;
    outDiffView[idx] = inDiffView[idx] * ((inDataView[idx] > 0)
                       + (inDataView[idx] <= 0) * slopeDataView[c]);
  });
}
template <>
void PReLUParamBackward(const int n, float *in_diff,
                        float *in_data, float *out_diff,
                        int in_diff_offset, int in_data_offset) {
  Concurrency::array_view<float, 1> inDataView =
    *((Concurrency::array_view<float, 1> *)(in_data));
  Concurrency::array_view<float, 1> inDiffView =
    *((Concurrency::array_view<float, 1> *)(in_diff));
  Concurrency::array_view<float, 1> outDiffView =
    *((Concurrency::array_view<float, 1> *)(out_diff));
  parallel_for_each(
    outDiffView.get_extent(),
  [ = ](Concurrency::index<1> idx) restrict(amp) {
    outDiffView[idx] = inDiffView[idx + in_diff_offset]
                       * inDataView[idx + in_data_offset]
                       * (inDataView[idx + in_data_offset] <= 0);
  });
}
template <>
void PReLUParamBackward(const int n, double *in_diff,
                        double *in_data, double *out_diff,
                        int in_diff_offset, int in_data_offset) {
  Concurrency::array_view<double, 1> inDataView =
    *((Concurrency::array_view<double, 1> *)(in_data));
  Concurrency::array_view<double, 1> inDiffView =
    *((Concurrency::array_view<double, 1> *)(in_diff));
  Concurrency::array_view<double, 1> outDiffView =
    *((Concurrency::array_view<double, 1> *)(out_diff));
  parallel_for_each(
    outDiffView.get_extent(),
  [ = ](Concurrency::index<1> idx) restrict(amp) {
    outDiffView[idx] = inDiffView[idx + in_diff_offset]
                       * inDataView[idx + in_data_offset]
                       * (inDataView[idx + in_data_offset] <= 0);
  });
}
