#include "hc.hpp"
#include "hc_am.hpp"
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
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = static_cast<int>(idx[0]);
    int c = (index / dim) % channels / div_factor;
    out[idx[0]] = in[idx[0]] > 0 ? in[idx[0]]
                   : in[idx[0]] * slope_data[c];
  }).wait();
}
template <>
void PReLUForward(const int n, const int channels, const int dim,
                  double *in, double *out, double *slope_data,
                  const int div_factor) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = static_cast<int>(idx[0]);
    int c = (index / dim) % channels / div_factor;
    out[idx[0]] = in[idx[0]] > 0 ? in[idx[0]]
                   : in[idx[0]] * slope_data[c];
  }).wait();
}

template <>
void PReLUBackward(const int n, const int channels, const int dim,
                   float *in_diff, float *in_data, float *out_diff,
                   float *slope_data, const int div_factor) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out_diff);
  long numOutElts = outInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
  grdExt,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    int c = (idx[0] / dim) % channels / div_factor;
    out_diff[idx[0]] = in_diff[idx[0]] * ((in_data[idx[0]] > 0)
                       + (in_data[idx[0]] <= 0) * slope_data[c]);
  }).wait();
}
template <>
void PReLUBackward(const int n, const int channels, const int dim,
                   double *in_diff, double *in_data, double *out_diff,
                   double *slope_data, const int div_factor) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out_diff);
  long numOutElts = outInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    int c = (idx[0] / dim) % channels / div_factor;
    out_diff[idx[0]] = in_diff[idx[0]] * ((in_data[idx[0]] > 0)
                       + (in_data[idx[0]] <= 0) * slope_data[c]);
  }).wait();
}
template <>
void PReLUParamBackward(const int n, float *in_diff,
                        float *in_data, float *out_diff,
                        int in_diff_offset, int in_data_offset) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out_diff);
  long numOutElts = outInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    out_diff[idx[0]] = in_diff[idx[0] + in_diff_offset]
                       * in_data[idx[0] + in_data_offset]
                       * (in_data[idx[0] + in_data_offset] <= 0);
  }).wait();
}
template <>
void PReLUParamBackward(const int n, double *in_diff,
                        double *in_data, double *out_diff,
                        int in_diff_offset, int in_data_offset) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out_diff);
  long numOutElts = outInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    out_diff[idx[0]] = in_diff[idx[0] + in_diff_offset]
                       * in_data[idx[0] + in_data_offset]
                       * (in_data[idx[0] + in_data_offset] <= 0);
  }).wait();
}
