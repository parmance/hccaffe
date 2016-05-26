#include <algorithm>
#include <vector>
#include "hc.hpp"
#include "hc_am.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void ReLUForward(const int N, Dtype* in, Dtype* out,
  Dtype negative_slope);

template <typename Dtype>
void ReLUBackward(const int N, Dtype* in_diff,
  Dtype* in_data, Dtype* out_diff, Dtype negative_slope);


template <>
void ReLUForward(const int N, float* in, float* out,
  float negative_slope) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out[idx[0]] = in[idx[0]] > 0 ? in[idx[0]] : in[idx[0]] * negative_slope;
  }).wait();
}

template <>
void ReLUBackward(const int N, float* in_diff,
  float* in_data, float* out_diff, float negative_slope) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outDiffInfo, out_diff);
  long numOutElts = outDiffInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out_diff[idx[0]] = in_diff[idx[0]] * ((in_data[idx[0]] > 0)
      + (in_data[idx[0]] <= 0) * negative_slope);
  }).wait();
}

template <>
void ReLUForward(const int N, double* in, double* out,
  double negative_slope) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out[idx[0]] = in[idx[0]] > 0 ? in[idx[0]] : in[idx[0]] * negative_slope;
  }).wait();
}

template <>
void ReLUBackward(const int N, double* in_diff,
  double* in_data, double* out_diff, double negative_slope) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outDiffInfo, out_diff);
  long numOutElts = outDiffInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
    out_diff[idx[0]] = in_diff[idx[0]] * ((in_data[idx[0]] > 0)
      + (in_data[idx[0]] <= 0) * negative_slope);
  }).wait();
}






