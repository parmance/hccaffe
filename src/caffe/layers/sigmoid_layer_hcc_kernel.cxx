#include "hc.hpp"
#include "hc_am.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

template <typename Dtype>
void SigmoidForward(const int N, Dtype* in, Dtype* out);

template <typename Dtype>
void SigmoidBackward(const int N, Dtype* in_diff,
                     Dtype* out_data, Dtype* out_diff);



template <>
void SigmoidForward<float>(const int N, float* in, float* out) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
     grdExt,
     [=](hc::index<1> idx) __attribute__((hc, cpu)) {
       out[idx[0]] = 1. / (1. + hc::fast_math::exp(-in[idx[0]]));
     }).wait();
}

template <>
void SigmoidForward<double>(const int N, double* in, double* out) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
     grdExt,
     [=](hc::index<1> idx) __attribute__((hc, cpu)) {
       out[idx[0]] =
         1. / (1. + hc::fast_math::exp(-in[idx[0]]));
   }).wait();
}


template <>
void SigmoidBackward<float>(const int N, float* in_diff,
                            float* out_data, float* out_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outDiffInfo, out_diff);
  long numOutDiffElts = outDiffInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutDiffElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      const float sigmoid_x = out_data[idx[0]];
      out_diff[idx[0]] =
        in_diff[idx[0]] * sigmoid_x * (1 - sigmoid_x);
    }).wait();
}

template <>
void SigmoidBackward<double>(const int N, double* in_diff,
                             double* out_data, double* out_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outDiffInfo, out_diff);
  long numOutDiffElts = outDiffInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutDiffElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      const double sigmoid_x = out_data[idx[0]];
      out_diff[idx[0]] = in_diff[idx[0]] * sigmoid_x * (1 - sigmoid_x);
    }).wait();
}

