#include "hc.hpp"
#include "hc_am.hpp"
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
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
        grdExt,
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        out[idx[0]] =
          in[idx[0]] * (mask[idx[0]] > threshold) * scale;
   }).wait();
}
template <>
void DropoutForward(int n, double* in,
                    unsigned int* mask, int threshold, float scale,
                    double* out) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
        grdExt,
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        out[idx[0]] = in[idx[0]] * (mask[idx[0]] > threshold) * scale;
   }).wait();
}
template <>
void DropoutBackward(int n,  float* in_diff,
                     unsigned int* mask,   int threshold,  float scale,
                     float* out_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outDiffInfo, out_diff);
  long numOutDiffElts = outDiffInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutDiffElts);
  parallel_for_each(
        grdExt,
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        out_diff[idx[0]] =
          in_diff[idx[0]] * scale * (mask[idx[0]] > threshold);
   }).wait();
}
template <>
void DropoutBackward(int n, double* in_diff,
                     unsigned int* mask, int threshold, float scale,
                     double* out_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outDiffInfo, out_diff);
  long numOutDiffElts = outDiffInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutDiffElts);
    parallel_for_each(
        grdExt,
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        out_diff[idx[0]] =
          in_diff[idx[0]] * scale * (mask[idx[0]] > threshold);
    }).wait();
}

