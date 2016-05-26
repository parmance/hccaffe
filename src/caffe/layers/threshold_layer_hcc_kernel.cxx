#include "hc.hpp"
#include "hc_am.hpp"
#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void ThresholdForwardKernel(const int N, Dtype threshold,
                            Dtype* in, Dtype* out);
template <>
void ThresholdForwardKernel(const int N, float threshold,
                            float* in, float* out) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
        grdExt,
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
        out[idx[0]] = in[idx[0]] > threshold ? 1 : 0;
    }).wait();
}
template <>
void ThresholdForwardKernel(const int N, double threshold,
                            double* in, double* out) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo outInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&outInfo, out);
  long numOutElts = outInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numOutElts);
  parallel_for_each(
        grdExt,
        [=](hc::index<1> idx) __attribute__((hc, cpu)){
       out[idx[0]] = in[idx[0]] > threshold ? 1 : 0;
  }).wait();
}
