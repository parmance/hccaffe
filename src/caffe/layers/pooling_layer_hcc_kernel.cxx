#include "hc.hpp"
#include "hc_am.hpp"
#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
template <typename Dtype>
void MaxPoolForward(int top_count, int boottom_count, Dtype *bottom_data,
                    const int num, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width, const int kernel_h,
                    const int kernel_w, const int stride_h,
                    const int stride_w, const int pad_h,
                    const int pad_w, Dtype *top_data,
                    int *mask, Dtype *top_mask);
template <typename Dtype>
void AvePoolForward(int top_count, int boottom_count, Dtype *bottom_data,
                    const int num, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width, const int kernel_h,
                    const int kernel_w, const int stride_h,
                    const int stride_w, const int pad_h,
                    const int pad_w, Dtype *top_data);
template <typename Dtype>
void StoPoolForwardTrain(int top_count, int boottom_count,
                         Dtype *bottom_data,
                         const int num, const int channels, const int height,
                         const int width, const int pooled_height,
                         const int pooled_width, const int kernel_h,
                         const int kernel_w, const int stride_h,
                         const int stride_w, Dtype *rand_idx, Dtype *top_data);
template <typename Dtype>
void StoPoolForwardTest(int top_count, int boottom_count,
                        Dtype *bottom_data,
                        const int num, const int channels, const int height,
                        const int width, const int pooled_height,
                        const int pooled_width, const int kernel_h,
                        const int kernel_w, const int stride_h,
                        const int stride_w, Dtype *top_data);
template <typename Dtype>
void MaxPoolBackward(int top_count, int boottom_count, Dtype *top_diff,
                     int *mask, Dtype *top_mask, const int num,
                     const int channels, const int height,
                     const int width, const int pooled_height,
                     const int pooled_width, const int kernel_h,
                     const int kernel_w, const int stride_h,
                     const int stride_w, const int pad_h, const int pad_w,
                     Dtype *bottom_diff);
template <typename Dtype>
void AvePoolBackward(int top_count, int boottom_count, Dtype *top_diff,
                     const int num, const int channels, const int height,
                     const int width, const int pooled_height,
                     const int pooled_width, const int kernel_h,
                     const int kernel_w, const int stride_h,
                     const int stride_w, const int pad_h, const int pad_w,
                     Dtype *bottom_diff);
template <typename Dtype>
void StoPoolBackward(int top_count, int boottom_count,
                     Dtype *rand_idx, Dtype *top_diff,
                     const int num, const int channels, const int height,
                     const int width, const int pooled_height,
                     const int pooled_width, const int kernel_h,
                     const int kernel_w, const int stride_h,
                     const int stride_w, Dtype *bottom_diff);
template <>
void MaxPoolForward(int top_count, int boottom_count, float *bottom_data,
                    const int num, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width, const int kernel_h,
                    const int kernel_w, const int stride_h,
                    const int stride_w, const int pad_h,
                    const int pad_w, float *top_data,
                    int *mask, float *top_mask) {

  bool maskTag = (mask == NULL ? false : true);
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numTopElts);

  if (maskTag) {
    //     delte mask;
    parallel_for_each(
      grdExt,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = hc::fast_math::fmin(hstart + kernel_h, height);
      int wend = hc::fast_math::fmin(wstart + kernel_w, width);
      hstart = hc::fast_math::fmax(hstart, 0);
      wstart = hc::fast_math::fmax(wstart, 0);
      float maxval = -FLT_MAX;
      double maxidx = -1;
      int locan = (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (bottom_data[locan + h * width + w] > maxval) {
            maxidx = h * width + w;
            maxval = bottom_data[locan + (int)maxidx];
          }
        }
      }
      top_data[index] = maxval;
      mask[index] = maxidx;
    }).wait();
  } else {
    parallel_for_each(
      grdExt,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = hc::fast_math::fmin(hstart + kernel_h, height);
      int wend = hc::fast_math::fmin(wstart + kernel_w, width);
      hstart = hc::fast_math::fmax(hstart, 0);
      wstart = hc::fast_math::fmax(wstart, 0);
      float maxval = -FLT_MAX;
      double maxidx = -1;
      int locan = (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (bottom_data[locan + h * width + w] > maxval) {
            maxidx = h * width + w;
            maxval = bottom_data[locan + (int)maxidx];
          }
        }
      }
      top_data[index] = maxval;
      top_mask[index] = maxidx;
    }).wait();
  }
}
template <>
void MaxPoolForward(int top_count,  int boottom_count, double *bottom_data,
                    const int num, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width,
                    const int kernel_h, const int kernel_w,
                    const int stride_h,
                    const int stride_w, const int pad_h, const int pad_w,
                    double *top_data,
                    int *mask, double *top_mask) {
  bool maskTag = (mask == NULL ? false : true);
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numTopElts);
  if (maskTag) {
    parallel_for_each(
      grdExt,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = hc::fast_math::fmin(hstart + kernel_h, height);
      int wend = hc::fast_math::fmin(wstart + kernel_w, width);
      hstart = hc::fast_math::fmax(hstart, 0);
      wstart = hc::fast_math::fmax(wstart, 0);
      double maxval = -FLT_MAX;
      double maxidx = -1;
      int locan = (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (bottom_data[locan + h * width + w] > maxval) {
            maxidx = h * width + w;
            maxval = bottom_data[locan + (int)maxidx];
          }
        }
      }
      top_data[index] = maxval;
      mask[index] = maxidx;
    }).wait();
  } else {
    parallel_for_each(
      grdExt,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = hc::fast_math::fmin(hstart + kernel_h, height);
      int wend = hc::fast_math::fmin(wstart + kernel_w, width);
      hstart = hc::fast_math::fmax(hstart, 0);
      wstart = hc::fast_math::fmax(wstart, 0);
      double maxval = -FLT_MAX;
      double maxidx = -1;
      int locan = (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (bottom_data[locan + h * width + w] > maxval) {
            maxidx = h * width + w;
            maxval = bottom_data[locan + (int)maxidx];
          }
        }
      }
      top_data[index] = maxval;
      top_mask[index] = maxidx;
    }).wait();
  }
}
template <>
void AvePoolForward(int top_count, int boottom_count, float *bottom_data,
                    const int num, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width,
                    const int kernel_h, const int kernel_w, const int stride_h,
                    const int stride_w, const int pad_h, const int pad_w,
                    float *top_data) {

  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numTopElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = hc::fast_math::fmin(hstart + kernel_h, height + pad_h);
    int wend = hc::fast_math::fmin(wstart + kernel_w, width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = hc::fast_math::fmax(hstart, 0);
    wstart = hc::fast_math::fmax(wstart, 0);
    hend =  hc::fast_math::fmin(hend, height);
    wend =  hc::fast_math::fmin(wend, width);
    float aveval = 0;
    int bottom_count = (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[bottom_count + h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }).wait();
}
template <>
void AvePoolForward(int top_count, int boottom_count, double *bottom_data,
                    const int num, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width,
                    const int kernel_h, const int kernel_w,
                    const int stride_h,
                    const int stride_w, const int pad_h,
                    const int pad_w, double *top_data) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numTopElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = hc::fast_math::fmin(hstart + kernel_h, height + pad_h);
    int wend = hc::fast_math::fmin(wstart + kernel_w, width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = hc::fast_math::fmax(hstart, 0);
    wstart = hc::fast_math::fmax(wstart, 0);
    hend = hc::fast_math::fmin(hend, height);
    wend = hc::fast_math::fmin(wend, width);
    double aveval = 0;
    int bottom_count = (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[bottom_count + h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }).wait();
}
template <>
void StoPoolForwardTrain(int top_count, int boottom_count,
                         float *bottom_data,
                         const int num, const int channels,
                         const int height,
                         const int width, const int pooled_height,
                         const int pooled_width,
                         const int kernel_h, const int kernel_w,
                         const int stride_h,
                         const int stride_w, float *rand_idx,
                         float *top_data) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numTopElts);
  parallel_for_each(
   grdExt,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = hc::fast_math::fmin(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = hc::fast_math::fmin(wstart + kernel_w, width);
    float cumsum = 0;
    int bottom_count = (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[bottom_count + h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[bottom_count + h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[bottom_count + h * width + w];
          return;
        }
      }
    }
  }).wait();;
}
template <>
void StoPoolForwardTrain(int top_count, int boottom_count,
                         double *bottom_data,
                         const int num, const int channels,
                         const int height,
                         const int width, const int pooled_height,
                         const int pooled_width,
                         const int kernel_h, const int kernel_w,
                         const int stride_h,
                         const int stride_w,
                         double *rand_idx,
                         double *top_data) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numTopElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = hc::fast_math::fmin(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = hc::fast_math::fmin(wstart + kernel_w, width);
    double cumsum = 0;
    int bottom_count = (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[bottom_count + h * width + w];
      }
    }
    double thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[bottom_count + h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] =
            ((n * channels + c) * height + h) * width + w;
          top_data[index] =
            bottom_data[bottom_count + h * width + w];
          return;
        }
      }
    }
  }).wait();
}
template <>
void StoPoolForwardTest(int top_count, int boottom_count,
                        float *bottom_data,
                        const int num, const int channels,
                        const int height,
                        const int width, const int pooled_height,
                        const int pooled_width,
                        const int kernel_h, const int kernel_w,
                        const int stride_h,
                        const int stride_w, float *top_data) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numTopElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = hc::fast_math::fmin(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = hc::fast_math::fmin(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    float cumsum = FLT_MIN;
    float cumvalues = 0.;
    int bottom_count = (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[bottom_count + h * width + w];
        cumvalues += bottom_data[bottom_count + h * width + w]
            * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }).wait();
}
template <>
void StoPoolForwardTest(int top_count, int boottom_count,
                        double *bottom_data,
                        const int num, const int channels,
                        const int height,
                        const int width, const int pooled_height,
                        const int pooled_width,
                        const int kernel_h, const int kernel_w,
                        const int stride_h,
                        const int stride_w, double *top_data) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo topInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&topInfo, top_data);
  long numTopElts = topInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numTopElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = hc::fast_math::fmin(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = hc::fast_math::fmin(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    double cumsum = FLT_MIN;
    double cumvalues = 0.;
    int bottom_count = (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[bottom_count + h * width + w];
        cumvalues += bottom_data[bottom_count + h * width + w]
          * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }).wait();
}
template <>
void MaxPoolBackward(int top_count, int boottom_count, float *top_diff,
                     int *mask, float *top_mask,
                     const int num, const int channels,
                     const int height, const int width,
                     const int pooled_height,
                     const int pooled_width, const int kernel_h,
                     const int kernel_w,
                     const int stride_h, const int stride_w,
                     const int pad_h, const int pad_w,
                     float *bottom_diff) {
  bool maskTag = (mask == NULL ? false : true);
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo bottomDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&bottomDiffInfo, bottom_diff);
  long numBotDiffElts = bottomDiffInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numBotDiffElts);
  if (maskTag) {
    parallel_for_each(
      grdExt,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int w = index % width;
      int h = (index / width) % height;
      int c = (index / width / height) % channels;
      int n = index / width / height / channels;
      int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
      int phend =
        hc::fast_math::fmin((h + pad_h) /
        stride_h + 1, pooled_height);
      int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
      int pwend =
        hc::fast_math::fmin((w + pad_w) /
        stride_w + 1, pooled_width);
      float gradient = 0;
      int offset = (n * channels + c) * pooled_height * pooled_width;
      int top_diff_offset = offset;
      int mask_offset = offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[mask_offset + ph * pooled_width + pw] ==
                h * width + w) {
            gradient += top_diff[top_diff_offset + ph * pooled_width + pw];
          }
        }
      }
      bottom_diff[index] = gradient;
    }).wait();

  } else {
    parallel_for_each(
      grdExt,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int w = index % width;
      int h = (index / width) % height;
      int c = (index / width / height) % channels;
      int n = index / width / height / channels;
      int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
      int phend =
        hc::fast_math::fmin((h + pad_h) /
        stride_h + 1, pooled_height);
      int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
      int pwend =
        hc::fast_math::fmin((w + pad_w) /
        stride_w + 1, pooled_width);
      float gradient = 0;
      int offset =
        (n * channels + c) * pooled_height * pooled_width;
      int top_diff_offset = offset;
      int top_mask_offset = offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[top_mask_offset + ph * pooled_width + pw] ==
                h * width + w) {
            gradient +=
              top_diff[top_diff_offset + ph * pooled_width + pw];
          }
        }
      }
      bottom_diff[index] = gradient;
    }).wait();
  }
}
template <>
void MaxPoolBackward(int top_count, int boottom_count, double *top_diff,
                     int *mask, double *top_mask,
                     const int num, const int channels,
                     const int height, const int width,
                     const int pooled_height,
                     const int pooled_width,
                     const int kernel_h, const int kernel_w,
                     const int stride_h, const int stride_w,
                     const int pad_h, const int pad_w,
                     double *bottom_diff) {
  bool maskTag = (mask == NULL ? false : true);
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo bottomDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&bottomDiffInfo, bottom_diff);
  long numBotDiffElts = bottomDiffInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numBotDiffElts);
  if (maskTag) {
    parallel_for_each(
      grdExt,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int w = index % width;
      int h = (index / width) % height;
      int c = (index / width / height) % channels;
      int n = index / width / height / channels;
      int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
      int phend =
        hc::fast_math::fmin((h + pad_h) /
        stride_h + 1, pooled_height);
      int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
      int pwend = hc::fast_math::fmin((w + pad_w) /
        stride_w + 1, pooled_width);
      double gradient = 0;
      int offset = (n * channels + c) * pooled_height * pooled_width;
      int top_diff_offset = offset;
      int mask_offset = offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[mask_offset + ph * pooled_width + pw] ==
            h * width + w) {
            gradient += top_diff[top_diff_offset + ph * pooled_width + pw];
          }
        }
      }
      bottom_diff[index] = gradient;
    }).wait();
  } else {
    parallel_for_each(
     grdExt,
    [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
      int index = idx[0];
      int w = index % width;
      int h = (index / width) % height;
      int c = (index / width / height) % channels;
      int n = index / width / height / channels;
      int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
      int phend =
        hc::fast_math::fmin((h + pad_h) /
        stride_h + 1, pooled_height);
      int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
      int pwend =
        hc::fast_math::fmin((w + pad_w) /
        stride_w + 1, pooled_width);
      double gradient = 0;
      int offset = (n * channels + c) * pooled_height * pooled_width;
      int top_diff_offset = offset;
      int top_mask_offset = offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[top_mask_offset + ph * pooled_width + pw] ==
              h * width + w) {
            gradient +=
              top_diff[top_diff_offset + ph * pooled_width + pw];
          }
        }
      }
      bottom_diff[index] = gradient;
    }).wait();
  }
}
template <>
void AvePoolBackward(int top_count, int boottom_count,
                     float *top_diff,
                     const int num, const int channels, const int height,
                     const int width,
                     const int pooled_height, const int pooled_width,
                     const int kernel_h, const int kernel_w,
                     const int stride_h,
                     const int stride_w, const int pad_h, const int pad_w,
                     float *bottom_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo bottomDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&bottomDiffInfo, bottom_diff);
  long numBotDiffElts = bottomDiffInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numBotDiffElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend =
      hc::fast_math::fmin(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend =
      hc::fast_math::fmin(w / stride_w + 1, pooled_width);
    float gradient = 0;
    int top_diff_count =
      (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend =
          hc::fast_math::fmin(hstart + kernel_h, height + pad_h);
        int wend =
          hc::fast_math::fmin(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient +=
          top_diff[top_diff_count + ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }).wait();
}
template <>
void AvePoolBackward(int top_count, int boottom_count, double *top_diff,
                     const int num, const int channels, const int height,
                     const int width, const int pooled_height,
                     const int pooled_width,
                     const int kernel_h, const int kernel_w,
                     const int stride_h,
                     const int stride_w, const int pad_h, const int pad_w,
                     double *bottom_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo bottomDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&bottomDiffInfo, bottom_diff);
  long numBotDiffElts = bottomDiffInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numBotDiffElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = hc::fast_math::fmin(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = hc::fast_math::fmin(w / stride_w + 1, pooled_width);
    double gradient = 0;
    int top_diff_count = (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend =
          hc::fast_math::fmin(hstart + kernel_h, height + pad_h);
        int wend =
          hc::fast_math::fmin(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient +=
          top_diff[top_diff_count + ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }).wait();
}
template <>
void StoPoolBackward(int top_count, int boottom_count,
                     float *rand_idx, float *top_diff,
                     const int num, const int channels,
                     const int height,
                     const int width, const int pooled_height,
                     const int pooled_width,
                     const int kernel_h, const int kernel_w,
                     const int stride_h,
                     const int stride_w, float *bottom_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo bottomDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&bottomDiffInfo, bottom_diff);
  long numBotDiffElts = bottomDiffInfo._sizeBytes/sizeof(float);
  hc::extent<1> grdExt(numBotDiffElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = hc::fast_math::fmin(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = hc::fast_math::fmin(w / stride_w + 1, pooled_width);
    float gradient = 0;
    int rand_idx_count = (n * channels + c) * pooled_height * pooled_width;
    int top_diff_count = (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[top_diff_count + ph * pooled_width + pw] *
          (index == static_cast<int>(rand_idx[rand_idx_count +
             ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }).wait();
}
template <>
void StoPoolBackward(int top_count, int boottom_count,
                     double *rand_idx, double *top_diff,
                     const int num, const int channels, const int height,
                     const int width, const int pooled_height,
                     const int pooled_width,
                     const int kernel_h, const int kernel_w,
                     const int stride_h,
                     const int stride_w, double *bottom_diff) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo bottomDiffInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&bottomDiffInfo, bottom_diff);
  long numBotDiffElts = bottomDiffInfo._sizeBytes/sizeof(double);
  hc::extent<1> grdExt(numBotDiffElts);
  parallel_for_each(
    grdExt,
    [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int index = idx[0];
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend =
      hc::fast_math::fmin(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend =
      hc::fast_math::fmin(w / stride_w + 1, pooled_width);
    double gradient = 0;
    int rand_idx_count =
      (n * channels + c) * pooled_height * pooled_width;
    int top_diff_count =
      (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[top_diff_count + ph * pooled_width + pw] *
                    (index ==
                      static_cast<int>(rand_idx[rand_idx_count +
                      ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }).wait();
}
