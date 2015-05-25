#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace Concurrency;

template <typename Dtype>
void im2col_amp_kernel(const int N, Dtype* data_im,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  Dtype* data_col) ;

template <typename Dtype>
 void col2im_amp_kernel(const int N, Dtype* data_col,
  const int height, const int width, const int channels,
  const int patch_h, const int patch_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  Dtype* data_im) ;


template <>
void im2col_amp_kernel(const int N, float* data_im,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  float* data_col) {

  array_view<float, 1> data_imView(N, data_im);
  array_view<float, 1> data_colView(N, data_col);
  parallel_for_each(
    data_colView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    int w_out = idx[0] % width_col;
    int h_index = idx[0] / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    int data_col_num = 0;
    data_col_num += (channel_out * height_col + h_out) * width_col + w_out;
    int data_im_num = 0;
    data_im_num += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        data_colView[data_col_num] = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_imView[data_im_num + i * width + j] : 0;
        data_col_num += height_col * width_col;
      }
    }
  }
  );
  data_colView.synchronize();
}

template <>
 void col2im_amp_kernel(const int N, float* data_col,
  const int height, const int width, const int channels,
  const int patch_h, const int patch_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  float* data_im) {
   array_view<float, 1> data_imView(N, data_im);
   array_view<float, 1> data_colView(N, data_col);
   parallel_for_each(
     data_imView.get_extent(),
     [=](index<1> idx) restrict(amp)
   {
     float val = 0;
     int w = idx[0] % width + pad_w;
     int h = (idx[0] / width) % height + pad_h;
     int c = idx[0] / (width * height);
     // compute the start and end of the output
     int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
     int w_col_end = min(w / stride_w + 1, width_col);
     int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
     int h_col_end = min(h / stride_h + 1, height_col);
     // equivalent implementation
     int offset =
       (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
     int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
     int coeff_w_col = (1 - stride_w * height_col * width_col);
     for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
         val += data_imView[offset + h_col * coeff_h_col + w_col * coeff_w_col];
       }
     }
     data_imView[idx] = val;
   }
   );
   data_imView.synchronize();
    
}

template <>
void im2col_amp_kernel(const int N, double* data_im,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  double* data_col) {

  array_view<double, 1> data_imView(N, data_im);
  array_view<double, 1> data_colView(N, data_col);
  parallel_for_each(
    data_colView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    int w_out = idx[0] % width_col;
    int h_index = idx[0] / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    int data_col_num = 0;
    data_col_num += (channel_out * height_col + h_out) * width_col + w_out;
    int data_im_num = 0;
    data_im_num += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        data_colView[data_col_num] = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_imView[data_im_num + i * width + j] : 0;
        data_col_num += height_col * width_col;
      }
    }
  }
  );
  data_colView.synchronize();
}

template <>
 void col2im_amp_kernel(const int N, double* data_col,
  const int height, const int width, const int channels,
  const int patch_h, const int patch_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  double* data_im) {
   array_view<double, 1> data_imView(N, data_im);
   array_view<double, 1> data_colView(N, data_col);
   parallel_for_each(
     data_imView.get_extent(),
     [=](index<1> idx) restrict(amp)
   {
     double val = 0;
     int w = idx[0] % width + pad_w;
     int h = (idx[0] / width) % height + pad_h;
     int c = idx[0] / (width * height);
     // compute the start and end of the output
     int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
     int w_col_end = min(w / stride_w + 1, width_col);
     int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
     int h_col_end = min(h / stride_h + 1, height_col);
     // equivalent implementation
     int offset =
       (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
     int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
     int coeff_w_col = (1 - stride_w * height_col * width_col);
     for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
         val += data_imView[offset + h_col * coeff_h_col + w_col * coeff_w_col];
       }
     }
     data_imView[idx] = val;
   }
   );
   data_imView.synchronize();
    
}
