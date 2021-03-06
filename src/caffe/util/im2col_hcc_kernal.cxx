#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "hc.hpp"
#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

template <typename Dtype>
void im2col_hcc_kernel(const int N,
    Dtype* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col, const int im_offset, const int col_offset);


template <typename Dtype>
void col2im_hcc_kernel(const int N, Dtype* data_col,
  const int height, const int width, const int channels,
  const int patch_h, const int patch_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  Dtype* data_im, const int col_offset, const int im_offset);

template <>
void im2col_hcc_kernel(const int N,
    float* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col, const int im_offset, const int col_offset) {
  hc::extent<1> e(N);
  hc::parallel_for_each(e,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
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
        data_col[col_offset + data_col_num] =
          (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im[im_offset + data_im_num + i * width + j] : 0;
        data_col_num += height_col * width_col;
      }
    }
  }).wait();
}

template <>
void im2col_hcc_kernel(const int N,
    double* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    double* data_col, const int im_offset, const int col_offset) {
  hc::extent<1> e(N);
  hc::parallel_for_each(e,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
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
        data_col[col_offset + data_col_num] =
          (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im[im_offset + data_im_num + i * width + j] : 0;
        data_col_num += height_col * width_col;
      }
    }
  }).wait();
}

template <>
void col2im_hcc_kernel(const int N, float* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_im, const int col_offset, const int im_offset) {
  hc::extent<1> e(N);
  hc::parallel_for_each(e,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    float val = 0;
    int w = idx[0] % width + pad_w;
    int h = (idx[0] / width) % height + pad_h;
    int c = idx[0] / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = hc::fast_math::fmin(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = hc::fast_math::fmin(h / stride_h + 1, height_col);
    // equivalent implementation
    int offset =
      (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val +=
         data_col[col_offset + offset + h_col * coeff_h_col + w_col *
                  coeff_w_col];
      }
    }
    data_im[im_offset + idx[0]] = val;
  }).wait();
}

template <>
void col2im_hcc_kernel(const int N, double* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    double* data_im, const int col_offset, const int im_offset) {
  hc::extent<1> e(N);
  hc::parallel_for_each(e,
      [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    double val = 0;
    int w = idx[0] % width + pad_w;
    int h = (idx[0] / width) % height + pad_h;
    int c = idx[0] / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = hc::fast_math::fmin(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = hc::fast_math::fmin(h / stride_h + 1, height_col);
    // equivalent implementation
    int offset =
      (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val +=
        data_col[col_offset + offset + h_col * coeff_h_col + w_col *
                 coeff_w_col];
      }
    }
    data_im[im_offset + idx[0]] = val;
  }).wait();
}
template <typename Dtype>
void im2col_hcc_kernel_opt(const int n, Dtype* data_im,
  const int channels, const int img_offset, const int height, const int width,
  const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
  const int stride_h, const int stride_w, const int height_col,
  const int width_col, Dtype* data_col, const int col_offset,
  const int optnum);
template <typename Dtype>
void col2im_hcc_kernel_opt(const int n, Dtype* data_col, const int col_offset,
  const int height, const int width, const int channels, const int kernel_h,
  const int kernel_w, const int pad_h, const int pad_w,
  const int stride_h, const int stride_w, const int height_col,
  const int width_col, Dtype* data_im, const int img_offset, const int optnum);

template <>
void im2col_hcc_kernel_opt(const int n, float* data_im,
  const int channels, const int img_offset, const int height, const int width,
  const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
  const int stride_h, const int stride_w, const int height_col,
  const int width_col, float* data_col, const int col_offset,
  const int optnum) {
    hc::extent<1> e(n);
    parallel_for_each(e, [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int x_out = idx[0] % width_col;
    int y_out = (idx[0] / width_col) % height_col;
    int channel_in = (idx[0] / width_col / height_col) % channels;
    int channel_out = channel_in * kernel_h * kernel_w;
    int im_id = idx[0] / width_col / height_col / channels;

    int y_in = y_out * stride_h - pad_h;
    int x_in = x_out * stride_w - pad_w;
    int offset_col = channel_out * optnum * height_col *
      width_col + im_id * height_col * width_col;
    int offset_im = im_id * channels * height * width + channel_in
      * height * width;

    for (int k_h = 0; k_h < kernel_h; k_h++) {
        for (int k_w = 0; k_w < kernel_w; k_w++) {
            int x_im = x_in + k_w;
            int y_im = y_in + k_h;
            int index_im = y_im * width + x_im;
            int index_col = (k_h * kernel_w + k_w) * optnum *
              height_col * width_col + y_out * width_col + x_out;
            if (y_im >= 0 && y_im < height && x_im >= 0 && x_im < width)
                data_col[col_offset+offset_col + index_col] =
                  data_im[img_offset+offset_im + index_im];
            else
                data_col[col_offset+offset_col + index_col] = 0;
        }
    }
  }).wait();
}

template <>
void im2col_hcc_kernel_opt(const int n, double* data_im,
  const int channels, const int img_offset, const int height, const int width,
  const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
  const int stride_h, const int stride_w, const int height_col,
  const int width_col, double* data_col, const int col_offset,
  const int optnum) {
    hc::extent<1> e(n);
    parallel_for_each(e, [=](hc::index<1> idx) __attribute__((hc, cpu)) {
    int x_out = idx[0] % width_col;
    int y_out = (idx[0] / width_col) % height_col;
    int channel_in = (idx[0] / width_col / height_col) % channels;
    int channel_out = channel_in * kernel_h * kernel_w;
    int im_id = idx[0] / width_col / height_col / channels;

    int y_in = y_out * stride_h - pad_h;
    int x_in = x_out * stride_w - pad_w;
    int offset_col = channel_out * optnum * height_col *
      width_col + im_id * height_col * width_col;
    int offset_im = im_id * channels * height * width + channel_in
      * height * width;

      for (int k_h = 0; k_h < kernel_h; k_h++) {
        for (int k_w = 0; k_w < kernel_w; k_w++) {
            int x_im = x_in + k_w;
            int y_im = y_in + k_h;
            int index_im = y_im * width + x_im;
            int index_col = (k_h * kernel_w + k_w) * optnum *
              height_col * width_col + y_out * width_col + x_out;
            if (y_im >= 0 && y_im < height && x_im >= 0 && x_im < width)
                data_col[col_offset+offset_col + index_col] =
                  data_im[img_offset+offset_im + index_im];
            else
                data_col[col_offset+offset_col + index_col] = 0;
        }
      }
    }).wait();
  }

template <>
void col2im_hcc_kernel_opt(const int n, float* data_col,
  const int col_offset, const int height, const int width,
  const int channels, const int kernel_h,
  const int kernel_w, const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  float* data_im, const int img_offset, const int optnum) {
    hc::extent<1> e(n);
    parallel_for_each(e, [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      float val = 0;
      int w = idx[0] % width + pad_w;
      int h = (idx[0] / width) % height + pad_h;
      int c = idx[0] / (width * height) % channels;
      int im = idx[0] / width / height / channels;
      // compute the start and end of the output
      int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
      int w_col_end = ((w / stride_w + 1) < width_col)?
                       w / stride_w + 1 : width_col;
      int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
      int h_col_end = ((h / stride_h + 1) < height_col) ? h / stride_h + 1
                      : height_col;
      // equivalent implementation
      int offset = (c * kernel_h * kernel_w + h * kernel_w + w) * height_col *
        width_col * optnum + im * height_col * width_col;
      int coeff_h_col = (1 - stride_h * kernel_w
        * height_col * optnum) * width_col;
      int coeff_w_col = (1 - stride_w * height_col * width_col * optnum);
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[col_offset+offset + h_col * coeff_h_col
                 + w_col * coeff_w_col];
        }
      }
      data_im[img_offset+idx[0]] = val;
    }).wait();
  }

template <>
void col2im_hcc_kernel_opt(const int n, double* data_col, const int col_offset,
  const int height, const int width, const int channels, const int kernel_h,
  const int kernel_w, const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  double* data_im, const int img_offset, const int optnum) {
    hc::extent<1> e(n);
    parallel_for_each(e, [=](hc::index<1> idx) __attribute__((hc, cpu)) {
      double val = 0;
      int w = idx[0] % width + pad_w;
      int h = (idx[0] / width) % height + pad_h;
      int c = idx[0] / (width * height) % channels;
      int im = idx[0] / width / height / channels;
      // compute the start and end of the output
      int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
      int w_col_end = ((w / stride_w + 1) < width_col)?
                       w / stride_w + 1 : width_col;
      int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
      int h_col_end = ((h / stride_h + 1) < height_col) ? h / stride_h + 1
                      : height_col;
      // equivalent implementation
      int offset = (c * kernel_h * kernel_w + h * kernel_w + w) * height_col *
        width_col * optnum + im * height_col * width_col;
      int coeff_h_col = (1 - stride_h * kernel_w * height_col
        * optnum) * width_col;
      int coeff_w_col = (1 - stride_w * height_col * width_col * optnum);
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[col_offset+offset + h_col * coeff_h_col
                 + w_col * coeff_w_col];
        }
      }
      data_im[img_offset+idx[0]] = val;
    }).wait();
  }
