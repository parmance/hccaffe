#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"


#ifdef HCC_BACKEND
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

namespace caffe {
template <typename Dtype>
void im2col_gpu(const Dtype* data_im,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col, const int im_offset, const int col_offset) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  Dtype* data_im_amp = const_cast<Dtype*>(data_im);
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_hcc_kernel(num_kernels,
      data_im_amp,
      height, width,
      kernel_h, kernel_w,
      pad_h, pad_w,
      stride_h, stride_w,
      height_col, width_col,
      data_col, im_offset, col_offset);
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col, const int im_offset, const int col_offset);

template void im2col_gpu<double>(const double* data_im,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    double* data_col, const int im_offset, const int col_offset);


template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im,
    const int col_offset, const int im_offset) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  Dtype * data_col_amp = const_cast<Dtype*>(data_col);
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_hcc_kernel(num_kernels, data_col_amp,
      height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im,
      col_offset, im_offset);
}

template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im,
    const int col_offset, const int im_offset);

template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    double* data_im,
    const int col_offset, const int im_offset);

template <typename Dtype>
void col2im_gpu_opt(const Dtype* data_col, const int col_offset,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im, const int img_offset, int optnum) {
      int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
      int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
      int num_kernels = channels * height * width  * optnum;
      Dtype * data_col_amp = const_cast<Dtype*>(data_col);

      col2im_hcc_kernel_opt(num_kernels, data_col_amp, col_offset,
        height, width, channels, kernel_h, kernel_w, pad_h,
        pad_w, stride_h, stride_w, height_col, width_col,
        data_im, img_offset, optnum);
}

template void col2im_gpu_opt<float>(const float* data_col,
    const int col_offset, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im, const int img_offset, int optnum);
template void col2im_gpu_opt<double>(const double* data_col,
    const int col_offset, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    double* data_im, const int img_offset, int optnum);

template <typename Dtype>
void im2col_gpu_opt(const Dtype* data_im, const int img_offset,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col, const int col_offset,
    int optnum) {
      int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
      int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
      int num_kernels = optnum * channels * height_col * width_col;
      Dtype* data_im_amp = const_cast<Dtype*>(data_im);
      im2col_hcc_kernel_opt(num_kernels, data_im_amp, channels, img_offset,
        height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, height_col, width_col, data_col, col_offset, optnum);
}
template void im2col_gpu_opt<float>(const float* data_im,
    const int img_offset, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col, const int col_offset,
    int optnum);
template void im2col_gpu_opt<double>(const double* data_im,
    const int img_offset, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col, const int col_offset,
    int optnum);

}  // namespace caffe
#endif  // HCC_BACKEND
