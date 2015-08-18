#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"


#ifdef USE_CPPAMP
template <typename Dtype>
void im2col_amp_kernel2(const int N,
    Dtype* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col, const int im_offset, const int col_offset);

template <typename Dtype>
void im2col_amp_kernel(const int N, Dtype* data_im,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  Dtype* data_col);

template <typename Dtype>
void col2im_amp_kernel2(const int N, Dtype* data_col,
  const int height, const int width, const int channels,
  const int patch_h, const int patch_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  Dtype* data_im, const int col_offset, const int im_offset);

template <typename Dtype>
void col2im_amp_kernel(const int N, Dtype* data_col,
  const int height, const int width, const int channels,
  const int patch_h, const int patch_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  Dtype* data_im);

namespace caffe {
template <typename Dtype>
void im2col_gpu2(const Dtype* data_im,
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
  im2col_amp_kernel2(num_kernels,
      data_im_amp,
      height, width,
      kernel_h, kernel_w,
      pad_h, pad_w,
      stride_h, stride_w,
      height_col, width_col,
      data_col, im_offset, col_offset);
}

// Explicit instantiation
template void im2col_gpu2<float>(const float* data_im,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col, const int im_offset, const int col_offset);

template void im2col_gpu2<double>(const double* data_im,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    double* data_col, const int im_offset, const int col_offset);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  Dtype* data_im_amp = const_cast<Dtype*>(data_im);
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_amp_kernel(
      num_kernels, data_im_amp, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
}

template void im2col_gpu<float>(const float* data_im, const int channels,
   const int height, const int width, const int kernel_h, const int kernel_w,
   const int pad_h, const int pad_w, const int stride_h, const int stride_w,
   float* data_col);

template void im2col_gpu<double>(const double* data_im, const int channels,
   const int height, const int width, const int kernel_h, const int kernel_w,
   const int pad_h, const int pad_w, const int stride_h, const int stride_w,
   double* data_col);

template <typename Dtype>
void col2im_gpu2(const Dtype* data_col, const int channels,
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
  col2im_amp_kernel2(num_kernels, data_col_amp,
      height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im,
      col_offset, im_offset);
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  Dtype * data_col_amp = const_cast<Dtype*>(data_col);
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_amp_kernel(
      num_kernels, data_col_amp, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);

template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

template void col2im_gpu2<float>(const float* data_col, const int channels,
    const int height, const int width,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im,
    const int col_offset, const int im_offset);

template void col2im_gpu2<double>(const double* data_col, const int channels,
    const int height, const int width,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    double* data_im,
    const int col_offset, const int im_offset);

}  // namespace caffe
#endif  // USE_CPPAMP
