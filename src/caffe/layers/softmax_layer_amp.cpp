#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#ifdef USE_CPPAMP
template <typename Dtype>
void kernel_channel_max(int count, const int N, const int channels,
  const int spatial_dim, Dtype* data, Dtype* out);
template <typename Dtype>
void kernel_channel_subtract(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_max, Dtype* data);
template <typename Dtype>
void kernel_exp(const int N, Dtype* data, Dtype* out);
template <typename Dtype>
void kernel_channel_sum(int count, const int N, const int channels,
    const int spatial_dim, Dtype* data, Dtype* channel_sum);
template <typename Dtype>
void kernel_channel_div(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_sum, Dtype* data);
template <typename Dtype>
void kernel_channel_dot(int count, const int N, const int channels,
    const int spatial_dim, Dtype* data_1, Dtype* data_2, Dtype* channel_dot);
namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = bottom[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max<Dtype>(count, outer_num_, channels, inner_num_, top_data,
      scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype>(count, top_data, top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype>(count, outer_num_, channels, inner_num_, top_data,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* top_diff = const_cast <Dtype*>(top[0]->gpu_diff());
  Dtype* top_data = const_cast <Dtype*>(top[0]->gpu_data());
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = top[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype>(count, outer_num_, channels, inner_num_, top_diff,
      top_data, scale_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype>(count, outer_num_, channels, inner_num_,
      scale_data, bottom_diff);
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);
}  // namespace caffe
#endif  // USE_CPPAMP
