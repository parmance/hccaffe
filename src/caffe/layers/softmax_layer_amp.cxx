#include <algorithm>
#include <cfloat>
#include <vector>

//#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;

namespace caffe {

template <typename Dtype>
void kernel_channel_max(const int N, const int channels,
  const int spatial_dim, Dtype* data, Dtype* out) {
  array_view<Dtype, 1> dataView(N, data);
  array_view<Dtype, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      Dtype maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = max(dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    }
  );
}

template <typename Dtype>
void kernel_channel_subtract(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_max, Dtype* data) {
  array_view<Dtype, 1> channel_maxView(N, channel_max);
  array_view<Dtype, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    }
  );
}

template <typename Dtype>
void kernel_exp(const int N, Dtype* data, Dtype* out) {
  array_view<Dtype, 1> dataView(N, data);
  array_view<Dtype, 1> outView(N, out);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = Concurrency::fast_math::exp(dataView[idx]);
    }
  );
}

template <typename Dtype>
void kernel_channel_sum(const int N, const int channels,
    const int spatial_dim, Dtype* data, Dtype* channel_sum) {
  array_view<Dtype, 1> dataView(N, data);
  array_view<Dtype, 1> channel_sumView(N, channel_sum);
  parallel_for_each(
    channel_sumView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      Dtype sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += dataView[(n * channels + c) * spatial_dim + s];
      }
      channel_sumView[idx] = sum;
    }
  );
}

template <typename Dtype>
void kernel_channel_div(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_sum, Dtype* data) {
  array_view<Dtype, 1> channel_sumView(N, channel_sum);
  array_view<Dtype, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    }
  );
}

template <typename Dtype>
void kernel_channel_dot(const int N, const int channels, const int spatial_dim,
    Dtype* data_1, Dtype* data_2, Dtype* channel_dot) {
  array_view<Dtype, 1> data_1View(N, data_1);
  array_view<Dtype, 1> data_2View(N, data_2);
  array_view<Dtype, 1> channel_dotView(N, channel_dot);
  parallel_for_each(
    channel_dotView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      Dtype dot = 0;
      for (int c = 0; c < channels; ++c) {
        dot += (data_1View[(n * channels + c) * spatial_dim + s]
          * data_2View[(n * channels + c) * spatial_dim + s]);
      }
      channel_dotView[idx] = dot;
    }
  );
}

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
  kernel_channel_max(outer_num_, channels, inner_num_, top_data, scale_data);

  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract(count, outer_num_, channels, inner_num_, scale_data, top_data);

  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp(count, top_data, top_data);

  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum(outer_num_, channels, inner_num_, top_data, scale_data);

  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div(count, outer_num_, channels, inner_num_, scale_data, top_data);
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
  kernel_channel_dot(outer_num_, channels, inner_num_, top_diff, top_data, scale_data);

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract(count, outer_num_, channels, inner_num_, scale_data, bottom_diff);

  // elementwise multiplication
  caffe_gpu_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);


}  // namespace caffe
