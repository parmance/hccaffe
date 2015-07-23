#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_CPPAMP
template <typename Dtype>
void LRNFillScale(const int N, Dtype* in,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype alpha_over_size,
  const Dtype k, Dtype* scale, int count);

template <typename Dtype>
void LRNComputeOutput(const int N, Dtype* in,
  Dtype* scale, const Dtype negative_beta, Dtype* out, int count);

template <typename Dtype>
void LRNComputeDiff(const int N, Dtype* bottom_data,
  Dtype* top_data, Dtype* scale, Dtype* top_diff,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype negative_beta,
  const Dtype cache_ratio, Dtype* bottom_diff, int count);

namespace caffe {

template <typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_gpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScale(n_threads, const_cast<Dtype*>(bottom_data), num_, channels_,
      height_, width_, size_, alpha_ / size_, k_, scale_data, scale_.count());

  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput(n_threads, const_cast<Dtype*>(bottom_data), scale_data,
      -beta_, top_data, scale_.count());
}

template void LRNLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template <typename Dtype>
void LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff(n_threads, const_cast<Dtype*>(bottom[0]->gpu_data()),
      const_cast<Dtype*>(top[0]->gpu_data()),
      const_cast<Dtype*>(scale_.gpu_data()),
      const_cast<Dtype*>(top[0]->gpu_diff()), num_, channels_, height_, width_,
      size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
      bottom[0]->mutable_gpu_diff(), bottom[0]->count());
}
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);

INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe

#endif  // USE_CPPAMP
