#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "amp.h"
#include "amp_math.h"
using namespace concurrency;

namespace caffe {

template <typename Dtype>
void LRNFillScale(const int N, Dtype* in,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype alpha_over_size,
  const Dtype k, Dtype* scale) {
  array_view<Dtype, 1> inView(N, in);
  array_view<Dtype, 1> scaleView(N, scale);
  parallel_for_each(
    scaleView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int w = idx[0] % width;
      int h = (idx[0] / width) % height;
      int n = idx[0] / width / height;
      int offset = (n * channels * height + h) * width + w;
      int step = height * width;
      int head = 0;
      int pre_pad = (size - 1) / 2;
      int post_pad = size - pre_pad - 1;
      Dtype accum_scale = 0;
      // accumulate values
      while (head < post_pad && head < channels) {
        accum_scale += inView[offset + head * step] * inView[offset + head * step];
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        accum_scale += inView[offset + head * step] * inView[offset + head * step];
        if (head - size >= 0) {
          accum_scale -= inView[offset + (head - size) * step] * inView[offset + (head - size) * step];
        }
        scaleView[offset + (head - post_pad) * step] = k + accum_scale * alpha_over_size;
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          accum_scale -= inView[offset + (head - size) * step] * inView[offset + (head - size) * step];
        }
        scaleView[offset + (head - post_pad) * step] = k + accum_scale * alpha_over_size;
        ++head;
      }
    }
  );
  scaleView.synchronize();
}

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

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
void LRNComputeOutput(const int N, Dtype* in,
  Dtype* scale, const Dtype negative_beta, Dtype* out) {
  array_view<Dtype, 1> inView(N, in);
  array_view<Dtype, 1> scaleView(N, scale);
  array_view<Dtype, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = inView[idx] * Concurrency::fast_math::pow(scaleView[idx], negative_beta);
    }
  );
  outView.synchronize();
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
  LRNFillScale(n_threads, const_cast<Dtype*>(bottom_data), num_, channels_, height_, width_, size_,
      alpha_ / size_, k_, scale_data);

  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput(n_threads, const_cast<Dtype*>(bottom_data), scale_data, -beta_, top_data);
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
void LRNComputeDiff(const int N, Dtype* bottom_data,
  Dtype* top_data, Dtype* scale, Dtype* top_diff,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype negative_beta,
  const Dtype cache_ratio, Dtype* bottom_diff) {
  array_view<Dtype, 1> bottom_dataView(N, bottom_data);
  array_view<Dtype, 1> top_dataView(N, top_data);
  array_view<Dtype, 1> scaleView(N, scale);
  array_view<Dtype, 1> top_diffView(N, top_diff);
  array_view<Dtype, 1> bottom_diffView(N, bottom_diff);
  parallel_for_each(
    bottom_diffView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int w = idx[0] % width;
      int h = (idx[0] / width) % height;
      int n = idx[0] / width / height;
      int offset = (n * channels * height + h) * width + w;
      int step = height * width;
      int head = 0;
      int pre_pad = size - (size + 1) / 2;
      int post_pad = size - pre_pad - 1;
      Dtype accum_ratio = 0;
      // accumulate values
      while (head < post_pad && head < channels) {
        accum_ratio += top_diffView[offset + head * step] * top_dataView[offset + head * step] /
              scaleView[offset + head * step];
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        accum_ratio += top_diffView[offset + head * step] * top_dataView[offset + head * step] /
          scaleView[offset + head * step];
        if (head - size >= 0) {
          accum_ratio -= top_diffView[offset + (head - size) * step] * top_dataView[offset + (head - size) * step] /
            scaleView[offset + (head - size) * step];
        }
        bottom_diffView[offset + (head - post_pad) * step] = top_diffView[offset + (head - post_pad) * step] * 
          Concurrency::fast_math::pow(scaleView[offset + (head - post_pad) * step], negative_beta) -
          cache_ratio * bottom_dataView[offset + (head - post_pad) * step] * accum_ratio;
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          accum_ratio -= top_diffView[offset + (head - size) * step] *
            top_dataView[offset + (head - size) * step] / scaleView[offset + (head - size) * step];
        }
        bottom_diffView[offset + (head - post_pad) * step] = top_diffView[offset + (head - post_pad) * step] *
          Concurrency::fast_math::pow(scaleView[offset + (head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_dataView[offset + (head - post_pad) * step] * accum_ratio;
        ++head;
      }
    }
  );
  bottom_diffView.synchronize();
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff(n_threads, const_cast<Dtype*>(bottom[0]->gpu_data()), const_cast<Dtype*>(top[0]->gpu_data()),
      const_cast<Dtype*>(scale_.gpu_data()), const_cast<Dtype*>(top[0]->gpu_diff()), num_, channels_, height_, width_,
      size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
      bottom[0]->mutable_gpu_diff());
}
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);



INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe
