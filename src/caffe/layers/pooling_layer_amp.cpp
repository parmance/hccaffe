#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <cfloat>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#ifdef USE_CPPAMP
template <typename Dtype>
void MaxPoolForward(int top_count,int boottom_count, Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data,
    int* mask, Dtype* top_mask);
template <typename Dtype>
void AvePoolForward(int top_count, int boottom_count, Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data);
template <typename Dtype>
void StoPoolForwardTrain(int top_count, int boottom_count,
    Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* rand_idx, Dtype* top_data);
template <typename Dtype>
void StoPoolForwardTest(int top_count, int boottom_count,
    Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* top_data);
template <typename Dtype>
void MaxPoolBackward(int top_count, int boottom_count, Dtype* top_diff,
    int* mask, Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff);
template <typename Dtype>
void AvePoolBackward(int top_count, int boottom_count, Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff);
template <typename Dtype>
void StoPoolBackward(int top_count, int boottom_count,
    Dtype* rand_idx, Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* bottom_diff);
namespace caffe {
using std::min;
using std::max;
  template <typename Dtype>
    void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
      /*Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
      Dtype* top_data = const_cast <Dtype*>(top[0]->mutable_gpu_data());
      int count = top[0]->count();
      int top_count = top[0]->count();
      int bottom_count = bottom[0]->count();
      // We'll output the mask to top[1] if it's of size >1.
      const bool use_top_mask = top.size() > 1;
      int* mask = NULL;
      Dtype* top_mask = NULL;
      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX:
          if (use_top_mask) {
            top_mask = top[1]->mutable_gpu_data();
          }
          else {
            mask = max_idx_.mutable_gpu_data();
          }
          // NOLINT_NEXT_LINE(whitespace/operators)
          MaxPoolForward(
              top_count, bottom_count,bottom_data, bottom[0]->num(), channels_,
              height_, width_, pooled_height_, pooled_width_, kernel_h_,
              kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
              mask, top_mask);
          break;
        case PoolingParameter_PoolMethod_AVE:
          // NOLINT_NEXT_LINE(whitespace/operators)
          AvePoolForward(
              top_count, bottom_count, bottom_data, bottom[0]->num(), channels_,
              height_, width_, pooled_height_, pooled_width_, kernel_h_,
              kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
          break;
        case PoolingParameter_PoolMethod_STOCHASTIC:
          if (this->phase_ == TRAIN) {
            // We need to create the random index as well.
            caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                rand_idx_.mutable_gpu_data());
            // NOLINT_NEXT_LINE(whitespace/operators)
            StoPoolForwardTrain(
                top_count, bottom_count, bottom_data, bottom[0]->num(), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, stride_h_, stride_w_,
                rand_idx_.mutable_gpu_data(), top_data);
          }
          else {
            // NOLINT_NEXT_LINE(whitespace/operators)
            StoPoolForwardTest(
                top_count, bottom_count, bottom_data, bottom[0]->num(), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, stride_h_, stride_w_, top_data);
          }
          break;
        default:
          LOG(FATAL) << "Unknown pooling method.";
      }*/
    }
  template <typename Dtype>
    void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  /*if (!propagate_down[0]) {
        return;
      }
      Dtype* top_diff = const_cast <Dtype*>(top[0]->gpu_diff());
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const int count = bottom[0]->count();
      int top_count = top[0]->count();
      int bottom_count = bottom[0]->count();
      caffe_gpu_set(count, Dtype(0.), bottom_diff);
      // We'll output the mask to top[1] if it's of size >1.
      const bool use_top_mask = top.size() > 1;
      int* mask = NULL;
      Dtype* top_mask = NULL;

      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX:
          if (use_top_mask) {
            top_mask = const_cast <Dtype*>(top[1]->gpu_data());
          }
          else {
            mask = const_cast <int*>(max_idx_.gpu_data());
          }
          // NOLINT_NEXT_LINE(whitespace/operators)
          MaxPoolBackward(
              top_count, bottom_count, top_diff, mask, top_mask, top[0]->num(), channels_,
              height_, width_, pooled_height_, pooled_width_,
              kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
              bottom_diff);
          break;
        case PoolingParameter_PoolMethod_AVE:
          // NOLINT_NEXT_LINE(whitespace/operators)
          AvePoolBackward(
              top_count, bottom_count, top_diff, top[0]->num(), channels_,
              height_, width_, pooled_height_, pooled_width_, kernel_h_,
              kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
          break;
        case PoolingParameter_PoolMethod_STOCHASTIC:
          // NOLINT_NEXT_LINE(whitespace/operators)
          StoPoolBackward(
              top_count, bottom_count, const_cast <Dtype *>(rand_idx_.gpu_data()), top_diff,
              top[0]->num(), channels_, height_, width_, pooled_height_,
              pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
              bottom_diff);
          break;
        default:
          LOG(FATAL) << "Unknown pooling method.";
      }*/
    }
  INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);
};
#endif
