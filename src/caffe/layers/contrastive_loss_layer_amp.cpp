#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_CPPAMP
template <typename Dtype>
void CLLForward(const int N,
    const int channels,
    const Dtype margin,
    const Dtype alpha,
    const int y_count,
    Dtype* y, Dtype* diff, Dtype* dist_sq, Dtype* bottom_diff);


namespace caffe {

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv2(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(), 0,  // (a_i-b_i)^2
      summer_vec_.gpu_data(), 0,
      Dtype(0.0),
      dist_sq_.mutable_gpu_data(), 0);  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLForward(count, channels, margin, alpha, bottom[2]->count(),
          const_cast <Dtype*>(bottom[2]->gpu_data()),
          // pair similarity 0 or 1
          const_cast <Dtype*>(diff_.gpu_data()),
          // the cached eltwise difference between a and b
          const_cast <Dtype*>(dist_sq_.gpu_data()),
          // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ContrastiveLossLayer);

}  // namespace caffe

#endif  // USE_CPPAMP
