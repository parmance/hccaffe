#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#ifdef USE_CPPAMP
template <typename Dtype>
void SoftmaxLossForwardGPU(int N, const int nthreads,
                           Dtype* prob_data, Dtype* label, Dtype* loss,
                           const int num, const int dim, const int spatial_dim,
                           const bool has_ignore_label_,
                           const int ignore_label_, Dtype* counts);
template <typename Dtype>
void SoftmaxLossBackwardGPU(int N, const int nthreads, Dtype* top,
                            Dtype* label, Dtype* bottom_diff, const int num,
                            const int dim, const int spatial_dim,
                            const bool has_ignore_label_,
                            const int ignore_label_, Dtype* counts);
namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  Dtype* prob_data = const_cast <Dtype*>(prob_.gpu_data());
  Dtype* label = const_cast <Dtype*>(bottom[1]->gpu_data());
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = const_cast <Dtype*>(bottom[0]->mutable_gpu_diff());
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU(prob_.count(), nthreads, prob_data, label, loss_data,
                        outer_num_, dim, inner_num_,
                        has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(nthreads, counts, &count);
    loss /= count;
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = const_cast <Dtype*>(bottom[0]->mutable_gpu_diff());
    Dtype* prob_data =const_cast <Dtype*>(prob_.gpu_data());
    Dtype* top_data = const_cast <Dtype*>(top[0]->gpu_data());
    caffe_amp_D2D((void*)prob_data, (void*)bottom_diff, sizeof(Dtype), false);
    Dtype* label = const_cast <Dtype*>(bottom[1]->gpu_data());
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU(prob_.count(), nthreads, top_data, label,
        bottom_diff, outer_num_, dim, inner_num_, has_ignore_label_,
        ignore_label_, counts);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
       caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
#endif  // USE_CPPAMP
