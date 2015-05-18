#include <algorithm>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <amp.h>
#include <amp_math.h>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using namespace concurrency;
namespace caffe {

template <typename Dtype>
void SoftmaxLossForwardGPU(const int nthreads,
                           Dtype* prob_data,  Dtype* label, Dtype* loss,
                           const int num, const int dim, const int spatial_dim,
                           const bool has_ignore_label_, const int ignore_label_,
                           Dtype* counts) {
    array_view<Dtype, 1> probDataView(nthreads, prob_data);
    array_view<Dtype, 1> labelView(nthreads, label);
    array_view<Dtype, 1> countsView(nthreads, counts);
    array_view<Dtype, 1> lossView(nthreads, loss);
    parallel_for_each(
        lossView.get_extent(),
        [=](index<1> idx) restrict(amp) {
      const int n = idx[0] / spatial_dim;
      const int s = idx[0] % spatial_dim;
      Dtype data_temp;
      int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
      if (has_ignore_label_ && (label_value == ignore_label_)) {
        lossView[idx] = 0;
        countsView[idx] = 0;
      } else {
          data_temp = max(probDataView[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN));
          lossView[idx] = -concurrency::fast_math::log(data_temp);
          countsView[idx] = 1;
        }
    }
    );
}
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
  SoftmaxLossForwardGPU(nthreads, prob_data, label, loss_data,
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
void SoftmaxLossBackwardGPU(const int nthreads,  Dtype* top,
	 Dtype* label, Dtype* bottom_diff, const int num, const int dim,
	const int spatial_dim, const bool has_ignore_label_,
	const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;
  array_view<Dtype, 1> labelView(nthreads, label);
  array_view<Dtype, 1> bottomDiffView(nthreads, bottom_diff);
  array_view<Dtype, 1> countsView(nthreads, counts);
  parallel_for_each(bottomDiffView.get_extent(),
		    [=](index<1> idx) restrict(amp)
  {
    const int n = idx[0] / spatial_dim;
    const int s = idx[0] % spatial_dim;
    const int label_value = static_cast<int>(labelView[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottomDiffView[n * dim + c * spatial_dim + s] = 0;
      }
      countsView[idx] = 0;
    } else {
        bottomDiffView[n * dim + label_value * spatial_dim + s] -= 1;
        countsView[idx] = 1;
    }
  }
  );
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = const_cast <Dtype*>(bottom[0]->mutable_gpu_diff());
    Dtype* prob_data =const_cast <Dtype*>(prob_.gpu_data());
    Dtype* top_data = const_cast <Dtype*>(top[0]->gpu_data());
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    Dtype* label = const_cast <Dtype*>(bottom[1]->gpu_data());
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU(nthreads, top_data, label, bottom_diff,
                           outer_num_, dim, inner_num_, has_ignore_label_, 
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
