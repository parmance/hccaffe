#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"



template <typename Dtype>
void SigmoidForward(const int N, Dtype* in, Dtype* out);

template <typename Dtype>
void SigmoidBackward(const int N, Dtype* in_diff, Dtype* out_data, Dtype* out_diff);



namespace caffe {

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidForward(count, bottom_data, top_data);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_data = const_cast <Dtype*>(top[0]->gpu_data());
    Dtype* top_diff = const_cast <Dtype*>(top[0]->gpu_diff());
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidBackward(count, top_diff, top_data, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);


}  // namespace caffe
