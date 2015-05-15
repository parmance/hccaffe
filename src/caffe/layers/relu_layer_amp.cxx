#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "amp.h"
#include "amp_math.h"

using namespace Concurrency;
namespace caffe {

template <typename Dtype>
void ReLUForward(const int N, Dtype* in, Dtype* out,
  Dtype negative_slope) {
  array_view<Dtype, 1> inView(N, in);
  array_view<Dtype, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    outView[idx] = inView[idx] > 0 ? inView[idx] : inView[idx] * negative_slope;
  }
  );
  outView.synchronize();
}

template <typename Dtype>
void ReLUBackward(const int N, Dtype* in_diff,
  Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  array_view<Dtype, 1> in_diffView(N, in_diff);
  array_view<Dtype, 1> out_diffView(N, out_diff);
  array_view<Dtype, 1> int_dataView(N, in_data);
  parallel_for_each(
    out_diffView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    out_diffView[idx] = in_diffView[idx] * ((int_dataView[idx] > 0) + (int_dataView[idx] <= 0) * negative_slope);
  }
  );
  out_diffView.synchronize();
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = const_cast<Dtype*>(bottom[0]->gpu_data());
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  ReLUForward(count, bottom_data, top_data, negative_slope);
}


template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_data =  const_cast<Dtype*>(bottom[0]->gpu_data());
    Dtype* top_diff =  const_cast<Dtype*>(top[0]->gpu_diff());
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    ReLUBackward(count, top_diff, bottom_data, bottom_diff, negative_slope);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
