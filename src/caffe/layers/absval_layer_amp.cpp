#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_CPPAMP

namespace caffe {

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* bottom_data = const_cast<Dtype*>(bottom[0]->gpu_data());
  caffe_gpu_abs(count, bottom_data, top_data);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  Dtype* top_data = const_cast<Dtype*>(top[0]->gpu_data());
  Dtype* top_diff = const_cast<Dtype*>(top[0]->gpu_diff());
  if (propagate_down[0]) {
    Dtype* bottom_data =const_cast<Dtype*>(bottom[0]->gpu_data());
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_sign(count, bottom_data, bottom_diff);
    caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsValLayer);
}  // namespace caffe
#endif

