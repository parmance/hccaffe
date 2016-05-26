#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef HCC_BACKEND

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    for (int n = 0; n < num_concats_; ++n) {
      caffe_amp_copy<Dtype>(bottom_concat_axis * concat_input_size_,
        static_cast<void*>(const_cast<Dtype*>(bottom_data)),
        static_cast<void*>(top_data),
        n * bottom_concat_axis * concat_input_size_,
        (n * top_concat_axis + offset_concat_axis)
          * concat_input_size_);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    if (!propagate_down[i]) { continue; }
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    for (int n = 0; n < num_concats_; ++n) {
      caffe_amp_copy<Dtype>(bottom_concat_axis * concat_input_size_,
          static_cast<void*>(const_cast<Dtype*>(top_diff)),
          static_cast<void*>(bottom_diff),
          (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
          n * bottom_concat_axis * concat_input_size_);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe

#endif
