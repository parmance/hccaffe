#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#ifdef HCC_BACKEND
namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      const_cast<Dtype*>(bottom_data), 0,  const_cast<Dtype*>(weight), 0 , (Dtype)0., top_data, 0);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        const_cast<Dtype*>(bias_multiplier_.gpu_data()), 0,
        const_cast<Dtype*>(this->blobs_[1]->gpu_data()), 0, (Dtype)1., top_data, 0);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        const_cast<Dtype*>(top_diff), 0, const_cast<Dtype*>(bottom_data), 0, (Dtype)0.,
        this->blobs_[0]->mutable_gpu_diff(), 0);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., const_cast<Dtype*>(top_diff), 0,
        const_cast<Dtype*>(bias_multiplier_.gpu_data()), 0, (Dtype)0.,
        this->blobs_[1]->mutable_gpu_diff(), 0);
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        const_cast<Dtype*>(top_diff), 0, const_cast<Dtype*>(this->blobs_[0]->gpu_data()), 0, (Dtype)0.,
        bottom[0]->mutable_gpu_diff(), 0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe

#endif  // HCC_BACKEND
