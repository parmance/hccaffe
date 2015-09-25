#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
#ifdef USE_CPPAMP
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

#if 0
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom[i]->count()-bottom[i]->offset(n),
          top[i]->count()-top[i]->offset(n),
          bottom_data , bottom[i]->offset(n),
          weight, top_data , top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data, top[i]->offset(n), bias);
      }
    }
  }
#else
  if (this->is_1x1_){
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom[i]->count()-bottom[i]->offset(n),
          top[i]->count()-top[i]->offset(n),
          bottom_data , bottom[i]->offset(n),
          weight, top_data , top[i]->offset(n));
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data, top[i]->offset(n), bias);
        }
      }
    }
  } else{
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      this->opt_num2 = global_packing_N;
      this->weight_offset_ = this->M_ * this->K_;
      for (int n = 0; n < this->num_; n += this->opt_num2) {
        this->opt_num2 = this->opt_num2 > (this->num_ - n)? (this->num_ - n) : this->opt_num2;
         //intermediate variables to pass offset
        this->top_offset_opt = this->M_ * this->N_ * this->opt_num2;
        this->top_offset_ = top[i]->offset(n);
        this->col_offset_ = this->K_ * this->N_ * this->opt_num2;
        this->bottom_offset_ = bottom[i]->offset(n);
        this->forward_gpu_gemm_opt(bottom_data, weight,
            top_data);
          if (this->bias_term_) {
            const Dtype* bias = this->blobs_[1]->gpu_data();
              this->forward_gpu_bias_opt(top_data, bias);
          }
        }  
      }
    }
#endif

}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#if 0
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_gpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff, top[i]->offset(n));
      }
    }

    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom[i]->count()-bottom[i]->offset(n),
              top[i]->count()-top[i]->offset(n),
              bottom_data , bottom[i]->offset(n),
              top_diff , top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top[i]->count()-top[i]->offset(n),
              bottom[i]->count()-bottom[i]->offset(n),
              top_diff , top[i]->offset(n), weight,
              bottom_diff , bottom[i]->offset(n));
        }
      }
    }
  }
#else
  if (this->is_1x1_){
    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    if (this->param_propagate_down_[0]) {
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
    }
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      caffe_gpu_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_gpu_diff());
    }
    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();
      // Bias gradient, if necessary.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          this->backward_gpu_bias(bias_diff, top_diff, top[i]->offset(n));
        }
      }

      if (this->param_propagate_down_[0] || propagate_down[i]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm(bottom[i]->count()-bottom[i]->offset(n),
                top[i]->count()-top[i]->offset(n),
                bottom_data , bottom[i]->offset(n),
                top_diff , top[i]->offset(n), weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_gpu_gemm(top[i]->count()-top[i]->offset(n),
              bottom[i]->count()-bottom[i]->offset(n),
              top_diff , top[i]->offset(n), weight,
              bottom_diff , bottom[i]->offset(n));
          }
        }
      }
    }
  }else{

  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[1]->count(), (Dtype)(0.), bias_diff);
      for (int n = 0; n < this->num_; ++n) {
        this->top_offset_ = top[i]->offset(n);
        this->backward_gpu_bias_opt(bias_diff, top_diff);
      }
     }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      this->weight_offset_ = this->M_ * this->K_;
      this->opt_num2 = global_packing_N;
      for (int n = 0; n < this->num_; n += this->opt_num2) {
        this->opt_num2 = this->opt_num2 > (this->num_ - n)? (this->num_ - n) : this->opt_num2;
        this->top_offset_ = top[i]->offset(n);
        this->bottom_offset_ = bottom[i]->offset(n);
        this->col_offset_ = this->K_ * (this->N_ * this->opt_num2);
        this->top_offset_opt = this->M_ * (this->N_ * this->opt_num2);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm_opt(bottom_data,
              top_diff, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_gpu_gemm_opt(top_diff, weight,
                bottom_diff);
          }
        }
      }
    }
  }
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);
#endif
}  // namespace caffe
