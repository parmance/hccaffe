#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#ifdef USE_CPPAMP

template <typename Dtype>
void MaxForward(const int N, Dtype* a, Dtype* b, const int blob_idx, Dtype* y,
    int* mask);

template <typename Dtype>
void MaxBackward(const int N, Dtype* top_diff, int blob_idx, int* mask,
    Dtype* bottom_diff);

namespace caffe {

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Dtype(0.), top_data);
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i],
          const_cast <Dtype*>(bottom[i]->gpu_data()), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward(count, const_cast <Dtype*>(bottom[0]->gpu_data()),
        const_cast <Dtype*>(bottom[1]->gpu_data()), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward(count, top_data, const_cast <Dtype*>(bottom[i]->gpu_data()),
          i-1, top_data, mask);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* top_diff = const_cast <Dtype*>(top[0]->gpu_diff());
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                            bottom_diff);
            }
          }
        } else {
          caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = const_cast <int*>(max_idx_.gpu_data());
        MaxBackward(count, top_diff, i, mask, bottom_diff);
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe

#endif  // USE_CPPAMP
