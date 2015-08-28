#include <boost/type_traits/is_same.hpp>

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#ifdef USE_CPPAMP
namespace caffe {

template <typename Dtype>
void PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_gpu_set(count, value, top_data);
    return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_amp_D2D(reinterpret_cast<void *>(bottom_data),
    reinterpret_cast<void *>(top_data), sizeof(Dtype), 
    boost::is_same<Dtype, int>::value);
  if (scale_ != Dtype(1)) {
    caffe_gpu_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_gpu_add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_gpu_powx(count, top_data, power_, top_data);
  }
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_gpu_set(count, diff_scale_, bottom_diff);
    } else {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      if (power_ == Dtype(2)) {
        caffe_gpu_axpby(count, diff_scale_ * scale_, bottom_data,
            Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(count, diff_scale_ * shift_, bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        caffe_gpu_scal(count, power_, bottom_diff);
      } else {
        caffe_amp_D2D((void *)bottom_data, (void *)bottom_diff, sizeof(Dtype) ,
          boost::is_same<Dtype, int>::value);
        if (scale_ != Dtype(1)) {
          caffe_gpu_scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(count, shift_, bottom_diff);
        }
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_gpu_scal(count, diff_scale_, bottom_diff);
        }
      }
    }
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PowerLayer);


}  // namespace caffe

#endif  // USE_CPPAMP
