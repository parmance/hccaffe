#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#ifdef USE_CPPAMP
template <typename Dtype>
void BNLLForward(const int n, Dtype* in, Dtype* out);
template <typename Dtype>
void BNLLBackward(const int n, Dtype* in_diff,
                  Dtype* in_data, Dtype* out_diff);
namespace caffe {
const float kBNLL_THRESHOLD = 50.;
template <typename Dtype>
void BNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
  Dtype* top_data = const_cast <Dtype*>(top[0]->mutable_gpu_data());
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BNLLForward(count, bottom_data, top_data);
}

template <typename Dtype>
void BNLLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
    Dtype* top_diff = const_cast <Dtype*>(top[0]->gpu_diff());
    Dtype* bottom_diff = const_cast <Dtype*>(bottom[0]->mutable_gpu_diff());
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    BNLLBackward(count, top_diff, bottom_data, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLLLayer);

}  // namespace caffe
#endif  // USE_CPPAMP
