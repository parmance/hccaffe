#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_CPPAMP
template <typename Dtype>
void ThresholdForwardKernel(const int N, Dtype threshold,
  Dtype* in, Dtype* out);
namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ThresholdForwardKernel(count, threshold_, bottom_data, top_data);
}

INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);

}  // namespace caffe
#endif
