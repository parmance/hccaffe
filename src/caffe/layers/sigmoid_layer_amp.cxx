#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;

namespace caffe {

template <typename Dtype>
void SigmoidForward(const int N, Dtype* in, Dtype* out) {
  array_view<Dtype, 1> inView(N, in);
  array_view<Dtype, 1> outView(N, out);
  parallel_for_each(
     outView.get_extent(),
     [=](index<1> idx) restrict(amp)
     {
       outView[idx] = 1 / (1 + Concurrency::fast_math::exp(-inView[idx]));
     }
  );
  outView.synchronize();
}

//template void SigmoidForward<int>(const int N, const int* in, int* out);
//template void SigmoidForward<float>(const int N, const float* in, float* out);
//template void SigmoidForward<double>(const int N, const double* in, double* out);

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
void SigmoidBackward(const int N, Dtype* in_diff, Dtype* out_data, Dtype* out_diff) {
  array_view<Dtype, 1> in_diffView(N, in_diff);
  array_view<Dtype, 1> out_dataView(N, out_data);
  array_view<Dtype, 1> out_diffView(N, out_diff);
  parallel_for_each(
    out_diffView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      const Dtype sigmoid_x = out_dataView[idx];
      out_diffView[idx] = in_diffView[idx] * sigmoid_x * (1 - sigmoid_x);
    }
  );
  out_diffView.synchronize();
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
