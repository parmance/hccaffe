#include <algorithm>
#include <vector>
#include "amp.h"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;
namespace caffe {
template <typename Dtype>
void ThresholdForward(const int N,  Dtype threshold,
                      Dtype* in, Dtype* out) {
    array_view<Dtype, 1> inView(N, in);
    array_view<Dtype, 1> outView(N, out);
    parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
        outView[idx] = inView[idx] > threshold ? 1 : 0;
    }
    );
    outView.synchronize();
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ThresholdForward(count, threshold_, bottom_data, top_data);
}


INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);


}  // namespace caffe
