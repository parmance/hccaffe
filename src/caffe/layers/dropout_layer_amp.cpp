#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#ifdef USE_CPPAMP
template <typename Dtype>
void DropoutForward(int n, Dtype* in,
                    unsigned int* mask, int threshold, float scale,
                    Dtype* out);
template <typename Dtype>
void DropoutBackward(int n, Dtype* in_diff,
                     unsigned int* mask, int threshold, float scale,
                     Dtype* out_diff);
namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
    Dtype* top_data = const_cast <Dtype*>(top[0]->mutable_gpu_data());
    int count = bottom[0]->count();
    if (this->phase_ == TRAIN) {
        unsigned int* mask = const_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
        caffe_gpu_rng_uniform(count, mask);
        // set thresholds
        // NOLINT_NEXT_LINE(whitespace/operators)
        DropoutForward(count, bottom_data, mask, uint_thres_, scale_, top_data);
    } else {
        caffe_copy(count, bottom_data, top_data);
    }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        Dtype* top_diff = const_cast <Dtype*>(top[0]->gpu_diff());
        Dtype* bottom_diff = const_cast <Dtype*>(bottom[0]->mutable_gpu_diff());
        if (this->phase_ == TRAIN) {
            unsigned int* mask =
                const_cast <unsigned int*>(rand_vec_.gpu_data());
            int count = bottom[0]->count();
            // NOLINT_NEXT_LINE(whitespace/operators)
            DropoutBackward(count, top_diff, mask, uint_thres_, scale_, bottom_diff);
        } else {
            caffe_copy(top[0]->count(), top_diff, bottom_diff);
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);


}  // namespace caffe
#endif