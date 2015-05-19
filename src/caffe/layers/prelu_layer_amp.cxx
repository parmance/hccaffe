#include <algorithm>
#include <vector>
#include <amp.h>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;
namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
void PReLUForward(const int n, const int channels, const int dim,
                  Dtype* in, Dtype* out, Dtype* slope_data,
                  const int div_factor) {
    array_view<Dtype, 1> inView(n, in);
    array_view<Dtype, 1> outView(n, out);
    array_view<Dtype, 1> slopeDataView(n, slope_data);
    parallel_for_each(
        outView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = (int)idx[0];//.global;
        int c = (index / dim) % channels / div_factor;
        outView[idx] = inView[idx] > 0 ? inView[idx] : inView[idx] * slopeDataView[c];
    }
    );
    outView.synchronize();
}
// CUDA kernel for bottom backward
template <typename Dtype>
void PReLUBackward(const int n, const int channels, const int dim,
                   Dtype* in_diff, Dtype* in_data, Dtype* out_diff,
                   Dtype* slope_data, const int div_factor) {
    array_view<Dtype, 1> inDataView(n, in_data);
    array_view<Dtype, 1> inDiffView(n, in_diff);
    array_view<Dtype, 1> outDiffView(n, out_diff);
    array_view<Dtype, 1> slopeDataView(n, slope_data);
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {

        int c = (idx[0] / dim) % channels / div_factor;
        outDiffView[idx] = inDiffView[idx] * ((inDataView[idx] > 0)
                                              + (inDataView[idx] <= 0) * slopeDataView[c]);
    }
    );
    outDiffView.synchronize();
}

template <typename Dtype>
void PReLUParamBackward(const int n, Dtype* in_diff,
                        Dtype* in_data, Dtype* out_diff) {
    array_view<Dtype, 1> inDataView(n, in_data);
    array_view<Dtype, 1> inDiffView(n, in_diff);
    array_view<Dtype, 1> outDiffView(n, out_diff);
    parallel_for_each(
        outDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        outDiffView[idx] = inDiffView[idx] * inDataView[idx] * (inDataView[idx] <= 0);
    }
    );
    outDiffView.synchronize();
}

template <typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
    Dtype* bottom_data = const_cast <Dtype *>(bottom[0]->gpu_data());
    Dtype* top_data = const_cast <Dtype *>(top[0]->mutable_gpu_data());
    const int count = bottom[0]->count();
    const int dim = bottom[0]->count(2);
    const int channels = bottom[0]->channels();
    Dtype* slope_data =const_cast <Dtype *>(this->blobs_[0]->gpu_data());
    const int div_factor = channel_shared_ ? channels : 1;

    // For in-place computation
    if (top[0] == bottom[0]) {
        caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
    }

    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUForward(count, channels, dim, bottom_data, top_data, slope_data, div_factor);
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_data = const_cast <Dtype *>(bottom[0]->gpu_data());
    Dtype* top_diff = const_cast <Dtype *>(top[0]->gpu_diff());
    const int count = bottom[0]->count();
    const int dim = bottom[0]->count(2);
    const int channels = bottom[0]->channels();

    // For in-place computation
    if (top[0] == bottom[0]) {
        bottom_data = const_cast <Dtype *>(bottom_memory_.gpu_data());
    }

    // Propagte to param
    // Since to write bottom diff will affect top diff if top and bottom blobs
    // are identical (in-place computaion), we first compute param backward to
    // keep top_diff unchanged.
    if (this->param_propagate_down_[0]) {
        Dtype* slope_diff =const_cast <Dtype *>(this->blobs_[0]->mutable_gpu_diff());
        // slope_diff is set as 0, then accumulated over batches
        caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), slope_diff);
        int cdim = channels * dim;
        Dtype dsum = 0.;
        for (int n = 0; n < bottom[0]->num(); ++n) {
            Dtype* temp_buff = multiplier_.mutable_gpu_diff();
            // compute element-wise diff
            // NOLINT_NEXT_LINE(whitespace/operators)
            PReLUParamBackward(cdim, top_diff + top[0]->offset(n),
                               bottom_data + bottom[0]->offset(n), multiplier_.mutable_gpu_diff());
            if (channel_shared_) {
                Dtype d;
                caffe_gpu_dot<Dtype>(channels * dim, multiplier_.gpu_diff(),
                                     multiplier_.gpu_data(), &d);
                dsum += d;
            }
            else {
                caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
                                      multiplier_.gpu_diff(), multiplier_.gpu_data(), 1.,
                                      slope_diff);
            }
        }
        if (channel_shared_) {
            caffe_gpu_set(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
        }
    }
    // Propagate to bottom
    if (propagate_down[0]) {
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        Dtype* slope_data = const_cast <Dtype *>(this->blobs_[0]->gpu_data());
        int div_factor = channel_shared_ ? channels : 1;
        // NOLINT_NEXT_LINE(whitespace/operators)
        PReLUBackward(count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
                      div_factor);
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);


}  // namespace caffe