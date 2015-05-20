#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <cfloat>
#include <amp.h>
#include <amp_math.h>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;
namespace caffe {
template <typename Dtype>
void MaxPoolForward( int nthreads,  Dtype* bottom_data,
                     const int num, const int channels, const int height,
                     const int width, const int pooled_height, const int pooled_width,
                     const int kernel_h, const int kernel_w, const int stride_h,
                     const int stride_w, const int pad_h, const int pad_w, Dtype* top_data,
                     int* mask, Dtype* top_mask) {
    array_view<Dtype, 1> bottomDataView(nthreads, bottom_data);
    array_view<Dtype, 1> topDataView(nthreads, top_data);
    array_view<int, 1> maskView(nthreads, mask);
    array_view<Dtype, 1> outView(nthreads, top_mask);
    parallel_for_each(
        topDataView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = idx[0];
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        Dtype aveval = 0;
        int locan = (n * channels + c) * height * width;
        //bottomDataView += (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                aveval += bottomDataView[locan+h * width + w];
            }
        }
        topDataView[index] = aveval / pool_size;
    }
    );
     topDataView.synchronize();
}
template <typename Dtype>
void AvePoolForward(const int nthreads, Dtype* bottom_data,
                    const int num, const int channels, const int height,
                    const int width, const int pooled_height, const int pooled_width,
                    const int kernel_h, const int kernel_w, const int stride_h,
                    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data) {

    array_view<Dtype, 1> bottomDataView(nthreads, bottom_data);
    array_view<Dtype, 1> topDataView(nthreads, top_data);
    parallel_for_each(
        topDataView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = idx[0];
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        Dtype aveval = 0;
        int bottom_count = (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                aveval += bottomDataView[bottom_count + h * width + w];
            }
        }
        topDataView[index] = aveval / pool_size;

    }
    );
    topDataView.synchronize();
}
template <typename Dtype>
void StoPoolForwardTrain(const int nthreads,
                         Dtype* bottom_data,
                         const int num, const int channels, const int height,
                         const int width, const int pooled_height, const int pooled_width,
                         const int kernel_h, const int kernel_w, const int stride_h,
                         const int stride_w, Dtype* rand_idx, Dtype* top_data) {
    array_view<Dtype, 1> bottomDataView(nthreads, bottom_data);
    array_view<Dtype, 1> randIdxView(nthreads, rand_idx);
    array_view<Dtype, 1> topDataView(nthreads, top_data);
    parallel_for_each(
        topDataView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = idx[0];
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h;
        int hend = min(hstart + kernel_h, height);
        int wstart = pw * stride_w;
        int wend = min(wstart + kernel_w, width);
        Dtype cumsum = 0;
        int bottom_count = (n * channels + c) * height * width;
        // First pass: get sum
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                cumsum += bottomDataView[bottom_count+h * width + w];
            }
        }
        float thres = randIdxView[index] * cumsum;
        // Second pass: get value, and set index.
        cumsum = 0;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                cumsum += bottomDataView[h * width + w];
                if (cumsum >= thres) {
                    randIdxView[index] = ((n * channels + c) * height + h) * width + w;
                    topDataView[index] = bottomDataView[h * width + w];
                    return;
                }
            }
        }
    }
    );
    topDataView.synchronize();   
}
template <typename Dtype>
void StoPoolForwardTest(const int nthreads,
                        Dtype* bottom_data,
                        const int num, const int channels, const int height,
                        const int width, const int pooled_height, const int pooled_width,
                        const int kernel_h, const int kernel_w, const int stride_h,
                        const int stride_w, Dtype* top_data) {
    array_view<Dtype, 1> bottomDataView(nthreads, bottom_data);
    array_view<Dtype, 1> topDataView(nthreads, top_data);
    parallel_for_each(
        topDataView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = idx[0];
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h;
        int hend = min(hstart + kernel_h, height);
        int wstart = pw * stride_w;
        int wend = min(wstart + kernel_w, width);
        // We set cumsum to be 0 to avoid divide-by-zero problems
        Dtype cumsum = FLT_MIN;
        Dtype cumvalues = 0.;
        int bottom_count = (n * channels + c) * height * width;
        // First pass: get sum
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                cumsum += bottomDataView[bottom_count + h * width + w];
                cumvalues += bottomDataView[bottom_count + h * width + w] * bottomDataView[h * width + w];
            }
        }
        topDataView[index] = cumvalues / cumsum;
    }
    );
    topDataView.synchronize();
}
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data());
    Dtype* top_data = const_cast <Dtype*>(top[0]->mutable_gpu_data());
    int count = top[0]->count();
    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    int* mask = NULL;
    Dtype* top_mask = NULL;
    switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX:
        if (use_top_mask) {
            top_mask = top[1]->mutable_gpu_data();
        }
        else {
            mask = max_idx_.mutable_gpu_data();
        }
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxPoolForward(
            count, bottom_data, bottom[0]->num(), channels_,
            height_, width_, pooled_height_, pooled_width_, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
            mask, top_mask);
        break;
    case PoolingParameter_PoolMethod_AVE:
        // NOLINT_NEXT_LINE(whitespace/operators)
        AvePoolForward(
            count, bottom_data, bottom[0]->num(), channels_,
            height_, width_, pooled_height_, pooled_width_, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
        break;
    case PoolingParameter_PoolMethod_STOCHASTIC:
        if (this->phase_ == TRAIN) {
            // We need to create the random index as well.
            caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                                  rand_idx_.mutable_gpu_data());
            // NOLINT_NEXT_LINE(whitespace/operators)
            StoPoolForwardTrain(
                count, bottom_data, bottom[0]->num(), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, stride_h_, stride_w_,
                rand_idx_.mutable_gpu_data(), top_data);
        }
        else {
            // NOLINT_NEXT_LINE(whitespace/operators)
            StoPoolForwardTest(
                count, bottom_data, bottom[0]->num(), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, stride_h_, stride_w_, top_data);
        }
        break;
    default:
        LOG(FATAL) << "Unknown pooling method.";
    }
}
template <typename Dtype>
void MaxPoolBackward(const int nthreads,  Dtype* top_diff,
                     int* mask,  Dtype* top_mask, const int num, const int channels,
                     const int height, const int width, const int pooled_height,
                     const int pooled_width, const int kernel_h, const int kernel_w,
                     const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                     Dtype* bottom_diff) {
    array_view<Dtype, 1> bottomDiffView(nthreads, bottom_diff);
    array_view<Dtype, 1> topDiffView(nthreads, top_diff);
    array_view<Dtype, 1> topMaskView(nthreads, top_mask);
    array_view<int, 1>  maskView(nthreads, mask);
    parallel_for_each(
        bottomDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = idx[0];
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;
        int phstart =
            (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
        int phend = min((h + pad_h) / stride_h + 1, pooled_height);
        int pwstart =
            (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
        int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
        Dtype gradient = 0;
        int offset = (n * channels + c) * pooled_height * pooled_width;
        int top_diff_offset = offset;
        if (maskView.data()) {
            int mask_offset = offset;
            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    if (maskView[mask_offset + ph * pooled_width + pw] == h * width + w) {
                        gradient += topDiffView[top_diff_offset + ph * pooled_width + pw];
                    }
                }
            }
        }
        else {
            int top_mask_offset = offset;
            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    if (topMaskView[top_mask_offset + ph * pooled_width + pw] == h * width + w) {
                        gradient += topDiffView[top_diff_offset + ph * pooled_width + pw];
                    }
                }
            }
        }
        bottomDiffView[index] = gradient;
    }
    );
    bottomDiffView.synchronize();
}
template <typename Dtype>
void AvePoolBackward(const int nthreads, Dtype* top_diff,
                     const int num, const int channels, const int height,
                     const int width, const int pooled_height, const int pooled_width,
                     const int kernel_h, const int kernel_w, const int stride_h,
                     const int stride_w, const int pad_h, const int pad_w,
                     Dtype* bottom_diff) {
    array_view<Dtype, 1> bottomDiffView(nthreads, bottom_diff);
    array_view<Dtype, 1> topDiffView(nthreads, top_diff);
    parallel_for_each(
        bottomDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = idx[0];
        int w = index % width + pad_w;
        int h = (index / width) % height + pad_h;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;
        int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        int phend = min(h / stride_h + 1, pooled_height);
        int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        int pwend = min(w / stride_w + 1, pooled_width);
        Dtype gradient = 0;
        //topDiffView += (n * channels + c) * pooled_height * pooled_width;
        int top_diff_count = (n * channels + c) * pooled_height * pooled_width;
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                // figure out the pooling size
                int hstart = ph * stride_h - pad_h;
                int wstart = pw * stride_w - pad_w;
                int hend = min(hstart + kernel_h, height + pad_h);
                int wend = min(wstart + kernel_w, width + pad_w);
                int pool_size = (hend - hstart) * (wend - wstart);
                gradient += topDiffView[top_diff_count + ph * pooled_width + pw] / pool_size;
            }
        }
        bottomDiffView[index] = gradient;
    }
    );
    bottomDiffView.synchronize();
}
template <typename Dtype>
void StoPoolBackward(const int nthreads,
                     Dtype* rand_idx,  Dtype* top_diff,
                     const int num, const int channels, const int height,
                     const int width, const int pooled_height, const int pooled_width,
                     const int kernel_h, const int kernel_w, const int stride_h,
                     const int stride_w, Dtype* bottom_diff) {
    array_view<Dtype, 1> bottomDiffView(nthreads, bottom_diff);
    array_view<Dtype, 1> randIdxView(nthreads, rand_idx);
    array_view<Dtype, 1> topDiffView(nthreads, top_diff);
    parallel_for_each(
        bottomDiffView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        int index = idx[0];
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;
        int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        int phend = min(h / stride_h + 1, pooled_height);
        int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        int pwend = min(w / stride_w + 1, pooled_width);
        Dtype gradient = 0;
        int rand_idx_count = (n * channels + c) * pooled_height * pooled_width;
        int top_diff_count = (n * channels + c) * pooled_height * pooled_width;
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                gradient += topDiffView[top_diff_count + ph * pooled_width + pw] *
                            (index == static_cast<int>(randIdxView[rand_idx_count + ph * pooled_width + pw]));
            }
        }
        bottomDiffView[index] = gradient;
    }
    );
    bottomDiffView.synchronize();
}
template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
        return;
    }
    Dtype* top_diff = const_cast <Dtype*>(top[0]->gpu_diff());
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    caffe_gpu_set(count, Dtype(0.), bottom_diff);
    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    int* mask = NULL;
    Dtype* top_mask = NULL;
    switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX:
        if (use_top_mask) {
            top_mask = const_cast <Dtype*>(top[1]->gpu_data());
        }
        else {
            mask = const_cast <int*>(max_idx_.gpu_data());
        }
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxPoolBackward(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            height_, width_, pooled_height_, pooled_width_,
            kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff);
        break;
    case PoolingParameter_PoolMethod_AVE:
        // NOLINT_NEXT_LINE(whitespace/operators)
        AvePoolBackward(
            count, top_diff, top[0]->num(), channels_,
            height_, width_, pooled_height_, pooled_width_, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
        break;
    case PoolingParameter_PoolMethod_STOCHASTIC:
        // NOLINT_NEXT_LINE(whitespace/operators)
        StoPoolBackward(
            count, const_cast <Dtype *>(rand_idx_.gpu_data()), top_diff,
            top[0]->num(), channels_, height_, width_, pooled_height_,
            pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
            bottom_diff);
        break;
    default:
        LOG(FATAL) << "Unknown pooling method.";
    }

}
INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);
};
