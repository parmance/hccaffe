#include <algorithm>
#include <vector>
#include <amp.h>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;
namespace caffe {

	const float kBNLL_THRESHOLD = 50.;

	template <typename Dtype>
	void BNLLForward(const int n, const Dtype* in, Dtype* out) {

		array_view<Dtype, 1> inView(n, in);
		array_view<Dtype, 1> outView(n, out);
		parallel_for_each(
			yView.get_extent(),
			[=](index<1> idx) restrict(amp)
		{
			outView[idx] = inView[idx] > 0 ?
				inView[idx] + log(1. + exp(-inView[idx])) :
				log(1. + exp(inView[idx]));
		}
		);
	}
	template <typename Dtype>
	void BNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	    Dtype* bottom_data = const_cast <Dtype*>(bottom[0]->gpu_data()£©;
		Dtype* top_data = const_cast <Dtype*>(top[0]->mutable_gpu_data()£©;
		const int count = bottom[0]->count();
		// NOLINT_NEXT_LINE(whitespace/operators)
		BNLLForward(count, bottom_data, top_data);
	}

	template <typename Dtype>
	void BNLLBackward(const int n, const Dtype* in_diff,
		const Dtype* in_data, Dtype* out_diff) {
		array_view<Dtype, 1> inDiffView(n, in_diff);
		array_view<Dtype, 1> inDataView(n, in_data);
		array_view<Dtype, 1> outDiffView(n, out_diff);
		parallel_for_each(
			yView.get_extent(),
			[=](index<1> idx) restrict(amp)
		{
			Dtype expval = exp(min(inDataView[idx], Dtype(kBNLL_THRESHOLD)));
			outDiffView[idx] = inDiffView[idx] * expval / (expval + 1.);
		}
		);
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
