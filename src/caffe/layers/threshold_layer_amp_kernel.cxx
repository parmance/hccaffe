#include <algorithm>
#include <vector>
#include "amp.h"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;
template <typename Dtype>
void ThresholdForwardKernel(const int N, Dtype threshold,
                      Dtype* in, Dtype* out);
template <>
void ThresholdForwardKernel(const int N, float threshold,
	float* in, float* out) {
	array_view<float, 1> inView(N, in);
	array_view<float, 1> outView(N, out);
    parallel_for_each(
        outView.get_extent(),
        [=](index<1> idx) restrict(amp)
    {
        outView[idx] = inView[idx] > threshold ? 1 : 0;
    }
    );
    outView.synchronize();
}
template <>
void ThresholdForwardKernel(const int N, double threshold,
	double* in, double* out) {
	array_view<double, 1> inView(N, in);
	array_view<double, 1> outView(N, out);
	parallel_for_each(
		outView.get_extent(),
		[=](index<1> idx) restrict(amp)
	{
		outView[idx] = inView[idx] > threshold ? 1 : 0;
	}
	);
	outView.synchronize();
}
