#include <algorithm>
#include <vector>
#include <amp.h>
#include <amp_math.h>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
using namespace Concurrency;

const float kBNLL_THRESHOLD = 50.;
template <typename Dtype>
void BNLLForward(const int n, Dtype* in, Dtype* out);
template <typename Dtype>
void BNLLBackward(const int n, Dtype* in_diff,
                  Dtype* in_data, Dtype* out_diff);

template <>
void BNLLForward(const int n, float* in, float* out) {
	array_view<float, 1> inView = *((Concurrency::array_view<float, 1>*)(in));
  array_view<float, 1> outView = *((Concurrency::array_view<float, 1>*)(out));
  Concurrency::extent<1> e(n);
    parallel_for_each(
        e,
        [=](index<1> idx) restrict(amp)
    {
        outView[idx] = inView[idx] > 0 ?
                       inView[idx] + fast_math::log(1. + fast_math::exp(-inView[idx])) :
                       fast_math::log(1. + fast_math::exp(inView[idx]));
    }
    );
}
template <>
void BNLLForward(const int n, double* in, double* out) {
	array_view<double, 1> inView = *((Concurrency::array_view<double, 1>*)(in));
  array_view<double, 1> outView = *((Concurrency::array_view<double, 1>*)(out));
  Concurrency::extent<1> e(n);
	parallel_for_each(
		e,
		[=](index<1> idx) restrict(amp)
	{
		outView[idx] = inView[idx] > 0 ?
			inView[idx] + fast_math::log(1. + fast_math::exp(-inView[idx])) :
			fast_math::log(1. + fast_math::exp(inView[idx]));
	}
	);
}
template <>
void BNLLBackward(const int n,  float* in_diff,
                  float* in_data, float* out_diff) {
	array_view<float, 1> inDiffView = *((Concurrency::array_view<float, 1>*)(in_diff));
  array_view<float, 1> inDataView = *((Concurrency::array_view<float, 1>*)(in_data));
  array_view<float, 1> outDiffView = *((Concurrency::array_view<float, 1>*)(out_diff));
  Concurrency::extent<1> e(n);
    parallel_for_each(
        e,
        [=](index<1> idx) restrict(amp)
    {
		float expval = fast_math::exp(Concurrency::fast_math::fmin(inDataView[idx], float(kBNLL_THRESHOLD)));
        outDiffView[idx] = inDiffView[idx] * expval / (expval + 1.);
    }
    );
}
template <>
void BNLLBackward(const int n, double* in_diff,
	double* in_data, double* out_diff) {
	array_view<double, 1> inDiffView = *((Concurrency::array_view<double, 1>*)(in_diff));
  array_view<double, 1> inDataView = *((Concurrency::array_view<double, 1>*)(in_data));
  array_view<double, 1> outDiffView = *((Concurrency::array_view<double, 1>*)(out_diff));
  Concurrency::extent<1> e(n);
	parallel_for_each(
		e,
		[=](index<1> idx) restrict(amp)
	{
		double expval = fast_math::exp(Concurrency::fast_math::fmin(inDataView[idx], double(kBNLL_THRESHOLD)));
		outDiffView[idx] = inDiffView[idx] * expval / (expval + 1.);
	}
	);
}


