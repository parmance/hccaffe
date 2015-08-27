#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "amp.h"
#include "amp_math.h"
using namespace concurrency;



template <typename Dtype>
void LRNFillScale(const int N, Dtype* in,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype alpha_over_size,
  const Dtype k, Dtype* scale,int count);

template <typename Dtype>
void LRNComputeOutput(const int N, Dtype* in,
  Dtype* scale, const Dtype negative_beta, Dtype* out,int count);

template <typename Dtype>
void LRNComputeDiff(const int N, Dtype* bottom_data,
  Dtype* top_data, Dtype* scale, Dtype* top_diff,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype negative_beta,
  const Dtype cache_ratio, Dtype* bottom_diff,int count);



template <>
void LRNFillScale<float>(const int N, float* in,
  const int num, const int channels, const int height,
  const int width, const int size, const float alpha_over_size,
  const float k, float* scale,int count) {
  array_view<float, 1> inView = *((Concurrency::array_view<float, 1>*)(in));
  array_view<float, 1> scaleView = *((Concurrency::array_view<float, 1>*)(scale));
  extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
    {
      int w = idx[0] % width;
      int h = (idx[0] / width) % height;
      int n = idx[0] / width / height;
      int offset = (n * channels * height + h) * width + w;
      int step = height * width;
      int head = 0;
      int pre_pad = (size - 1) / 2;
      int post_pad = size - pre_pad - 1;
      float accum_scale = 0;
      // accumulate values
      while (head < post_pad && head < channels) {
        accum_scale += inView[offset + head * step] * inView[offset + head * step];
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        accum_scale += inView[offset + head * step] * inView[offset + head * step];
        if (head - size >= 0) {
          accum_scale -= inView[offset + (head - size) * step] * inView[offset + (head - size) * step];
        }
        scaleView[offset + (head - post_pad) * step] = k + accum_scale * alpha_over_size;
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          accum_scale -= inView[offset + (head - size) * step] * inView[offset + (head - size) * step];
        }
        scaleView[offset + (head - post_pad) * step] = k + accum_scale * alpha_over_size;
        ++head;
      }
    }
  );
}

template <>
void LRNFillScale<double>(const int N, double* in,
  const int num, const int channels, const int height,
  const int width, const int size, const double alpha_over_size,
  const double k, double* scale,int count) {
  array_view<double, 1> inView = *((Concurrency::array_view<double, 1>*)(in));
  array_view<double, 1> scaleView = *((Concurrency::array_view<double, 1>*)(scale));
  extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
    {
      int w = idx[0] % width;
      int h = (idx[0] / width) % height;
      int n = idx[0] / width / height;
      int offset = (n * channels * height + h) * width + w;
      int step = height * width;
      int head = 0;
      int pre_pad = (size - 1) / 2;
      int post_pad = size - pre_pad - 1;
      double accum_scale = 0;
      // accumulate values
      while (head < post_pad && head < channels) {
        accum_scale += inView[offset + head * step] * inView[offset + head * step];
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        accum_scale += inView[offset + head * step] * inView[offset + head * step];
        if (head - size >= 0) {
          accum_scale -= inView[offset + (head - size) * step] * inView[offset + (head - size) * step];
        }
        scaleView[offset + (head - post_pad) * step] = k + accum_scale * alpha_over_size;
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          accum_scale -= inView[offset + (head - size) * step] * inView[offset + (head - size) * step];
        }
        scaleView[offset + (head - post_pad) * step] = k + accum_scale * alpha_over_size;
        ++head;
      }
    }
  );
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <>
void LRNComputeOutput<float>(const int N, float* in,
  float* scale, const float negative_beta, float* out,int count) {
  array_view<float, 1> inView = *((Concurrency::array_view<float, 1>*)(in));
  array_view<float, 1> scaleView = *((Concurrency::array_view<float, 1>*)(scale));
  array_view<float, 1> outView = *((Concurrency::array_view<float, 1>*)(out));
  extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = inView[idx] * Concurrency::fast_math::pow(scaleView[idx], negative_beta);
    }
  );
  //outView.synchronize();
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <>
void LRNComputeOutput<double>(const int N, double* in,
  double* scale, const double negative_beta, double* out,int count) {
  array_view<double, 1> inView = *((Concurrency::array_view<double, 1>*)(in));
  array_view<double, 1> scaleView = *((Concurrency::array_view<double, 1>*)(scale));
  array_view<double, 1> outView = *((Concurrency::array_view<double, 1>*)(out));
  extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = inView[idx] * Concurrency::fast_math::pow(scaleView[idx], negative_beta);
    }
  );
}

template <>
void LRNComputeDiff<float>(const int N, float* bottom_data,
  float* top_data, float* scale, float* top_diff,
  const int num, const int channels, const int height,
  const int width, const int size, const float negative_beta,
  const float cache_ratio, float* bottom_diff,int count) {
  array_view<float, 1> bottom_dataView = *((Concurrency::array_view<float, 1>*)(bottom_data));
  array_view<float, 1> top_dataView = *((Concurrency::array_view<float, 1>*)(top_data));
  array_view<float, 1> scaleView = *((Concurrency::array_view<float, 1>*)(scale));
  array_view<float, 1> top_diffView = *((Concurrency::array_view<float, 1>*)(top_diff));
  array_view<float, 1> bottom_diffView = *((Concurrency::array_view<float, 1>*)(bottom_diff));
  extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
    {
      int w = idx[0] % width;
      int h = (idx[0] / width) % height;
      int n = idx[0] / width / height;
      int offset = (n * channels * height + h) * width + w;
      int step = height * width;
      int head = 0;
      int pre_pad = size - (size + 1) / 2;
      int post_pad = size - pre_pad - 1;
      float accum_ratio = 0;
      // accumulate values
      while (head < post_pad && head < channels) {
        accum_ratio += top_diffView[offset + head * step] * top_dataView[offset + head * step] /
              scaleView[offset + head * step];
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        accum_ratio += top_diffView[offset + head * step] * top_dataView[offset + head * step] /
          scaleView[offset + head * step];
        if (head - size >= 0) {
          accum_ratio -= top_diffView[offset + (head - size) * step] * top_dataView[offset + (head - size) * step] /
            scaleView[offset + (head - size) * step];
        }
        bottom_diffView[offset + (head - post_pad) * step] = top_diffView[offset + (head - post_pad) * step] * 
          Concurrency::fast_math::pow(scaleView[offset + (head - post_pad) * step], negative_beta) -
          cache_ratio * bottom_dataView[offset + (head - post_pad) * step] * accum_ratio;
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          accum_ratio -= top_diffView[offset + (head - size) * step] *
            top_dataView[offset + (head - size) * step] / scaleView[offset + (head - size) * step];
        }
        bottom_diffView[offset + (head - post_pad) * step] = top_diffView[offset + (head - post_pad) * step] *
          Concurrency::fast_math::pow(scaleView[offset + (head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_dataView[offset + (head - post_pad) * step] * accum_ratio;
        ++head;
      }
    }
  );
}

template <>
void LRNComputeDiff<double>(const int N, double* bottom_data,
  double* top_data, double* scale, double* top_diff,
  const int num, const int channels, const int height,
  const int width, const int size, const double negative_beta,
  const double cache_ratio, double* bottom_diff,int count) {
  array_view<double, 1> bottom_dataView = *((Concurrency::array_view<double, 1>*)(bottom_data));
  array_view<double, 1> top_dataView = *((Concurrency::array_view<double, 1>*)(top_data));
  array_view<double, 1> scaleView = *((Concurrency::array_view<double, 1>*)(scale));
  array_view<double, 1> top_diffView = *((Concurrency::array_view<double, 1>*)(top_diff));
  array_view<double, 1> bottom_diffView = *((Concurrency::array_view<double, 1>*)(bottom_diff));
  extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
    {
      int w = idx[0] % width;
      int h = (idx[0] / width) % height;
      int n = idx[0] / width / height;
      int offset = (n * channels * height + h) * width + w;
      int step = height * width;
      int head = 0;
      int pre_pad = size - (size + 1) / 2;
      int post_pad = size - pre_pad - 1;
      double accum_ratio = 0;
      // accumulate values
      while (head < post_pad && head < channels) {
        accum_ratio += top_diffView[offset + head * step] * top_dataView[offset + head * step] /
              scaleView[offset + head * step];
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        accum_ratio += top_diffView[offset + head * step] * top_dataView[offset + head * step] /
          scaleView[offset + head * step];
        if (head - size >= 0) {
          accum_ratio -= top_diffView[offset + (head - size) * step] * top_dataView[offset + (head - size) * step] /
            scaleView[offset + (head - size) * step];
        }
        bottom_diffView[offset + (head - post_pad) * step] = top_diffView[offset + (head - post_pad) * step] * 
          Concurrency::fast_math::pow(scaleView[offset + (head - post_pad) * step], negative_beta) -
          cache_ratio * bottom_dataView[offset + (head - post_pad) * step] * accum_ratio;
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          accum_ratio -= top_diffView[offset + (head - size) * step] *
            top_dataView[offset + (head - size) * step] / scaleView[offset + (head - size) * step];
        }
        bottom_diffView[offset + (head - post_pad) * step] = top_diffView[offset + (head - post_pad) * step] *
          Concurrency::fast_math::pow(scaleView[offset + (head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_dataView[offset + (head - post_pad) * step] * accum_ratio;
        ++head;
      }
    }
  );
}

