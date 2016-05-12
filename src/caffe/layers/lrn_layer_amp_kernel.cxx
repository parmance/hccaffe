#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "hc.hpp"
template <typename Dtype>
void LRNFillScale(const int N, Dtype *in,
                  const int num, const int channels, const int height,
                  const int width, const int size, const Dtype alpha_over_size,
                  const Dtype k, Dtype *scale, int count);

template <typename Dtype>
void LRNComputeOutput(const int N, Dtype *in,
                      Dtype *scale, const Dtype negative_beta,
                      Dtype *out, int count);

template <typename Dtype>
void LRNComputeDiff(const int N, Dtype *bottom_data,
                    Dtype *top_data, Dtype *scale, Dtype *top_diff,
                    const int num, const int channels, const int height,
                    const int width, const int size,
                    const Dtype negative_beta,
                    const Dtype cache_ratio, Dtype *bottom_diff, int count);



template <>
void LRNFillScale<float>(const int N, float *in,
                         const int num, const int channels, const int height,
                         const int width, const int size,
                         const float alpha_over_size,
                         const float k, float *scale, int count) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
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
      accum_scale += in[offset + head * step] *
        in[offset + head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale +=
        in[offset + head * step] *
        in[offset + head * step];
      if (head - size >= 0) {
        accum_scale -=
          in[offset + (head - size) * step] *
          in[offset + (head - size) * step];
      }
      scale[offset + (head - post_pad) * step] =
        k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in[offset + (head - size) * step]
          * in[offset + (head - size) * step];
      }
      scale[offset + (head - post_pad) * step] =
        k + accum_scale * alpha_over_size;
      ++head;
    }
  }).wait();
}

template <>
void LRNFillScale<double>(const int N, double *in,
                          const int num, const int channels,
                          const int height,
                          const int width, const int size,
                          const double alpha_over_size,
                          const double k, double *scale, int count) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
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
      accum_scale +=
        in[offset + head * step] *
        in[offset + head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale +=
        in[offset + head * step] *
        in[offset + head * step];
      if (head - size >= 0) {
        accum_scale -=
          in[offset + (head - size) * step] *
          in[offset + (head - size) * step];
      }
      scale[offset + (head - post_pad) * step] =
        k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -=
          in[offset + (head - size) * step] *
          in[offset + (head - size) * step];
      }
      scale[offset + (head - post_pad) * step] =
        k + accum_scale * alpha_over_size;
      ++head;
    }
  }).wait();
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <>
void LRNComputeOutput<float>(const int N, float *in,
                             float *scale, const float negative_beta,
                             float *out, int count) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    out[idx[0]] =
      in[idx[0]] *
      hc::fast_math::pow(scale[idx[0]], negative_beta);
  }).wait();
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <>
void LRNComputeOutput<double>(const int N, double *in,
                              double *scale, const double negative_beta,
                              double *out, int count) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    out[idx[0]] =
      in[idx[0]] *
      hc::fast_math::pow(
      scale[idx[0]], negative_beta);
  }).wait();
}

template <>
void LRNComputeDiff<float>(const int N, float *bottom_data,
                           float *top_data, float *scale, float *top_diff,
                           const int num, const int channels, const int height,
                           const int width, const int size,
                           const float negative_beta,
                           const float cache_ratio,
                           float *bottom_diff, int count) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
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
      accum_ratio += top_diff[offset + head * step] *
        top_data[offset + head * step] /
        scale[offset + head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio +=
        top_diff[offset + head * step] *
        top_data[offset + head * step] /
        scale[offset + head * step];
      if (head - size >= 0) {
        accum_ratio -=
          top_diff[offset + (head - size) * step] *
          top_data[offset + (head - size) * step] /
          scale[offset + (head - size) * step];
      }
      bottom_diff[offset + (head - post_pad) * step] =
        top_diff[offset+ (head - post_pad) * step] *
        hc::fast_math::pow(scale[offset +
        (head - post_pad) * step], negative_beta) -
        cache_ratio * bottom_data[offset +
        (head - post_pad) * step] *
        accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -=
          top_diff[offset + (head - size) * step] *
          top_data[offset + (head - size) * step] /
          scale[offset + (head - size) * step];
      }
      bottom_diff[offset + (head - post_pad) * step] =
        top_diff[offset + (head - post_pad) * step] *
        hc::fast_math::pow(scale[offset +
        (head - post_pad) * step], negative_beta) - cache_ratio *
        bottom_data[offset + (head - post_pad) *
        step] * accum_ratio;
      ++head;
    }
  }).wait();
}

template <>
void LRNComputeDiff<double>(const int N, double *bottom_data,
                            double *top_data, double *scale, double *top_diff,
                            const int num, const int channels, const int height,
                            const int width, const int size,
                            const double negative_beta,
                            const double cache_ratio,
                            double *bottom_diff, int count) {
  hc::extent<1> e(N);
  parallel_for_each(
    e,
  [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
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
      accum_ratio +=
        top_diff[offset + head * step] *
        top_data[offset + head * step] /
        scale[offset + head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio +=
        top_diff[offset + head * step] *
        top_data[offset + head * step] /
        scale[offset + head * step];
      if (head - size >= 0) {
        accum_ratio -=
          top_diff[offset + (head - size) * step] *
          top_data[offset + (head - size) * step] /
          scale[offset + (head - size) * step];
      }
      bottom_diff[offset + (head- post_pad)* step] =
        top_diff[offset +(head - post_pad) * step] *
        hc::fast_math::pow(scale[offset +
        (head - post_pad) * step], negative_beta) -
        cache_ratio * bottom_data[offset +
        (head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -=
          top_diff[offset + (head - size) * step] *
          top_data[offset + (head - size) * step] /
          scale[offset + (head - size) * step];
      }
      bottom_diff[offset + (head - post_pad) * step] =
        top_diff[offset +(head - post_pad) * step] *
        hc::fast_math::pow(scale[offset +
        (head - post_pad) * step],
        negative_beta) - cache_ratio *
        bottom_data[offset + (head - post_pad) *
        step] * accum_ratio;
      ++head;
    }
  });
}

