#include <cfloat>
#include "hc.hpp"

template <typename Dtype>
void kernel_channel_max(int count, const int N, const int channels,
  const int spatial_dim, Dtype* data, Dtype* out);
template <typename Dtype>
void kernel_channel_subtract(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_max, Dtype* data);
template <typename Dtype>
void kernel_exp(const int N, Dtype* data, Dtype* out);
template <typename Dtype>
void kernel_channel_sum(int count, const int N, const int channels,
    const int spatial_dim, Dtype* data, Dtype* channel_sum);
template <typename Dtype>
void kernel_channel_div(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_sum, Dtype* data);
template <typename Dtype>
void kernel_channel_dot(int count, const int N,
    const int channels, const int spatial_dim,
    Dtype* data_1, Dtype* data_2, Dtype* channel_dot);

template <>
void kernel_channel_max<float>(int count, const int N, const int channels,
  const int spatial_dim, float* data, float* out) {
  hc::array_view<float, 1> dataView =
    *((hc::array_view<float, 1>*)(data));
  hc::array_view<float, 1> outView =
    *((hc::array_view<float, 1>*)(out));

  hc::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      float maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = hc::fast_math::fmax(
           dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    });
}

template <>
void kernel_channel_subtract<float>(const int N,
    const int num, const int channels,
    const int spatial_dim, float* channel_max, float* data) {

  hc::array_view<float, 1> channel_maxView =
    *((hc::array_view<float, 1>*)(channel_max));
  hc::array_view<float, 1> dataView =
    *((hc::array_view<float, 1>*)(data));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    });
}

template <>
void kernel_exp<float>(const int N, float* data, float* out) {
  hc::array_view<float, 1> dataView =
    *((hc::array_view<float, 1>*)(data));
  hc::array_view<float, 1> outView =
    *((hc::array_view<float, 1>*)(out));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      outView[idx] = hc::fast_math::exp(dataView[idx]);
    });
}

template <>
void kernel_channel_sum<float>(int count, const int N, const int channels,
    const int spatial_dim, float* data, float* channel_sum) {

  hc::array_view<float, 1> dataView =
    *((hc::array_view<float, 1>*)(data));
  hc::array_view<float, 1> channel_sumView =
    *((hc::array_view<float, 1>*)(channel_sum));

  hc::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      float sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += dataView[(n * channels + c) * spatial_dim + s];
      }
      channel_sumView[idx] = sum;
    });
}

template <>
void kernel_channel_div<float>(const int N, const int num, const int channels,
    const int spatial_dim, float* channel_sum, float* data) {

  hc::array_view<float, 1> dataView =
    *((hc::array_view<float, 1>*)(data));
  hc::array_view<float, 1> channel_sumView =
    *((hc::array_view<float, 1>*)(channel_sum));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    });
}

template <>
void kernel_channel_dot<float>(int count, const int N,
    const int channels, const int spatial_dim,
    float* data_1, float* data_2, float* channel_dot) {
  hc::array_view<float, 1> data_1View =
    *((hc::array_view<float, 1>*)(data_1));
  hc::array_view<float, 1> data_2View =
    *((hc::array_view<float, 1>*)(data_2));
  hc::array_view<float, 1> channel_dotView =
    *((hc::array_view<float, 1>*)(channel_dot));


  hc::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      float dot = 0;
      for (int c = 0; c < channels; ++c) {
        dot += (data_1View[(n * channels + c) * spatial_dim + s]
          * data_2View[(n * channels + c) * spatial_dim + s]);
      }
      channel_dotView[idx] = dot;
    });
}

template <>
void kernel_channel_max<double>(int count, const int N, const int channels,
  const int spatial_dim, double* data, double* out) {
  hc::array_view<double, 1> dataView =
    *((hc::array_view<double, 1>*)(data));
  hc::array_view<double, 1> outView =
    *((hc::array_view<double, 1>*)(out));

  hc::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      double maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = hc::fast_math::fmax(
            dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    });
}

template <>
void kernel_channel_subtract<double>(const int N,
    const int num, const int channels,
    const int spatial_dim, double* channel_max, double* data) {

  hc::array_view<double, 1> channel_maxView =
    *((hc::array_view<double, 1>*)(channel_max));
  hc::array_view<double, 1> dataView =
    *((hc::array_view<double, 1>*)(data));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    });
}


template <>
void kernel_exp<double>(const int N, double* data, double* out) {
  hc::array_view<double, 1> dataView =
    *((hc::array_view<double, 1>*)(data));
  hc::array_view<double, 1> outView =
    *((hc::array_view<double, 1>*)(out));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      outView[idx] = hc::fast_math::exp(dataView[idx]);
    });
}


template <>
void kernel_channel_sum<double>(int count, const int N, const int channels,
    const int spatial_dim, double* data, double* channel_sum) {

  hc::array_view<double, 1> dataView =
    *((hc::array_view<double, 1>*)(data));
  hc::array_view<double, 1> channel_sumView =
    *((hc::array_view<double, 1>*)(channel_sum));

  hc::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      double sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += dataView[(n * channels + c) * spatial_dim + s];
      }
      channel_sumView[idx] = sum;
    });
}

template <>
void kernel_channel_div<double>(const int N, const int num, const int channels,
    const int spatial_dim, double* channel_sum, double* data) {
  hc::array_view<double, 1> channel_sumView =
    *((hc::array_view<double, 1>*)(channel_sum));
  hc::array_view<double, 1> dataView =
    *((hc::array_view<double, 1>*)(data));

  hc::extent<1> e(N);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    });
}

template <>
void kernel_channel_dot<double>(int count, const int N,
    const int channels, const int spatial_dim,
    double* data_1, double* data_2, double* channel_dot) {
  hc::array_view<double, 1> data_1View =
    *((hc::array_view<double, 1>*)(data_1));
  hc::array_view<double, 1> data_2View =
    *((hc::array_view<double, 1>*)(data_2));
  hc::array_view<double, 1> channel_dotView =
    *((hc::array_view<double, 1>*)(channel_dot));

  hc::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](hc::index<1> idx) __attribute__((hc, cpu)){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      double dot = 0;
      for (int c = 0; c < channels; ++c) {
        dot += (data_1View[(n * channels + c) * spatial_dim + s]
          * data_2View[(n * channels + c) * spatial_dim + s]);
      }
      channel_dotView[idx] = dot;
    });
}
