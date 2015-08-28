#include <cfloat>
#include "amp.h"
#include "amp_math.h"

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
  Concurrency::array_view<float, 1> dataView =
    *((Concurrency::array_view<float, 1>*)(data));
  Concurrency::array_view<float, 1> outView =
    *((Concurrency::array_view<float, 1>*)(out));

  Concurrency::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      float maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = Concurrency::fast_math::fmax(
           dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    });
}

template <>
void kernel_channel_subtract<float>(const int N,
    const int num, const int channels,
    const int spatial_dim, float* channel_max, float* data) {

  Concurrency::array_view<float, 1> channel_maxView =
    *((Concurrency::array_view<float, 1>*)(channel_max));
  Concurrency::array_view<float, 1> dataView =
    *((Concurrency::array_view<float, 1>*)(data));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    });
}

template <>
void kernel_exp<float>(const int N, float* data, float* out) {
  Concurrency::array_view<float, 1> dataView =
    *((Concurrency::array_view<float, 1>*)(data));
  Concurrency::array_view<float, 1> outView =
    *((Concurrency::array_view<float, 1>*)(out));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      outView[idx] = Concurrency::fast_math::exp(dataView[idx]);
    });
}

template <>
void kernel_channel_sum<float>(int count, const int N, const int channels,
    const int spatial_dim, float* data, float* channel_sum) {

  Concurrency::array_view<float, 1> dataView =
    *((Concurrency::array_view<float, 1>*)(data));
  Concurrency::array_view<float, 1> channel_sumView =
    *((Concurrency::array_view<float, 1>*)(channel_sum));

  Concurrency::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
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

  Concurrency::array_view<float, 1> dataView =
    *((Concurrency::array_view<float, 1>*)(data));
  Concurrency::array_view<float, 1> channel_sumView =
    *((Concurrency::array_view<float, 1>*)(channel_sum));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    });
}

template <>
void kernel_channel_dot<float>(int count, const int N,
    const int channels, const int spatial_dim,
    float* data_1, float* data_2, float* channel_dot) {
  Concurrency::array_view<float, 1> data_1View =
    *((Concurrency::array_view<float, 1>*)(data_1));
  Concurrency::array_view<float, 1> data_2View =
    *((Concurrency::array_view<float, 1>*)(data_2));
  Concurrency::array_view<float, 1> channel_dotView =
    *((Concurrency::array_view<float, 1>*)(channel_dot));


  Concurrency::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
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
  Concurrency::array_view<double, 1> dataView =
    *((Concurrency::array_view<double, 1>*)(data));
  Concurrency::array_view<double, 1> outView =
    *((Concurrency::array_view<double, 1>*)(out));

  Concurrency::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      double maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = Concurrency::fast_math::fmax(
            dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    });
}

template <>
void kernel_channel_subtract<double>(const int N,
    const int num, const int channels,
    const int spatial_dim, double* channel_max, double* data) {

  Concurrency::array_view<double, 1> channel_maxView =
    *((Concurrency::array_view<double, 1>*)(channel_max));
  Concurrency::array_view<double, 1> dataView =
    *((Concurrency::array_view<double, 1>*)(data));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    });
}


template <>
void kernel_exp<double>(const int N, double* data, double* out) {
  Concurrency::array_view<double, 1> dataView =
    *((Concurrency::array_view<double, 1>*)(data));
  Concurrency::array_view<double, 1> outView =
    *((Concurrency::array_view<double, 1>*)(out));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      outView[idx] = Concurrency::fast_math::exp(dataView[idx]);
    });
}


template <>
void kernel_channel_sum<double>(int count, const int N, const int channels,
    const int spatial_dim, double* data, double* channel_sum) {

  Concurrency::array_view<double, 1> dataView =
    *((Concurrency::array_view<double, 1>*)(data));
  Concurrency::array_view<double, 1> channel_sumView =
    *((Concurrency::array_view<double, 1>*)(channel_sum));

  Concurrency::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
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
  Concurrency::array_view<double, 1> channel_sumView =
    *((Concurrency::array_view<double, 1>*)(channel_sum));
  Concurrency::array_view<double, 1> dataView =
    *((Concurrency::array_view<double, 1>*)(data));

  Concurrency::extent<1> e(N);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    });
}

template <>
void kernel_channel_dot<double>(int count, const int N,
    const int channels, const int spatial_dim,
    double* data_1, double* data_2, double* channel_dot) {
  Concurrency::array_view<double, 1> data_1View =
    *((Concurrency::array_view<double, 1>*)(data_1));
  Concurrency::array_view<double, 1> data_2View =
    *((Concurrency::array_view<double, 1>*)(data_2));
  Concurrency::array_view<double, 1> channel_dotView =
    *((Concurrency::array_view<double, 1>*)(channel_dot));

  Concurrency::extent<1> e(N*spatial_dim);

  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp){
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
