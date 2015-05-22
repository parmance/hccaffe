#include <cfloat>
#include "amp.h"
#include "amp_math.h"
using namespace concurrency;
/*template <typename Dtype>
void kernel_channel_max(const int N, const int channels,
  const int spatial_dim, Dtype* data, Dtype* out) {
  array_view<Dtype, 1> dataView(N, data);
  array_view<Dtype, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      Dtype maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = max(dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    }
  );
  outView.synchronize();
}

template <typename Dtype>
void kernel_channel_subtract(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_max, Dtype* data) {
  array_view<Dtype, 1> channel_maxView(N, channel_max);
  array_view<Dtype, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    }
  );
  dataView.synchronize();
}

template <typename Dtype>
void kernel_exp(const int N, Dtype* data, Dtype* out) {
  array_view<Dtype, 1> dataView(N, data);
  array_view<Dtype, 1> outView(N, out);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = Concurrency::fast_math::exp(dataView[idx]);
    }
  );
  dataView.synchronize();
}

template <typename Dtype>
void kernel_channel_sum(const int N, const int channels,
    const int spatial_dim, Dtype* data, Dtype* channel_sum) {
  array_view<Dtype, 1> dataView(N, data);
  array_view<Dtype, 1> channel_sumView(N, channel_sum);
  parallel_for_each(
    channel_sumView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      Dtype sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += dataView[(n * channels + c) * spatial_dim + s];
      }
      channel_sumView[idx] = sum;
    }
  );
  channel_sumView.synchronize();
}

template <typename Dtype>
void kernel_channel_div(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_sum, Dtype* data) {
  array_view<Dtype, 1> channel_sumView(N, channel_sum);
  array_view<Dtype, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    }
  );
  dataView.synchronize();
}

template <typename Dtype>
void kernel_channel_dot(const int N, const int channels, const int spatial_dim,
    Dtype* data_1, Dtype* data_2, Dtype* channel_dot) {
  array_view<Dtype, 1> data_1View(N, data_1);
  array_view<Dtype, 1> data_2View(N, data_2);
  array_view<Dtype, 1> channel_dotView(N, channel_dot);
  parallel_for_each(
    channel_dotView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      Dtype dot = 0;
      for (int c = 0; c < channels; ++c) {
        dot += (data_1View[(n * channels + c) * spatial_dim + s]
          * data_2View[(n * channels + c) * spatial_dim + s]);
      }
      channel_dotView[idx] = dot;
    }
  );
  channel_dotView.synchronize();
}
*/
template <typename Dtype>
void kernel_channel_max(const int N, const int channels,
  const int spatial_dim, Dtype* data, Dtype* out);
template <typename Dtype>
void kernel_channel_subtract(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_max, Dtype* data);
template <typename Dtype>
void kernel_exp(const int N, Dtype* data, Dtype* out);
template <typename Dtype>
void kernel_channel_sum(const int N, const int channels,
    const int spatial_dim, Dtype* data, Dtype* channel_sum);
template <typename Dtype>
void kernel_channel_div(const int N, const int num, const int channels,
    const int spatial_dim, Dtype* channel_sum, Dtype* data);
template <typename Dtype>
void kernel_channel_dot(const int N, const int channels, const int spatial_dim,
    Dtype* data_1, Dtype* data_2, Dtype* channel_dot);

template <>
void kernel_channel_max<float>(const int N, const int channels,
  const int spatial_dim, float* data, float* out) {
  array_view<float, 1> dataView(N, data);
  array_view<float, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      float maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = max(dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    }
  );
  outView.synchronize();
}

template <>
void kernel_channel_subtract<float>(const int N, const int num, const int channels,
    const int spatial_dim, float* channel_max, float* data) {
  array_view<float, 1> channel_maxView(N, channel_max);
  array_view<float, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    }
  );
  dataView.synchronize();
}

template <>
void kernel_exp<float>(const int N, float* data, float* out) {
  array_view<float, 1> dataView(N, data);
  array_view<float, 1> outView(N, out);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = Concurrency::fast_math::exp(dataView[idx]);
    }
  );
  dataView.synchronize();
}

template <>
void kernel_channel_sum<float>(const int N, const int channels,
    const int spatial_dim, float* data, float* channel_sum) {
  array_view<float, 1> dataView(N, data);
  array_view<float, 1> channel_sumView(N, channel_sum);
  parallel_for_each(
    channel_sumView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      float sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += dataView[(n * channels + c) * spatial_dim + s];
      }
      channel_sumView[idx] = sum;
    }
  );
  channel_sumView.synchronize();
}

template <>
void kernel_channel_div<float>(const int N, const int num, const int channels,
    const int spatial_dim, float* channel_sum, float* data) {
  array_view<float, 1> channel_sumView(N, channel_sum);
  array_view<float, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    }
  );
  dataView.synchronize();
}

template <>
void kernel_channel_dot<float>(const int N, const int channels, const int spatial_dim,
    float* data_1, float* data_2, float* channel_dot) {
  array_view<float, 1> data_1View(N, data_1);
  array_view<float, 1> data_2View(N, data_2);
  array_view<float, 1> channel_dotView(N, channel_dot);
  parallel_for_each(
    channel_dotView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      float dot = 0;
      for (int c = 0; c < channels; ++c) {
        dot += (data_1View[(n * channels + c) * spatial_dim + s]
          * data_2View[(n * channels + c) * spatial_dim + s]);
      }
      channel_dotView[idx] = dot;
    }
  );
  channel_dotView.synchronize();
}
template <>
void kernel_channel_max<double>(const int N, const int channels,
  const int spatial_dim, double* data, double* out) {
  array_view<double, 1> dataView(N, data);
  array_view<double, 1> outView(N, out);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      double maxval = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        maxval = max(dataView[(n * channels + c) * spatial_dim + s], maxval);
      }
      outView[idx] = maxval;
    }
  );
  outView.synchronize();
}

template <>
void kernel_channel_subtract<double>(const int N, const int num, const int channels,
    const int spatial_dim, double* channel_max, double* data) {
  array_view<double, 1> channel_maxView(N, channel_max);
  array_view<double, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] -= channel_maxView[n * spatial_dim + s];
    }
  );
  dataView.synchronize();
}

template <>
void kernel_exp<double>(const int N, double* data, double* out) {
  array_view<double, 1> dataView(N, data);
  array_view<double, 1> outView(N, out);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = Concurrency::fast_math::exp(dataView[idx]);
    }
  );
  dataView.synchronize();
}

template <>
void kernel_channel_sum<double>(const int N, const int channels,
    const int spatial_dim, double* data, double* channel_sum) {
  array_view<double, 1> dataView(N, data);
  array_view<double, 1> channel_sumView(N, channel_sum);
  parallel_for_each(
    channel_sumView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      double sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += dataView[(n * channels + c) * spatial_dim + s];
      }
      channel_sumView[idx] = sum;
    }
  );
  channel_sumView.synchronize();
}

template <>
void kernel_channel_div<double>(const int N, const int num, const int channels,
    const int spatial_dim, double* channel_sum, double* data) {
  array_view<double, 1> channel_sumView(N, channel_sum);
  array_view<double, 1> dataView(N, data);
  parallel_for_each(
    dataView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / channels / spatial_dim;
      int s = idx[0] % spatial_dim;
      dataView[idx] /= channel_sumView[n * spatial_dim + s];
    }
  );
  dataView.synchronize();
}

template <>
void kernel_channel_dot<double>(const int N, const int channels, const int spatial_dim,
    double* data_1, double* data_2, double* channel_dot) {
  array_view<double, 1> data_1View(N, data_1);
  array_view<double, 1> data_2View(N, data_2);
  array_view<double, 1> channel_dotView(N, channel_dot);
  parallel_for_each(
    channel_dotView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      int n = idx[0] / spatial_dim;
      int s = idx[0] % spatial_dim;
      double dot = 0;
      for (int c = 0; c < channels; ++c) {
        dot += (data_1View[(n * channels + c) * spatial_dim + s]
          * data_2View[(n * channels + c) * spatial_dim + s]);
      }
      channel_dotView[idx] = dot;
    }
  );
  channel_dotView.synchronize();
}
