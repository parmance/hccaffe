#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <limits>
#include "caffe/util/math_functions.hpp"
#include "amp.h"
#include "amp_math.h"
#include "caffe/common.hpp"
#include "cppamp/ampblaslib.h"

using namespace concurrency;


namespace caffe {

#ifdef USE_CPPAMP
template <typename Dtype>
void caffe_amp_abs(const int N, Dtype* a, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = aView[idx] >= 0 ? aView[idx] : -1 * aView[idx];
  }
  );
  yView.synchronize();
}                                                                                                                                                

template <typename Dtype>
void caffe_amp_sign(const int N, Dtype* a, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = aView[idx] == 0 ? 0 : (aView[idx] < 0 ? -1 : 1);
  }
  );
  yView.synchronize();
}

template <typename Dtype>
void sgnbit_kernel(const int N, Dtype* a, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    yView[idx] = Concurrency::fast_math::signbit(aView[idx]);
  }
  );
  yView.synchronize();
}

template<>
void caffe_gpu_sgnbit<float>(const int n, const float* x, float* y){
  sgnbit_kernel(n, const_cast <float*>(x),  y);
}
template<>
void caffe_gpu_sgnbit<double>(const int n, const double* x, double* y){
  sgnbit_kernel(n, const_cast <double*>(x),  y);
}

template <typename Dtype>
void caffe_amp_mul(const int N, Dtype* a, Dtype* b, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> bView(N, b);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      yView[idx] = (aView[idx] * bView[idx]);
    }
  );
  yView.synchronize();
}

template <>
void caffe_gpu_abs<float>(const int N,const float* a, float* y) {
  caffe_amp_abs(N, const_cast <float*>(a), y);
}

template <>
void caffe_gpu_abs<double>(const int N,const double* a, double* y) {
  caffe_amp_abs(N, const_cast <double*>(a), y);
}

template <>
void caffe_gpu_sign<float>(const int N,const float* a, float* y) {
  caffe_amp_sign(N, const_cast <float*>(a), y);
}

template <>
void caffe_gpu_sign<double>(const int N,const double* a, double* y) {
  caffe_amp_sign(N, const_cast <double*>(a), y);
}



template <>
void caffe_gpu_mul<float>(const int N, const float* a,
  const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_amp_mul(N, const_cast <float*>(a), const_cast <float*>(b), y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
  const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_amp_mul(N, const_cast <double*>(a), const_cast <double*>(b), y);
}

template <typename Dtype>
void div_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> bView(N, b);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      yView[idx] = (aView[idx] / bView[idx]);
    }
  );
  yView.synchronize();
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
  const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel(N, const_cast <float*>(a), const_cast <float*>(b), y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
  const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel(N, const_cast <double*>(a), const_cast <double*>(b), y);
}

template <typename Dtype>
void add_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> bView(N, b);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      yView[idx] = (aView[idx] + bView[idx]);
    }
  );
  yView.synchronize();
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel(N, const_cast <float*>(a), const_cast <float*>(b), y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel(N, const_cast <double*>(a), const_cast <double*>(b), y);
}

template <typename Dtype>
void sub_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> bView(N, b);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      yView[idx] = (aView[idx] - bView[idx]);
    }
  );
  yView.synchronize();
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
  float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel(N, const_cast <float*>(a), const_cast <float*>(b), y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
  double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel(N, const_cast <double*>(a), const_cast <double*>(b), y);
}

template <typename Dtype>
void set_kernel(const int N, const Dtype alpha, Dtype* y) {
  array_view<Dtype, 1> outView(N, y);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] = alpha;
    }
  );
  outView.synchronize();
}

template <>
void caffe_gpu_set<float>(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel(N, alpha, Y);
}

template <>
void caffe_gpu_set<double>(const int N, const double alpha, double* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(double) * N);
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel(N, alpha, Y);
}

template <typename Dtype>
void exp_kernel(const int N, Dtype* a, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      yView[idx] = Concurrency::fast_math::exp(aView[idx]);
    }
  );
  yView.synchronize();
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel(N, const_cast <float*>(a), y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel(N, const_cast <double*>(a), y);
}

template <typename Dtype>
void add_scalar_kernel(const int N, const Dtype alpha, Dtype* y) {
  array_view<Dtype, 1> outView(N, y);
  parallel_for_each(
    outView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      outView[idx] += alpha;
    }
  );
  outView.synchronize();
}

template <>
void caffe_gpu_add_scalar<float>(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel(N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar<double>(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel(N, alpha, Y);
}

template <typename Dtype>
void powx_kernel(const int N, Dtype* a, Dtype alpha, Dtype* y) {
  array_view<Dtype, 1> aView(N, a);
  array_view<Dtype, 1> yView(N, y);
  parallel_for_each(
    yView.get_extent(),
    [=](index<1> idx) restrict(amp)
    {
      yView[idx] = Concurrency::fast_math::pow(aView[idx], alpha);
    }
  );
  yView.synchronize();
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
  const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel(N, const_cast <float*>(a), alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
  const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel(N, const_cast <double*>(a), alpha, y);
}



template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
  float* out) {
  //todo cpu version
  *out = cblas_sdot(n, x, 1, y, 1);
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
  double * out) {
  //todo cpu version
  *out = cblas_ddot(n, x, 1, y, 1);
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
        float* Y) {
  amp_axpy(N, alpha, const_cast <float*>(X), Y);
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
        double* Y) {
  amp_axpy(N, alpha, const_cast <double*>(X), Y);
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                                float* y)
{
  amp_copy(n, const_cast <float*>(x), y);
  amp_scale(n, alpha, y);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                                 double* y) {
  amp_copy(n, const_cast <double*>(x), y);
  amp_scale(n, alpha, y);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  amp_scale(N, alpha, X);
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  amp_scale(N, alpha, X);
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
  const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
  const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const float alpha, const float* A, const float* x,
  const float beta, float* y) {
  const enum CBLAS_ORDER Order=CblasRowMajor;
  //todo cpu version
  cblas_sgemv(Order, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const double alpha, const double* A, const double* x,
  const double beta, double* y) {
  const enum CBLAS_ORDER Order=CblasRowMajor;
  //todo cpu version
  cblas_dgemv(Order, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}


template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta, float* C) {
  // todo cpu version
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const double alpha, const double* A, const double* B, const double beta, double* C) {
  // todo cpu version
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}
template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  *y = cblas_sasum(n, x, 1);
}
template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  *y = cblas_dasum(n, x, 1);
}
void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  caffe_rng_uniform(n,r);
} 
template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  caffe_rng_uniform(n, a, b, r);
};
template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  caffe_rng_uniform(n, a, b, r);
};
template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  uint32_t result[n];
  memset(result, 0, sizeof(uint32_t)*n);
  array_view<uint32_t, 1> resultView(n, result);
  array_view<float, 1> xView(n, const_cast <float*>(x));
  array_view<float, 1> yView(n, const_cast <float*>(y));
  parallel_for_each(
    resultView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    uint32_t ret = 0;
    uint32_t u = static_cast<uint32_t>(xView[idx]) ^
                 static_cast<uint32_t>(yView[idx]);
    while(u)
    {
      u = u & (u - 1);
      ret ++;
    }
    resultView[idx] = ret;
  }
  );
  resultView.synchronize();
  uint32_t sum = 0;
  for(int i = 0; i < n; i++ ) {
    sum+=result[i];
  }
  return sum;
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  uint32_t result[n];
  memset(result, 0, sizeof(uint32_t)*n);
  array_view<uint32_t, 1> resultView(n, result);
  array_view<double, 1> xView(n, const_cast <double*>(x));
  array_view<double, 1> yView(n, const_cast <double*>(y));
  parallel_for_each(
    resultView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    uint32_t ret = 0;
    uint64_t u = static_cast<uint64_t>(xView[idx]) ^
                 static_cast<uint64_t>(yView[idx]);
    while(u)
    {
      u = u & (u - 1);
      ret ++;
    }
    resultView[idx] = ret;
  }
  );
  resultView.synchronize();
  uint32_t sum = 0;
  for(int i = 0; i < n; i++ ) {
    sum+=result[i];
  }
  return sum;
}
void caffe_gpu_memcpy(const size_t N, const void *X, void *Y)
{
  memcpy(Y,X,N);
}


template <>
    void caffe_gpu_rng_gaussian(const int N, const float mu, const float sigma, float* r) {
      float * v = new float[N];
      float * s = new float[N];
      float tempV2;
      float tempS;
      int flag = 0;
      int i = 0;
      while (i < N)
      {
        if (flag == 0)
        {
          float U1 = (float)rand() / RAND_MAX;
          float U2 = (float)rand() / RAND_MAX;
    
          float V1 = 2 * U1 - 1;
          float V2 = 2 * U2 - 1;
          float S = V1 * V1 + V2 * V2;
          if (S >= 1 || S == 0)
            continue;
          v[i] = V1;
          s[i] = S;
          tempV2 = V2;
          tempS = S;
        }
        else
        {
          v[i] = tempV2;
          s[i] = tempS;
        }
        flag = 1 - flag;
        i++;
      }
      array_view<float, 1> rView(N, r);
      array_view<float, 1> vView(N, v);
      array_view<float, 1> sView(N, s);
      parallel_for_each(
        rView.get_extent(),
        [=](index<1> idx) restrict(amp)
      {
        
        //rView[idx] = vView[idx] * Concurrency::fast_math::sqrt((float)-2 * fast_math::log((float)sView[idx]) / (float)sView[idx]) * sigma + mu;
      }
      );
      delete[] v;
      delete[] s;
      rView.synchronize();
    }



template <>
    void caffe_gpu_rng_gaussian(const int N, const double mu, const double sigma, double* r) {
      double * v = new double[N];
      double * s = new double[N];
      double tempV2;
      double tempS;
      int flag = 0;
      int i = 0;
      while (i < N)
      {
        if (flag == 0)
        {
          double U1 = (double)rand() / RAND_MAX;
          double U2 = (double)rand() / RAND_MAX;
    
          double V1 = 2 * U1 - 1;
          double V2 = 2 * U2 - 1;
          double S = V1 * V1 + V2 * V2;
          if (S >= 1 || S == 0)
            continue;
          v[i] = V1;
          s[i] = S;
          tempV2 = V2;
          tempS = S;
        }
        else
        {
          v[i] = tempV2;
          s[i] = tempS;
        }
        flag = 1 - flag;
        i++;
      }
      array_view<double, 1> rView(N, r);
      array_view<double, 1> vView(N, v);
      array_view<double, 1> sView(N, s);
      parallel_for_each(
        rView.get_extent(),
        [=](index<1> idx) restrict(amp)
      {
        //rView[idx] = vView[idx] * Concurrency::fast_math::sqrt((float)-2 * fast_math::log((float)sView[idx]) / (float)sView[idx]) * sigma + mu;
      }
      );
      delete[] v;
      delete[] s;
      rView.synchronize();
    }


#endif //USE_CPPAMP
}  // namespace caffe

