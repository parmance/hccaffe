#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <limits>
#include "caffe/util/math_functions.hpp"
#include "amp.h"
#include "amp_math.h"
#include "caffe/common.hpp"
#include "cppamp/ampblaslib.h"

using namespace concurrency;
using namespace Concurrency::fast_math;


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
  array_view<float, 1> xView(n, const_cast <float*>(x));
  array_view<float, 1> yView(n, const_cast <float*>(y));
  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<float,1> global_buffer(tile_count);
  concurrency::array_view<float,1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp)
  {
    // shared tile buffer
    tile_static float local_buffer[TILE_SIZE];
    // indexes
    int idx = tid.global[0];
    // this threads's shared memory pointer
    float& smem = local_buffer[ tid.local[0] ];
    // initialize local buffer
    smem = 0.0f;
    // fold data into local buffer
    while (idx < n) {
      // reduction of smem and X[idx] with results stored in smem
      smem += xView[concurrency::index<1>(idx)] * yView[concurrency::index<1>(idx)] ;
      // next chunk
      idx += thread_count;
    }
    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    float *mem = &smem;
    // unrolled for performance
    if (local < 128) { mem[0] = mem[0] + mem[128]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; } tid.barrier.wait_with_tile_static_memory_fence();
    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  });
  // 2nd pass reduction
  std::vector<float> host_buffer(global_buffer);
  *out = *std::max_element(host_buffer.begin(), host_buffer.end());
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
  double * out) {
  array_view<double, 1> xView(n, const_cast <double*>(x));
  array_view<double, 1> yView(n, const_cast <double*>(y));
  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<double,1> global_buffer(tile_count);
  concurrency::array_view<double,1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp)
  {
    // shared tile buffer
    tile_static double local_buffer[TILE_SIZE];
    // indexes
    int idx = tid.global[0];
    // this threads's shared memory pointer
    double& smem = local_buffer[ tid.local[0] ];
    // initialize local buffer
    smem = 0.0f;
    // fold data into local buffer
    while (idx < n) {
      // reduction of smem and X[idx] with results stored in smem
      smem += xView[concurrency::index<1>(idx)] * yView[concurrency::index<1>(idx)];
      // next chunk
      idx += thread_count;
    }
    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    double *mem = &smem;
    // unrolled for performance
    if (local < 128) { mem[0] = mem[0] + mem[128]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; } tid.barrier.wait_with_tile_static_memory_fence();
    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  });
  // 2nd pass reduction
  std::vector<double> host_buffer(global_buffer);
  *out = *std::max_element(host_buffer.begin(), host_buffer.end());
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
                                float* y){
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
  AMPBLAS_TRANS ampTransA = trans;
  Ampblaslibrary amp;
  if(TransA == CblasTrans)
  {
      ampTransA = noTrans;
  }
  if(TransA == CblasConjTrans)
  {
      ampTransA = conjugate;
  }
  amp.ampblas_sgemv(ampTransA, N, M, &alpha, const_cast<float*>(A), 0, N, const_cast<float*>(x), 0, 1, &beta, y, 0, 1);
}


template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const double alpha, const double* A, const double* x,
  const double beta, double* y) {
  
  const enum CBLAS_ORDER Order=CblasRowMajor;
  AMPBLAS_TRANS ampTransA = trans;
  Ampblaslibrary amp;
  if(TransA == CblasTrans)
  {
      ampTransA = noTrans;
  }
  if(TransA == CblasConjTrans)
  {
      ampTransA = conjugate;
  }
  //cblas_sgemv(Order, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
  amp.ampblas_dgemv(ampTransA, N, M, &alpha, const_cast<double*>(A), 0, N, const_cast<double*>(x), 0, 1, &beta, y, 0, 1);
}


template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta, float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  AMPBLAS_TRANS ampTransA = noTrans;
  AMPBLAS_TRANS ampTransB = noTrans;
  Ampblaslibrary amp;
  if(TransA == CblasTrans)
  {
      ampTransA = trans;
  }
  if(TransA == CblasConjTrans)
  {
      ampTransA = conjugate;
  }

  if(TransB == CblasTrans)
  {
      ampTransB = trans;
  }

  if(TransB == CblasConjTrans)
  {
      ampTransB = conjugate;
  }
  amp.ampblas_sgemm(ampTransB, ampTransA, N, M, K, &alpha, const_cast<float*>(B),
                ldb, const_cast<float*>(A), lda, &beta, C, N, 0, 0, 0);
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const double alpha, const double* A, const double* B, const double beta, double* C) {
   int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  AMPBLAS_TRANS ampTransA = noTrans;
  AMPBLAS_TRANS ampTransB = noTrans;
  Ampblaslibrary amp;
  if(TransA == CblasTrans)
  {
      ampTransA = trans;
  }
  if(TransA == CblasConjTrans)
  {
      ampTransA = conjugate;
  }

  if(TransB == CblasTrans)
  {
      ampTransB = trans;
  }

  if(TransB == CblasConjTrans)
  {
      ampTransB = conjugate;
  }
  amp.ampblas_dgemm(ampTransB, ampTransA, N, M, K, &alpha, const_cast<double*>(B),
                ldb, const_cast<double*>(A), lda, &beta, C, N, 0, 0, 0);

}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  array_view<float, 1> xView(n, const_cast <float*>(x));
  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<float,1> global_buffer(tile_count);
  concurrency::array_view<float,1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp)
  {
    // shared tile buffer
    tile_static float local_buffer[TILE_SIZE];
    // indexes
    int idx = tid.global[0];
    // this threads's shared memory pointer
    float& smem = local_buffer[ tid.local[0] ];
    // initialize local buffer
    smem = 0.0f;
    // fold data into local buffer
    while (idx < n) {
      // reduction of smem and X[idx] with results stored in smem
      smem += Concurrency::fast_math::fabs(xView[concurrency::index<1>(idx)]);
      // next chunk
      idx += thread_count;
    }
    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    float *mem = &smem;
    // unrolled for performance
    if (local < 128) { mem[0] = mem[0] + mem[128]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; } tid.barrier.wait_with_tile_static_memory_fence();
    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  } );
  // 2nd pass reduction
  std::vector<float> host_buffer(global_buffer);
  *y = *std::max_element(host_buffer.begin(), host_buffer.end());
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  array_view<double, 1> xView(n, const_cast <double*>(x));
  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<double,1> global_buffer(tile_count);
  concurrency::array_view<double,1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp)
  {
    // shared tile buffer
    tile_static double local_buffer[TILE_SIZE];
    // indexes
    int idx = tid.global[0];
    // this threads's shared memory pointer
    double& smem = local_buffer[ tid.local[0] ];
    // initialize local buffer
    smem = 0.0f;
    // fold data into local buffer
    while (idx < n) {
      // reduction of smem and X[idx] with results stored in smem
      smem += Concurrency::fast_math::fabs(xView[concurrency::index<1>(idx)]);
      // next chunk
      idx += thread_count;
    }
    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    double *mem = &smem;
    // unrolled for performance
    if (local < 128) { mem[0] = mem[0] + mem[128]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; } tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; } tid.barrier.wait_with_tile_static_memory_fence();
    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  } );
  // 2nd pass reduction
  std::vector<double> host_buffer(global_buffer);
  *y = *std::max_element(host_buffer.begin(), host_buffer.end());
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  caffe_rng_uniform(n,r);
}

#define MAX 65536
#define FACTOR 2053
#define CONSTANT 13849
float srnd_kernel(float &ri) restrict(amp){
  int temp;
  temp = (int)(ri / (float)MAX);
  ri = ri - temp*(float)MAX;
  ri = (float)FACTOR * ri + (float)CONSTANT;
  temp = (int)(ri / (float)MAX);
  ri = ri - temp * (float)MAX;
  return ri / (float)MAX;
};

double drnd_kernel(double &ri) restrict(amp){
  int temp;
  temp = (int)(ri / (double)MAX);
  ri = ri - temp*(double)MAX;
  ri = (double)FACTOR * ri + (double)CONSTANT;
  temp = (int)(ri / (double)MAX);
  ri = ri - temp * (double)MAX;
  return ri / (double)MAX;
};

template <>
void caffe_gpu_rng_uniform<float>(const int N, const float a, const float b,float* r) {
  array_view<float, 1> rView(N, r);
  int rnd = (int)((long)r);
  int coefficient = (rnd % MAX);
  parallel_for_each(
    rView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    float seed = (float)idx[0] * coefficient;
    rView[idx] = srnd_kernel(seed) * (b - a) + a;
  } );
  rView.synchronize();
};

template <>
void caffe_gpu_rng_uniform<double>(const int N, const double a, const double b,double* r) {
  array_view<double, 1> rView(N, r);
  int rnd = (int)((long)r);
  int coefficient = (rnd % MAX);
  parallel_for_each(
    rView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    double seed = (double)idx[0] * coefficient;
    rView[idx] = drnd_kernel(seed) * (b - a) + a;
  } );
  rView.synchronize();
};

template <>
void caffe_gpu_rng_gaussian(const int N, const float mu, const float sigma, float* r) {
  array_view<float, 1> rView(N, r);
  float coefficient =  (float)rand() / RAND_MAX;
  parallel_for_each(
    rView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    float seed = (float)idx[0] * coefficient;
    float V1 = 0.0, V2 = 0.0, S=0.0;
    do {
      V1 = 2 * srnd_kernel(seed) - 1;
      V2 = 2 * srnd_kernel(seed) - 1;
      S = V1 * V1 + V2 * V2;
    } while ((S >= 1.0) || (S == 0.0));
	float temp = sqrt(-2.0 * log(S) / S) * sigma ;
    if (2 * idx[0] < N)
      rView[2 * idx] = V1 * temp + mu;
    if (2*idx[0] + 1 < N)
      rView[2 * idx + 1] = V2 * temp + mu;
  } );
  rView.synchronize();
}


template <>
void caffe_gpu_rng_gaussian(const int N, const double mu, const double sigma, double* r) {
  array_view<double, 1> rView(N, r);
  double coefficient = (double)rand() / RAND_MAX;
  parallel_for_each(
    rView.get_extent(),
    [=](index<1> idx) restrict(amp)
  {
    double seed = (double)idx[0] * coefficient;
    double V1 = 0.0, V2 = 0.0, S=0.0;
    do {
      V1 = 2 * drnd_kernel(seed) - 1;
      V2 = 2 * drnd_kernel(seed) - 1;
      S = V1 * V1 + V2 * V2;
    } while ((S >= 1.0) || (S == 0.0));
	double temp = sqrt(-2.0 * log(S) / S) * sigma ;
    if (2 * idx[0] < N)
      rView[2 * idx] = V1 * temp + mu;
    if (2*idx[0] + 1 < N)
      rView[2 * idx + 1] = V2 * temp + mu;
  } );
  rView.synchronize();
}



template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  uint32_t result[n];
  uint32_t ax[n];
  uint32_t ay[n];
  for(int i = 0; i < n; ++i ) {
    ax[i] = static_cast<uint32_t>(x[i]);
    ay[i] = static_cast<uint32_t>(y[i]);
  }

  array_view<uint32_t, 1> resultView(n, result);
  array_view<uint32_t, 1> xView(n, ax);
  array_view<uint32_t, 1> yView(n, ay);
  parallel_for_each(resultView.get_extent(), [=](index<1> idx) restrict(amp) {
    uint32_t ret = 0;
    uint32_t u = xView[idx] ^ yView[idx];
    while(u) {
      u = u & (u - 1);
      ret ++;
    }
    resultView[idx] = ret;
  } );
  resultView.synchronize();
  uint32_t sum = 0;
  for(int i = 0; i < n; ++i ) {
    sum+=result[i];
  }
  return sum;
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  uint32_t result[n];
  uint64_t ax[n];
  uint64_t ay[n];
  for(int i = 0; i < n; ++i ) {
    ax[i] = static_cast<uint64_t>(x[i]);
    ay[i] = static_cast<uint64_t>(y[i]);
  }
  array_view<uint32_t, 1> resultView(n, result);
  array_view<uint64_t, 1> xView(n, ax);
  array_view<uint64_t, 1> yView(n, ay);
  parallel_for_each(resultView.get_extent(), [=](index<1> idx) restrict(amp) {
    uint32_t ret = 0;
    uint64_t u = xView[idx] ^ yView[idx];
    while(u) {
      u = u & (u - 1);
      ret ++;
    }
    resultView[idx] = ret;
  } );
  resultView.synchronize();
  uint32_t sum = 0;
  for(int i = 0; i < n; ++i ) {
    sum+=result[i];
  }
  return sum;
}

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y){
  memcpy(Y,X,N);
}

#endif //USE_CPPAMP
}  // namespace caffe

