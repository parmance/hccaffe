#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <glog/logging.h>
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
//The size is the total memory size
void caffe_amp_malloc(void** ptr, size_t size, size_t element_size,
    bool is_int){
  if(is_int) {
    Concurrency::extent<1> eA(size/sizeof(int));
      // Allocating device array of given size
      // Use default device
      concurrency::accelerator currentAcc(accelerator::default_accelerator);
      Concurrency::array<int, 1> arr =
         Concurrency::array<int, 1>(eA, currentAcc.get_default_view());
      Concurrency::array_view<int>* avData =
        new Concurrency::array_view<int>(arr);
      avData->discard_data();
      *ptr = (void*)avData;
  } else {
    if(element_size == sizeof(float)){
      Concurrency::extent<1> eA(size/sizeof(float));
      // Allocating device array of given size
      // Use default device
      concurrency::accelerator currentAcc(accelerator::default_accelerator);
      Concurrency::array<float, 1> arr =
         Concurrency::array<float, 1>(eA, currentAcc.get_default_view());
      Concurrency::array_view<float>* avData =
        new Concurrency::array_view<float>(arr);
      avData->discard_data();
      *ptr = (void*)avData;
    } else if(element_size == sizeof(double)){
      Concurrency::extent<1> eA(size/sizeof(double));
      // Allocating device array of given size
      // Use default device
      concurrency::accelerator currentAcc(accelerator::default_accelerator);
     Concurrency::array<double, 1> arr =
       Concurrency::array<double, 1>(eA, currentAcc.get_default_view());
     Concurrency::array_view<double>* avData =
       new Concurrency::array_view<double>(arr);
     avData->discard_data();
     *ptr = (void*)avData;
    }
  }
}

void caffe_amp_free(void* ptr, size_t element_size, bool is_int){
  if (ptr) {
    if(is_int){
        delete (Concurrency::array_view<int> *)ptr;
    } else {
      if(element_size == sizeof(float))
        delete (Concurrency::array_view<float> *)ptr;
      else if(element_size == sizeof(double))
        delete (Concurrency::array_view<double> *)ptr;
    }
    ptr = NULL;
  }
}

void caffe_amp_D2H(void* src, void* dst, size_t element_size, bool is_int){
  if(is_int){
    Concurrency::array_view<int, 1>* avSrc =
      (Concurrency::array_view<int, 1>*)(src);
    Concurrency::copy(*avSrc, (int*)dst);

  } else {
    if(element_size == sizeof(float)){
      Concurrency::array_view<float, 1>* avSrc =
        (Concurrency::array_view<float, 1>*)(src);
      Concurrency::copy(*avSrc, (float*)dst);
    } else if (element_size == sizeof(double)){
      Concurrency::array_view<double, 1>* avSrc =
        (Concurrency::array_view<double, 1>*)(src);
      Concurrency::copy(*avSrc, (double*)dst);
    }
  }
}

void caffe_amp_H2D(void* src, void* dst, size_t element_size, bool is_int){
  if(is_int){
    Concurrency::array_view<int, 1>* avDst =
      (Concurrency::array_view<int, 1>*)(dst);
    Concurrency::copy((int*)src, *avDst);
  } else {
    if(element_size == sizeof(float)){
      Concurrency::array_view<float, 1>* avDst =
        (Concurrency::array_view<float, 1>*)(dst);
      Concurrency::copy((float*)src, *avDst);
    } else if (element_size == sizeof(double)){
      Concurrency::array_view<double, 1>* avDst =
        (Concurrency::array_view<double, 1>*)(dst);
      Concurrency::copy((double*)src, *avDst);
    }
  }
}

void caffe_amp_D2D(void* src, void* dst, size_t element_size, bool is_int){
  if(is_int){
    Concurrency::array_view<int, 1>* avSrc =
      (Concurrency::array_view<int, 1>*)(src);
    Concurrency::array_view<int, 1>* avDst =
      (Concurrency::array_view<int, 1>*)(dst);
    Concurrency::copy(*avSrc, *avDst);
  } else {
    if(element_size == sizeof(float)){
      Concurrency::array_view<float, 1>* avSrc =
        (Concurrency::array_view<float, 1>*)(src);
      Concurrency::array_view<float, 1>* avDst =
        (Concurrency::array_view<float, 1>*)(dst);
      Concurrency::copy(*avSrc, *avDst);
    } else if (element_size == sizeof(double)){
      Concurrency::array_view<double, 1>* avSrc =
        (Concurrency::array_view<double, 1>*)(src);
      Concurrency::array_view<double, 1>* avDst =
        (Concurrency::array_view<double, 1>*)(dst);
      Concurrency::copy(*avSrc, *avDst);
    }
  }
}

template <typename Dtype>
void abs_kernel(const int N, Dtype* a, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = aView[idx] >= 0 ? aView[idx] : -1 * aView[idx];
  } );
}

template <typename Dtype>
void sign_kernel(const int N, Dtype* a, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = aView[idx] == 0 ? 0 : (aView[idx] < 0 ? -1 : 1);
  } );
}

template <typename Dtype>
void sgnbit_kernel(const int N, Dtype* a, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = Concurrency::fast_math::signbit(aView[idx]);
  } );
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
void mul_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> bView =
      *((Concurrency::array_view<Dtype, 1>*)(b));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
      yView[idx] = (aView[idx] * bView[idx]);
  } );
}

template <>
void caffe_gpu_abs<float>(const int N,const float* a, float* y) {
  abs_kernel(N, const_cast <float*>(a), y);
}

template <>
void caffe_gpu_abs<double>(const int N,const double* a, double* y) {
  abs_kernel(N, const_cast <double*>(a), y);
}

template <>
void caffe_gpu_sign<float>(const int N,const float* a, float* y) {
  sign_kernel(N, const_cast <float*>(a), y);
}

template <>
void caffe_gpu_sign<double>(const int N,const double* a, double* y) {
  sign_kernel(N, const_cast <double*>(a), y);
}



template <>
void caffe_gpu_mul<float>(const int N, const float* a,
  const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel(N, const_cast <float*>(a), const_cast <float*>(b), y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
  const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel(N, const_cast <double*>(a), const_cast <double*>(b), y);
}

template <typename Dtype>
void div_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> bView =
      *((Concurrency::array_view<Dtype, 1>*)(b));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = (aView[idx] / bView[idx]);
  } );
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
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> bView =
      *((Concurrency::array_view<Dtype, 1>*)(b));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = (aView[idx] + bView[idx]);
  } );
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
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> bView =
      *((Concurrency::array_view<Dtype, 1>*)(b));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = (aView[idx] - bView[idx]);
  } );
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
  Concurrency::array_view<Dtype, 1> outView =
    *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp){
    outView[idx] = alpha;
  } );
}

template <>
void caffe_gpu_set<float>(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel(N, alpha, Y);
}

template <>
void caffe_gpu_set<double>(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel(N, alpha, Y);
}

template <typename Dtype>
void exp_kernel(const int N, Dtype* a, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = Concurrency::fast_math::exp(aView[idx]);
  } );
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
  Concurrency::array_view<Dtype, 1> outView =
    *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    outView[idx] += alpha;
  } );
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
  Concurrency::array_view<Dtype, 1> aView =
      *((Concurrency::array_view<Dtype, 1>*)(a));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = Concurrency::fast_math::pow(aView[idx], alpha);
  } );
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
  caffe_amp_D2D((void*)x, (void*)y, sizeof(float), false);
  amp_scale(n, alpha, y);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                                 double* y) {
  caffe_amp_D2D((void*)x, (void*)y, sizeof(double), false);
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
template <typename Dtype>
void caffe_gpu_gemv2(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const Dtype alpha, const Dtype* A, const int offseta,
  const Dtype* x, const int offsetx,
  const Dtype beta, Dtype* y, const int offsety);
template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const float alpha, const float* A, const float* x,
  const float beta, float* y) {
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
  amp.ampblas_dgemv(ampTransA, N, M, &alpha, const_cast<double*>(A), 0, N, const_cast<double*>(x), 0, 1, &beta, y, 0, 1);
}
template <>
void caffe_gpu_gemv2<float>(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const float alpha, const float* A, const int offseta,
  const float* x, const int offsetx,
  const float beta, float* y, const int offsety) {
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
  Concurrency::array_view<float, 1> A_mat =
    *((Concurrency::array_view<float, 1>*)(A));
  Concurrency::array_view<float, 1> X_mat =
    *((Concurrency::array_view<float, 1>*)(x));
  Concurrency::array_view<float, 1> Y_mat =
    *((Concurrency::array_view<float, 1>*)(y));

  amp.ampblas_sgemv2(ampTransA, N, M, &alpha, A_mat, offseta, N, X_mat, offsetx, 1, &beta, Y_mat, offsety, 1);
}


template <>
void caffe_gpu_gemv2<double>(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const double alpha, const double* A, const int offseta,
  const double* x, const int offsetx,
  const double beta, double* y, const int offsety) {
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
  Concurrency::array_view<double, 1> A_mat =
    *((Concurrency::array_view<double, 1>*)(A));
  Concurrency::array_view<double, 1> X_mat =
    *((Concurrency::array_view<double, 1>*)(x));
  Concurrency::array_view<double, 1> Y_mat =
    *((Concurrency::array_view<double, 1>*)(y));

  amp.ampblas_dgemv2(ampTransA, N, M, &alpha, A_mat, offseta, N, X_mat, offsetx, 1, &beta, Y_mat, offsety, 1);

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
  amp.ampblas_sgemm(colMajor, ampTransB, ampTransA, N, M, K, &alpha, const_cast<float*>(B),
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
  amp.ampblas_dgemm(colMajor, ampTransB, ampTransA, N, M, K, &alpha, const_cast<double*>(B),
                ldb, const_cast<double*>(A), lda, &beta, C, N, 0, 0, 0);

}
template <typename Dtype>
void caffe_gpu_gemm2(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB,
  const int M, const int N, const int K,
  const Dtype alpha, const Dtype* A, const int offet_A,const Dtype* B,
  const int offset_B, const Dtype beta, Dtype* C, const int offset_C);



template <>
void caffe_gpu_gemm2<float>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB,
  const int M, const int N, const int K,
  const float alpha, const float* A, const int offset_A,const float* B,
  const int offset_B, const float beta, float* C, const int offset_C) {
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
  Concurrency::array_view<float, 1> A_mat =
    *((Concurrency::array_view<float, 1>*)(A));
  Concurrency::array_view<float, 1> B_mat =
    *((Concurrency::array_view<float, 1>*)(B));
  Concurrency::array_view<float, 1> C_mat =
    *((Concurrency::array_view<float, 1>*)(C));
    //PPAStartCpuEventFunc(GPU_GEMM);
    amp.ampblas_sgemm2(colMajor,ampTransB, ampTransA, N, M, K, &alpha, B_mat,
                ldb, A_mat, lda, &beta, C_mat, N, offset_B, offset_A, offset_C);
   // PPAStopCpuEventFunc(GPU_GEMM);
}


template <>
void caffe_gpu_gemm2<double>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB,
  const int M, const int N, const int K,
  const double alpha, const double* A, const int offset_A,const double* B,
  const int offset_B, const double beta, double* C, const int offset_C) {
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
  Concurrency::array_view<double, 1> A_mat =
    *((Concurrency::array_view<double, 1>*)(A));
  Concurrency::array_view<double, 1> B_mat =
    *((Concurrency::array_view<double, 1>*)(B));
  Concurrency::array_view<double, 1> C_mat =
    *((Concurrency::array_view<double, 1>*)(C));
  //Concurrency::array_view<float> A_mat(all_A, const_cast<float*>(A));
  //Concurrency::array_view<float> B_mat(all_B, const_cast<float*>(B));
  //Concurrency::array_view<float> C_mat(all_C, C);
  //printf("======All_A = %d,ALL_B = %d, ALL_C = %d, group = %d, offset_A=%d,offset_B=%d,offset_C=%d,M=%d,N=%d,K=%d\n",all_A,all_B,all_C,group,offset_A,offset_B,offset_C,M,N,K);
    //PPAStartCpuEventFunc(GPU_GEMM);
    amp.ampblas_dgemm2(colMajor, ampTransB, ampTransA, N, M, K, &alpha, B_mat,
                ldb, A_mat, lda, &beta, C_mat, N, offset_B, offset_A, offset_C);
   // PPAStopCpuEventFunc(GPU_GEMM);
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

#define MAX 65536
#define FACTOR 2053
#define CONSTANT 13849
unsigned int uirnd_kernel(unsigned int &ri) restrict(amp){
  int temp;
  temp = (int)(ri / (unsigned int)MAX);
  ri = ri - temp*(unsigned int)MAX;
  ri = (unsigned int)FACTOR * ri + (unsigned int)CONSTANT;
  temp = (int)(ri / (unsigned int)MAX);
  ri = ri - temp * (unsigned int)MAX;
  return ri / (unsigned int)MAX;
}

float srnd_kernel(float &ri) restrict(amp){
  int temp;
  temp = (int)(ri / (float)MAX);
  ri = ri - temp*(float)MAX;
  ri = (float)FACTOR * ri + (float)CONSTANT;
  temp = (int)(ri / (float)MAX);
  ri = ri - temp * (float)MAX;
  return ri / (float)MAX;
}

double drnd_kernel(double &ri) restrict(amp){
  int temp;
  temp = (int)(ri / (double)MAX);
  ri = ri - temp*(double)MAX;
  ri = (double)FACTOR * ri + (double)CONSTANT;
  temp = (int)(ri / (double)MAX);
  ri = ri - temp * (double)MAX;
  return ri / (double)MAX;
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
 unsigned int temp[n];
  caffe_rng_uniform(n,temp);
  array_view<unsigned int, 1> tempView(n, temp);
  array_view<unsigned int, 1> rView = *((Concurrency::array_view<unsigned int, 1>*)(r));
  Concurrency::extent<1> e(n);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
  {
    rView[idx] = tempView(idx);
  } );
}


template <>
void caffe_gpu_rng_uniform<float>(const int N, const float a, const float b,float* r) {
  array_view<float, 1> rView = *((Concurrency::array_view<float, 1>*)(r));
  float coefficient =  (float)rand() / RAND_MAX;
  Concurrency::extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
  {
    float seed = (float)idx[0] * coefficient;
    rView[idx] = srnd_kernel(seed) * (b - a) + a;
  } );
};

template <>
void caffe_gpu_rng_uniform<double>(const int N, const double a, const double b, double* r) {
  array_view<double, 1> rView = *((Concurrency::array_view<double, 1>*)(r));
  double coefficient =  (double)rand() / RAND_MAX;
  Concurrency::extent<1> e(N);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp)
  {
    double seed = (double)idx[0] * coefficient;
    rView[idx] = drnd_kernel(seed) * (b - a) + a;
  } );
};

template <>
void caffe_gpu_rng_gaussian(const int N, const float mu, const float sigma, float* r) {
  array_view<float, 1> rView = *((Concurrency::array_view<float, 1>*)(r));
  float coefficient =  (float)rand() / RAND_MAX;
  Concurrency::extent<1> e(N);
  parallel_for_each(
    e,
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
}


template <>
void caffe_gpu_rng_gaussian(const int N, const double mu, const double sigma, double* r) {
  array_view<double, 1> rView = *((Concurrency::array_view<double, 1>*)(r));
  double coefficient = (double)rand() / RAND_MAX;
  Concurrency::extent<1> e(N);
  parallel_for_each(
    e,
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
  LOG(FATAL) << "Instead of caffe_gpu_memcpy with caffe_amp_X2X.";
}

#endif //USE_CPPAMP
}  // namespace caffe

