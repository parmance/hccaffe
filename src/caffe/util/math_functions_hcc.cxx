#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/type_traits/is_same.hpp>
#include <glog/logging.h>
#include <limits>
#include <vector>
#include "hc.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "hcblaslib.h"
#include "hc_math.hpp"
#include "hc_am.hpp"
using namespace hc::fast_math;
#define TILE_SIZE 256
#define MAX_TILES 1024

namespace caffe {
template <typename Dtype>
void transform_gpu(void* src, void* dst, int top_offset, int N_,
                   int M_, int packing_num) {
  hc::extent<1> e(M_ * packing_num);
  Dtype* dstt = static_cast<Dtype*>(dst);
  Dtype* srct = static_cast<Dtype*>(src);
  parallel_for_each(e, [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    int i = 0;

    for (i = 0 ; i < N_; i++) {
      if (packing_num == 1)
        dstt[top_offset + (idx[0] / packing_num)* N_ + i] =
          srct[idx[0] * N_ + i];
      else
        dstt[top_offset + (idx[0] % packing_num * M_ + idx[0] / packing_num)
             * N_ + i] = srct[idx[0] * N_ + i];
    }
  }).wait();
}
template void transform_gpu<float>(void* src, void* dst, int top_offset,
                                   int N_, int M_, int packing_num);
template void transform_gpu<double>(void* src, void* dst, int top_offset,
                                    int N_, int M_, int packing_num);

template <typename Dtype>
void opttrans(void* data_im, int im_offset, int channels,
              int height, int width, void* data_opt, int opt_offset, int packing_num) {
  int num_kernels = channels * height * width * packing_num;
  hc::extent<1> e(num_kernels);
  Dtype* data_optt = static_cast<Dtype*>(data_opt);
  Dtype* data_imt = static_cast<Dtype*>(data_im);
  parallel_for_each(e, [ = ](hc::index<1> idx) __attribute__((hc, cpu)) {
    int w = idx[0] % width;
    int h = (idx[0] / width) % height;
    int c = idx[0] / (width * height) % channels;
    int im = idx[0] / width / height / channels;
    int opt_index = c * height * packing_num * width +
                    h * packing_num * width + im * width + w;
    data_optt[opt_offset + opt_index] = data_imt[im_offset + idx[0]];
  }).wait();
}
template void opttrans<float>(void* data_im, int im_offset, int channels,
                              int height, int width, void* data_opt, int opt_offset, int optnum);
template void opttrans<double>(void* data_im, int im_offset, int channels,
                               int height, int width, void* data_opt, int opt_offset, int optnum);



// The size is the total memory size
void caffe_hcc_malloc(void** ptr, size_t size, size_t element_size,
                      bool is_int) {
  // Use default device
  hc::accelerator currentAcc(L"default");

  if (is_int) {
    if (element_size == sizeof(int)) {
      int* data = hc::am_alloc(size, currentAcc, 0);
      *ptr = static_cast<void*>(data);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_malloc.";
    }
  } else {
    if (element_size == sizeof(float)) {
      float* data = hc::am_alloc(size, currentAcc, 0);
      *ptr = static_cast<void*>(data);
    } else if (element_size == sizeof(double)) {
      double* data = hc::am_alloc(size, currentAcc, 0);
      *ptr = static_cast<void*>(data);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_malloc.";
    }
  }
}

void caffe_hcc_malloc(void** ptr, void* src, size_t size, size_t element_size,
                      bool is_int) {
  // Use default device
  hc::accelerator currentAcc(L"default");

  if (is_int) {
    if (element_size == sizeof(int)) {
      int* data = hc::am_alloc(size, currentAcc, 0);
      hc::am_copy(data, src, size);
      *ptr = static_cast<void*>(data);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_malloc.";
    }
  } else {
    float* data = hc::am_alloc(size, currentAcc, 0);
    hc::am_copy(data, src, size);
    *ptr = static_cast<void*>(data);

    if (element_size == sizeof(float)) {
    } else if (element_size == sizeof(double)) {
      double* data = hc::am_alloc(size, currentAcc, 0);
      hc::am_copy(data, src, size);
      *ptr = static_cast<void*>(data);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_malloc.";
    }
  }
}

void caffe_hcc_free(void* ptr, size_t element_size, bool is_int) {
  if (ptr) {
    if (is_int) {
      if (element_size == sizeof(int)) {
        hc::am_free(ptr);
      } else {
        LOG(FATAL) << "Wrong element size for caffe_hcc_free.";
      }
    } else {
      if (element_size == sizeof(float)) {
        hc::am_free(ptr);
      } else if (element_size == sizeof(double)) {
        hc::am_free(ptr);
      } else {
        LOG(FATAL) << "Wrong element size for caffe_hcc_free.";
      }
    }

    ptr = NULL;
  }
}

void caffe_hcc_D2H(size_t size, void* src, void* dst, size_t element_size, bool is_int) {
  if (src == NULL || dst == NULL) {
    LOG(FATAL) << "Wrong source or destination for caffe_hcc_D2H.";
  }

  if (is_int) {
    if (element_size == sizeof(int)) {
      hc::am_copy(dst, src, size);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_D2H.";
    }
  } else {
    if (element_size == sizeof(float)) {
      hc::am_copy(dst, src, size);
    } else if (element_size == sizeof(double)) {
      hc::am_copy(dst, src, size);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_D2H.";
    }
  }
}

void caffe_hcc_H2D(void* src, void* dst, size_t element_size, bool is_int) {
  if (src == NULL || dst == NULL) {
    LOG(FATAL) << "Wrong source or destination for caffe_hcc_H2D.";
  }

  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo dstInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&dstInfo, dst);
  size_t size = dstInfo._sizeBytes;

  if (is_int) {
    if (element_size == sizeof(int)) {
      hc::am_copy(dst, src, size);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_H2D.";
    }
  } else {
    if (element_size == sizeof(float)) {
      hc::am_copy(dst, src, size);
    } else if (element_size == sizeof(double)) {
      hc::am_copy(dst, src, size);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_H2D.";
    }
  }
}

void caffe_hcc_D2D(void* src, void* dst, size_t element_size, bool is_int) {
  if (src == NULL || dst == NULL) {
    LOG(FATAL) << "Wrong source or destination for caffe_hcc_D2D.";
  }

  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo dstInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&dstInfo, dst);
  size_t size = dstInfo._sizeBytes;

  if (is_int) {
    if (element_size == sizeof(int)) {
      hc::am_copy(dst, src, size);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_D2D.";
    }
  } else {
    if (element_size == sizeof(float)) {
      hc::am_copy(dst, src, size);
    } else if (element_size == sizeof(double)) {
      hc::am_copy(dst, src, size);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_hcc_D2D.";
    }
  }
}

template <typename Dtype>
void caffe_hcc_copy(int N, void* src, void* dst,
                    int srcOffset, int dstOffset) {
  Dtype* dstt = static_cast<Dtype*>(dst);
  Dtype* srct = static_cast<Dtype*>(src);
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo dstInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&dstInfo, dst);
  hc::AmPointerInfo srcInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&srcInfo, src);
  size_t numSrcElts = srcInfo._sizeBytes / sizeof(Dtype);
  size_t numDestElts = dstInfo._sizeBytes / sizeof(Dtype);

  if (src == NULL || dst == NULL ||
      N > numSrcElts - srcOffset ||
      N > numDestElts - dstOffset) {
    LOG(FATAL) << "Wrong Parameters for caffe_hcc_copy.";
  }

  if (srcOffset == 0 && dstOffset == 0 &&
      N == numSrcElts &&
      N <= numDestElts) {
    caffe_hcc_D2D(src, dst, sizeof(Dtype), boost::is_same<Dtype, int>::value);
  } else {
    hc::am_copy(dstt + dstOffset, srct + srcOffset, N * sizeof(Dtype));
  }
}

template void caffe_hcc_copy<int>(int N, void* src, void* dst,
                                  int srcOffset, int dstOffset);
template void caffe_hcc_copy<float>(int N, void* src, void* dst,
                                    int srcOffset, int dstOffset);
template void caffe_hcc_copy<double>(int N, void* src, void* dst,
                                     int srcOffset, int dstOffset);

template <typename Dtype>
void caffe_hcc_copy_H2D(int N, void* src, void* dst, int dstOffset) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo dstInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&dstInfo, dst);
  size_t numDestElts = dstInfo._sizeBytes / sizeof(Dtype);
  Dtype* dstt = static_cast<Dtype*>(dst);

  if (src == NULL || dst == NULL ||
      N > numDestElts - dstOffset) {
    LOG(FATAL) << "Wrong Parameters for caffe_hcc_copy_H2D.";
  }

  hc::am_copy(dstt + dstOffset, src, N * sizeof(Dtype));
}

template void caffe_hcc_copy_H2D<int>(int N, void* src, void* dst,
                                      int dstOffset);
template void caffe_hcc_copy_H2D<float>(int N, void* src, void* dst,
                                        int dstOffset);
template void caffe_hcc_copy_H2D<double>(int N, void* srci, void* dst,
    int dstOffset);

template <typename Dtype>
void caffe_hcc_copy_D2H(int N, void* src, void* dst, int srcOffset) {
  hc::accelerator currentAcc(L"default");
  hc::AmPointerInfo srcInfo(0, 0, 0, currentAcc, 0, 0);
  hc::am_memtracker_getinfo(&srcInfo, src);
  size_t numSrcElts = srcInfo._sizeBytes / sizeof(Dtype);

  if (src == NULL || dst == NULL ||
      N > numSrcElts - srcOffset) {
    LOG(FATAL) << "Wrong Parameters for caffe_hcc_copy_D2H.";
  }

  Dtype* srct = static_cast<Dtype*>(src);
  hc::am_copy(dst, srct + srcOffset, N * sizeof(Dtype));
}


template void caffe_hcc_copy_D2H<int>(int N, void* src, void* dst,
                                      int srcOffset);
template void caffe_hcc_copy_D2H<float>(int N, void* src, void* dst,
                                        int srcOffset);
template void caffe_hcc_copy_D2H<double>(int N, void* src, void* dst,
    int srcOffset);

template <typename Dtype>
void abs_kernel(const int N, Dtype* a, Dtype* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = a[idx[0]] >= 0 ? a[idx[0]] : -1 * a[idx[0]];
  }).wait();
}

template <typename Dtype>
void sign_kernel(const int N, Dtype* a, Dtype* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = a[idx[0]] == 0 ? 0 : (a[idx[0]] < 0 ? -1 : 1);
  }).wait();
}

template <typename Dtype>
void sgnbit_kernel(const int N, Dtype* a, Dtype* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = hc::fast_math::signbit(a[idx[0]]);
  }).wait();
}

template<>
void caffe_gpu_sgnbit<float>(const int n, const float* x, float* y) {
  sgnbit_kernel(n, const_cast <float*>(x),  y);
}
template<>
void caffe_gpu_sgnbit<double>(const int n, const double* x, double* y) {
  sgnbit_kernel(n, const_cast <double*>(x),  y);
}

template <typename Dtype>
void mul_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = (a[idx[0]] * b[idx[0]]);
  }).wait();
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  abs_kernel(N, const_cast <float*>(a), y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  abs_kernel(N, const_cast <double*>(a), y);
}

template <>
void caffe_gpu_sign<float>(const int N, const float* a, float* y) {
  sign_kernel(N, const_cast <float*>(a), y);
}

template <>
void caffe_gpu_sign<double>(const int N, const double* a, double* y) {
  sign_kernel(N, const_cast <double*>(a), y);
}



template <>
void caffe_gpu_mul<float>(const int N, const float* a,
                          const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel(N, const_cast<float*>(a), const_cast<float*>(b), y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
                           const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel(N, const_cast<double*>(a), const_cast<double*>(b), y);
}

template <typename Dtype>
void div_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = (a[idx[0]] / b[idx[0]]);
  }).wait();
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
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = (a[idx[0]] + b[idx[0]]);
  }).wait();
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
                          float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel(N, const_cast <float*>(a), const_cast <float*>(b), y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
                           double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel(N, const_cast <double*>(a), const_cast <double*>(b), y);
}

template <typename Dtype>
void sub_kernel(const int N, Dtype* a, Dtype* b, Dtype* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = (a[idx[0]] - b[idx[0]]);
  }).wait();
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
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = alpha;
  }).wait();
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
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = hc::fast_math::exp(a[idx[0]]);
  }).wait();
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
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] += alpha;
  }).wait();
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
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    y[idx[0]] = hc::fast_math::pow(a[idx[0]], alpha);
  }).wait();
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
  // runtime sizes
  unsigned int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count : MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  hc::array<float, 1> global_buffer(tile_count);
  hc::array_view<float, 1> global_buffer_view(global_buffer);
  // configuration
  hc::extent<1> extent(thread_count);
  hc::parallel_for_each(
    extent.tile(TILE_SIZE),
  [ = ] (hc::tiled_index<1>& tid) __attribute__((hc, cpu)) {
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
      smem += x[idx] *
              y[idx];
      // next chunk
      idx += thread_count;
    }

    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    float* mem = &smem;

    // unrolled for performance
    if (local < 128) {
      mem[0] = mem[0] + mem[128];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  64) {
      mem[0] = mem[0] + mem[ 64];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  32) {
      mem[0] = mem[0] + mem[ 32];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  16) {
      mem[0] = mem[0] + mem[ 16];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   8) {
      mem[0] = mem[0] + mem[  8];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   4) {
      mem[0] = mem[0] + mem[  4];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   2) {
      mem[0] = mem[0] + mem[  2];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   1) {
      mem[0] = mem[0] + mem[  1];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  }).wait();
  // 2nd pass reduction
  std::vector<float> host_buffer(global_buffer);
  *out = *std::max_element(host_buffer.begin(), host_buffer.end());
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
                           double* out) {
  // runtime sizes
  unsigned int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count : MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  hc::array<double, 1> global_buffer(tile_count);
  hc::array_view<double, 1> global_buffer_view(global_buffer);
  // configuration
  hc::extent<1> extent(thread_count);
  hc::parallel_for_each(
    extent.tile(TILE_SIZE),
  [ = ] (hc::tiled_index<1>& tid) __attribute__((hc, cpu)) {
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
      smem += x[idx] *
              y[idx];
      // next chunk
      idx += thread_count;
    }

    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    double* mem = &smem;

    // unrolled for performance
    if (local < 128) {
      mem[0] = mem[0] + mem[128];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  64) {
      mem[0] = mem[0] + mem[ 64];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  32) {
      mem[0] = mem[0] + mem[ 32];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  16) {
      mem[0] = mem[0] + mem[ 16];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   8) {
      mem[0] = mem[0] + mem[  8];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   4) {
      mem[0] = mem[0] + mem[  4];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   2) {
      mem[0] = mem[0] + mem[  2];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   1) {
      mem[0] = mem[0] + mem[  1];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  }).wait();
  // 2nd pass reduction
  std::vector<double> host_buffer(global_buffer);
  *out = *std::max_element(host_buffer.begin(), host_buffer.end());
}

void hc_axpy(int N, float alpha, float* x, float* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    int i = idx[0];

    if (i < N) {
      y[i] = alpha * x[i] + y[i];
    }
  }).wait();
}

void hc_axpy(int N, double alpha, double* x, double* y) {
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    int i = idx[0];

    if (i < N) {
      y[i] = alpha * x[i] + y[i];
    }
  }).wait();
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
                           float* Y) {
  hc_axpy(N, alpha, const_cast <float*>(X), Y);
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
                            double* Y) {
  hc_axpy(N, alpha, const_cast <double*>(X), Y);
}


template <typename Dtype>
void hc_scale(const int N, const Dtype value, Dtype* x)
{
  hc::extent<1> e(N);
	hc::parallel_for_each(
      e, [=](hc::index<1> idx) __attribute__((hc, cpu)) {
		x[idx[0]] *= value;
	}	);
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float* x,
                            float* y) {
  caffe_hcc_D2D(static_cast<void*>(const_cast<float*>(x)),
                static_cast<void*>(const_cast<float*>(y)), sizeof(float),
                false);
  hc_scale(n, alpha, y);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double* x,
                             double* y) {
  caffe_hcc_D2D(static_cast<void*>(const_cast<double*>(x)),
                static_cast<void*>(const_cast<double*>(y)), sizeof(double),
                false);
  hc_scale(n, alpha, y);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X) {
  hc_scale(N, alpha, X);
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X) {
  hc_scale(N, alpha, X);
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
                           const int N, const float alpha, float* A, const int offseta,
                           float* x, const int offsetx,
                           const float beta, float* y, const int offsety) {
  hcblasTranspose ampTransA = Trans;
  Hcblaslibrary hc;

  if (TransA == CblasTrans) {
    ampTransA = NoTrans;
  }

  if (TransA == CblasConjTrans) {
    ampTransA = NoTrans;
  }

  // target the default accelerator
  hc::accelerator currentAcc(L"default");
  hc::accelerator_view accl_view = currentAcc.get_default_view();
  hc.hcblas_sgemv(accl_view, ColMajor, ampTransA, N, M, alpha, A, offseta, N, x,
                  offsetx, 1, beta, y, offsety, 1);
}


template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const double alpha, double* A, const int offseta,
                            double* x, const int offsetx,
                            const double beta, double* y, const int offsety) {
  hcblasTranspose ampTransA = Trans;
  Hcblaslibrary hc;

  if (TransA == CblasTrans) {
    ampTransA = NoTrans;
  }

  if (TransA == CblasConjTrans) {
    ampTransA = NoTrans;
  }

  // target the default accelerator
  hc::accelerator currentAcc(L"default");
  hc::accelerator_view accl_view = currentAcc.get_default_view();
  hc.hcblas_dgemv(accl_view, ColMajor, ampTransA, N, M, alpha, A, offseta, N, x,
                  offsetx, 1, beta, y, offsety, 1);
}
template <>
void caffe_gpu_gemv2<float>(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const float alpha, float* A, size_t offA, int lda,
                            float* x, size_t offx, const float beta, int incx,
                            float* y, size_t offy, int incy) {
  hcblasTranspose ampTransA = Trans;
  Hcblaslibrary hc;

  if (TransA == CblasTrans) {
    ampTransA = NoTrans;
  }

  if (TransA == CblasConjTrans) {
    ampTransA = NoTrans;
  }

  // target the default accelerator
  hc::accelerator currentAcc(L"default");
  hc::accelerator_view accl_view = currentAcc.get_default_view();
  hc.hcblas_sgemv(accl_view, ColMajor, ampTransA, N, M, alpha, A, offA, lda, x,
                  offx, incx, beta, y, offy, incy);
}

template <>
void caffe_gpu_gemv2<double>(const CBLAS_TRANSPOSE TransA, const int M,
                             const int N, const double alpha, double* A, size_t offA, int lda,
                             double* x, size_t offx, const double beta, int incx,
                             double* y, size_t offy, int incy) {
  hcblasTranspose ampTransA = Trans;
  Hcblaslibrary hc;

  if (TransA == CblasTrans) {
    ampTransA = NoTrans;
  }

  if (TransA == CblasConjTrans) {
    ampTransA = NoTrans;
  }

  // target the default accelerator
  hc::accelerator currentAcc(L"default");
  hc::accelerator_view accl_view = currentAcc.get_default_view();
  hc.hcblas_dgemv(accl_view, ColMajor, ampTransA, N, M, alpha, A, offA, lda, x,
                  offx, incx, beta, y, offy, incy);
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB,
                           const int M, const int N, const int K,
                           const float alpha, float* A, const int offset_A, float* B,
                           const int offset_B, const float beta, float* C, const int offset_C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  hcblasTranspose ampTransA = NoTrans;
  hcblasTranspose ampTransB = NoTrans;
  Hcblaslibrary hc;

  if (TransA == CblasTrans) {
    ampTransA = Trans;
  }

  if (TransA == CblasConjTrans) {
    ampTransA = NoTrans;
  }

  if (TransB == CblasTrans) {
    ampTransB = Trans;
  }

  if (TransB == CblasConjTrans) {
    ampTransB = NoTrans;
  }

  // target the default accelerator
  hc::accelerator currentAcc(L"default");
  hc::accelerator_view accl_view = currentAcc.get_default_view();
  hc.hcblas_sgemm(accl_view, ColMajor, ampTransB, ampTransA, N, M, K, alpha, B,
                  ldb, A, lda, beta, C, N, offset_B, offset_A,
                  offset_C);
}


template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB,
                            const int M, const int N, const int K,
                            const double alpha, double* A, const int offset_A, double* B,
                            const int offset_B, const double beta, double* C, const int offset_C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  hcblasTranspose ampTransA = NoTrans;
  hcblasTranspose ampTransB = NoTrans;
  Hcblaslibrary hc;

  if (TransA == CblasTrans) {
    ampTransA = Trans;
  }

  if (TransA == CblasConjTrans) {
    ampTransA = NoTrans;
  }

  if (TransB == CblasTrans) {
    ampTransB = Trans;
  }

  if (TransB == CblasConjTrans) {
    ampTransB = NoTrans;
  }

  // target the default accelerator
  hc::accelerator currentAcc(L"default");
  hc::accelerator_view accl_view = currentAcc.get_default_view();
  hc.hcblas_dgemm(accl_view, ColMajor, ampTransB, ampTransA, N, M, K, alpha, B,
                  ldb, A, lda, beta, C, N, offset_B, offset_A, offset_C);
}
template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  // runtime sizes
  unsigned int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count : MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  hc::array<float, 1> global_buffer(tile_count);
  hc::array_view<float, 1> global_buffer_view(global_buffer);
  // configuration
  hc::extent<1> extent(thread_count);
  hc::parallel_for_each(
    extent.tile(TILE_SIZE),
  [ = ] (hc::tiled_index<1>& tid) __attribute__((hc, cpu)) {
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
      smem += hc::fast_math::fabs(x[idx]);
      // next chunk
      idx += thread_count;
    }

    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    float* mem = &smem;

    // unrolled for performance
    if (local < 128) {
      mem[0] = mem[0] + mem[128];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  64) {
      mem[0] = mem[0] + mem[ 64];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  32) {
      mem[0] = mem[0] + mem[ 32];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  16) {
      mem[0] = mem[0] + mem[ 16];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   8) {
      mem[0] = mem[0] + mem[  8];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   4) {
      mem[0] = mem[0] + mem[  4];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   2) {
      mem[0] = mem[0] + mem[  2];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   1) {
      mem[0] = mem[0] + mem[  1];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  }).wait();
  // 2nd pass reduction
  std::vector<float> host_buffer(global_buffer);
  *y = *std::max_element(host_buffer.begin(), host_buffer.end());
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  // runtime sizes
  unsigned int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count : MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  hc::array<double, 1> global_buffer(tile_count);
  hc::array_view<double, 1> global_buffer_view(global_buffer);
  // configuration
  hc::extent<1> extent(thread_count);
  hc::parallel_for_each(
    extent.tile(TILE_SIZE),
  [ = ] (hc::tiled_index<1>& tid) __attribute__((hc, cpu)) {
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
      smem += hc::fast_math::fabs(x[idx]);
      // next chunk
      idx += thread_count;
    }

    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    double* mem = &smem;

    // unrolled for performance
    if (local < 128) {
      mem[0] = mem[0] + mem[128];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  64) {
      mem[0] = mem[0] + mem[ 64];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  32) {
      mem[0] = mem[0] + mem[ 32];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  16) {
      mem[0] = mem[0] + mem[ 16];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   8) {
      mem[0] = mem[0] + mem[  8];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   4) {
      mem[0] = mem[0] + mem[  4];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   2) {
      mem[0] = mem[0] + mem[  2];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   1) {
      mem[0] = mem[0] + mem[  1];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  }).wait();
  // 2nd pass reduction
  std::vector<double> host_buffer(global_buffer);
  *y = *std::max_element(host_buffer.begin(), host_buffer.end());
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  unsigned int temp[n];
  caffe_rng_uniform(n, temp);
  array_view<unsigned int, 1> tempView(n, temp);
  hc::extent<1> e(n);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    r[idx[0]] = tempView(idx[0]);
  }).wait();
}

template <>
void caffe_gpu_rng_uniform<float>(const int N, const float a, const float b,
                                  float* r) {
  float temp[N];
  caffe_rng_uniform(N, a, b, temp);
  array_view<float, 1> tempView(N, temp);
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    r[idx[0]] = tempView(idx[0]);
  }).wait();
}

template <>
void caffe_gpu_rng_uniform<double>(const int N, const double a, const double b,
                                   double* r) {
  double temp[N];
  caffe_rng_uniform(N, a, b, temp);
  array_view<double, 1> tempView(N, temp);
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    r[idx[0]] = tempView(idx[0]);
  }).wait();
}

template <>
void caffe_gpu_rng_gaussian(const int N, const float mu, const float sigma,
                            float* r) {
  float temp[N];
  caffe_rng_gaussian(N, mu, sigma, temp);
  array_view<float, 1> tempView(N, temp);
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    r[idx[0]] = tempView(idx[0]);
  }).wait();
}


template <>
void caffe_gpu_rng_gaussian(const int N, const double mu,
                            const double sigma, double* r) {
  double temp[N];
  caffe_rng_gaussian(N, mu, sigma, temp);
  array_view<double, 1> tempView(N, temp);
  hc::extent<1> e(N);
  parallel_for_each(e, [ = ](index<1> idx) __attribute__((hc, cpu)) {
    r[idx[0]] = tempView(idx[0]);
  }).wait();
}

template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
    const float* y) {
  hc::accelerator currentAcc(L"default");
  uint32_t* dresult = hc::am_alloc(n * sizeof(uint32_t), currentAcc, 0);
  uint32_t* result = (uint32_t*)(malloc(sizeof(uint32_t) * n));
  hc::extent<1> e(n);
  hc::parallel_for_each(e, [ = ] (hc::index<1>& idx) __attribute__((hc, cpu)) {
    uint32_t ret = 0;
    int ux = x[idx[0]];
    int uy = y[idx[0]];
    uint32_t u = ux ^ uy;

    while (u) {
      u = u & (u - 1);
      ret++;
    }

    dresult[idx[0]] = ret;
  }).wait();
  // Copy Result back to CPU
  hc::am_copy(result, dresult,  n * sizeof(uint32_t));
  uint32_t sum = 0;

  for (int i = 0; i < n; ++i) {
    sum += result[i];
  }

  hc::am_free(dresult);
  free(result);
  return sum;
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
    const double* y) {
  hc::accelerator currentAcc(L"default");
  uint32_t* dresult = hc::am_alloc(n * sizeof(uint64_t), currentAcc, 0);
  uint32_t* result = (uint32_t*)(malloc(sizeof(uint64_t) * n));
  hc::extent<1> e(n);
  hc::parallel_for_each(e, [ = ] (hc::index<1>& idx) __attribute__((hc, cpu)) {
    uint64_t ret = 0;
    long ux = x[idx[0]];
    long uy = y[idx[0]];
    uint64_t u = ux ^ uy;

    while (u) {
      u = u & (u - 1);
      ret++;
    }

    dresult[idx[0]] = ret;
  }).wait();
  // Copy Result back to CPU
  hc::am_copy(result, dresult,  n * sizeof(uint64_t));
  uint64_t sum = 0;

  for (int i = 0; i < n; ++i) {
    sum += result[i];
  }

  hc::am_free(dresult);
  free(result);
  return sum;
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  LOG(FATAL) << "Instead of caffe_gpu_memcpy with caffe_hcc_X2X.";
}

}  // namespace caffe

