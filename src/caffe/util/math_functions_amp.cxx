#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/type_traits/is_same.hpp>
#include <glog/logging.h>
#include <limits>
#include <vector>
#include "amp.h"
#include "amp_math.h"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "cppamp/ampblaslib.h"

namespace caffe {
template <typename Dtype>
void transform_gpu(void* src, void* dst, int top_offset, int N_,
  int M_, int packing_num) {
  Concurrency::array_view<Dtype, 1> avSrc =
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(src));
  Concurrency::array_view<Dtype, 1> avDst =
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(dst));
  Concurrency::extent<1> e(M_*packing_num);
  parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) {
    int i = 0;
    for (i = 0 ; i < N_; i++){
      if (packing_num == 1)
        avDst[top_offset + (idx / packing_num)* N_ + i] =
          avSrc[idx * N_ + i];
      else
       avDst[top_offset+(idx % packing_num * M_ + idx / packing_num)
         * N_ + i] = avSrc[idx * N_ + i];
    }
  });
}
template void transform_gpu<float>(void* src, void* dst, int top_offset,
  int N_, int M_, int packing_num);
template void transform_gpu<double>(void* src, void* dst, int top_offset,
  int N_, int M_, int packing_num);

template <typename Dtype>
void opttrans(void* data_im, int im_offset, int channels,
    int height, int width, void* data_opt, int opt_offset, int packing_num) {
  Concurrency::array_view<Dtype, 1> avIm =
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(data_im));
  Concurrency::array_view<Dtype, 1> avOpt =
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(data_opt));

  int num_kernels = channels * height * width * packing_num;

  Concurrency::extent<1> e(num_kernels);
  parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) {
    int w = idx[0] % width;
    int h = (idx[0] / width) % height;
    int c = idx[0] / (width * height) % channels;
    int im = idx[0] / width / height / channels;
    int opt_index = c * height * packing_num * width +
      h * packing_num * width + im * width + w;
        avOpt[opt_offset+opt_index] = avIm[im_offset+idx];
  });
}
template void opttrans<float>(void* data_im, int im_offset, int channels,
    int height, int width, void* data_opt, int opt_offset, int optnum);
template void opttrans<double>(void* data_im, int im_offset, int channels,
    int height, int width, void* data_opt, int opt_offset, int optnum);



// The size is the total memory size
void caffe_amp_malloc(void** ptr, size_t size, size_t element_size,
    bool is_int) {
  // Use default device
  concurrency::accelerator currentAcc(accelerator::default_accelerator);
  if (is_int) {
    if (element_size == sizeof(int)) {
      Concurrency::extent<1> eA(size/sizeof(int));
      // Allocating device array of given size
      Concurrency::array<int, 1> arr =
        Concurrency::array<int, 1>(eA, currentAcc.get_default_view());
      Concurrency::array_view<int>* avData =
        new Concurrency::array_view<int>(arr);
      avData->discard_data();
      *ptr = static_cast<void*>(avData);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_malloc.";
    }
  } else {
    if (element_size == sizeof(float)) {
      Concurrency::extent<1> eA(size/sizeof(float));
      Concurrency::array<float, 1> arr =
        Concurrency::array<float, 1>(eA, currentAcc.get_default_view());
      Concurrency::array_view<float>* avData =
        new Concurrency::array_view<float>(arr);
      avData->discard_data();
      *ptr = static_cast<void*>(avData);
    } else if (element_size == sizeof(double)) {
      Concurrency::extent<1> eA(size/sizeof(double));
      Concurrency::array<double, 1> arr =
        Concurrency::array<double, 1>(eA, currentAcc.get_default_view());
      Concurrency::array_view<double>* avData =
        new Concurrency::array_view<double>(arr);
     avData->discard_data();
     *ptr = static_cast<void*>(avData);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_malloc.";
    }
  }
}

void caffe_amp_malloc(void** ptr, void* src, size_t size, size_t element_size,
    bool is_int) {
  // Use default device
  if (is_int) {
    if (element_size == sizeof(int)) {
      Concurrency::extent<1> eA(size/sizeof(int));
      // Allocating device array of given size
      Concurrency::array_view<int>* avData =
        new Concurrency::array_view<int>(eA, static_cast<int*>(src));
      avData->discard_data();
      *ptr = static_cast<void*>(avData);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_malloc.";
    }
  } else {
    if (element_size == sizeof(float)) {
      Concurrency::extent<1> eA(size/sizeof(float));
      Concurrency::array_view<float>* avData =
        new Concurrency::array_view<float>(eA, static_cast<float*>(src));
      avData->discard_data();
      *ptr = static_cast<void*>(avData);
    } else if (element_size == sizeof(double)) {
      Concurrency::extent<1> eA(size/sizeof(double));
      Concurrency::array_view<double>* avData =
        new Concurrency::array_view<double>(eA, static_cast<double*>(src));
      avData->discard_data();
     *ptr = static_cast<void*>(avData);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_malloc.";
    }
  }
}

void caffe_amp_free(void* ptr, size_t element_size, bool is_int) {
  if (ptr) {
    if (is_int) {
      if (element_size == sizeof(int)) {
        delete static_cast<Concurrency::array_view<int> *>(ptr);
      } else {
        LOG(FATAL) << "Wrong element size for caffe_amp_free.";
      }
    } else {
      if (element_size == sizeof(float)) {
        delete static_cast<Concurrency::array_view<float> *>(ptr);
      } else if (element_size == sizeof(double)) {
        delete static_cast<Concurrency::array_view<double> *>(ptr);
      } else {
        LOG(FATAL) << "Wrong element size for caffe_amp_free.";
      }
    }
    ptr = NULL;
  }
}

void caffe_amp_D2H(void* src, void* dst, size_t element_size, bool is_int) {
  if (src == NULL || dst == NULL) {
    LOG(FATAL) << "Wrong source or destination for caffe_amp_D2H.";
  }
  if (is_int) {
    if (element_size == sizeof(int)) {
      Concurrency::array_view<int, 1>* avSrc =
       static_cast<Concurrency::array_view<int, 1>*>(src);
      Concurrency::copy(*avSrc, static_cast<int*>(dst));
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_D2H.";
    }
  } else {
    if (element_size == sizeof(float)) {
      Concurrency::array_view<float, 1>* avSrc =
        static_cast<Concurrency::array_view<float, 1>*>(src);
      Concurrency::copy(*avSrc, static_cast<float*>(dst));
    } else if (element_size == sizeof(double)) {
      Concurrency::array_view<double, 1>* avSrc =
        (Concurrency::array_view<double, 1>*)(src);
      Concurrency::copy(*avSrc, static_cast<double*>(dst));
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_D2H.";
    }
  }
}

void caffe_amp_H2D(void* src, void* dst, size_t element_size, bool is_int) {
  if (src == NULL || dst == NULL) {
    LOG(FATAL) << "Wrong source or destination for caffe_amp_H2D.";
  }
  if (is_int) {
    if (element_size == sizeof(int)) {
      Concurrency::array_view<int, 1>* avDst =
        static_cast<Concurrency::array_view<int, 1>*>(dst);
      Concurrency::copy(static_cast<int*>(src), *avDst);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_H2D.";
    }
  } else {
    if (element_size == sizeof(float)) {
      Concurrency::array_view<float, 1>* avDst =
        (Concurrency::array_view<float, 1>*)(dst);
      Concurrency::copy(static_cast<float*>(src), *avDst);
    } else if (element_size == sizeof(double)) {
      Concurrency::array_view<double, 1>* avDst =
        static_cast<Concurrency::array_view<double, 1>*>(dst);
      Concurrency::copy(static_cast<double*>(src), *avDst);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_H2D.";
    }
  }
}

void caffe_amp_D2D(void* src, void* dst, size_t element_size, bool is_int) {
  if (src == NULL || dst == NULL) {
    LOG(FATAL) << "Wrong source or destination for caffe_amp_D2D.";
  }
  if (is_int) {
    if (element_size == sizeof(int)) {
      Concurrency::array_view<int, 1>* avSrc =
        static_cast<Concurrency::array_view<int, 1>*>(src);
      Concurrency::array_view<int, 1>* avDst =
        static_cast<Concurrency::array_view<int, 1>*>(dst);
      Concurrency::copy(*avSrc, *avDst);
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_D2D.";
    }
  } else {
    if (element_size == sizeof(float)) {
      Concurrency::array_view<float, 1>* avSrc =
        static_cast<Concurrency::array_view<float, 1>*>(src);
      Concurrency::array_view<float, 1>* avDst =
        static_cast<Concurrency::array_view<float, 1>*>(dst);
      Concurrency::copy(*avSrc, *avDst);
    } else if (element_size == sizeof(double)) {
      Concurrency::array_view<double, 1>* avSrc =
        static_cast<Concurrency::array_view<double, 1>*>(src);
      Concurrency::array_view<double, 1>* avDst =
        static_cast<Concurrency::array_view<double, 1>*>(dst);
      Concurrency::copy(*avSrc, *avDst);  // NOLINT(build/include_what_you_use)
    } else {
      LOG(FATAL) << "Wrong element size for caffe_amp_D2D.";
    }
  }
}

template <typename Dtype>
void caffe_amp_copy(int N, void* src, void* dst,
    int srcOffset, int dstOffset) {
  Concurrency::array_view<Dtype, 1> avSrc =
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(src));
  Concurrency::array_view<Dtype, 1> avDst =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(dst));
  if (src == NULL || dst == NULL ||
      N > avSrc.get_extent().size() - srcOffset ||
      N > avDst.get_extent().size() - dstOffset) {
    LOG(FATAL) << "Wrong Parameters for caffe_amp_copy.";
  }

  if (srcOffset == 0 && dstOffset== 0 &&
      N == avSrc.get_extent().size() &&
      N <= avDst.get_extent().size()) {
    caffe_amp_D2D(src, dst, sizeof(Dtype), boost::is_same<Dtype, int>::value);
  } else {
    Concurrency::extent<1> e(N);
    parallel_for_each(e, [=](index<1> idx) restrict(amp) {
      avDst[dstOffset + idx] = avSrc[srcOffset + idx];
    });
  }
}

template void caffe_amp_copy<int>(int N, void* src, void* dst,
    int srcOffset, int dstOffset);
template void caffe_amp_copy<float>(int N, void* src, void* dst,
    int srcOffset, int dstOffset);
template void caffe_amp_copy<double>(int N, void* src, void* dst,
    int srcOffset, int dstOffset);

template <typename Dtype>
void caffe_amp_copy_H2D(int N, void* src, void* dst, int dstOffset) {
  Concurrency::array_view<Dtype, 1> avDst =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(dst));
  if (src == NULL || dst == NULL ||
      N > avDst.get_extent().size() - dstOffset) {
    LOG(FATAL) << "Wrong Parameters for caffe_amp_copy_H2D.";
  }
  Concurrency::array_view<Dtype, 1> avSrc(N, static_cast<Dtype*>(src));

  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    avDst[dstOffset + idx] = avSrc[idx];
  });
}

template void caffe_amp_copy_H2D<int>(int N, void* src, void* dst,
    int dstOffset);
template void caffe_amp_copy_H2D<float>(int N, void* src, void* dst,
    int dstOffset);
template void caffe_amp_copy_H2D<double>(int N, void* srci, void* dst,
    int dstOffset);

template <typename Dtype>
void caffe_amp_copy_D2H(int N, void* src, void* dst, int srcOffset) {
  Concurrency::array_view<Dtype, 1> avSrc =
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(src));
  if (src == NULL || dst == NULL ||
      N > avSrc.get_extent().size() - srcOffset) {
    LOG(FATAL) << "Wrong Parameters for caffe_amp_copy_D2H.";
  }
  Concurrency::array_view<Dtype, 1> avDst(N, static_cast<Dtype*>(dst));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    avDst[idx] = avSrc[srcOffset + idx];
  });
  avDst.synchronize();
}

template void caffe_amp_copy_D2H<int>(int N, void* src, void* dst,
    int srcOffset);
template void caffe_amp_copy_D2H<float>(int N, void* src, void* dst,
    int srcOffset);
template void caffe_amp_copy_D2H<double>(int N, void* src, void* dst,
    int srcOffset);

template <typename Dtype>
void abs_kernel(const int N, Dtype* a, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = aView[idx] >= 0 ? aView[idx] : -1 * aView[idx];
  });
}

template <typename Dtype>
void sign_kernel(const int N, Dtype* a, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = aView[idx] == 0 ? 0 : (aView[idx] < 0 ? -1 : 1);
  });
}

template <typename Dtype>
void sgnbit_kernel(const int N, Dtype* a, Dtype* y) {
  Concurrency::array_view<Dtype, 1> aView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = Concurrency::fast_math::signbit(aView[idx]);
  });
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
  Concurrency::array_view<Dtype, 1> aView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> bView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(b)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
      yView[idx] = (aView[idx] * bView[idx]);
  });
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
  Concurrency::array_view<Dtype, 1> aView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> bView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(b)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = (aView[idx] / bView[idx]);
  });
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
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> bView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(b)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = (aView[idx] + bView[idx]);
  });
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
  Concurrency::array_view<Dtype, 1> aView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> bView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(b)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = (aView[idx] - bView[idx]);
  });
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
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp){
    outView[idx] = alpha;
  });
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
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = Concurrency::fast_math::exp(aView[idx]);
  });
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
    *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    outView[idx] += alpha;
  });
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
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(a)));
  Concurrency::array_view<Dtype, 1> yView =
      *(static_cast<Concurrency::array_view<Dtype, 1>*>(static_cast<void*>(y)));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    yView[idx] = Concurrency::fast_math::pow(aView[idx], alpha);
  });
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
  Concurrency::array_view<float, 1> xView =
      *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
              const_cast<float*>(x))));
  Concurrency::array_view<float, 1> yView =
      *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
              const_cast<float*>(y))));

  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<float, 1> global_buffer(tile_count);
  concurrency::array_view<float, 1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp) {
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
      smem += xView[concurrency::index<1>(idx)] *
        yView[concurrency::index<1>(idx)];
      // next chunk
      idx += thread_count;
    }
    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    float *mem = &smem;
    // unrolled for performance
    if (local < 128) { mem[0] = mem[0] + mem[128]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; }
    tid.barrier.wait_with_tile_static_memory_fence();
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
  Concurrency::array_view<double, 1> xView =
      *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(x))));
  Concurrency::array_view<double, 1> yView =
      *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(y))));

  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<double, 1> global_buffer(tile_count);
  concurrency::array_view<double, 1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp) {
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
      smem += xView[concurrency::index<1>(idx)] *
        yView[concurrency::index<1>(idx)];
      // next chunk
      idx += thread_count;
    }
    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    double *mem = &smem;
    // unrolled for performance
    if (local < 128) { mem[0] = mem[0] + mem[128]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; }
    tid.barrier.wait_with_tile_static_memory_fence();
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
                                float* y) {
  caffe_amp_D2D(static_cast<void*>(const_cast<float*>(x)),
      static_cast<void*>(const_cast<float*>(y)), sizeof(float),
      false);
  amp_scale(n, alpha, y);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                                 double* y) {
  caffe_amp_D2D(static_cast<void*>(const_cast<double*>(x)),
      static_cast<void*>(const_cast<double*>(y)), sizeof(double),
      false);
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
  const int N, const float alpha, const float* A, const int offseta,
  const float* x, const int offsetx,
  const float beta, float* y, const int offsety) {
  AMPBLAS_TRANS ampTransA = trans;
  Ampblaslibrary amp;
  if (TransA == CblasTrans) {
      ampTransA = noTrans;
  }
  if (TransA == CblasConjTrans) {
      ampTransA = conjugate;
  }
  Concurrency::array_view<float, 1> A_mat =
    *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
            const_cast<float*>(A))));
  Concurrency::array_view<float, 1> X_mat =
    *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
            const_cast<float*>(x))));
  Concurrency::array_view<float, 1> Y_mat =
    *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(y)));

  amp.ampblas_sgemv2(ampTransA, N, M, &alpha, A_mat, offseta, N, X_mat,
      offsetx, 1, &beta, Y_mat, offsety, 1);
}


template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const double alpha, const double* A, const int offseta,
  const double* x, const int offsetx,
  const double beta, double* y, const int offsety) {
  AMPBLAS_TRANS ampTransA = trans;
  Ampblaslibrary amp;
  if (TransA == CblasTrans) {
      ampTransA = noTrans;
  }
  if (TransA == CblasConjTrans) {
      ampTransA = conjugate;
  }
  Concurrency::array_view<double, 1> A_mat =
    *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(A))));
  Concurrency::array_view<double, 1> X_mat =
    *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(x))));
  Concurrency::array_view<double, 1> Y_mat =
    *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(y)));
  amp.ampblas_dgemv2(ampTransA, N, M, &alpha, A_mat, offseta, N, X_mat,
      offsetx, 1, &beta, Y_mat, offsety, 1);
}
template <>
void caffe_gpu_gemv2<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, size_t offA, int lda,
    const float* x, size_t offx, const float beta, int incx,
    float* y, size_t offy, int incy) {
    AMPBLAS_TRANS ampTransA = trans;
    Ampblaslibrary amp;
    if (TransA == CblasTrans) {
        ampTransA = noTrans;
    }
    if (TransA == CblasConjTrans) {
        ampTransA = conjugate;
    }
    Concurrency::array_view<float, 1> A_mat =
      *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
              const_cast<float*>(A))));
    Concurrency::array_view<float, 1> X_mat =
      *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
              const_cast<float*>(x))));
    Concurrency::array_view<float, 1> Y_mat =
      *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(y)));

    amp.ampblas_sgemv2(ampTransA, N, M, &alpha, A_mat, offA, lda, X_mat,
        offx, incx, &beta, Y_mat, offy, incy);
}

template <>
void caffe_gpu_gemv2<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, size_t offA, int lda,
    const double* x, size_t offx, const double beta, int incx,
    double* y, size_t offy, int incy) {
      AMPBLAS_TRANS ampTransA = trans;
      Ampblaslibrary amp;
      if (TransA == CblasTrans) {
          ampTransA = noTrans;
      }
      if (TransA == CblasConjTrans) {
          ampTransA = conjugate;
      }
      Concurrency::array_view<double, 1> A_mat =
        *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(A))));
      Concurrency::array_view<double, 1> X_mat =
        *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(x))));
      Concurrency::array_view<double, 1> Y_mat =
        *(static_cast<Concurrency::array_view<double, 1>*>
          (static_cast<void*>(y)));
      amp.ampblas_dgemv2(ampTransA, N, M, &alpha, A_mat, offA, lda, X_mat,
        offx, incx, &beta, Y_mat, offy, incy);
}
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB,
  const int M, const int N, const int K,
  const float alpha, const float* A, const int offset_A, const float* B,
  const int offset_B, const float beta, float* C, const int offset_C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  AMPBLAS_TRANS ampTransA = noTrans;
  AMPBLAS_TRANS ampTransB = noTrans;
  Ampblaslibrary amp;
  if (TransA == CblasTrans) {
      ampTransA = trans;
  }
  if (TransA == CblasConjTrans) {
      ampTransA = conjugate;
  }

  if (TransB == CblasTrans) {
      ampTransB = trans;
  }

  if (TransB == CblasConjTrans) {
      ampTransB = conjugate;
  }
  Concurrency::array_view<float, 1> A_mat =
    *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
            const_cast<float*>(A))));
  Concurrency::array_view<float, 1> B_mat =
    *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
            const_cast<float*>(B))));
  Concurrency::array_view<float, 1> C_mat =
    *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(C)));
    amp.ampblas_sgemm2(colMajor, ampTransB, ampTransA, N, M, K, &alpha, B_mat,
                ldb, A_mat, lda, &beta, C_mat, N, offset_B, offset_A,
                offset_C);
}


template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB,
  const int M, const int N, const int K,
  const double alpha, const double* A, const int offset_A, const double* B,
  const int offset_B, const double beta, double* C, const int offset_C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  AMPBLAS_TRANS ampTransA = noTrans;
  AMPBLAS_TRANS ampTransB = noTrans;
  Ampblaslibrary amp;
  if (TransA == CblasTrans) {
      ampTransA = trans;
  }
  if (TransA == CblasConjTrans) {
      ampTransA = conjugate;
  }

  if (TransB == CblasTrans) {
      ampTransB = trans;
  }

  if (TransB == CblasConjTrans) {
      ampTransB = conjugate;
  }
  Concurrency::array_view<double, 1> A_mat =
    *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(A))));
  Concurrency::array_view<double, 1> B_mat =
    *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(B))));
  Concurrency::array_view<double, 1> C_mat =
    *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(C)));
    amp.ampblas_dgemm2(colMajor, ampTransB, ampTransA, N, M, K, &alpha, B_mat,
                ldb, A_mat, lda, &beta, C_mat, N, offset_B, offset_A, offset_C);
}
template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  Concurrency::array_view<float, 1> xView =
    *(static_cast<Concurrency::array_view<float, 1>*>(static_cast<void*>(
            const_cast<float*>(x))));

  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<float, 1> global_buffer(tile_count);
  concurrency::array_view<float, 1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp) {
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
    if (local < 128) { mem[0] = mem[0] + mem[128]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  });
  // 2nd pass reduction
  std::vector<float> host_buffer(global_buffer);
  *y = *std::max_element(host_buffer.begin(), host_buffer.end());
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  Concurrency::array_view<double, 1> xView =
    *(static_cast<Concurrency::array_view<double, 1>*>(static_cast<void*>(
            const_cast<double*>(x))));

  // runtime sizes
  unsigned int tile_count = (n+TILE_SIZE-1) / TILE_SIZE;
  tile_count = tile_count < MAX_TILES ? tile_count:MAX_TILES;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  concurrency::array<double, 1> global_buffer(tile_count);
  concurrency::array_view<double, 1> global_buffer_view(global_buffer);
  // configuration
  concurrency::extent<1> extent(thread_count);
  concurrency::parallel_for_each(
    extent.tile<TILE_SIZE>(),
    [=] (concurrency::tiled_index<TILE_SIZE> tid) restrict(amp) {
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
    if (local < 128) { mem[0] = mem[0] + mem[128]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  64) { mem[0] = mem[0] + mem[ 64]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  32) { mem[0] = mem[0] + mem[ 32]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <  16) { mem[0] = mem[0] + mem[ 16]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   8) { mem[0] = mem[0] + mem[  8]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   4) { mem[0] = mem[0] + mem[  4]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   2) { mem[0] = mem[0] + mem[  2]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    if (local <   1) { mem[0] = mem[0] + mem[  1]; }
    tid.barrier.wait_with_tile_static_memory_fence();
    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      global_buffer_view[ tid.tile[0] ] = smem;
    }
  });
  // 2nd pass reduction
  std::vector<double> host_buffer(global_buffer);
  *y = *std::max_element(host_buffer.begin(), host_buffer.end());
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  unsigned int temp[n];
  caffe_rng_uniform(n, temp);
  array_view<unsigned int, 1> tempView(n, temp);
  array_view<unsigned int, 1> rView =
    *((Concurrency::array_view<unsigned int, 1>*)(r));
  Concurrency::extent<1> e(n);
  parallel_for_each(e, [=](index<1> idx) restrict(amp){
    rView[idx] = tempView(idx);
  });
}

template <>
void caffe_gpu_rng_uniform<float>(const int N, const float a, const float b,
    float* r) {
  float temp[N];
  caffe_rng_uniform(N, a, b, temp);
  array_view<float, 1> tempView(N, temp);
  array_view<float, 1> rView =
    *((Concurrency::array_view<float, 1>*)(r));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp){
    rView[idx] = tempView(idx);
  });
}

template <>
void caffe_gpu_rng_uniform<double>(const int N, const double a, const double b,
    double* r) {
  double temp[N];
  caffe_rng_uniform(N, a, b, temp);
  array_view<double, 1> tempView(N, temp);
  array_view<double, 1> rView =
    *((Concurrency::array_view<double, 1>*)(r));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp){
    rView[idx] = tempView(idx);
  });
}

template <>
void caffe_gpu_rng_gaussian(const int N, const float mu, const float sigma,
    float* r) {
  float temp[N];
  caffe_rng_gaussian(N, mu, sigma, temp);
  array_view<float, 1> tempView(N, temp);
  array_view<float, 1> rView =
    *((Concurrency::array_view<float, 1>*)(r));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp){
    rView[idx] = tempView(idx);
  });
}


template <>
void caffe_gpu_rng_gaussian(const int N, const double mu,
    const double sigma, double* r) {
  double temp[N];
  caffe_rng_gaussian(N, mu, sigma, temp);
  array_view<double, 1> tempView(N, temp);
  array_view<double, 1> rView =
    *((Concurrency::array_view<double, 1>*)(r));
  Concurrency::extent<1> e(N);
  parallel_for_each(e, [=](index<1> idx) restrict(amp){
    rView[idx] = tempView(idx);
  });
}

template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  array_view<float, 1> axView =
    *(static_cast<Concurrency::array_view<float, 1>*>(
          (static_cast<void*>(const_cast<float*>(x)))));
  array_view<float, 1> ayView =
    *(static_cast<Concurrency::array_view<float, 1>*>(
          (static_cast<void*>(const_cast<float*>(y)))));

  uint32_t* result = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * n));
  uint32_t* ax = static_cast<uint32_t*>(
      malloc(sizeof(uint32_t) * axView.get_extent().size()));
  uint32_t* ay = static_cast<uint32_t*>(
      malloc(sizeof(uint32_t) * ayView.get_extent().size()));

  for (int i = 0; i < n; ++i) {
    ax[i] = static_cast<uint32_t>(axView[i]);
    ay[i] = static_cast<uint32_t>(ayView[i]);
  }

  array_view<uint32_t, 1> resultView(n, result);
  array_view<uint32_t, 1> xView(axView.get_extent().size(), ax);
  array_view<uint32_t, 1> yView(ayView.get_extent().size(), ay);

  Concurrency::extent<1> e(n);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    uint32_t ret = 0;
    uint32_t u = xView[idx] ^ yView[idx];
    while (u) {
      u = u & (u - 1);
      ret++;
    }
    resultView[idx] = ret;
  });
  resultView.synchronize();
  xView.synchronize();
  yView.synchronize();
  uint32_t sum = 0;
  for (int i = 0; i < n; ++i) {
    sum+=result[i];
  }
  free(result);
  free(ax);
  free(ay);
  return sum;
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  array_view<double, 1> axView =
    *(static_cast<Concurrency::array_view<double, 1>*>(
          (static_cast<void*>(const_cast<double*>(x)))));
  array_view<double, 1> ayView =
    *(static_cast<Concurrency::array_view<double, 1>*>(
          (static_cast<void*>(const_cast<double*>(y)))));

  uint32_t* result = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * n));
  uint64_t* ax = static_cast<uint64_t*>(
      malloc(sizeof(uint64_t) * axView.get_extent().size()));
  uint64_t* ay = static_cast<uint64_t*>(
      malloc(sizeof(uint64_t) * ayView.get_extent().size()));
  for (int i = 0; i < n; ++i) {
    ax[i] = static_cast<uint64_t>(axView[i]);
    ay[i] = static_cast<uint64_t>(ayView[i]);
  }
  array_view<uint32_t, 1> resultView(n, result);
  array_view<uint64_t, 1> xView(axView.get_extent().size(), ax);
  array_view<uint64_t, 1> yView(ayView.get_extent().size(), ay);
  Concurrency::extent<1> e(n);
  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    uint32_t ret = 0;
    uint64_t u = xView[idx] ^ yView[idx];
    while (u) {
      u = u & (u - 1);
      ret++;
    }
    resultView[idx] = ret;
  });
  resultView.synchronize();
  xView.synchronize();
  yView.synchronize();
  uint32_t sum = 0;
  for (int i = 0; i < n; ++i) {
    sum+=result[i];
  }
  free(result);
  free(ax);
  free(ay);
  return sum;
}

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y) {
  LOG(FATAL) << "Instead of caffe_gpu_memcpy with caffe_amp_X2X.";
}

}  // namespace caffe

