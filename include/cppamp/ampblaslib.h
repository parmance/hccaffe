/*
*
*  FILENAME : ampblas.h
*  This file is the top level header file which includes the Ampblaslilbrary class
*
*/

#ifndef AMPBLAS_LIB_H
#define AMPBLAS_LIB_H

#include<iostream>
//todo cpu version
//#include<cblas.h>

#include "amp.h"
#include "amp_math.h"
using namespace concurrency;

/* enumerator to indicate the status of  blas operation */
enum ampblasStatus {
    AMPBLAS_SUCCESS = 0,
    AMPBLAS_INVALID = -1,
    AMPBLAS_ERROR = -2
};

/* enumerator to define the layout of  input matrix for blas operation */
enum AMPBLAS_ORDER {
    rowMajor ,
    colMajor
};

/* enumerator to define the type of operation to be performed on the input matrix
 ( NO_TRANSPOSE, TRANSPOSE, CONJUGATE) */
enum AMPBLAS_TRANS {
    noTrans = 'n',
    trans = 't',
    conjugate = 'c'
};
struct ampComplex
{
     float real;
     float img;
};
template <typename Dtype>
void amp_axpy(const int N, const Dtype alpha, Dtype* x, Dtype* y)
{
  array_view<Dtype, 1> xView(N, x);
  array_view<Dtype, 1> yView(N, y);
  concurrency::parallel_for_each(
    yView.get_extent(),
    [=] (concurrency::index<1> idx) restrict(amp) 
    {
      yView[idx] += alpha * xView[idx];
    }
  );
  yView.synchronize();
}

template <typename Dtype>
void amp_copy(const int N, Dtype* x, Dtype* y)
{
  array_view<Dtype, 1> xView(N, x);
  array_view<Dtype, 1> yView(N, y);
  concurrency::parallel_for_each(
    yView.get_extent(), 
    [=] (concurrency::index<1> idx) restrict(amp) 
    {
      yView[idx] = xView[idx];
    }
  );
  yView.synchronize();
}

template <typename Dtype>
void amp_scale(const int N, const Dtype value, Dtype* x)
{
  array_view<Dtype, 1> xView(N, x);
  concurrency::parallel_for_each(
    xView.get_extent(), 
    [=] (concurrency::index<1> idx) restrict(amp) 
    {
      xView[idx] *= value;
    }
  );
  xView.synchronize();
}

/*                  Y = alpha * op(A) * X + beta * Y                     */
ampblasStatus  ampblas_sgemv(const enum AMPBLAS_TRANS type, const int M,
                             const int N, const float *alpha, float *A,
                             const long aOffset,const int lda, float *X,
                             const long xOffset, const int incX,
                             const float *beta, float *Y,const long yOffset,
                             const int incY);

ampblasStatus  ampblas_dgemv(const enum AMPBLAS_TRANS type, const int M,
                             const int N, const double *alpha, double *A,
                             const long aOffset,const int lda, double *X,
                             const long xOffset, const int incX,
                             const double *beta, double *Y,const long yOffset,
                             const int incY);

/*                  C = alpha * op(A) * op(B) + beta * C                 */
ampblasStatus  ampblas_sgemm(const enum AMPBLAS_TRANS typeA,
                             const enum AMPBLAS_TRANS typeB, const int M,
                             const int N, const int K, const float *alpha,
                             float *A, const long lda, float *B,
                             const long ldb, const float *beta, float *C,
                             const long ldc, const long aOffset,
                             const long bOffset, const long cOffset);

ampblasStatus  ampblas_dgemm(const enum AMPBLAS_TRANS typeA,
                             const enum AMPBLAS_TRANS typeB, const int M,
                             const int N, const int K, const double *alpha,
                             double *A, const long lda, double *B,
                             const long ldb, const double *beta, double *C,
                             const long ldc, const long aOffset,
                             const long bOffset, const long cOffset);

#endif
