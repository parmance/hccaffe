/*
*
*  FILENAME : hcblas.h
*  This file is the top level header file which includes the Hcblaslilbrary class
*  for performing five blas operations ( saxpy, sger, sgemv, sgemm, cgemm )
*
*/

#ifndef HCBLAS_H
#define HCBLAS_H

#include <iostream>
#include "hc.hpp"
#include "hc_short_vector.hpp"

using namespace hc;
using namespace hc::short_vector;
using namespace std;
/* enumerator to indicate the status of  blas operation */
enum hcblasStatus {
    HCBLAS_SUCCESS = 0,
    HCBLAS_INVALID = -1
};

/* enumerator to define the layout of  input matrix for blas operation */
enum hcblasOrder {
    RowMajor ,
    ColMajor
};

/* enumerator to define the type of operation to be performed on the input matrix
 ( NO_TRANSPOSE, TRANSPOSE, CONJUGATE) */
enum hcblasTranspose {
    noTrans = 'n',
    trans = 't',
    conjugate = 'c'
};

struct hcComplex
{
     float real;
     float img;
};

template <typename Dtype>
void hc_axpy(const int N, const Dtype alpha, Dtype* x, Dtype* y)
{
  hc::extent<1> e(N);
	hc::parallel_for_each(
      e,	[=](hc::index<1> idx) __attribute__((hc, cpu)) {
		y[idx[0]] += alpha * x[idx[0]];
	} );
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
/* Class which implements the blas ( SGEMM, CGEMM, SGEMV, SGER, SAXPY )  */
class Hcblaslibrary
{
    public:
/*                  Y = alpha * X + Y                                    */
    hcblasStatus hcblas_saxpy(const int N, const float *alpha, float *X, const int incX,
                              float *Y, const int incY, const long xOffset, const long yOffset);

/* SAXPY - Overloaded function with arguments of type hc::array_view */

    hcblasStatus hcblas_saxpy(hc::accelerator_view &accl_view, const int N, const float &alpha,
			      float* &X, const int incX,
                              float* &Y, const int incY, 
			      const long xOffset, const long yOffset);

/* SAXPY - Overloaded function with arguments related to batch processing */

    hcblasStatus hcblas_saxpy(hc::accelerator_view &accl_view,
                              const int N, const float &alpha,
                              float* &X, const int incX, const long X_batchOffset,
                              float* &Y, const int incY, const long Y_batchOffset,
                              const long xOffset, const long yOffset, const int batchSize);


/*                  A = alpha * X * Y' + A                               */
    hcblasStatus hcblas_sger(hcblasOrder order, const int M, const int N, const float *alpha,
                             float *X, const long xOffset, const int incX,
                             float *Y, const long yOffset, const int incY,
                             float *A, const long aOffset, const int lda);

/* SGER - Overloaded function with arguments of type hc::array_view */

    hcblasStatus hcblas_sger(hc::accelerator_view &accl_view,
			     hcblasOrder order, const int M, const int N, const float &alpha,
                             float* &X, const long xOffset, const int incX,
                             float* &Y, const long yOffset, const int incY,
                             float* &A, const long aOffset, const int lda);

/* SGER - Overloaded function with arguments related to batch processing */

    hcblasStatus hcblas_sger(hc::accelerator_view &accl_view,
                             hcblasOrder order, const int M, const int N, const float &alpha,
                             float* &X, 
                             const long xOffset, const long X_batchOffset, const int incX,
                             float* &Y, 
                             const long yOffset, const long Y_batchOffset, const int incY,
                             float* &A, 
                             const long aOffset, const long A_batchOffset, const int lda, const int batchSize);

/*                  Y = alpha * op(A) * X + beta * Y                     */
    hcblasStatus hcblas_sgemv(hcblasOrder order, hcblasTranspose type, const int M,
                              const int N, const float *alpha, float *A,
                              const long aOffset,const int lda, float *X,
                              const long xOffset, const int incX,
                              const float *beta, float *Y,const long yOffset,
                              const int incY);

    hcblasStatus hcblas_sgemv2(hcblasOrder order, hcblasTranspose type, 
                               const int M, const int N, 
                               const float *alpha, float* &A_mat, const long aOffset,
                               const int lda, float* &X_mat, const long xOffset,
                               const int incX, const float *beta,
                               float* &Y_mat,const long yOffset, const int incY);

/*                  Y = alpha * op(A) * X + beta * Y                     */
    hcblasStatus hcblas_dgemv(hcblasOrder order, hcblasTranspose type, const int M,
	                      const int N, const double *alpha, double *A,
		              const long aOffset, const int lda, double *X,
		              const long xOffset, const int incX,
		              const double *beta, double  *Y, const long yOffset,
		              const int incY);

    hcblasStatus hcblas_dgemv2(hcblasOrder order, hcblasTranspose type,
                               const int M, const int N,
                               const double *alpha, double* &A_mat, const long aOffset,
                               const int lda, double* &X_mat, const long xOffset,
                               const int incX, const double *beta,
                               double* &Y_mat, const long yOffset, const int incY);

/* SGEMV- Overloaded function with arguments of type hc::array_view */
    hcblasStatus hcblas_sgemv(hc::accelerator_view &accl_view, hcblasOrder order,
			      hcblasTranspose type, const int M,
                              const int N, const float &alpha, 
                              float* &A, const long aOffset, const int lda, 
			      float* &X, const long xOffset, const int incX,
                              const float &beta,  
			      float* &Y, const long yOffset, const int incY);

    hcblasStatus hcblas_dgemv(hc::accelerator_view &accl_view, hcblasOrder order,
	                      hcblasTranspose type, const int M,
                 	      const int N, const double &alpha,
                 	      double* &A, const long aOffset, const int lda,
		              double* &X, const long xOffset, const int incX,
		              const double &beta,
		              double* &Y, const long yOffset, const int incY);

/* SGEMV- Overloaded function with arguments related to batch processing */
    hcblasStatus hcblas_sgemv(hc::accelerator_view &accl_view, hcblasOrder order,
                              hcblasTranspose type, const int M,
                              const int N, const float &alpha, float* &A, 
                              const long aOffset, const long A_batchOffset, const int lda,
                              float* &X, 
                              const long xOffset, const long X_batchOffset, const int incX,
                              const float &beta, float* &Y, 
                              const long yOffset, const long Y_batchOffset, const int incY, const int batchSize);

    hcblasStatus hcblas_dgemv(hc::accelerator_view &accl_view, hcblasOrder order,
	                      hcblasTranspose type, const int M,
		              const int N, const double &alpha, double* &A,
		              const long aOffset, const long A_batchOffset, const int lda,
		              double* &X,
		              const long xOffset, const long X_batchOffset, const int incX,
		              const double &beta, double* &Y,
		              const long yOffset, const long Y_batchOffset, const int incY, const int batchSize);

/*                  C = alpha * op(A) * op(B) + beta * C                 */
    hcblasStatus hcblas_sgemm(hcblasOrder order, hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M,
                              const int N, const int K, const float *alpha,
                              float *A, const long lda, float *B,
                              const long ldb, const float *beta, float *C,
                              const long ldc, const long aOffset,
                              const long bOffset, const long cOffset);

    hcblasStatus hcblas_sgemm2(hcblasOrder order, hcblasTranspose typeA, 
                               hcblasTranspose typeB, const int M, const int N,
                               const int K, const float *alpha,
                               float* &A_mat, const long lda,
                               float* &B_mat, const long ldb,
                               const float *beta, float* &C_mat,
                               const long ldc, const long aOffset,
                               const long bOffset, const long cOffset);

    hcblasStatus hcblas_dgemm(hcblasOrder order, hcblasTranspose typeA,
               		      hcblasTranspose typeB, const int M,
                	      const int N, const int K, const double *alpha,
                	      double *A, const long lda, double *B,
                 	      const long ldb, const double *beta, double *C,
		              const long ldc, const long aOffset,
                  	      const long bOffset, const long cOffset);

    hcblasStatus hcblas_dgemm2(hcblasOrder order, hcblasTranspose typeA, 
                              hcblasTranspose typeB, const int M, const int N,
                              const int K, const double *alpha,
                              double* &A_mat, const long lda,
                              double* &B_mat, const long ldb,
                              const double *beta, double* &C_mat,
                              const long ldc, const long aOffset,
                              const long bOffset, const long cOffset);

/* SGEMM- Overloaded function with arguments of type hc::array_view */
    hcblasStatus hcblas_sgemm(hc::accelerator_view &accl_view,
                   	      hcblasOrder order,hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M,
                              const int N, const int K, const float &alpha,
                              float* &A, const long lda, 
		              float* &B, const long ldb, 
			      const float &beta,  
			      float* &C, const long ldc, 
			      const long aOffset, const long bOffset, const long cOffset);

    hcblasStatus hcblas_dgemm(hc::accelerator_view &accl_view,
 		              hcblasOrder order, hcblasTranspose typeA,
  		              hcblasTranspose typeB, const int M,
		              const int N, const int K, const double &alpha,
		              double* &A, const long lda,
		              double* &B, const long ldb,
		              const double &beta,
		              double* &C, const long ldc,
		              const long aOffset, const long bOffset, const long cOffset);

/* SGEMM- Overloaded function with arguments related to batch processing */

    hcblasStatus hcblas_sgemm(hc::accelerator_view &accl_view,
                              hcblasOrder order, hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M,
                              const int N, const int K, const float &alpha,
                              float* &A, const long lda, const long A_batchOffset,
                              float* &B, const long ldb, const long B_batchOffset,
                              const float &beta,
                              float* &C, const long ldc, const long C_batchOffset,
                              const long aOffset, const long bOffset, const long cOffset, const int batchSize);

    hcblasStatus hcblas_dgemm(hc::accelerator_view &accl_view,
		              hcblasOrder order, hcblasTranspose typeA,
		              hcblasTranspose typeB, const int M,
		              const int N, const int K, const double &alpha,
		              double* &A, const long lda, const long A_batchOffset,
		              double* &B, const long ldb, const long B_batchOffset,
		              const double &beta,
		              double* &C, const long ldc, const long C_batchOffset,
		              const long aOffset, const long bOffset, const long cOffset, const int batchSize);

/*                  C = alpha * op(A) * op(B) + beta * C                   */
    hcblasStatus hcblas_cgemm(hcblasOrder order, hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M, 
                              const int N, const int K,
                              const hcComplex *alpha,
                              const hcComplex *A, const long aOffset, const long lda,
                              const hcComplex *B, const long bOffset, const long ldb,
                              const hcComplex *beta, hcComplex *C,
                              const long cOffset, const long ldc);

/* CGEMM - Overloaded function with arguments of type hc::array_view */     
   hcblasStatus hcblas_cgemm(hc::accelerator_view &accl_view,
			     hcblasOrder order, hcblasTranspose typeA,
                             hcblasTranspose typeB, const int M,
                             const int N, const int K,
                             const float_2 &alpha,
                             float_2* &A, const long aOffset, const long lda,
                             float_2* &B, const long bOffset, const long ldb,
                             const float_2 &beta, 
                             float_2* &C, const long cOffset, const long ldc);

/* CGEMM - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_cgemm(hc::accelerator_view &accl_view,
                             hcblasOrder order, hcblasTranspose typeA,
                             hcblasTranspose typeB, const int M,
                             const int N, const int K,
                             const float_2 &alpha,
                             float_2* &A, 
                             const long aOffset, const long A_batchOffset, const long lda,
                             float_2* &B, 
			     const long bOffset, const long B_batchOffset, const long ldb,
                             const float_2 &beta,
                             float_2* &C, 
			     const long cOffset, const long C_batchOffset, const long ldc, const int batchSize);

};
#define TILE_SIZE 256
#define MAX_TILES 1024
#endif
