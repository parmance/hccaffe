/*
*
*  FILENAME : ampblas.h
*  This file is the top level header file which includes the Ampblaslilbrary class
*  for performing five blas operations ( saxpy, sger, sgemv, sgemm, cgemm )
*
*/

#ifndef AMPBLASLIB_H
#define AMPBLASLIB_H

#include <iostream>
#include <amp.h>
#include <amp_short_vectors.h>

using namespace Concurrency;
using namespace Concurrency::graphics;
using namespace std;
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
  Concurrency::array_view<Dtype, 1> xView =
      *((Concurrency::array_view<Dtype, 1>*)(x));
  Concurrency::array_view<Dtype, 1> yView =
      *((Concurrency::array_view<Dtype, 1>*)(y));
  Concurrency::extent<1> e(N);
	concurrency::parallel_for_each(
      e,	[=](concurrency::index<1> idx) restrict(amp) {
		yView[idx] += alpha * xView[idx];
	} );
}

template <typename Dtype>
void amp_scale(const int N, const Dtype value, Dtype* x)
{
  Concurrency::array_view<Dtype, 1> xView =
      *((Concurrency::array_view<Dtype, 1>*)(x));
  Concurrency::extent<1> e(N);
	concurrency::parallel_for_each(
      e, [=](concurrency::index<1> idx) restrict(amp){
		xView[idx] *= value;
	}	);
}
/* Class which implements the blas ( SGEMM, CGEMM, SGEMV, SGER, SAXPY )  */
class Ampblaslibrary
{
    public:
/*                  Y = alpha * X + Y                                    */
    ampblasStatus ampblas_saxpy(const int N,
                                const float *alpha, float *X, const int incX,
                                float *Y, const int incY, const long xOffset,
                                const long yOffset);

/* SAXPY - Overloaded function with arguments of type Concurrency::array_view */

    ampblasStatus ampblas_saxpy(Concurrency::accelerator_view &accl_view,
				const int N, const float &alpha,
				Concurrency::array_view<float> &X, const int incX,
                                Concurrency::array_view<float> &Y, const int incY, 
				const long xOffset, const long yOffset);

/* SAXPY - Overloaded function with arguments related to batch processing */

    ampblasStatus ampblas_saxpy(Concurrency::accelerator_view &accl_view,
                                const int N, const float &alpha,
                                Concurrency::array_view<float> &X, const int incX, const long X_batchOffset,
                                Concurrency::array_view<float> &Y, const int incY, const long Y_batchOffset,
                                const long xOffset, const long yOffset, const int batchSize);


/*                  A = alpha * X * Y' + A                               */
    ampblasStatus ampblas_sger(const int order, const int M, const int N, const float *alpha,
                               float *X, const long xOffset, const int incX,
                               float *Y, const long yOffset, const int incY,
                               float *A, const long aOffset, const int lda);

/* SGER - Overloaded function with arguments of type Concurrency::array_view */

    ampblasStatus ampblas_sger(Concurrency::accelerator_view &accl_view,
			       const int order, const int M, const int N, const float &alpha,
                               Concurrency::array_view<float> &X, const long xOffset, const int incX,
                               Concurrency::array_view<float> &Y, const long yOffset, const int incY,
                               Concurrency::array_view<float> &A, const long aOffset, const int lda);

/* SGER - Overloaded function with arguments related to batch processing */

    ampblasStatus ampblas_sger(Concurrency::accelerator_view &accl_view,
                               const int order, const int M, const int N, const float &alpha,
                               Concurrency::array_view<float> &X, 
                               const long xOffset, const long X_batchOffset, const int incX,
                               Concurrency::array_view<float> &Y, 
                               const long yOffset, const long Y_batchOffset, const int incY,
                               Concurrency::array_view<float> &A, 
                               const long aOffset, const long A_batchOffset, const int lda, const int batchSize);

/*                  Y = alpha * op(A) * X + beta * Y                     */
    ampblasStatus ampblas_sgemv(const enum AMPBLAS_TRANS type, const int M,
                                const int N, const float *alpha, float *A,
                                const long aOffset,const int lda, float *X,
                                const long xOffset, const int incX,
                                const float *beta, float *Y,const long yOffset,
                                const int incY);
    ampblasStatus ampblas_sgemv2(const enum AMPBLAS_TRANS type,
        const int M, const int N,
        const float *alpha, Concurrency::array_view<float> &A_mat, const long aOffset,
        const int lda, Concurrency::array_view<float> &X_mat, const long xOffset,
        const int incX, const float *beta,
        Concurrency::array_view<float> &Y_mat,const long yOffset, const int incY);

	/*                  Y = alpha * op(A) * X + beta * Y                     */
	ampblasStatus ampblas_dgemv(const enum AMPBLAS_TRANS type, const int M,
		const int N, const double *alpha, double *A,
		const long aOffset, const int lda, double *X,
		const long xOffset, const int incX,
		const double *beta, double  *Y, const long yOffset,
		const int incY);
     ampblasStatus ampblas_dgemv2(const enum AMPBLAS_TRANS type,
        const int M, const int N,
        const double *alpha, Concurrency::array_view<double> &A_mat, const long aOffset,
        const int lda, Concurrency::array_view<double> &X_mat, const long xOffset,
        const int incX, const double *beta,
        Concurrency::array_view<double> &Y_mat, const long yOffset, const int incY);

/* SGEMV- Overloaded function with arguments of type Concurrency::array_view */
    ampblasStatus ampblas_sgemv(Concurrency::accelerator_view &accl_view,
				const enum AMPBLAS_TRANS type, const int M,
                                const int N, const float &alpha, 
                                Concurrency::array_view<float> &A, const long aOffset, const int lda, 
				Concurrency::array_view<float> &X, const long xOffset, const int incX,
                                const float &beta,  
				Concurrency::array_view<float> &Y, const long yOffset, const int incY);
	ampblasStatus ampblas_dgemv(Concurrency::accelerator_view &accl_view,
		const enum AMPBLAS_TRANS type, const int M,
		const int N, const float &alpha,
		Concurrency::array_view<double> &A, const long aOffset, const int lda,
		Concurrency::array_view<double> &X, const long xOffset, const int incX,
		const double &beta,
		Concurrency::array_view<double> &Y, const long yOffset, const int incY);

/* SGEMV- Overloaded function with arguments related to batch processing */
    ampblasStatus ampblas_sgemv(Concurrency::accelerator_view &accl_view,
                                const enum AMPBLAS_TRANS type, const int M,
                                const int N, const float &alpha, Concurrency::array_view<float> &A, 
                                const long aOffset, const long A_batchOffset, const int lda,
                                Concurrency::array_view<float> &X, 
                                const long xOffset, const long X_batchOffset, const int incX,
                                const float &beta, Concurrency::array_view<float> &Y, 
                                const long yOffset, const long Y_batchOffset, const int incY, const int batchSize);
	ampblasStatus ampblas_dgemv(Concurrency::accelerator_view &accl_view,
		const enum AMPBLAS_TRANS type, const int M,
		const int N, const double &alpha, Concurrency::array_view<double> &A,
		const long aOffset, const long A_batchOffset, const int lda,
		Concurrency::array_view<double> &X,
		const long xOffset, const long X_batchOffset, const int incX,
		const double &beta, Concurrency::array_view<double> &Y,
		const long yOffset, const long Y_batchOffset, const int incY, const int batchSize);

/*                  C = alpha * op(A) * op(B) + beta * C                 */
    ampblasStatus ampblas_sgemm(const enum AMPBLAS_ORDER order, const enum AMPBLAS_TRANS typeA,
                                const enum AMPBLAS_TRANS typeB, const int M,
                                const int N, const int K, const float *alpha,
                                float *A, const long lda, float *B,
                                const long ldb, const float *beta, float *C,
                                const long ldc, const long aOffset,
                                const long bOffset, const long cOffset);
    ampblasStatus ampblas_sgemm2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_TRANS typeA, const enum AMPBLAS_TRANS typeB,
        const int M, const int N,
        const int K, const float *alpha,
        Concurrency::array_view<float> &A_mat, const long lda,
        Concurrency::array_view<float> &B_mat, const long ldb,
        const float *beta, Concurrency::array_view<float> &C_mat,
        const long ldc, const long aOffset,
        const long bOffset,
        const long cOffset);
   ampblasStatus ampblas_dgemm(const enum AMPBLAS_ORDER order,
                const enum AMPBLAS_TRANS typeA,
		const enum AMPBLAS_TRANS typeB, const int M,
		const int N, const int K, const double *alpha,
		double *A, const long lda, double *B,
		const long ldb, const double *beta, double *C,
		const long ldc, const long aOffset,
		const long bOffset, const long cOffset);
   ampblasStatus ampblas_dgemm2(const enum AMPBLAS_ORDER order,
     const enum AMPBLAS_TRANS typeA, const enum AMPBLAS_TRANS typeB,
        const int M, const int N,
        const int K, const double *alpha,
        Concurrency::array_view<double> &A_mat, const long lda,
        Concurrency::array_view<double> &B_mat, const long ldb,
        const double *beta, Concurrency::array_view<double> &C_mat,
        const long ldc, const long aOffset,
        const long bOffset,
        const long cOffset);
/* SGEMM- Overloaded function with arguments of type Concurrency::array_view */
    ampblasStatus ampblas_sgemm(Concurrency::accelerator_view &accl_view,
 				const enum AMPBLAS_ORDER order,const enum AMPBLAS_TRANS typeA,
                                const enum AMPBLAS_TRANS typeB, const int M,
                                const int N, const int K, const float &alpha,
                                Concurrency::array_view<float> &A, const long lda, 
		                Concurrency::array_view<float> &B, const long ldb, 
				const float &beta,  
				Concurrency::array_view<float> &C, const long ldc, 
				const long aOffset, const long bOffset, const long cOffset);
	ampblasStatus ampblas_dgemm(Concurrency::accelerator_view &accl_view,
		const enum AMPBLAS_ORDER order, const enum AMPBLAS_TRANS typeA,
		const enum AMPBLAS_TRANS typeB, const int M,
		const int N, const int K, const double &alpha,
		Concurrency::array_view<double> &A, const long lda,
		Concurrency::array_view<double> &B, const long ldb,
		const double &beta,
		Concurrency::array_view<double> &C, const long ldc,
		const long aOffset, const long bOffset, const long cOffset);

/* SGEMM- Overloaded function with arguments related to batch processing */
    ampblasStatus ampblas_sgemm(Concurrency::accelerator_view &accl_view,
                                const enum AMPBLAS_ORDER order, const enum AMPBLAS_TRANS typeA,
                                const enum AMPBLAS_TRANS typeB, const int M,
                                const int N, const int K, const float &alpha,
                                Concurrency::array_view<float> &A, const long lda, const long A_batchOffset,
                                Concurrency::array_view<float> &B, const long ldb, const long B_batchOffset,
                                const float &beta,
                                Concurrency::array_view<float> &C, const long ldc, const long C_batchOffset,
                                const long aOffset, const long bOffset, const long cOffset, const int batchSize);
	ampblasStatus ampblas_dgemm(Concurrency::accelerator_view &accl_view,
		const enum AMPBLAS_ORDER order, const enum AMPBLAS_TRANS typeA,
		const enum AMPBLAS_TRANS typeB, const int M,
		const int N, const int K, const double &alpha,
		Concurrency::array_view<double> &A, const long lda, const long A_batchOffset,
		Concurrency::array_view<double> &B, const long ldb, const long B_batchOffset,
		const double &beta,
		Concurrency::array_view<double> &C, const long ldc, const long C_batchOffset,
		const long aOffset, const long bOffset, const long cOffset, const int batchSize);

/*                  C = alpha * op(A) * op(B) + beta * C                   */
    ampblasStatus ampblas_cgemm(const int order, const enum AMPBLAS_TRANS typeA,
                                const enum AMPBLAS_TRANS typeB, const int M, 
                                const int N, const int K,
                                const ampComplex *alpha,
                                const ampComplex *A, const long aOffset, const long lda,
                                const ampComplex *B, const long bOffset, const long ldb,
                                const ampComplex *beta, ampComplex *C,
                                const long cOffset, const long ldc);

/* CGEMM - Overloaded function with arguments of type Concurrency::array_view */     
   ampblasStatus ampblas_cgemm(Concurrency::accelerator_view &accl_view,
			       const int order, const enum AMPBLAS_TRANS typeA,
                               const enum AMPBLAS_TRANS typeB, const int M,
                               const int N, const int K,
                               const Concurrency::graphics::float_2 &alpha,
                               Concurrency::array_view<float_2> &A, const long aOffset, const long lda,
                               Concurrency::array_view<float_2> &B, const long bOffset, const long ldb,
                               const Concurrency::graphics::float_2 &beta, 
                               Concurrency::array_view<float_2> &C, const long cOffset, const long ldc);

/* CGEMM - Overloaded function with arguments related to batch processing */
   ampblasStatus ampblas_cgemm(Concurrency::accelerator_view &accl_view,
                               const int order, const enum AMPBLAS_TRANS typeA,
                               const enum AMPBLAS_TRANS typeB, const int M,
                               const int N, const int K,
                               const Concurrency::graphics::float_2 &alpha,
                               Concurrency::array_view<float_2> &A, 
                               const long aOffset, const long A_batchOffset, const long lda,
                               Concurrency::array_view<float_2> &B, 
			       const long bOffset, const long B_batchOffset, const long ldb,
                               const Concurrency::graphics::float_2 &beta,
                               Concurrency::array_view<float_2> &C, 
			       const long cOffset, const long C_batchOffset, const long ldc, const int batchSize);

};
#define TILE_SIZE 256
#define MAX_TILES 1024
#endif
