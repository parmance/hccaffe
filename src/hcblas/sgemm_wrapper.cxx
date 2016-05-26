#include "hcblas/sgemm_kernels.h"

// Sgemm Wrapper routine that invokes the appropriate kernel routines depending on the input dimension M N and K
hcblasStatus gemm_HC(hc::accelerator_view &accl_view,
                     const int order, char TransA, char TransB,
                     const int M, const int N, const int K,
                     const float alpha, float* &A,
                     long aOffset, long lda,
                     float* &B,
                     long bOffset, long ldb, const float beta,
                     float* &C,
                     long cOffset, long ldc,
                     long A_batchOffset = 0, long B_batchOffset = 0, long C_batchOffset = 0, int batchSize = 0) {
  int i, j, k;
  hcblasStatus status = HCBLAS_SUCCESS;

  // Quick return if possible
  if (!M || !N || !K) {
    return HCBLAS_INVALID;
  }

  // For alpha = 0
  if (alpha == 0) {
    if (beta == 0) {
     for (k = 0; k <= batchSize; ++k) {
      for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
          C[cOffset + C_batchOffset * k + i + j * ldc] = 0;
        }
      }
     }
    } else {
     for(k = 0; k <= batchSize; ++k) {
      for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
          C[cOffset + C_batchOffset * k + i + j * ldc] *= beta;
        }
      }
     }
    }

    return status;
  }

    // Start the operations

  if (order) {
    if(batchSize > 0) {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        } else {
          status = gemm_NoTransB(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      } else {
        status = gemm_TransAB(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      }
    } else {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        } else {
          status = gemm_NoTransB(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      } else {
        status = gemm_TransAB(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      }
    }
  } else {
    if(batchSize > 0) {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_rMajor(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        } else {
          status = gemm_NoTransB_rMajor(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_rMajor(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      } else {
        status = gemm_TransAB_rMajor(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      }
    } else {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_rMajor(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        } else {
          status = gemm_NoTransB_rMajor(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_rMajor(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      } else {
        status = gemm_TransAB_rMajor(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      }
    }
  }

  return status;
}

hcblasStatus gemm_HC_d(hc::accelerator_view &accl_view,
                     const int order, char TransA, char TransB,
                     const int M, const int N, const int K,
                     const double alpha, double* &A,
                     long aOffset, long lda,
                     double* &B,
                     long bOffset, long ldb, const double beta,
                     double* &C,
                     long cOffset, long ldc,
                     long A_batchOffset = 0, long B_batchOffset = 0, long C_batchOffset = 0, int batchSize = 0) {
  int i, j, k;
  hcblasStatus status = HCBLAS_SUCCESS;

  // Quick return if possible
  if (!M || !N || !K) {
    return HCBLAS_INVALID;
  }

  // For alpha = 0
  if (alpha == 0) {
    if (beta == 0) {
     for (k = 0; k <= batchSize; ++k) {
      for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
          C[cOffset + C_batchOffset * k + i + j * ldc] = 0;
        }
      }
     }
    } else {
     for(k = 0; k <= batchSize; ++k) {
      for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
          C[cOffset + C_batchOffset * k + i + j * ldc] *= beta;
        }
      }
     }
    }

    return status;
  }

    // Start the operations

  if (order) {
    if(batchSize > 0) {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        } else {
          status = gemm_NoTransB_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      } else {
        status = gemm_TransAB_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      }
    } else {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        } else {
          status = gemm_NoTransB_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      } else {
        status = gemm_TransAB_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      }
    }
  } else {
    if(batchSize > 0) {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_rMajor_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        } else {
          status = gemm_NoTransB_rMajor_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_rMajor_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      } else {
        status = gemm_TransAB_rMajor_d(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      }
    } else {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_rMajor_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        } else {
          status = gemm_NoTransB_rMajor_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_rMajor_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      } else {
        status = gemm_TransAB_rMajor_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      }
    }
  }

  return status;
}


// Sgemm Call Type 1: Inputs and Outputs are host float pointers
hcblasStatus Hcblaslibrary :: hcblas_sgemm(hcblasOrder order,
                                           hcblasTranspose typeA,
                                           hcblasTranspose typeB,
                                           const int M, const int N,
                                           const int K, const float *alpha,
                                           float *A, const long lda,
                                           float *B, const long ldb,
                                           const float *beta, float *C,
                                           const long ldc, const long aOffset,
                                           const long bOffset,
                                           const long cOffset) {
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view());

    hcblasStatus status = gemm_HC(accl_view, order, typeA, typeB, M, N, K, *alpha,
                                  A, aOffset, lda, B, bOffset, ldb,
                                  *beta, C, cOffset, ldc);

    return status;
}

// Dgemm Call Type 1: Inputs and Outputs are host double pointers
hcblasStatus Hcblaslibrary::hcblas_dgemm(hcblasOrder order,
                                         hcblasTranspose typeA,
                                         hcblasTranspose typeB,
                                         const int M, const int N,
                                         const int K, const double *alpha,
                                         double *A, const long lda,
                                         double *B, const long ldb,
                                         const double *beta, double *C,
                                         const long ldc, const long aOffset,
                                         const long bOffset,
                                         const long cOffset) {
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view());
    hcblasStatus status = gemm_HC_d(accl_view, order, typeA, typeB, M, N, K, *alpha,
                                    A, aOffset, lda, B, bOffset, ldb,
                                    *beta, C, cOffset, ldc);
    return status;
}

// Sgemm Call Type II: Inputs and outputs are HCC C++ float array_View containers
hcblasStatus  Hcblaslibrary :: hcblas_sgemm(hc::accelerator_view &accl_view,
                                            hcblasOrder order,
                                            hcblasTranspose typeA,
                                            hcblasTranspose typeB, const int M,
                                            const int N, const int K, const float &alpha,
                                            float* &A, const long lda,
                                            float* &B, const long ldb,
                                            const float &beta,
                                            float* &C, const long ldc,
                                            const long aOffset, const long bOffset, const long cOffset) {
    hcblasStatus status = gemm_HC(accl_view, order, typeA, typeB, M, N, K, alpha, A,
                                  aOffset, lda, B, bOffset, ldb, beta, C,
                                  cOffset, ldc);
    return status;
}

// Dgemm Call Type II: Inputs and outputs are HCC C++ double array_View containers
hcblasStatus  Hcblaslibrary::hcblas_dgemm(hc::accelerator_view &accl_view,
                                          hcblasOrder order,
                                          hcblasTranspose typeA,
                                          hcblasTranspose typeB, const int M,
                                          const int N, const int K, const double &alpha,
                                          double* &A, const long lda,
                                          double* &B, const long ldb,
                                          const double &beta,
                                          double* &C, const long ldc,
                                          const long aOffset, const long bOffset, const long cOffset) {
    hcblasStatus status = gemm_HC_d(accl_view, order, typeA, typeB, M, N, K, alpha, A,
                                    aOffset, lda, B, bOffset, ldb, beta, C,
                                    cOffset, ldc);

    return status;
}

/* SGEMM- Overloaded function with arguments related to batch processing */
hcblasStatus Hcblaslibrary :: hcblas_sgemm(hc::accelerator_view &accl_view,
                                           hcblasOrder order,
                                           hcblasTranspose typeA,
                                           hcblasTranspose typeB, const int M,
                                           const int N, const int K, const float &alpha,
                                           float* &A, const long lda, const long A_batchOffset,
                                           float* &B, const long ldb, const long B_batchOffset,
                                           const float &beta,
                                           float* &C, const long ldc, const long C_batchOffset,
                                           const long aOffset, const long bOffset, const long cOffset, const int batchSize) {    
    hcblasStatus status = gemm_HC(accl_view, order, typeA, typeB, M, N, K, alpha, A, aOffset, lda, B,
                                  bOffset, ldb, beta, C, cOffset, ldc, A_batchOffset, B_batchOffset, C_batchOffset, batchSize);

    return status;
}

/* DGEMM- Overloaded function with arguments related to batch processing */
hcblasStatus Hcblaslibrary :: hcblas_dgemm(hc::accelerator_view &accl_view,
                                           hcblasOrder order,
                                           hcblasTranspose typeA,
                                           hcblasTranspose typeB, const int M,
                                           const int N, const int K, const double &alpha,
                                           double* &A, const long lda, const long A_batchOffset,
                                           double* &B, const long ldb, const long B_batchOffset,
                                           const double &beta,
                                           double* &C, const long ldc, const long C_batchOffset,
                                           const long aOffset, const long bOffset, const long cOffset, const int batchSize) {    
    hcblasStatus status = gemm_HC_d(accl_view, order, typeA, typeB, M, N, K, alpha, A, aOffset, lda, B,
                                    bOffset, ldb, beta, C, cOffset, ldc, A_batchOffset, B_batchOffset, C_batchOffset, batchSize);

    return status;
}

// Sgemm Call Type 1: Alpha and beta are host float pointers
hcblasStatus Hcblaslibrary :: hcblas_sgemm2(hcblasOrder order,
                                            hcblasTranspose typeA,
                                            hcblasTranspose typeB,
                                            const int M, const int N,
                                            const int K, const float *alpha,
                                            float* &A, const long lda,
                                            float* &B, const long ldb,
                                            const float *beta, float* &C,
                                            const long ldc, const long aOffset,
                                            const long bOffset,
                                            const long cOffset) {
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view());
    hcblasStatus status = gemm_HC(accl_view, order, typeA, typeB, M, N, K, *alpha,
                                  A, aOffset, lda, B, bOffset, ldb,
                                  *beta, C, cOffset, ldc);
    return status;
}

// Dgemm Call Type 1: Alpha and beta are host double pointers
hcblasStatus Hcblaslibrary :: hcblas_dgemm2(hcblasOrder order,
                                            hcblasTranspose typeA,
                                            hcblasTranspose typeB,
                                            const int M, const int N,
                                            const int K, const double *alpha,
                                            double* &A, const long lda,
                                            double* &B, const long ldb,
                                            const double *beta, double* &C,
                                            const long ldc, const long aOffset,
                                            const long bOffset, const long cOffset) {
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view());
    hcblasStatus status = gemm_HC_d(accl_view, order, typeA, typeB, M, N, K,
                                    *alpha, A, aOffset, lda, B, bOffset,
                                    ldb, *beta, C, cOffset, ldc);

    return status;
}
