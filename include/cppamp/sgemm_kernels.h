#include "ampblaslib.h"
#include <amp.h>
#include <amp_math.h>

using namespace Concurrency;

#define TILESIZE 8
#define STEPSIZE 8
#define MICROTILESIZE 2
#define BANKTILESIZE (TILESIZE + 1)
#define STEPTILERATIO STEPSIZE/TILESIZE
#define STEPTILEPROD STEPSIZE*TILESIZE
#define BANKNUMTILEELMTS TILESIZE*BANKTILESIZE
#define NUMTILEELMTS TILESIZE*TILESIZE
#define TOTMICROTILEPROD (TILESIZE*TILESIZE*MICROTILESIZE)
#define MICROTILEPROD (TILESIZE*MICROTILESIZE)
#define BANKMICROTILESIZE (TILESIZE*MICROTILESIZE+1)

#define  MS1x1(offsetA, offsetB)			\
            for(int iter = 0; iter < STEPSIZE/TILESIZE; ++iter) \
            {\
              rA[0][iter] = lA[offA + (TILESIZE * TILESIZE) * iter];	\
              rB[0][iter] = lB[offB + (TILESIZE * TILESIZE) * iter];	\
              rC[0][0] +=rA[0][iter] *rB[0][iter]; \
            }\
            offA += offsetA;			\
            offB += offsetB;			\
 
#define  MS1x1_NOBANK(offset)			\
            for(int iter = 0; iter < STEPTILERATIO; ++iter) \
            {\
              rA[0][iter] = lA[offA + (BANKNUMTILEELMTS) * iter];	\
              rB[0][iter] = lB[offB + (BANKNUMTILEELMTS) * iter];	\
              rC[0][0] +=rA[0][iter] *rB[0][iter]; \
            }\
            offA += offset;			\
            offB += offset;			\
 
#define  MTS                                                                \
           for(int iter = 0; iter < MICROTILESIZE ; iter++)                 \
           {                                                                \
             rA[0][iter] = lA[offA + (iter * TILESIZE)];                    \
             rB[0][iter] = lB[offB + (iter * TILESIZE)];                    \
           }                                                                \
           for(int rowIndex = 0; rowIndex < MICROTILESIZE ; rowIndex++)     \
           {                                                                \
           for(int colIndex = 0; colIndex < MICROTILESIZE ; colIndex++)     \
           {                                                                \
           rC[rowIndex][colIndex] = rA[0][rowIndex] * rB[0][colIndex] +     \
                                    rC[rowIndex][colIndex];                 \
           }                                                                \
           }                                                                \
           offA += (MICROTILESIZE * TILESIZE);                              \
           offB += (MICROTILESIZE * TILESIZE);                              \
 
#define  MTS_NOBANK                                                         \
           for(int iter = 0; iter < MICROTILESIZE ; iter++)                 \
           {                                                                \
             rA[0][iter] = lA[offA + (iter * TILESIZE)];                    \
             rB[0][iter] = lB[offB + (iter * TILESIZE)];                    \
           }                                                                \
           for(int rowIndex = 0; rowIndex < MICROTILESIZE ; rowIndex++)     \
           {                                                                \
           for(int colIndex = 0; colIndex < MICROTILESIZE ; colIndex++)     \
           {                                                                \
           rC[rowIndex][colIndex] = rA[0][rowIndex] * rB[0][colIndex] +     \
                                    rC[rowIndex][colIndex];                 \
           }                                                                \
           }                                                                \
           offA += BANKMICROTILESIZE;                              \
           offB += BANKMICROTILESIZE;                              \
 

/*
*  SGEMM kernels - column major Order
*/
ampblasStatus gemm_NoTransAB(Concurrency::accelerator_view &accl_view,
                             Concurrency::array_view<float, 1> &A, long aOffset,
                             Concurrency::array_view<float, 1> &B, long bOffset,
                             Concurrency::array_view<float, 1> &C, long cOffset,
                             int M, int N, int K, int lda, int ldb, int ldc,
                             float alpha, float beta);
ampblasStatus gemm_NoTransAB_d(Concurrency::accelerator_view &accl_view,
                               Concurrency::array_view<double, 1> &A, long aOffset,
                               Concurrency::array_view<double, 1> &B, long bOffset,
                               Concurrency::array_view<double, 1> &C, long cOffset,
                               int M, int N, int K, int lda, int ldb, int ldc,
                               double alpha, double beta);
ampblasStatus gemm_NoTransA(Concurrency::accelerator_view &accl_view,
                            Concurrency::array_view<float, 1> &A, long aOffset,
                            Concurrency::array_view<float, 1> &B, long bOffset,
                            Concurrency::array_view<float, 1> &C, long cOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            float alpha, float beta);
ampblasStatus gemm_NoTransA_d(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<double, 1> &A, long aOffset,
                              Concurrency::array_view<double, 1> &B, long bOffset,
                              Concurrency::array_view<double, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              double alpha, double beta);
ampblasStatus gemm_NoTransB(Concurrency::accelerator_view &accl_view,
                            Concurrency::array_view<float, 1> &A, long aOffset,
                            Concurrency::array_view<float, 1> &B, long bOffset,
                            Concurrency::array_view<float, 1> &C, long cOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            float alpha, float beta);
ampblasStatus gemm_NoTransB_d(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<double, 1> &A, long aOffset,
                              Concurrency::array_view<double, 1> &B, long bOffset,
                              Concurrency::array_view<double, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              double alpha, double beta);

ampblasStatus gemm_TransAB(Concurrency::accelerator_view &accl_view,
                           Concurrency::array_view<float, 1> &A, long aOffset,
                           Concurrency::array_view<float, 1> &B, long bOffset,
                           Concurrency::array_view<float, 1> &C, long cOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           float alpha, float beta);
ampblasStatus gemm_TransAB_d(Concurrency::accelerator_view &accl_view,
                             Concurrency::array_view<double, 1> &A, long aOffset,
                             Concurrency::array_view<double, 1> &B, long bOffset,
                             Concurrency::array_view<double, 1> &C, long cOffset,
                             int M, int N, int K, int lda, int ldb, int ldc,
                             double alpha, double beta);

/*
* SGEMM Kernels for Batch processing in column major order
*/

ampblasStatus gemm_NoTransAB(Concurrency::accelerator_view &accl_view,
                             Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                             Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                             Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                             int M, int N, int K, int lda, int ldb, int ldc,
                             float alpha, float beta, int batchSize);
ampblasStatus gemm_NoTransAB_d(Concurrency::accelerator_view &accl_view,
                               Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                               Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                               Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                               int M, int N, int K, int lda, int ldb, int ldc,
                               double alpha, double beta, int batchSize);
ampblasStatus gemm_NoTransA(Concurrency::accelerator_view &accl_view,
                            Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                            Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                            Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            float alpha, float beta, int batchSize);
ampblasStatus gemm_NoTransA_d(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                              Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                              Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              double alpha, double beta, int batchSize);
ampblasStatus gemm_NoTransB(Concurrency::accelerator_view &accl_view,
                            Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                            Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                            Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            float alpha, float beta, int batchSize);
ampblasStatus gemm_NoTransB_d(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                              Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                              Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              double alpha, double beta, int batchSize);
ampblasStatus gemm_TransAB(Concurrency::accelerator_view &accl_view,
                           Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                           Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                           Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           float alpha, float beta, int batchSize);
ampblasStatus gemm_TransAB_d(Concurrency::accelerator_view &accl_view,
                             Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                             Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                             Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                             int M, int N, int K, int lda, int ldb, int ldc,
                             double alpha, double beta, int batchSize);

/*
* SGEMM Kernels - Row major order
*/

ampblasStatus gemm_NoTransAB_rMajor(Concurrency::accelerator_view &accl_view,
                                    Concurrency::array_view<float, 1> &A, long aOffset,
                                    Concurrency::array_view<float, 1> &B, long bOffset,
                                    Concurrency::array_view<float, 1> &C, long cOffset,
                                    int M, int N, int K, int lda, int ldb, int ldc,
                                    float alpha, float beta);
ampblasStatus gemm_NoTransAB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                      Concurrency::array_view<double, 1> &A, long aOffset,
                                      Concurrency::array_view<double, 1> &B, long bOffset,
                                      Concurrency::array_view<double, 1> &C, long cOffset,
                                      int M, int N, int K, int lda, int ldb, int ldc,
                                      double alpha, double beta);
ampblasStatus gemm_NoTransA_rMajor(Concurrency::accelerator_view &accl_view,
                                   Concurrency::array_view<float, 1> &A, long aOffset,
                                   Concurrency::array_view<float, 1> &B, long bOffset,
                                   Concurrency::array_view<float, 1> &C, long cOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta);
ampblasStatus gemm_NoTransA_rMajor_d(Concurrency::accelerator_view &accl_view,
                                     Concurrency::array_view<double, 1> &A, long aOffset,
                                     Concurrency::array_view<double, 1> &B, long bOffset,
                                     Concurrency::array_view<double, 1> &C, long cOffset,
                                     int M, int N, int K, int lda, int ldb, int ldc,
                                     double alpha, double beta);
ampblasStatus gemm_NoTransB_rMajor(Concurrency::accelerator_view &accl_view,
                                   Concurrency::array_view<float, 1> &A, long aOffset,
                                   Concurrency::array_view<float, 1> &B, long bOffset,
                                   Concurrency::array_view<float, 1> &C, long cOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta);
ampblasStatus gemm_NoTransB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                     Concurrency::array_view<double, 1> &A, long aOffset,
                                     Concurrency::array_view<double, 1> &B, long bOffset,
                                     Concurrency::array_view<double, 1> &C, long cOffset,
                                     int M, int N, int K, int lda, int ldb, int ldc,
                                     double alpha, double beta);
ampblasStatus gemm_TransAB_rMajor(Concurrency::accelerator_view &accl_view,
                                  Concurrency::array_view<float, 1> &A, long aOffset,
                                  Concurrency::array_view<float, 1> &B, long bOffset,
                                  Concurrency::array_view<float, 1> &C, long cOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta);

ampblasStatus gemm_TransAB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                    Concurrency::array_view<double, 1> &A, long aOffset,
                                    Concurrency::array_view<double, 1> &B, long bOffset,
                                    Concurrency::array_view<double, 1> &C, long cOffset,
                                    int M, int N, int K, int lda, int ldb, int ldc,
                                    double alpha, double beta);

/*
* SGEMM Kernels for Batch-processing in Row major order
*/

ampblasStatus gemm_NoTransAB_rMajor(Concurrency::accelerator_view &accl_view,
                                    Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                                    Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                                    Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                                    int M, int N, int K, int lda, int ldb, int ldc,
                                    float alpha, float beta, int batchSize);
ampblasStatus gemm_NoTransAB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                      Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                                      Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                                      Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                                      int M, int N, int K, int lda, int ldb, int ldc,
                                      double alpha, double beta, int batchSize);
ampblasStatus gemm_NoTransA_rMajor(Concurrency::accelerator_view &accl_view,
                                   Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                                   Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                                   Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batchSize);
ampblasStatus gemm_NoTransA_rMajor_d(Concurrency::accelerator_view &accl_view,
                                     Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                                     Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                                     Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                                     int M, int N, int K, int lda, int ldb, int ldc,
                                     double alpha, double beta, int batchSize);
ampblasStatus gemm_NoTransB_rMajor(Concurrency::accelerator_view &accl_view,
                                   Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                                   Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                                   Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batchSize);
ampblasStatus gemm_NoTransB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                     Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                                     Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                                     Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                                     int M, int N, int K, int lda, int ldb, int ldc,
                                     double alpha, double beta, int batchSize);
ampblasStatus gemm_TransAB_rMajor(Concurrency::accelerator_view &accl_view,
                                  Concurrency::array_view<float, 1> &A, long aOffset, long A_batchOffset,
                                  Concurrency::array_view<float, 1> &B, long bOffset, long B_batchOffset,
                                  Concurrency::array_view<float, 1> &C, long cOffset, long C_batchOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta, int batchSize);

ampblasStatus gemm_TransAB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                    Concurrency::array_view<double, 1> &A, long aOffset, long A_batchOffset,
                                    Concurrency::array_view<double, 1> &B, long bOffset, long B_batchOffset,
                                    Concurrency::array_view<double, 1> &C, long cOffset, long C_batchOffset,
                                    int M, int N, int K, int lda, int ldb, int ldc,
                                    double alpha, double beta, int batchSize);