#include "cppamp/ampblaslib.h"
#include <amp.h>
#define BLOCK_SIZE 256

using namespace concurrency;

static void gemv_TransA(Concurrency::array_view<double> &A_mat, int aOffset,
                        Concurrency::array_view<double> &X_vec, long xOffset,
                        Concurrency::array_view<double> &Y_vec, long yOffset,
                        double alpha, double beta, int lenX, int lenY,
                        Concurrency::array_view<double> &tempBuf)
{
  if((lenX - lenY) > 5000)
  {
    int len_X = (lenX + (BLOCK_SIZE - 1)) & ~(BLOCK_SIZE - 1);
    int num_blocks = len_X / BLOCK_SIZE;
    Concurrency::extent<1> grdExt(len_X);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext,[=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
      tile_static double t[BLOCK_SIZE];
      for (int Col = 0; Col < lenY; Col++)
      {
        int blockIdx = tidx.tile[0];
        int threadIdx = tidx.local[0];
        tempBuf[Col * num_blocks + blockIdx] = 0;
        t[threadIdx] = 0;

        if (Col < lenY && blockIdx * BLOCK_SIZE + threadIdx < lenX)
          t[threadIdx] = X_vec[xOffset + blockIdx * BLOCK_SIZE + threadIdx] * A_mat[aOffset + Col * lenX + blockIdx * BLOCK_SIZE + threadIdx];

        tidx.barrier.wait();

        for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
        {
          if(threadIdx < stride)
            t[threadIdx] += t[threadIdx + stride];
        }
        tempBuf[Col * num_blocks + blockIdx] = t[0];
        tidx.barrier.wait();
      }

      if (tidx.tile[0] == 0)
      {
        for(int Col = 0; Col < lenY; Col++)
        {
          tile_static double sh[BLOCK_SIZE];
          int threadId = tidx.local[0];
          sh[tidx.local[0]] = 0;

          for (int i = threadId; i < num_blocks; i += tidx.tile_dim0)
            sh[threadId] += tempBuf[Col * num_blocks + i];

          tidx.barrier.wait();

          for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
          {
            if(threadId < stride)
              sh[threadId] += sh[threadId + stride];
          }
          tidx.barrier.wait();
          Y_vec[yOffset + Col] *= beta;
          Y_vec[yOffset + Col] += alpha * sh[0];
        }
      }
    });
  }
  else
  {
    Concurrency::extent<1> grdExt(lenY * BLOCK_SIZE);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
      int threadIdx = tidx.local[0];
      int blockIdx = tidx.tile[0];
      int Col = blockIdx;

      tile_static double sh[BLOCK_SIZE];
      sh[threadIdx] = 0;

      for (int tileId = 0; tileId < ((lenX + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1)) / BLOCK_SIZE; tileId++)
      {
        if (tileId * BLOCK_SIZE + threadIdx < lenX && Col < lenY)
          sh[threadIdx] += X_vec[xOffset + tileId * BLOCK_SIZE + threadIdx] * A_mat[aOffset + Col * lenX + tileId * BLOCK_SIZE + threadIdx];
      }
      tidx.barrier.wait();

      for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
      {
        if (threadIdx < stride)
          sh[threadIdx] += sh[threadIdx + stride];
        tidx.barrier.wait();
      }

      if(threadIdx == 0 && Col < lenY)
      {
        Y_vec[yOffset + Col] *= beta;
        Y_vec[yOffset + Col] += alpha * sh[0];
      }
    });
  }
}

#define TILE_SZ_A 16

static void gemv_TransA_register(Concurrency::array_view<double> &A_mat, int aOffset,
                        Concurrency::array_view<double> &X_vec, long xOffset,
                        Concurrency::array_view<double> &Y_vec, long yOffset,
                        double alpha, double beta, int lenX, int lenY,
                        Concurrency::array_view<double> &tempBuf)
{

    Concurrency::extent<2> grdExt(((lenX - 1) / TILE_SZ_A + 1) * TILE_SZ_A, ((lenY - 1) / TILE_SZ_A + 1)*TILE_SZ_A);
    Concurrency::tiled_extent <TILE_SZ_A, TILE_SZ_A> t_ext(grdExt);
    Concurrency::parallel_for_each(t_ext,
                                   [=] (Concurrency::tiled_index<TILE_SZ_A,TILE_SZ_A>tidx)
                                   restrict(amp) {

    // Shared memory for tiling input B array
    tile_static double A_s [TILE_SZ_A][TILE_SZ_A];

    // Macros for accessing flattened matrices
    #define A1(row,col) A_mat[(row) + (col) * lenX]
    const unsigned int row = tidx.local[0];
    const unsigned int col = tidx.global[1];

    double y_reg[TILE_SZ_A] = {(double)0};

    for(unsigned int tileIdx = 0; tileIdx < (lenX - 1)/TILE_SZ_A + 1; ++tileIdx) {
        if (tileIdx*TILE_SZ_A + row < lenX && col < lenY) {
            A_s[tidx.local[0]][tidx.local[1]] = A1(tileIdx*TILE_SZ_A + row, col);
        }
        else {
            A_s[tidx.local[0]][tidx.local[1]] = 0;
        }
        tidx.barrier.wait();

        for (unsigned int idx = 0; idx < TILE_SZ_A; ++idx) {
            double x_reg;
            if(tileIdx*TILE_SZ_A + idx < lenX) {
                x_reg = X_vec[tileIdx*TILE_SZ_A + idx];
            }
            else {
                x_reg = 0;
            }

            for(unsigned int outIdx = 0; outIdx < TILE_SZ_A; ++outIdx) {
                y_reg[outIdx] += x_reg*A_s[idx][outIdx];
            }
        }
          tidx.barrier.wait();
    }
    for (unsigned int outIdx = 0; outIdx < TILE_SZ_A; ++outIdx) {
        if (col < lenY) {
           Y_vec[tidx.tile[1] * TILE_SZ_A + outIdx] *= beta;
           Y_vec[tidx.tile[1] * TILE_SZ_A + outIdx] += y_reg[outIdx] * alpha;
        }
    }
});

}

static void gemv_NoTransA_register(Concurrency::array_view<double> &A, long aOffset,
                          Concurrency::array_view<double> &X, long xOffset,
                          Concurrency::array_view<double> &Y, long yOffset,
                          double alpha, double beta, int lenX, int lenY)
{
    Concurrency::extent<2> grdExt( ((lenX - 1) / TILE_SZ_A + 1)*TILE_SZ_A, ((lenY - 1) / TILE_SZ_A + 1)*TILE_SZ_A);
    Concurrency::tiled_extent <TILE_SZ_A, TILE_SZ_A> t_ext(grdExt);
    Concurrency::parallel_for_each(t_ext,
                                   [=] (Concurrency::tiled_index<TILE_SZ_A,TILE_SZ_A> tidx)
                                   restrict(amp) {

    tile_static double A_s [TILE_SZ_A][TILE_SZ_A];

    #define A(row,col) A[(row)*lenY + (col)]

    const unsigned int row = tidx.local[0];
    const unsigned int col = tidx.global[1];

    double y_reg[TILE_SZ_A] = {(double)0};

    for(unsigned int tileIdx = 0; tileIdx < (lenX - 1)/TILE_SZ_A + 1; ++tileIdx) {
        if (tileIdx*TILE_SZ_A + row < lenX && col < lenY) {
            A_s[tidx.local[0]][tidx.local[1]] = A(tileIdx*TILE_SZ_A + row, col);
        }
        else {
            A_s[tidx.local[0]][tidx.local[1]] = 0;
        }
        tidx.barrier.wait();

        for (unsigned int idx = 0; idx < TILE_SZ_A; ++idx) {
            double x_reg;
            if(tileIdx*TILE_SZ_A + idx < lenX) {
                x_reg = X[tileIdx*TILE_SZ_A + idx];
            }
            else {
                x_reg = 0;
            }

            for(unsigned int outIdx = 0; outIdx < TILE_SZ_A; ++outIdx) {
                y_reg[outIdx] += x_reg*A_s[idx][outIdx];
            }
        }
          tidx.barrier.wait();
    }
    for (unsigned int outIdx = 0; outIdx < TILE_SZ_A; ++outIdx) {
        if (col < lenY) {
           Y[tidx.tile[1] * TILE_SZ_A + outIdx] *= beta;
           Y[tidx.tile[1] * TILE_SZ_A + outIdx] += y_reg[outIdx] * alpha;
        }
    }
});


}
                                                                                                                                                                          




static void gemv_NoTransA(Concurrency::array_view<double> &A, long aOffset,
                          Concurrency::array_view<double> &X, long xOffset,
                          Concurrency::array_view<double> &Y, long yOffset,
                          double alpha, double beta, int lenX, int lenY)
{
  long size = (lenY + 255) & ~255;
  Concurrency::extent<1> compute_domain(size);

  Concurrency::parallel_for_each(compute_domain.tile<BLOCK_SIZE>(),[=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
  {
    int bx = tidx.tile[0];
    int tx = tidx.local[0];
    tile_static double Xds[BLOCK_SIZE];
    int Col = bx * BLOCK_SIZE + tx;
    double Pvalue = 0;

    for (int m = 0; m < (lenX - 1) / BLOCK_SIZE + 1; ++m)
    {
      if (m * BLOCK_SIZE + tx < lenX)
        Xds[tx] = X[xOffset + m * BLOCK_SIZE + tx];
      else
        Xds[tx]=0;

      tidx.barrier.wait();

      for (int k = 0; k < BLOCK_SIZE; k++)
        if (Col < lenY && m * BLOCK_SIZE + k < lenX)
          Pvalue += Xds[k] * A[aOffset + Col + (m * BLOCK_SIZE + k) * lenY];

      tidx.barrier.wait();
    }
    if (Col < lenY)
    {
      Y[yOffset + Col] *= beta;
      Y[yOffset + Col] += alpha * Pvalue;
    }
    tidx.barrier.wait();
  });
}

void gemv_AMP(char TransA, int M, int N, double alpha,
              Concurrency::array_view<double> &A, long aOffset,
              Concurrency::array_view<double> &X, long xOffset, long incX, double beta,
              Concurrency::array_view<double> &Y, long yOffset, long incY,
              Concurrency::array_view<double> &temp_buf)
{
  if (alpha == 0.0)
    return;

  int lenX, lenY;
  if (M == 0 || N == 0)
    return;

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (TransA == 'n')
  {
    lenX = N;
    lenY = M;
  }
  else
  {
    lenX = M;
    lenY = N;
  }

  if (TransA == 't')
    gemv_TransA_register(A, aOffset, X, xOffset, Y, yOffset, alpha, beta, lenX, lenY, temp_buf);
  else if (TransA == 'n')
    gemv_NoTransA_register(A, aOffset, X, xOffset, Y, yOffset, alpha, beta, lenX, lenY);
}

ampblasStatus   ampblas_dgemv(const enum AMPBLAS_TRANS type,
                                              const int M, const int N,
                                              const double *alpha, double *A, const long aOffset,
                                              const int lda, double *X, const long xOffset,
                                              const int incX, const double *beta,
                                              double *Y,const long yOffset, const int incY)
{

    if(alpha == NULL || X == NULL || Y == NULL || A == NULL || M <= 0 || N <= 0 || beta == NULL )
        return AMPBLAS_INVALID;

    long lenXn = 1 + (N - 1) * abs(incX);
    long lenXt = 1 + (M - 1) * abs(incX);
    long lenYn = 1 + (M - 1) * abs(incY);
    long lenYt = 1 + (N - 1) * abs(incY);
    
    array_view<double> aMat(M * N, A);
    int num_blocks = lenXt / BLOCK_SIZE;
    double* temp = (double*)malloc(num_blocks * lenYt * sizeof(double));
    Concurrency::array_view<double> tempBuf(num_blocks * lenYt, temp);

    if( type == 'n')
    {
    Concurrency::array_view<double> xView(lenXn, X);
    Concurrency::array_view<double> yView(lenYn, Y);  
    gemv_AMP(type, M, N, *alpha, aMat, aOffset, xView, xOffset, incX, *beta, yView, yOffset, incY, tempBuf);
    aMat.synchronize();
    /* Print Output */
/*    for (int i = 0 ;i < M; i++) {
        cout << "[Y" << i << "] " << yView[i] << endl;
    }*/
    }
    
    
    if( type == 't')
    {
    Concurrency::array_view<double> xView(lenXt, X);
    Concurrency::array_view<double> yView(lenYt, Y);
    gemv_AMP(type, M, N, *alpha, aMat, aOffset, xView, xOffset, incX, *beta, yView, yOffset, incY, tempBuf);
    aMat.synchronize();
    /* Print Output */
   /* for (int i = 0 ;i < lenYt; i++) {
        cout << "[Y" << i << "] "<< yView[i] << endl;
    }*/
    }

    return AMPBLAS_SUCCESS;
}
