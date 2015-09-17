#include "cppamp/sgemm_kernels.h"

/*
* SGEMM - NoTransAB case - Row major Access
* STEP with Non Bank Conflict Implmentation
* TILESIZE = 8 STEPSIZE = 8
*/
ampblasStatus gemm_NoTransAB_rMajor_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{(float)0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
        int i = 0;
        do
        {
            tidx.barrier.wait();
            for(int sec = 0; sec < STEPSIZE / TILESIZE; sec++ ) {

                if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT +(sec * TILESIZE) < K)
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = B[bOffset + gidy * TILESIZE + idxT + ((idyT + (sec * TILESIZE)) *ldb) + i * (ldb << shiftFactor)];
                }
                else
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = 0;
                }

                if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lA[(sec * BANKNUMTILEELMTS ) + idyT + idxT * BANKTILESIZE] = 0;
                }
            }
            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;
            for (int iter = 0; iter < TILESIZE; ++iter)
            {
                MS1x1_NOBANK(1);
            }
            i++;
        } while (--block_k > 0);


        tidx.barrier.wait();
        if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
            C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransAB_rMajor_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { (double)0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
        int i = 0;
        do
        {
            tidx.barrier.wait();
            for (int sec = 0; sec < STEPSIZE / TILESIZE; sec++) {

                if (gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = B[bOffset + gidy * TILESIZE + idxT + ((idyT + (sec * TILESIZE)) *ldb) + i * (ldb << shiftFactor)];
                }
                else
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = 0;
                }

                if (gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = A[aOffset + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = 0;
                }
            }
            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;
            for (int iter = 0; iter < TILESIZE; ++iter)
            {
                MS1x1_NOBANK(1);
            }
            i++;
        } while (--block_k > 0);


        tidx.barrier.wait();
        if (gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
            C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
/*
* SGEMM - NoTransAB case - Row major Access
* STEP with Non Bank Conflict Implmentation
* TILESIZE = 16 STEPSIZE = 16
*/

ampblasStatus gemm_NoTransAB_rMajor_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{(float)0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
        int i = 0;
        do
        {
            tidx.barrier.wait();
            for(int sec = 0; sec < STEPSIZE / TILESIZE; sec++ ) {

                if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT +(sec * TILESIZE) < K)
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = B[bOffset + gidy * TILESIZE + idxT + ((idyT + (sec * TILESIZE)) *ldb) + i * (ldb << shiftFactor)];
                }
                else
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = 0;
                }

                if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lA[(sec * BANKNUMTILEELMTS ) + idyT + idxT * BANKTILESIZE] = 0;
                }
            }
            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;
            for (int iter = 0; iter < TILESIZE; ++iter)
            {
                MS1x1_NOBANK(1);
            }
            i++;
        } while (--block_k > 0);


        tidx.barrier.wait();
        if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
            C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransAB_rMajor_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { (double)0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
        int i = 0;
        do
        {
            tidx.barrier.wait();
            for (int sec = 0; sec < STEPSIZE / TILESIZE; sec++) {

                if (gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = B[bOffset + gidy * TILESIZE + idxT + ((idyT + (sec * TILESIZE)) *ldb) + i * (ldb << shiftFactor)];
                }
                else
                {
                    lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = 0;
                }

                if (gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = A[aOffset + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = 0;
                }
            }
            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;
            for (int iter = 0; iter < TILESIZE; ++iter)
            {
                MS1x1_NOBANK(1);
            }
            i++;
        } while (--block_k > 0);


        tidx.barrier.wait();
        if (gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
            C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}

/*
* SGEMM - NoTransAB case - Row major Access
* SUBMICROTILE Implmentation
* TILESIZE = 16 MICROTILESIZE = 2
*/
ampblasStatus gemm_NoTransAB_rMajor_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k = 0;
        do
        {
            tidx.barrier.wait();
            for(int sec = 0; sec < MICROTILESIZE; ++sec)
            {
                if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                {
                    lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * ldb + block_k * (ldb * TILESIZE)];
                }
                else
                {
                    lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                }

                if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                {
                    lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * lda +  idyT + block_k * TILESIZE];
                }
                else
                {
                    lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                }
            }
            tidx.barrier.wait();

            int offA = idx;
            int offB = idy;
            for (int iter=0; iter < TILESIZE; ++iter)
            {
                MTS;
            }
            tidx.barrier.wait();
        } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1))/TILESIZE));

        int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if((xIndex/ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransAB_rMajor_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
        double rA[1][MICROTILESIZE];
        double rB[1][MICROTILESIZE];
        tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k = 0;
        do
        {
            tidx.barrier.wait();
            for (int sec = 0; sec < MICROTILESIZE; ++sec)
            {
                if (gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                {
                    lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * ldb + block_k * (ldb * TILESIZE)];
                }
                else
                {
                    lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                }

                if (gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                {
                    lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * lda + idyT + block_k * TILESIZE];
                }
                else
                {
                    lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                }
            }
            tidx.barrier.wait();

            int offA = idx;
            int offB = idy;
            for (int iter = 0; iter < TILESIZE; ++iter)
            {
                MTS;
            }
            tidx.barrier.wait();
        } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

        int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
        for (int row = 0; row < MICROTILESIZE; row++)
        {
            for (int col = 0; col < MICROTILESIZE; col++)
            {
                if ((xIndex / ldc) + (TILESIZE * col) < M && (yIndex)+(TILESIZE * row) < N)
                    C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;

}

/*
* SGEMM - NoTransA case - Row major Access
* STEP Implmentation
* TILESIZE = 8 STEPSIZE = 8
*/

ampblasStatus gemm_NoTransA_rMajor_STEP_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{(float)0}};
        float rA[1][STEPSIZE / TILESIZE];
        float rB[1][STEPSIZE / TILESIZE];
        tile_static float lA[TILESIZE + TILESIZE * STEPSIZE];
        tile_static float lB[TILESIZE + TILESIZE * STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k =((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;

        int i = 0;
        do
        {
            tidx.barrier.wait();
            for(int sec = 0; sec < STEPSIZE / TILESIZE; ++sec)
            {
                if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lB[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = B[bOffset + (gidy * TILESIZE + idxT) * ldb + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lB[(sec * TILESIZE * TILESIZE ) + idyT + idxT * TILESIZE] = 0;
                }

                if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE ) < K)
                {
                    lA[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = A[aOffset + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lA[(sec * TILESIZE * TILESIZE ) + idyT + idxT * TILESIZE] = 0;
                }
            }

            tidx.barrier.wait();

            int offA = idx * TILESIZE;
            int offB = idy * TILESIZE;
            int offset = 1;

            for (int iter=0; iter < TILESIZE; ++iter)
            {
                MS1x1(offset, offset);
            }
            i++;
        } while (--block_k > 0);

        tidx.barrier.wait();
        if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
            C[cOffset + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
    });

#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransA_rMajor_STEP_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { (double)0 } };
        double rA[1][STEPSIZE / TILESIZE];
        double rB[1][STEPSIZE / TILESIZE];
        tile_static double lA[TILESIZE + TILESIZE * STEPSIZE];
        tile_static double lB[TILESIZE + TILESIZE * STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = TILESIZE * idy + idx;
        int idxT = idt % TILESIZE;
        int idyT = idt / TILESIZE;
        int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;

        int i = 0;
        do
        {
            tidx.barrier.wait();
            for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec)
            {
                if (gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lB[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = B[bOffset + (gidy * TILESIZE + idxT) * ldb + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lB[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = 0;
                }

                if (gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                {
                    lA[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = A[aOffset + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
                }
                else
                {
                    lA[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = 0;
                }
            }

            tidx.barrier.wait();

            int offA = idx * TILESIZE;
            int offB = idy * TILESIZE;
            int offset = 1;

            for (int iter = 0; iter < TILESIZE; ++iter)
            {
                MS1x1(offset, offset);
            }
            i++;
        } while (--block_k > 0);

        tidx.barrier.wait();
        if (gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
            C[cOffset + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
    });

#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}

/*
* SGEMM - NoTransA case - Row major Access
* STEP with Non Bank Conflict Implmentation
* TILESIZE = 8 STEPSIZE = 8
*/

ampblasStatus gemm_NoTransA_rMajor_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
        int shiftfactor = (int)Concurrency::fast_math::log2(STEPSIZE);
        int block_k =((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
        float rC[1][1] = {{0.0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = (idy << tilemulshift) + idx; //(idy * TILESIZE + idx)
        int ids = (idy << shiftfactor) + idx; //(idy * STEPSIZE + idx)
        int idxS = ids & (STEPSIZE - 1);


        int idyT = (idt)>> tilemulshift;
        int gidyOffset = gidy << tilemulshift;
        int gidxOffset = gidx << tilemulshift;
        int idyTOffset = idyT * BANKTILESIZE;


        int i = 0;
        do
        {
            tidx.barrier.wait();
            int iOffset = i << shiftfactor;
            for(int sec = 0; sec < STEPTILERATIO; ++sec)
            {
                int secOffset  = sec << tilemulshift;
                int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                int localIdx = secStartPt + idxS + idyTOffset;
                int kIndex = iOffset + idxS + secOffset;

                // Initialize the local memory with zero
                lB[localIdx] = 0;
                lA[localIdx] = 0;

                if(gidyOffset + idyT < N && kIndex < K)
                {
                    lB[localIdx] = B[bOffset + (gidyOffset + idyT) * ldb + kIndex];
                }
                if(gidxOffset + idyT < M && kIndex < K)
                {
                    lA[localIdx] = A[aOffset + (gidxOffset + idyT) * lda + kIndex];
                }
            }

            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;

            for (int piter=0; piter < TILESIZE; ++piter)
            {
                MS1x1_NOBANK(1);
            }

            i++;

        } while (--block_k > 0);

        tidx.barrier.wait();

        int crow = (gidxOffset + idx) * ldc;
        int ccolprod = (gidyOffset + idy);
        if(crow/ldc < M && ccolprod < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransA_rMajor_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
        int shiftfactor = (int)Concurrency::fast_math::log2(STEPSIZE);
        int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
        double rC[1][1] = { { 0.0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = (idy << tilemulshift) + idx; //(idy * TILESIZE + idx)
        int ids = (idy << shiftfactor) + idx; //(idy * STEPSIZE + idx)
        int idxS = ids & (STEPSIZE - 1);


        int idyT = (idt) >> tilemulshift;
        int gidyOffset = gidy << tilemulshift;
        int gidxOffset = gidx << tilemulshift;
        int idyTOffset = idyT * BANKTILESIZE;


        int i = 0;
        do
        {
            tidx.barrier.wait();
            int iOffset = i << shiftfactor;
            for (int sec = 0; sec < STEPTILERATIO; ++sec)
            {
                int secOffset = sec << tilemulshift;
                int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                int localIdx = secStartPt + idxS + idyTOffset;
                int kIndex = iOffset + idxS + secOffset;

                // Initialize the local memory with zero
                lB[localIdx] = 0;
                lA[localIdx] = 0;

                if (gidyOffset + idyT < N && kIndex < K)
                {
                    lB[localIdx] = B[bOffset + (gidyOffset + idyT) * ldb + kIndex];
                }
                if (gidxOffset + idyT < M && kIndex < K)
                {
                    lA[localIdx] = A[aOffset + (gidxOffset + idyT) * lda + kIndex];
                }
            }

            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;

            for (int piter = 0; piter < TILESIZE; ++piter)
            {
                MS1x1_NOBANK(1);
            }

            i++;

        } while (--block_k > 0);

        tidx.barrier.wait();

        int crow = (gidxOffset + idx) * ldc;
        int ccolprod = (gidyOffset + idy);
        if (crow / ldc < M && ccolprod < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
/*
* SGEMM - NoTransA case - Row major Access
* STEP with Non Bank Conflict Implmentation
* TILESIZE = 16 STEPSIZE = 16
*/

ampblasStatus gemm_NoTransA_rMajor_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
        int shiftfactor = (int)Concurrency::fast_math::log2(STEPSIZE);
        int block_k =((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
        float rC[1][1] = {{0.0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = (idy << tilemulshift) + idx; //(idy * TILESIZE + idx)
        int ids = (idy << shiftfactor) + idx; //(idy * STEPSIZE + idx)
        int idxS = ids & (STEPSIZE - 1);


        int idyT = (idt)>> tilemulshift;
        int gidyOffset = gidy << tilemulshift;
        int gidxOffset = gidx << tilemulshift;
        int idyTOffset = idyT * BANKTILESIZE;


        int i = 0;
        do
        {
            tidx.barrier.wait();
            int iOffset = i << shiftfactor;
            for(int sec = 0; sec < STEPTILERATIO; ++sec)
            {
                int secOffset  = sec << tilemulshift;
                int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                int localIdx = secStartPt + idxS + idyTOffset;
                int kIndex = iOffset + idxS + secOffset;

                // Initialize the local memory with zero
                lB[localIdx] = 0;
                lA[localIdx] = 0;

                if(gidyOffset + idyT < N && kIndex < K)
                {
                    lB[localIdx] = B[bOffset + (gidyOffset + idyT) * ldb + kIndex];
                }
                if(gidxOffset + idyT < M && kIndex < K)
                {
                    lA[localIdx] = A[aOffset + (gidxOffset + idyT) * lda + kIndex];
                }
            }

            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;

            for (int piter=0; piter < TILESIZE; ++piter)
            {
                MS1x1_NOBANK(1);
            }

            i++;

        } while (--block_k > 0);

        tidx.barrier.wait();

        int crow = (gidxOffset + idx) * ldc;
        int ccolprod = (gidyOffset + idy);
        if(crow/ldc < M && ccolprod < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransA_rMajor_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
        int shiftfactor = (int)Concurrency::fast_math::log2(STEPSIZE);
        int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
        double rC[1][1] = { { 0.0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = (idy << tilemulshift) + idx; //(idy * TILESIZE + idx)
        int ids = (idy << shiftfactor) + idx; //(idy * STEPSIZE + idx)
        int idxS = ids & (STEPSIZE - 1);


        int idyT = (idt) >> tilemulshift;
        int gidyOffset = gidy << tilemulshift;
        int gidxOffset = gidx << tilemulshift;
        int idyTOffset = idyT * BANKTILESIZE;


        int i = 0;
        do
        {
            tidx.barrier.wait();
            int iOffset = i << shiftfactor;
            for (int sec = 0; sec < STEPTILERATIO; ++sec)
            {
                int secOffset = sec << tilemulshift;
                int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                int localIdx = secStartPt + idxS + idyTOffset;
                int kIndex = iOffset + idxS + secOffset;

                // Initialize the local memory with zero
                lB[localIdx] = 0;
                lA[localIdx] = 0;

                if (gidyOffset + idyT < N && kIndex < K)
                {
                    lB[localIdx] = B[bOffset + (gidyOffset + idyT) * ldb + kIndex];
                }
                if (gidxOffset + idyT < M && kIndex < K)
                {
                    lA[localIdx] = A[aOffset + (gidxOffset + idyT) * lda + kIndex];
                }
            }

            tidx.barrier.wait();

            int offA = idx * BANKTILESIZE;
            int offB = idy * BANKTILESIZE;

            for (int piter = 0; piter < TILESIZE; ++piter)
            {
                MS1x1_NOBANK(1);
            }

            i++;

        } while (--block_k > 0);

        tidx.barrier.wait();

        int crow = (gidxOffset + idx) * ldc;
        int ccolprod = (gidyOffset + idy);
        if (crow / ldc < M && ccolprod < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
/*
* SGEMM - NoTransA case - Row major Access
* SUBMICROTILE with Non Bank Conflict Implmentation
* TILESIZE = 16 MICROTILESIZE = 8
*/

ampblasStatus gemm_NoTransA_rMajor_MICRO_NBK_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2

    Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftTS = Concurrency::fast_math::log2(TILESIZE);
        float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TOTMICROTILEPROD + TILESIZE];
        tile_static float lB[TOTMICROTILEPROD + TILESIZE];
        int gidx = tidx.tile[0];
        int gidy = tidx.tile[1];
        int idx = tidx.local[0];
        int idy = tidx.local[1];
        int idt = ( idy << shiftTS ) + idx;
        int idxT = idt % TILESIZE ;
        int idyT = idt / TILESIZE;
        int block_k = 0;
        do
        {
            int colIndex =( block_k << shiftTS )+ idyT;
            int lIndex = (idyT * BANKMICROTILESIZE) + idxT;

            tidx.barrier.wait();
            for(int sec = 0; sec < MICROTILESIZE; ++sec)
            {
                int secVal = sec << shiftTS;
                int BrowIndex = ( gidy * MICROTILEPROD) + idxT + secVal;
                int ArowIndex = ( gidx * MICROTILEPROD) + idxT + secVal;

                tidx.barrier.wait();
                if( BrowIndex < N && colIndex < K)
                {
                    lB[ lIndex + secVal] = B[ bOffset + BrowIndex * ldb + colIndex ];
                }
                else
                {
                    lB[ lIndex + secVal] = 0;
                }

                if( ArowIndex < M && colIndex < K)
                {
                    lA[ lIndex + secVal] = A[aOffset + ArowIndex * lda +  colIndex];
                }
                else
                {
                    lA[ lIndex + secVal] = 0;
                }
            }
            tidx.barrier.wait();

            int offA = idx;
            int offB = idy;
            for (int iter=0; iter < TILESIZE; ++iter)
            {
                MTS_NOBANK;
            }
            tidx.barrier.wait();
        } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

        int xIndex = ((gidx * MICROTILEPROD) + idx) * ldc;
        int yIndex = ((gidy * MICROTILEPROD) + idy);
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if((xIndex/ldc) + (col << shiftTS) < M && (yIndex) + (row << shiftTS) < N)
                    C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS )] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS)];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
	return AMPBLAS_SUCCESS;

}
    ampblasStatus gemm_NoTransA_rMajor_MICRO_NBK_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 16
#define MICROTILESIZE 2

        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftTS = Concurrency::fast_math::log2(TILESIZE);
            double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
            double rA[1][MICROTILESIZE];
            double rB[1][MICROTILESIZE];
            tile_static double lA[TOTMICROTILEPROD + TILESIZE];
            tile_static double lB[TOTMICROTILEPROD + TILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = (idy << shiftTS) + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = 0;
            do
            {
                int colIndex = (block_k << shiftTS) + idyT;
                int lIndex = (idyT * BANKMICROTILESIZE) + idxT;

                tidx.barrier.wait();
                for (int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    int secVal = sec << shiftTS;
                    int BrowIndex = (gidy * MICROTILEPROD) + idxT + secVal;
                    int ArowIndex = (gidx * MICROTILEPROD) + idxT + secVal;

                    tidx.barrier.wait();
                    if (BrowIndex < N && colIndex < K)
                    {
                        lB[lIndex + secVal] = B[bOffset + BrowIndex * ldb + colIndex];
                    }
                    else
                    {
                        lB[lIndex + secVal] = 0;
                    }

                    if (ArowIndex < M && colIndex < K)
                    {
                        lA[lIndex + secVal] = A[aOffset + ArowIndex * lda + colIndex];
                    }
                    else
                    {
                        lA[lIndex + secVal] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MTS_NOBANK;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

            int xIndex = ((gidx * MICROTILEPROD) + idx) * ldc;
            int yIndex = ((gidy * MICROTILEPROD) + idy);
            for (int row = 0; row < MICROTILESIZE; row++)
            {
                for (int col = 0; col < MICROTILESIZE; col++)
                {
                    if ((xIndex / ldc) + (col << shiftTS) < M && (yIndex)+(row << shiftTS) < N)
                        C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;

    }

    /*
    * SGEMM - NoTransA case - Row major Access
    * SUBMICROTILE Implmentation
    * TILESIZE = 16 MICROTILESIZE = 2
    */

    ampblasStatus gemm_NoTransA_rMajor_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {

#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N / 2 + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
            float rA[1][MICROTILESIZE];
            float rB[1][MICROTILESIZE];
            tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
            tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = 0;
            do
            {
                tidx.barrier.wait();
                for(int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * ldb + idyT + block_k * TILESIZE];
                    }
                    else
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }

                    if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * lda +  idyT + block_k * TILESIZE];
                    }
                    else
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MTS;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1))/TILESIZE));

            int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
            int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
            for( int row = 0; row < MICROTILESIZE; row++)
            {
                for( int col = 0; col < MICROTILESIZE ; col++)
                {
                    if((xIndex / ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N)
                        C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) *  N) + yIndex + (TILESIZE * row)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;
    }
    ampblasStatus gemm_NoTransA_rMajor_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {

#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N / 2 + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            double rC[MICROTILESIZE][MICROTILESIZE] = { { (float)0 } };
            double rA[1][MICROTILESIZE];
            double rB[1][MICROTILESIZE];
            tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
            tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = 0;
            do
            {
                tidx.barrier.wait();
                for (int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    if (gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * ldb + idyT + block_k * TILESIZE];
                    }
                    else
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }

                    if (gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * lda + idyT + block_k * TILESIZE];
                    }
                    else
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MTS;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

            int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
            int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
            for (int row = 0; row < MICROTILESIZE; row++)
            {
                for (int col = 0; col < MICROTILESIZE; col++)
                {
                    if ((xIndex / ldc) + (TILESIZE * col) < M && (yIndex)+(TILESIZE * row) < N)
                        C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) *  N) + yIndex + (TILESIZE * row)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;
    }
    /*
    * SGEMM - NoTransB case - Row major Access
    * STEP with Non Bank Conflict Implmentation
    * TILESIZE = 16 STEPSIZE = 16
    */

    ampblasStatus gemm_NoTransB_rMajor_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 16
#define STEPSIZE 16
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
            int shiftfactor = Concurrency::fast_math::log2(STEPSIZE);
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
            float rC[1][1] = {{0.0}};
            float rA[1][STEPTILERATIO];
            float rB[1][STEPTILERATIO];
            tile_static float lA[STEPTILEPROD + STEPSIZE];
            tile_static float lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = (idy << tilemulshift) + idx;
            int idyT = idt >> tilemulshift;
            int idxT = idt & (TILESIZE - 1);
            int gidyOffset = gidy << tilemulshift;
            int gidxOffset = gidx << tilemulshift;
            int idyTOffset = idyT * BANKTILESIZE;

            int i = 0;
            do
            {
                tidx.barrier.wait();
                int iOffset = i << shiftfactor;
                for(int sec = 0; sec < STEPTILERATIO; ++sec)
                {
                    int secOffset  = sec << tilemulshift;
                    int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                    int localIdx = secStartPt + idxT + idyTOffset;
                    int kIndex = iOffset + idyT + secOffset;

                    // Initialize the local memory with zero
                    lB[localIdx] = 0;
                    lA[localIdx] = 0;

                    if(gidyOffset + idxT < N && kIndex < K)
                    {
                        lB[localIdx] = B[bOffset + gidyOffset + idxT + kIndex * ldb];
                    }

                    if(gidxOffset + idxT < M && kIndex < K)
                    {
                        lA[localIdx] = A[aOffset + gidxOffset + idxT + kIndex * lda];
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;

                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(BANKTILESIZE);
                }

                i++;
            } while (--block_k > 0);
            tidx.barrier.wait();
            int crow = (gidxOffset + idx)*ldc;
            int ccolprod = (gidyOffset + idy);
            if(crow/ldc < M && ccolprod < N)
                C[cOffset + crow + ccolprod] =  alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;
    }
    ampblasStatus gemm_NoTransB_rMajor_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 16
#define STEPSIZE 16
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
            int shiftfactor = Concurrency::fast_math::log2(STEPSIZE);
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
            double rC[1][1] = { { 0.0 } };
            double rA[1][STEPTILERATIO];
            double rB[1][STEPTILERATIO];
            tile_static double lA[STEPTILEPROD + STEPSIZE];
            tile_static double lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = (idy << tilemulshift) + idx;
            int idyT = idt >> tilemulshift;
            int idxT = idt & (TILESIZE - 1);
            int gidyOffset = gidy << tilemulshift;
            int gidxOffset = gidx << tilemulshift;
            int idyTOffset = idyT * BANKTILESIZE;

            int i = 0;
            do
            {
                tidx.barrier.wait();
                int iOffset = i << shiftfactor;
                for (int sec = 0; sec < STEPTILERATIO; ++sec)
                {
                    int secOffset = sec << tilemulshift;
                    int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                    int localIdx = secStartPt + idxT + idyTOffset;
                    int kIndex = iOffset + idyT + secOffset;

                    // Initialize the local memory with zero
                    lB[localIdx] = 0;
                    lA[localIdx] = 0;

                    if (gidyOffset + idxT < N && kIndex < K)
                    {
                        lB[localIdx] = B[bOffset + gidyOffset + idxT + kIndex * ldb];
                    }

                    if (gidxOffset + idxT < M && kIndex < K)
                    {
                        lA[localIdx] = A[aOffset + gidxOffset + idxT + kIndex * lda];
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;

                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(BANKTILESIZE);
                }

                i++;
            } while (--block_k > 0);
            tidx.barrier.wait();
            int crow = (gidxOffset + idx)*ldc;
            int ccolprod = (gidyOffset + idy);
            if (crow / ldc < M && ccolprod < N)
                C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;
    }

    /*
    * SGEMM - NoTransB case - Row major Access
    * SUBMICROTILE with Non Bank Conflict Implmentation
    * TILESIZE = 16 MICROTILESIZE = 2
    */

    ampblasStatus gemm_NoTransB_rMajor_MICRO_NBK_TS16XMTS2(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftTS = Concurrency::fast_math::log2(TILESIZE);
            float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
            float rA[1][MICROTILESIZE];
            float rB[1][MICROTILESIZE];
            tile_static float lA[TOTMICROTILEPROD + TILESIZE];
            tile_static float lB[TOTMICROTILEPROD + TILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = ( idy << shiftTS) + idx;
            int idxT = idt & ( TILESIZE - 1);
            int idyT = idt >> shiftTS;
            int block_k = 0;
            do
            {
                int colIndex = ( block_k << shiftTS ) + idyT;
                int lIndex = (idyT * BANKMICROTILESIZE) + idxT;

                tidx.barrier.wait();
                for(int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    int secVal = sec << shiftTS;
                    int BrowIndex = (gidy * MICROTILEPROD) + idxT + secVal;
                    int ArowIndex = (gidx * MICROTILEPROD) + idxT + secVal;

                    if( BrowIndex < N && colIndex < K)
                    {
                        lB[ lIndex + secVal] = B[bOffset + BrowIndex + colIndex * ldb];
                    }
                    else
                    {
                        lB[lIndex + secVal] = 0;
                    }

                    if(ArowIndex < M && colIndex < K)
                    {
                        lA[lIndex + secVal] = A[aOffset + ArowIndex + colIndex * lda];
                    }
                    else
                    {
                        lA[lIndex + secVal] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MTS_NOBANK;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

            int xIndex = ((gidx * MICROTILEPROD) + idx) * ldc;
            int yIndex = ((gidy * MICROTILEPROD) + idy);
            for( int row = 0; row < MICROTILESIZE; row++)
            {
                for( int col = 0; col < MICROTILESIZE ; col++)
                {
                    if((xIndex / ldc) + (col << shiftTS) < M && (yIndex) + (row << shiftTS) < N)
                        C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;

    }
    ampblasStatus gemm_NoTransB_rMajor_MICRO_NBK_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftTS = Concurrency::fast_math::log2(TILESIZE);
            double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
            double rA[1][MICROTILESIZE];
            double rB[1][MICROTILESIZE];
            tile_static double lA[TOTMICROTILEPROD + TILESIZE];
            tile_static double lB[TOTMICROTILEPROD + TILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = (idy << shiftTS) + idx;
            int idxT = idt & (TILESIZE - 1);
            int idyT = idt >> shiftTS;
            int block_k = 0;
            do
            {
                int colIndex = (block_k << shiftTS) + idyT;
                int lIndex = (idyT * BANKMICROTILESIZE) + idxT;

                tidx.barrier.wait();
                for (int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    int secVal = sec << shiftTS;
                    int BrowIndex = (gidy * MICROTILEPROD) + idxT + secVal;
                    int ArowIndex = (gidx * MICROTILEPROD) + idxT + secVal;

                    if (BrowIndex < N && colIndex < K)
                    {
                        lB[lIndex + secVal] = B[bOffset + BrowIndex + colIndex * ldb];
                    }
                    else
                    {
                        lB[lIndex + secVal] = 0;
                    }

                    if (ArowIndex < M && colIndex < K)
                    {
                        lA[lIndex + secVal] = A[aOffset + ArowIndex + colIndex * lda];
                    }
                    else
                    {
                        lA[lIndex + secVal] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MTS_NOBANK;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

            int xIndex = ((gidx * MICROTILEPROD) + idx) * ldc;
            int yIndex = ((gidy * MICROTILEPROD) + idy);
            for (int row = 0; row < MICROTILESIZE; row++)
            {
                for (int col = 0; col < MICROTILESIZE; col++)
                {
                    if ((xIndex / ldc) + (col << shiftTS) < M && (yIndex)+(row << shiftTS) < N)
                        C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;

    }
    /*
    * SGEMM - NoTransB case - Row major Access
    * STEP Implmentation
    * TILESIZE = 16 STEPSIZE = 16
    */

    ampblasStatus gemm_NoTransB_rMajor_STEP_TS16XSS16(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 16
#define STEPSIZE 16
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            float rC[1][1] = {{(float)0}};
            float rA[1][STEPSIZE / TILESIZE];
            float rB[1][STEPSIZE / TILESIZE];
            tile_static float lA[TILESIZE + TILESIZE * STEPSIZE];
            tile_static float lB[TILESIZE + TILESIZE * STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();
                for(int sec = 0; sec < STEPSIZE / TILESIZE; ++sec)
                {
                    if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                    {
                        lB[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = B[bOffset + gidy * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * ldb + i * (ldb << shiftFactor)];
                    }
                    else
                    {
                        lB[(idyT + (sec * TILESIZE )) * TILESIZE + idxT] = 0;
                    }

                    if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                    {
                        lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = A[aOffset + gidx * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * lda + i * (lda << shiftFactor)];
                    }
                    else
                    {
                        lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                int offset = TILESIZE;

                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MS1x1(offset, offset);
                }

                i++;
            } while (--block_k > 0);
            tidx.barrier.wait();
            if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
                C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)] =  alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;
    }
    ampblasStatus gemm_NoTransB_rMajor_STEP_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 16
#define STEPSIZE 16
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            double rC[1][1] = { { (double)0 } };
            double rA[1][STEPSIZE / TILESIZE];
            double rB[1][STEPSIZE / TILESIZE];
            tile_static double lA[TILESIZE + TILESIZE * STEPSIZE];
            tile_static double lB[TILESIZE + TILESIZE * STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();
                for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec)
                {
                    if (gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                    {
                        lB[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = B[bOffset + gidy * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * ldb + i * (ldb << shiftFactor)];
                    }
                    else
                    {
                        lB[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = 0;
                    }

                    if (gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K)
                    {
                        lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = A[aOffset + gidx * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * lda + i * (lda << shiftFactor)];
                    }
                    else
                    {
                        lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                int offset = TILESIZE;

                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MS1x1(offset, offset);
                }

                i++;
            } while (--block_k > 0);
            tidx.barrier.wait();
            if (gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N)
                C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx)*ldc + (gidy * TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;
    }
    /*
    * SGEMM - NoTransB case - Row major Access
    * SUBMICORTILE Implmentation
    * TILESIZE = 16 STEPSIZE = 2
    */

    ampblasStatus gemm_NoTransB_rMajor_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
            float rA[1][MICROTILESIZE];
            float rB[1][MICROTILESIZE];
            tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
            tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = 0;
            do
            {
                tidx.barrier.wait();
                for(int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * ldb + block_k * (ldb * TILESIZE)];
                    }
                    else
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }

                    if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) +  idyT * lda + block_k * (lda * TILESIZE)];
                    }
                    else
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MTS;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1))/TILESIZE));

            int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
            int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
            for( int row = 0; row < MICROTILESIZE; row++)
            {
                for( int col = 0; col < MICROTILESIZE ; col++)
                {
                    if((xIndex/ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N)
                        C[cOffset + (xIndex + (TILESIZE * col)*N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) * N ) + yIndex + (TILESIZE * row)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;
    }
    ampblasStatus gemm_NoTransB_rMajor_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
            double rA[1][MICROTILESIZE];
            double rB[1][MICROTILESIZE];
            tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
            tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = 0;
            do
            {
                tidx.barrier.wait();
                for (int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    if (gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * ldb + block_k * (ldb * TILESIZE)];
                    }
                    else
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }

                    if (gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * lda + block_k * (lda * TILESIZE)];
                    }
                    else
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MTS;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

            int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
            int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
            for (int row = 0; row < MICROTILESIZE; row++)
            {
                for (int col = 0; col < MICROTILESIZE; col++)
                {
                    if ((xIndex / ldc) + (TILESIZE * col) < M && (yIndex)+(TILESIZE * row) < N)
                        C[cOffset + (xIndex + (TILESIZE * col)*N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;
    }
    /*
    * SGEMM - NoTransB case - Row major Access
    * STEP with Non Bank Concflict Implmentation
    * TILESIZE = 8 STEPSIZE = 8
    */

    ampblasStatus gemm_NoTransB_rMajor_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 8
#define STEPSIZE 8
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
            int shiftfactor = Concurrency::fast_math::log2(STEPSIZE);
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
            float rC[1][1] = {{0.0}};
            float rA[1][STEPTILERATIO];
            float rB[1][STEPTILERATIO];
            tile_static float lA[STEPTILEPROD + STEPSIZE];
            tile_static float lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = (idy << tilemulshift) + idx;
            int idyT = idt >> tilemulshift;
            int idxT = idt & (TILESIZE - 1);
            int gidyOffset = gidy << tilemulshift;
            int gidxOffset = gidx << tilemulshift;
            int idyTOffset = idyT * BANKTILESIZE;

            int i = 0;
            do
            {
                tidx.barrier.wait();
                int iOffset = i << shiftfactor;
                for(int sec = 0; sec < STEPTILERATIO; ++sec)
                {
                    int secOffset  = sec << tilemulshift;
                    int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                    int localIdx = secStartPt + idxT + idyTOffset;
                    int kIndex = iOffset + idyT + secOffset;

                    // Initialize the local memory with zero
                    lB[localIdx] = 0;
                    lA[localIdx] = 0;

                    if(gidyOffset + idxT < N && kIndex < K)
                    {
                        lB[localIdx] = B[bOffset + gidyOffset + idxT + kIndex * ldb];
                    }

                    if(gidxOffset + idxT < M && kIndex < K)
                    {
                        lA[localIdx] = A[aOffset + gidxOffset + idxT + kIndex * lda];
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;

                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(BANKTILESIZE);
                }

                i++;
            } while (--block_k > 0);
            tidx.barrier.wait();
            int crow = (gidxOffset + idx)*ldc;
            int ccolprod = (gidyOffset + idy);
            if(crow/ldc < M && ccolprod < N)
                C[cOffset + crow + ccolprod] =  alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;
    }
    ampblasStatus gemm_NoTransB_rMajor_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 8
#define STEPSIZE 8
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int tilemulshift = (int)Concurrency::fast_math::log2(TILESIZE);
            int shiftfactor = Concurrency::fast_math::log2(STEPSIZE);
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
            double rC[1][1] = { { 0.0 } };
            double rA[1][STEPTILERATIO];
            double rB[1][STEPTILERATIO];
            tile_static double lA[STEPTILEPROD + STEPSIZE];
            tile_static double lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = (idy << tilemulshift) + idx;
            int idyT = idt >> tilemulshift;
            int idxT = idt & (TILESIZE - 1);
            int gidyOffset = gidy << tilemulshift;
            int gidxOffset = gidx << tilemulshift;
            int idyTOffset = idyT * BANKTILESIZE;

            int i = 0;
            do
            {
                tidx.barrier.wait();
                int iOffset = i << shiftfactor;
                for (int sec = 0; sec < STEPTILERATIO; ++sec)
                {
                    int secOffset = sec << tilemulshift;
                    int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
                    int localIdx = secStartPt + idxT + idyTOffset;
                    int kIndex = iOffset + idyT + secOffset;

                    // Initialize the local memory with zero
                    lB[localIdx] = 0;
                    lA[localIdx] = 0;

                    if (gidyOffset + idxT < N && kIndex < K)
                    {
                        lB[localIdx] = B[bOffset + gidyOffset + idxT + kIndex * ldb];
                    }

                    if (gidxOffset + idxT < M && kIndex < K)
                    {
                        lA[localIdx] = A[aOffset + gidxOffset + idxT + kIndex * lda];
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;

                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(BANKTILESIZE);
                }

                i++;
            } while (--block_k > 0);
            tidx.barrier.wait();
            int crow = (gidxOffset + idx)*ldc;
            int ccolprod = (gidyOffset + idy);
            if (crow / ldc < M && ccolprod < N)
                C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;
    }
    /*
    * SGEMM - TransAB case - Row major Access
    * STEP with Non Bank Conflict Implmentation
    * TILESIZE = 8 STEPSIZE = 8
    */

    ampblasStatus gemm_TransAB_rMajor_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 8
#define STEPSIZE 8
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            float rC[1][1] = {{0.0}};
            float rA[1][STEPTILERATIO];
            float rB[1][STEPTILERATIO];
            tile_static float lA[STEPTILEPROD + STEPSIZE];//8*8+8
            tile_static float lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();

                // Load Sections of A and B into respective shared memory slots
                for (int sec =0; sec < STEPSIZE/TILESIZE; ++sec)
                {
                    // Load Section 'sec' from global memory B onto shared lB
                    if(gidy*TILESIZE+idyT  < N && (idxT + i * STEPSIZE + (TILESIZE * sec)) < K)
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = B[bOffset + (gidy*TILESIZE+ idyT) * ldb + idxT + i * STEPSIZE + (TILESIZE * sec)];
                    else
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = 0;

                    // Load Section 'sec' from global memory A onto shared lA
                    if(gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K)
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = A[aOffset  + gidx*TILESIZE+ idxT + idyT*lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
                    else
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = 0;
                }
                tidx.barrier.wait();

                int offA = idx * BANKTILESIZE;
                int offB = idy * BANKTILESIZE;

                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(1);
                }

                i++;
            } while (--block_k > 0);


            tidx.barrier.wait();
            if(gidx*TILESIZE+idx < M && gidy*TILESIZE+idy < N)
                C[cOffset + (gidx*TILESIZE +idx) * ldc + (gidy*TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx*TILESIZE+idx) * ldc + (gidy*TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;

    }
    ampblasStatus gemm_TransAB_rMajor_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 8
#define STEPSIZE 8
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            double rC[1][1] = { { 0.0 } };
            double rA[1][STEPTILERATIO];
            double rB[1][STEPTILERATIO];
            tile_static double lA[STEPTILEPROD + STEPSIZE];//8*8+8
            tile_static double lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();

                // Load Sections of A and B into respective shared memory slots
                for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec)
                {
                    // Load Section 'sec' from global memory B onto shared lB
                    if (gidy*TILESIZE + idyT  < N && (idxT + i * STEPSIZE + (TILESIZE * sec)) < K)
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = B[bOffset + (gidy*TILESIZE + idyT) * ldb + idxT + i * STEPSIZE + (TILESIZE * sec)];
                    else
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = 0;

                    // Load Section 'sec' from global memory A onto shared lA
                    if (gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K)
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = A[aOffset + gidx*TILESIZE + idxT + idyT*lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
                    else
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = 0;
                }
                tidx.barrier.wait();

                int offA = idx * BANKTILESIZE;
                int offB = idy * BANKTILESIZE;

                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(1);
                }

                i++;
            } while (--block_k > 0);


            tidx.barrier.wait();
            if (gidx*TILESIZE + idx < M && gidy*TILESIZE + idy < N)
                C[cOffset + (gidx*TILESIZE + idx) * ldc + (gidy*TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx*TILESIZE + idx) * ldc + (gidy*TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;

    }

    /*
    * SGEMM - TransAB case - Row major Access
    * STEP with Non Bank Conflict Implmentation
    * TILESIZE = 16 STEPSIZE = 16
    */

    ampblasStatus gemm_TransAB_rMajor_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 16
#define STEPSIZE 16
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            float rC[1][1] = {{0.0}};
            float rA[1][STEPTILERATIO];
            float rB[1][STEPTILERATIO];
            tile_static float lA[STEPTILEPROD + STEPSIZE];//8*8+8
            tile_static float lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();

                // Load Sections of A and B into respective shared memory slots
                for (int sec =0; sec < STEPSIZE/TILESIZE; ++sec)
                {
                    // Load Section 'sec' from global memory B onto shared lB
                    if(gidy*TILESIZE+idyT  < N && (idxT + i * STEPSIZE + (TILESIZE * sec)) < K)
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = B[bOffset + (gidy*TILESIZE+ idyT) * ldb + idxT + i * STEPSIZE + (TILESIZE * sec)];
                    else
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = 0;

                    // Load Section 'sec' from global memory A onto shared lA
                    if(gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K)
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = A[aOffset  + gidx*TILESIZE+ idxT + idyT*lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
                    else
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = 0;
                }
                tidx.barrier.wait();

                int offA = idx * BANKTILESIZE;
                int offB = idy * BANKTILESIZE;

                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(1);
                }

                i++;
            } while (--block_k > 0);


            tidx.barrier.wait();
            if(gidx*TILESIZE+idx < M && gidy*TILESIZE+idy < N)
                C[cOffset + (gidx*TILESIZE +idx) * ldc + (gidy*TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx*TILESIZE+idx) * ldc + (gidy*TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;

    }
    ampblasStatus gemm_TransAB_rMajor_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 16
#define STEPSIZE 16
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            double rC[1][1] = { { 0.0 } };
            double rA[1][STEPTILERATIO];
            double rB[1][STEPTILERATIO];
            tile_static double lA[STEPTILEPROD + STEPSIZE];//8*8+8
            tile_static double lB[STEPTILEPROD + STEPSIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();

                // Load Sections of A and B into respective shared memory slots
                for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec)
                {
                    // Load Section 'sec' from global memory B onto shared lB
                    if (gidy*TILESIZE + idyT  < N && (idxT + i * STEPSIZE + (TILESIZE * sec)) < K)
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = B[bOffset + (gidy*TILESIZE + idyT) * ldb + idxT + i * STEPSIZE + (TILESIZE * sec)];
                    else
                        lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = 0;

                    // Load Section 'sec' from global memory A onto shared lA
                    if (gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K)
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = A[aOffset + gidx*TILESIZE + idxT + idyT*lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
                    else
                        lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = 0;
                }
                tidx.barrier.wait();

                int offA = idx * BANKTILESIZE;
                int offB = idy * BANKTILESIZE;

                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MS1x1_NOBANK(1);
                }

                i++;
            } while (--block_k > 0);


            tidx.barrier.wait();
            if (gidx*TILESIZE + idx < M && gidy*TILESIZE + idy < N)
                C[cOffset + (gidx*TILESIZE + idx) * ldc + (gidy*TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx*TILESIZE + idx) * ldc + (gidy*TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;

    }

    /*
    * SGEMM - TransAB case - Row major Access
    * SUBMICROTILE with Non Bank Concflict Implmentation
    * TILESIZE = 16 MICROITLESIZE = 2
    */

    ampblasStatus gemm_TransAB_rMajor_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
            float rA[1][MICROTILESIZE];
            float rB[1][MICROTILESIZE];
            tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
            tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = 0;
            do
            {
                tidx.barrier.wait();
                for(int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * ldb + idyT + block_k * TILESIZE];
                    }
                    else
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }

                    if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) +  idyT * lda + block_k * (lda * TILESIZE)];
                    }
                    else
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MTS;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1))/TILESIZE));

            int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
            int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
            for( int row = 0; row < MICROTILESIZE; row++)
            {
                for( int col = 0; col < MICROTILESIZE ; col++)
                {
                    if((xIndex/ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N)
                        C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;

    }
    ampblasStatus gemm_TransAB_rMajor_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 16
#define MICROTILESIZE 2
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
            double rA[1][MICROTILESIZE];
            double rB[1][MICROTILESIZE];
            tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
            tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = 0;
            do
            {
                tidx.barrier.wait();
                for (int sec = 0; sec < MICROTILESIZE; ++sec)
                {
                    if (gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K)
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + (gidy * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * ldb + idyT + block_k * TILESIZE];
                    }
                    else
                    {
                        lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }

                    if (gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K)
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + (gidx * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * lda + block_k * (lda * TILESIZE)];
                    }
                    else
                    {
                        lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
                    }
                }
                tidx.barrier.wait();

                int offA = idx;
                int offB = idy;
                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MTS;
                }
                tidx.barrier.wait();
            } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

            int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
            int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);
            for (int row = 0; row < MICROTILESIZE; row++)
            {
                for (int col = 0; col < MICROTILESIZE; col++)
                {
                    if ((xIndex / ldc) + (TILESIZE * col) < M && (yIndex)+(TILESIZE * row) < N)
                        C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row)];
                }
            }
        });
#undef TILESIZE
#undef MICROTILESIZE
        return AMPBLAS_SUCCESS;

    }
    /*
    * SGEMM - TransAB case - Row major Access
    * STEP Implmentation
    * TILESIZE = 8 STEPSIZE = 8
    */

    ampblasStatus gemm_TransAB_rMajor_STEP_TS8XSS8(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<float, 1> &A, long aOffset,
            Concurrency::array_view<float, 1> &B, long bOffset,
            Concurrency::array_view<float, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta)
    {
#define TILESIZE 8
#define STEPSIZE 8
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            float rC[1][1];
            float rA[1][STEPSIZE/TILESIZE];
            float rB[1][STEPSIZE/TILESIZE];
            tile_static float lA[TILESIZE + TILESIZE * STEPSIZE];//8*8+8
            tile_static float lB[TILESIZE + TILESIZE * STEPSIZE];
            rC[0][0] = 0;
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();

                // Load Sections of A and B into respective shared memory slots
                for (int sec =0; sec < STEPSIZE/TILESIZE; ++sec)
                {
                    // Load Section 'sec' from global memory B onto shared lB
                    if(gidy*TILESIZE+idxT  < N && (idyT + i * STEPSIZE + (TILESIZE * sec)) < K)
                        lB[idxT*TILESIZE+idyT + (TILESIZE * TILESIZE * sec)] = B[bOffset + (gidy*TILESIZE+ idxT) * ldb + idyT + i * STEPSIZE + (TILESIZE * sec)];
                    else
                        lB[idxT*TILESIZE+idyT + (TILESIZE * TILESIZE * sec)] = 0;

                    // Load Section 'sec' from global memory A onto shared lA
                    if(gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K)
                        lA[idxT*TILESIZE+idyT + (TILESIZE * TILESIZE * sec)] = A[aOffset  + gidx*TILESIZE+ idxT + idyT*lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
                    else
                        lA[idxT*TILESIZE+idyT + (TILESIZE * TILESIZE * sec)] = 0;
                }
                tidx.barrier.wait();

                int offA = idx * TILESIZE;
                int offB = idy * TILESIZE;
                int offset = 1;

                for (int iter=0; iter < TILESIZE; ++iter)
                {
                    MS1x1(offset, offset);
                }

                i++;
            } while (--block_k > 0);

            tidx.barrier.wait();
            if(gidx*TILESIZE+idx < M && gidy*TILESIZE+idy < N)
                C[cOffset + (gidx*TILESIZE +idx)*ldc + (gidy*TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx*TILESIZE+idx)*ldc + (gidy*TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;

    }
    ampblasStatus gemm_TransAB_rMajor_STEP_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
            Concurrency::array_view<double, 1> &A, long aOffset,
            Concurrency::array_view<double, 1> &B, long bOffset,
            Concurrency::array_view<double, 1> &C, long cOffset,
            int M, int N, int K, int lda, int ldb, int ldc,
            double alpha, double beta)
    {
#define TILESIZE 8
#define STEPSIZE 8
        Concurrency::extent<2> grdExt((M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
            double rC[1][1];
            double rA[1][STEPSIZE / TILESIZE];
            double rB[1][STEPSIZE / TILESIZE];
            tile_static double lA[TILESIZE + TILESIZE * STEPSIZE];//8*8+8
            tile_static double lB[TILESIZE + TILESIZE * STEPSIZE];
            rC[0][0] = 0;
            int gidx = tidx.tile[0];
            int gidy = tidx.tile[1];
            int idx = tidx.local[0];
            int idy = tidx.local[1];
            int idt = TILESIZE * idy + idx;
            int idxT = idt % TILESIZE;
            int idyT = idt / TILESIZE;
            int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
            int i = 0;
            do
            {
                tidx.barrier.wait();

                // Load Sections of A and B into respective shared memory slots
                for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec)
                {
                    // Load Section 'sec' from global memory B onto shared lB
                    if (gidy*TILESIZE + idxT  < N && (idyT + i * STEPSIZE + (TILESIZE * sec)) < K)
                        lB[idxT*TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = B[bOffset + (gidy*TILESIZE + idxT) * ldb + idyT + i * STEPSIZE + (TILESIZE * sec)];
                    else
                        lB[idxT*TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;

                    // Load Section 'sec' from global memory A onto shared lA
                    if (gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K)
                        lA[idxT*TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = A[aOffset + gidx*TILESIZE + idxT + idyT*lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
                    else
                        lA[idxT*TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;
                }
                tidx.barrier.wait();

                int offA = idx * TILESIZE;
                int offB = idy * TILESIZE;
                int offset = 1;

                for (int iter = 0; iter < TILESIZE; ++iter)
                {
                    MS1x1(offset, offset);
                }

                i++;
            } while (--block_k > 0);

            tidx.barrier.wait();
            if (gidx*TILESIZE + idx < M && gidy*TILESIZE + idy < N)
                C[cOffset + (gidx*TILESIZE + idx)*ldc + (gidy*TILESIZE + idy)] = alpha * rC[0][0] + beta * C[cOffset + (gidx*TILESIZE + idx)*ldc + (gidy*TILESIZE + idy)];
        });
#undef TILESIZE
#undef STEPSIZE
        return AMPBLAS_SUCCESS;

    }
    /*  TOP LEVEL FUNCITONS */
    ampblasStatus gemm_NoTransAB_rMajor(Concurrency::accelerator_view &accl_view,
                                        Concurrency::array_view<float, 1> &A, long aOffset,
                                        Concurrency::array_view<float, 1> &B, long bOffset,
                                        Concurrency::array_view<float, 1> &C, long cOffset,
                                        int M, int N, int K, int lda, int ldb, int ldc,
                                        float alpha, float beta)
    {
        if ((M < 600 && N < 600 && K < 10) || (M < 1800 && N < 600 && K < 600))
        {
            return gemm_NoTransAB_rMajor_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        if ((M < 600 && N < 600 && K < 1800) || (M < 1800 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))))
        {
            return gemm_NoTransAB_rMajor_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else
        {
            return gemm_NoTransAB_rMajor_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
    }
    ampblasStatus gemm_NoTransAB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                          Concurrency::array_view<double, 1> &A, long aOffset,
                                          Concurrency::array_view<double, 1> &B, long bOffset,
                                          Concurrency::array_view<double, 1> &C, long cOffset,
                                          int M, int N, int K, int lda, int ldb, int ldc,
                                          double alpha, double beta)
    {
        if ((M < 600 && N < 600 && K < 10) || (M < 1800 && N < 600 && K < 600))
        {
            return gemm_NoTransAB_rMajor_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        if ((M < 600 && N < 600 && K < 1800) || (M < 1800 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))))
        {
            return gemm_NoTransAB_rMajor_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else
        {
            return gemm_NoTransAB_rMajor_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
    }

    ampblasStatus gemm_NoTransA_rMajor(Concurrency::accelerator_view &accl_view,
                                       Concurrency::array_view<float, 1> &A, long aOffset,
                                       Concurrency::array_view<float, 1> &B, long bOffset,
                                       Concurrency::array_view<float, 1> &C, long cOffset,
                                       int M, int N, int K, int lda, int ldb, int ldc,
                                       float alpha, float beta)
    {
        if (((M >=10 && M < 600) && N < 600 && K < 10) || (M >=6000 && M < 10000 && N < 600 && K < 10))
        {
            return gemm_NoTransA_rMajor_STEP_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M > 10 && M < 600) && N < 10 && K < 10)
        {
            return gemm_NoTransA_rMajor_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((((M < 10 && N < 1800) || (M >=600 && M < 1800 && N < 10) || (M < 600 && N < 1800)) && K < 1800) || (M >=1800 && M < 6000 && N < 1800 && K < 10) || (M < 600 && N >=1800 && N < 6000 && K < 10) || (M < 1800  && N < 600 && K <600) || (M >= 6000 && M < 10000 && N < 600 && (K < 10 || M == K )) || (M < 600 && N >=1800 && N < 6000 && K < 1800))
        {
            return gemm_NoTransA_rMajor_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 1800 && N < 1800 && K < 600)
        {
            return  gemm_NoTransA_rMajor_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else
        {
            return  gemm_NoTransA_rMajor_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }

    }
    ampblasStatus gemm_NoTransA_rMajor_d(Concurrency::accelerator_view &accl_view,
                                         Concurrency::array_view<double, 1> &A, long aOffset,
                                         Concurrency::array_view<double, 1> &B, long bOffset,
                                         Concurrency::array_view<double, 1> &C, long cOffset,
                                         int M, int N, int K, int lda, int ldb, int ldc,
                                         double alpha, double beta)
    {
        if (((M >= 10 && M < 600) && N < 600 && K < 10) || (M >= 6000 && M < 10000 && N < 600 && K < 10))
        {
            return gemm_NoTransA_rMajor_STEP_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M > 10 && M < 600) && N < 10 && K < 10)
        {
            return gemm_NoTransA_rMajor_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((((M < 10 && N < 1800) || (M >= 600 && M < 1800 && N < 10) || (M < 600 && N < 1800)) && K < 1800) || (M >= 1800 && M < 6000 && N < 1800 && K < 10) || (M < 600 && N >= 1800 && N < 6000 && K < 10) || (M < 1800 && N < 600 && K <600) || (M >= 6000 && M < 10000 && N < 600 && (K < 10 || M == K)) || (M < 600 && N >= 1800 && N < 6000 && K < 1800))
        {
            return gemm_NoTransA_rMajor_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 1800 && N < 1800 && K < 600)
        {
            return  gemm_NoTransA_rMajor_MICRO_NBK_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else
        {
            return  gemm_NoTransA_rMajor_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }

    }

    ampblasStatus gemm_NoTransB_rMajor(Concurrency::accelerator_view &accl_view,
                                       Concurrency::array_view<float, 1> &A, long aOffset,
                                       Concurrency::array_view<float, 1> &B, long bOffset,
                                       Concurrency::array_view<float, 1> &C, long cOffset,
                                       int M, int N, int K, int lda, int ldb, int ldc,
                                       float alpha, float beta)
    {
        if ((M < 10 && N < 1800 && K < 600) || (M < 600 && ((N < 600 && K < 1800) || (N < 6000 && K < 10))) || (M < 6000 && N < 600 && K < 10))
        {
            return gemm_NoTransB_rMajor_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 600 && N < 1800 && K < 1800) || (M < 1800 && N < 1800 && K < 600))
        {
            return gemm_NoTransB_rMajor_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 10 && N < 1800 && K < 1800) || (M < 600 && (N < 10  || (N >= 1800 && N < 6000)) && K < 1800) || (M < 1800 && ((N < 10 && K < 1800)|| (N < 600 && K >=10 && K < 600))) || (M < 10000 && N < 600 && M == K))
        {
            return gemm_NoTransB_rMajor_STEP_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 1800 && N < 1800 && K < 1800)
        {
            return gemm_NoTransB_rMajor_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && K < 10))
        {
            return gemm_NoTransB_rMajor_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else {
            return gemm_NoTransB_rMajor_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }

    }
    ampblasStatus gemm_NoTransB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                         Concurrency::array_view<double, 1> &A, long aOffset,
                                         Concurrency::array_view<double, 1> &B, long bOffset,
                                         Concurrency::array_view<double, 1> &C, long cOffset,
                                         int M, int N, int K, int lda, int ldb, int ldc,
                                         double alpha, double beta)
    {
        if ((M < 10 && N < 1800 && K < 600) || (M < 600 && ((N < 600 && K < 1800) || (N < 6000 && K < 10))) || (M < 6000 && N < 600 && K < 10))
        {
            return gemm_NoTransB_rMajor_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 600 && N < 1800 && K < 1800) || (M < 1800 && N < 1800 && K < 600))
        {
            return gemm_NoTransB_rMajor_MICRO_NBK_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 10 && N < 1800 && K < 1800) || (M < 600 && (N < 10 || (N >= 1800 && N < 6000)) && K < 1800) || (M < 1800 && ((N < 10 && K < 1800) || (N < 600 && K >= 10 && K < 600))) || (M < 10000 && N < 600 && M == K))
        {
            return gemm_NoTransB_rMajor_STEP_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 1800 && N < 1800 && K < 1800)
        {
            return gemm_NoTransB_rMajor_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && K < 10))
        {
            return gemm_NoTransB_rMajor_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else {
            return gemm_NoTransB_rMajor_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }

    }
    ampblasStatus gemm_TransAB_rMajor(Concurrency::accelerator_view &accl_view,
                                      Concurrency::array_view<float, 1> &A, long aOffset,
                                      Concurrency::array_view<float, 1> &B, long bOffset,
                                      Concurrency::array_view<float, 1> &C, long cOffset,
                                      int M, int N, int K, int lda, int ldb, int ldc,
                                      float alpha, float beta)
    {
        if (M < 600 && N < 600 && K < 10)
        {
            return gemm_TransAB_rMajor_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 600 && ((N < 600 && K < 1800)|| (N < 1800 && K < 10) || (N < 6000 && (K < 10||(K>= 600 && K < 1800))))) || (M < 2000 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))) || (M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && M == K))
        {
            return gemm_TransAB_rMajor_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 1800 && N < 1800 && K < 1800)
        {
            return gemm_TransAB_rMajor_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 10000 && N < 600 && K < 10)
        {
            return gemm_TransAB_rMajor_STEP_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else {
            return gemm_TransAB_rMajor_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
    }
    ampblasStatus gemm_TransAB_rMajor_d(Concurrency::accelerator_view &accl_view,
                                        Concurrency::array_view<double, 1> &A, long aOffset,
                                        Concurrency::array_view<double, 1> &B, long bOffset,
                                        Concurrency::array_view<double, 1> &C, long cOffset,
                                        int M, int N, int K, int lda, int ldb, int ldc,
                                        double alpha, double beta)
    {
        if (M < 600 && N < 600 && K < 10)
        {
            return gemm_TransAB_rMajor_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if ((M < 600 && ((N < 600 && K < 1800) || (N < 1800 && K < 10) || (N < 6000 && (K < 10 || (K >= 600 && K < 1800))))) || (M < 2000 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))) || (M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && M == K))
        {
            return gemm_TransAB_rMajor_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 1800 && N < 1800 && K < 1800)
        {
            return gemm_TransAB_rMajor_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else if (M < 10000 && N < 600 && K < 10)
        {
            return gemm_TransAB_rMajor_STEP_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
        else {
            return gemm_TransAB_rMajor_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
    }
