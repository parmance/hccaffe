#include "cppamp/sgemm_kernels.h"

ampblasStatus gemm_NoTransAB_STEP_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1];
        float rA[1][STEPSIZE/TILESIZE];
        float rB[1][STEPSIZE/TILESIZE];
        tile_static float lA[TILESIZE * STEPSIZE];//8*8+8
        tile_static float lB[TILESIZE * STEPSIZE];
        rC[0][0] = 0;
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx*TILESIZE +idx + (gidy*TILESIZE + idy)*ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx*TILESIZE+idx + (gidy*TILESIZE + idy)*ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransAB_STEP_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1];
        double rA[1][STEPSIZE / TILESIZE];
        double rB[1][STEPSIZE / TILESIZE];
        tile_static double lA[TILESIZE * STEPSIZE];//8*8+8
        tile_static double lB[TILESIZE * STEPSIZE];
        rC[0][0] = 0;
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx*TILESIZE + idx + (gidy*TILESIZE + idy)*ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx*TILESIZE + idx + (gidy*TILESIZE + idy)*ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_NoTransAB_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{0.0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];//8*8+8
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx*TILESIZE +idx + (gidy*TILESIZE + idy)*ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx*TILESIZE+idx + (gidy*TILESIZE + idy)*ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransAB_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { 0.0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];//8*8+8
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx*TILESIZE + idx + (gidy*TILESIZE + idy)*ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx*TILESIZE + idx + (gidy*TILESIZE + idy)*ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransAB_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{0.0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];//8*8+8
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx*TILESIZE +idx + (gidy*TILESIZE + idy)*ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx*TILESIZE+idx + (gidy*TILESIZE + idy)*ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransAB_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { 0.0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];//8*8+8
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx*TILESIZE + idx + (gidy*TILESIZE + idy)*ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx*TILESIZE + idx + (gidy*TILESIZE + idy)*ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransAB_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if(xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransAB_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
        double rA[1][MICROTILESIZE];
        double rB[1][MICROTILESIZE];
        tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for (int row = 0; row < MICROTILESIZE; row++)
        {
            for (int col = 0; col < MICROTILESIZE; col++)
            {
                if (xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_NoTransA_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if(crow < M && ccolprod/ldc < N)
            C[cOffset + crow + ccolprod] =  alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}

ampblasStatus gemm_NoTransA_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if (crow < M && ccolprod / ldc < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransA_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if(crow < M && ccolprod/ldc < N)
            C[cOffset + crow + ccolprod] =  alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_NoTransA_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if (crow < M && ccolprod / ldc < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}



ampblasStatus gemm_NoTransA_STEP_TS16XSS16(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{(float)0}};
        float rA[1][STEPSIZE / TILESIZE];
        float rB[1][STEPSIZE / TILESIZE];
        tile_static float lA[TILESIZE * STEPSIZE];
        tile_static float lB[TILESIZE * STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc] =  alpha * rC[0][0] + beta * C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransA_STEP_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { (double)0 } };
        double rA[1][STEPSIZE / TILESIZE];
        double rB[1][STEPSIZE / TILESIZE];
        tile_static double lA[TILESIZE * STEPSIZE];
        tile_static double lB[TILESIZE * STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_NoTransA_MICRO_NBK_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftTS = Concurrency::fast_math::log2(TILESIZE);
        float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TOTMICROTILEPROD + TILESIZE];
        tile_static float lB[TOTMICROTILEPROD + TILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = (gidx * MICROTILEPROD) + idx;
        int yIndex = ((gidy * MICROTILEPROD) + idy) * ldc;
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if(xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N)
                    C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransA_MICRO_NBK_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftTS = Concurrency::fast_math::log2(TILESIZE);
        double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
        double rA[1][MICROTILESIZE];
        double rB[1][MICROTILESIZE];
        tile_static double lA[TOTMICROTILEPROD + TILESIZE];
        tile_static double lB[TOTMICROTILEPROD + TILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = (gidx * MICROTILEPROD) + idx;
        int yIndex = ((gidy * MICROTILEPROD) + idy) * ldc;
        for (int row = 0; row < MICROTILESIZE; row++)
        {
            for (int col = 0; col < MICROTILESIZE; col++)
            {
                if (xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N)
                    C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_NoTransA_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        float rC[MICROTILESIZE][MICROTILESIZE] =  {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if(xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransA_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
        double rA[1][MICROTILESIZE];
        double rB[1][MICROTILESIZE];
        tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for (int row = 0; row < MICROTILESIZE; row++)
        {
            for (int col = 0; col < MICROTILESIZE; col++)
            {
                if (xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_NoTransB_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if(crow < M && ccolprod/ldc < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransB_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if (crow < M && ccolprod / ldc < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransB_STEP_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{(float)0}};
        float rA[1][STEPSIZE / TILESIZE];
        float rB[1][STEPSIZE / TILESIZE];
        tile_static float lA[TILESIZE * STEPSIZE];
        tile_static float lB[TILESIZE * STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + (gidx * TILESIZE + idx) + (gidy * TILESIZE + idy) * ldc] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx) + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransB_STEP_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { (double)0 } };
        double rA[1][STEPSIZE / TILESIZE];
        double rB[1][STEPSIZE / TILESIZE];
        tile_static double lA[TILESIZE * STEPSIZE];
        tile_static double lB[TILESIZE * STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + (gidx * TILESIZE + idx) + (gidy * TILESIZE + idy) * ldc] = alpha * rC[0][0] + beta * C[cOffset + (gidx * TILESIZE + idx) + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransB_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if(crow < M && ccolprod/ldc < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransB_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
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
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int crow = gidxOffset + idx;
        int ccolprod = (gidyOffset + idy) * ldc;
        if (crow < M && ccolprod / ldc < N)
            C[cOffset + crow + ccolprod] = alpha * rC[0][0] + beta * C[cOffset + crow + ccolprod];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_NoTransB_MICRO_NBK_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftTS = Concurrency::fast_math::log2(TILESIZE);
        float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TOTMICROTILEPROD + TILESIZE];
        tile_static float lB[TOTMICROTILEPROD + TILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = (gidx * MICROTILEPROD) + idx;
        int yIndex = ((gidy * MICROTILEPROD) + idy) * ldc;
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if(xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N)
                    C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS ) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransB_MICRO_NBK_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M >> 1) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftTS = Concurrency::fast_math::log2(TILESIZE);
        double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
        double rA[1][MICROTILESIZE];
        double rB[1][MICROTILESIZE];
        tile_static double lA[TOTMICROTILEPROD + TILESIZE];
        tile_static double lB[TOTMICROTILEPROD + TILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = (gidx * MICROTILEPROD) + idx;
        int yIndex = ((gidy * MICROTILEPROD) + idy) * ldc;
        for (int row = 0; row < MICROTILESIZE; row++)
        {
            for (int col = 0; col < MICROTILESIZE; col++)
            {
                if (xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N)
                    C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}


ampblasStatus gemm_NoTransB_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{

#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if(xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;

}

ampblasStatus gemm_NoTransB_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{

#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
        double rA[1][MICROTILESIZE];
        double rB[1][MICROTILESIZE];
        tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for (int row = 0; row < MICROTILESIZE; row++)
        {
            for (int col = 0; col < MICROTILESIZE; col++)
            {
                if (xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;

}



ampblasStatus gemm_TransAB_STEP_NBK_TS8XSS8(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{(float)0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_STEP_NBK_TS8XSS8_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 8
#define STEPSIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { (double)0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_STEP_NBK_TS16XSS16(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{

#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        float rC[1][1] = {{(float)0}};
        float rA[1][STEPTILERATIO];
        float rB[1][STEPTILERATIO];
        tile_static float lA[STEPTILEPROD + STEPSIZE];
        tile_static float lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_TransAB_STEP_NBK_TS16XSS16_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{

#define TILESIZE 16
#define STEPSIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int shiftFactor = Concurrency::fast_math::log2(STEPSIZE);
        double rC[1][1] = { { (double)0 } };
        double rA[1][STEPTILERATIO];
        double rB[1][STEPTILERATIO];
        tile_static double lA[STEPTILEPROD + STEPSIZE];
        tile_static double lB[STEPTILEPROD + STEPSIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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
            C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc] = alpha * rC[0][0] + beta * C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc];
    });
#undef TILESIZE
#undef STEPSIZE
    return AMPBLAS_SUCCESS;

}
ampblasStatus gemm_TransAB_MICRO_TS16XMTS2(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<float, 1> &A, long aOffset,
        Concurrency::array_view<float, 1> &B, long bOffset,
        Concurrency::array_view<float, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        float alpha, float beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
        float rA[1][MICROTILESIZE];
        float rB[1][MICROTILESIZE];
        tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for( int row = 0; row < MICROTILESIZE; row++)
        {
            for( int col = 0; col < MICROTILESIZE ; col++)
            {
                if(xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_MICRO_TS16XMTS2_d(Concurrency::accelerator_view &accl_view,
        Concurrency::array_view<double, 1> &A, long aOffset,
        Concurrency::array_view<double, 1> &B, long bOffset,
        Concurrency::array_view<double, 1> &C, long cOffset,
        int M, int N, int K, int lda, int ldb, int ldc,
        double alpha, double beta)
{
#define TILESIZE 16
#define MICROTILESIZE 2
    Concurrency::extent<2> grdExt(((N / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1), ((M / 2) + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        double rC[MICROTILESIZE][MICROTILESIZE] = { { (double)0 } };
        double rA[1][MICROTILESIZE];
        double rB[1][MICROTILESIZE];
        tile_static double lA[TILESIZE * TILESIZE * MICROTILESIZE];
        tile_static double lB[TILESIZE * TILESIZE * MICROTILESIZE];
        int gidx = tidx.tile[1];
        int gidy = tidx.tile[0];
        int idx = tidx.local[1];
        int idy = tidx.local[0];
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

        int xIndex = gidx * TILESIZE * MICROTILESIZE + idx;
        int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy) * ldc;
        for (int row = 0; row < MICROTILESIZE; row++)
        {
            for (int col = 0; col < MICROTILESIZE; col++)
            {
                if (xIndex + (TILESIZE * col) < M && (yIndex / ldc) + (TILESIZE * row) < N)
                    C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc] = alpha * rC[col][row] + beta * C[cOffset + (xIndex + TILESIZE * col) + yIndex + (TILESIZE * row) * ldc];
            }
        }
    });
#undef TILESIZE
#undef MICROTILESIZE
    return AMPBLAS_SUCCESS;
}


// Kernel 1

/*
 * Inputs and Outputs are processed in Column major form
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */

ampblasStatus gemm_TransAB_K1(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end   = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp  = TILESIZE;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE * tile_x;

        // last index of first thread reading B in local workgroup -- unused in code
        //int b_end = b_bgn + N*(K-1);

        // step taken by each thread reading B in local workgroup
        int b_stp  = TILESIZE * N;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;

        // local memory for matrix A
        tile_static float localMemA[TILESIZE][TILESIZE];

        // local memory for matrix B
        tile_static float localMemB[TILESIZE][TILESIZE];


        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // each thread in workgroup reads one element of matrix A from global to local memory
            if ( thread_x + global_x < K )
            {
                localMemA[thread_y][thread_x] = alpha*A[a + aOffset +  K * thread_y + thread_x];
            }
            else
            {   // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }
            // each thread in workgroup reads one element of matrix B from global to local memory
            if ( thread_y + global_y < K )
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + N * thread_y + thread_x];
            }
            else
            {   // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();
            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if ( tidx.global[0] < N && tidx.global[1] < M )
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x * M < M*N )
            {
                C[c + cOffset + thread_y + thread_x * M] = sum + beta * C[c + cOffset + thread_y + thread_x * M];
            }
        }
    });
#undef TILESIZE

    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_K1_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp = TILESIZE;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE * tile_x;

        // last index of first thread reading B in local workgroup -- unused in code
        //int b_end = b_bgn + N*(K-1);

        // step taken by each thread reading B in local workgroup
        int b_stp = TILESIZE * N;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;

        // local memory for matrix A
        tile_static double localMemA[TILESIZE][TILESIZE];

        // local memory for matrix B
        tile_static double localMemB[TILESIZE][TILESIZE];


        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // each thread in workgroup reads one element of matrix A from global to local memory
            if (thread_x + global_x < K)
            {
                localMemA[thread_y][thread_x] = alpha*A[a + aOffset + K * thread_y + thread_x];
            }
            else
            {   // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }
            // each thread in workgroup reads one element of matrix B from global to local memory
            if (thread_y + global_y < K)
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + N * thread_y + thread_x];
            }
            else
            {   // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();
            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if (tidx.global[0] < N && tidx.global[1] < M)
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x * M < M*N)
            {
                C[c + cOffset + thread_y + thread_x * M] = sum + beta * C[c + cOffset + thread_y + thread_x * M];
            }
        }
    });
#undef TILESIZE

    return AMPBLAS_SUCCESS;
}


/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is no transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */
ampblasStatus gemm_NoTransAB_K2(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<float, 1> &A, long aOffset,
                                Concurrency::array_view<float, 1> &B, long bOffset,
                                Concurrency::array_view<float, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                float alpha, float beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];


        // first index of first thread reading A in local workgroup
        int a_bgn = TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + M * K;

        // step taken by each thread reading A
        int a_stp  = TILESIZE * M;

        // first index of first thread reading A in local workgroup
        int b_bgn = K * TILESIZE * tile_x;

        // step taken by each thread reading A
        int b_stp = TILESIZE;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;


        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // local memory for matrix A
            tile_static float localMemA[TILESIZE][TILESIZE];

            // local memory for matrix B
            tile_static float localMemB[TILESIZE][TILESIZE];

            // each thread in workgroup reads one element of matrix A from global to local memory
            if ( thread_x + global_x < K )
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + M * thread_x + thread_y];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            if ( thread_y + global_y < K )
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + K * thread_x + thread_y];
            }
            else
            {   // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if ( tidx.global[0] < N && tidx.global[1] < M )
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x * M < M*N )
            {
                C[c + cOffset + thread_y + thread_x * M ] = sum + beta * C[c + cOffset +  thread_y + thread_x * M];
            }
        }
    });
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_NoTransAB_K2_d(Concurrency::accelerator_view &accl_view,
                                  Concurrency::array_view<double, 1> &A, long aOffset,
                                  Concurrency::array_view<double, 1> &B, long bOffset,
                                  Concurrency::array_view<double, 1> &C, long cOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  double alpha, double beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];


        // first index of first thread reading A in local workgroup
        int a_bgn = TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + M * K;

        // step taken by each thread reading A
        int a_stp = TILESIZE * M;

        // first index of first thread reading A in local workgroup
        int b_bgn = K * TILESIZE * tile_x;

        // step taken by each thread reading A
        int b_stp = TILESIZE;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;


        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // local memory for matrix A
            tile_static double localMemA[TILESIZE][TILESIZE];

            // local memory for matrix B
            tile_static double localMemB[TILESIZE][TILESIZE];

            // each thread in workgroup reads one element of matrix A from global to local memory
            if (thread_x + global_x < K)
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + M * thread_x + thread_y];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            if (thread_y + global_y < K)
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + K * thread_x + thread_y];
            }
            else
            {   // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if (tidx.global[0] < N && tidx.global[1] < M)
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x * M < M*N)
            {
                C[c + cOffset + thread_y + thread_x * M] = sum + beta * C[c + cOffset + thread_y + thread_x * M];
            }
        }
    });
    return AMPBLAS_SUCCESS;
}

/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */
ampblasStatus gemm_TransAB_K3(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
    Concurrency::extent<2> grdExt((N/AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end   = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp  = TILESIZE;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE * tile_x * AMP_WPT;

        // step taken by each thread reading B in local workgroup
        int b_stp  = TILESIZE * N;

        int global_x = 0;
        int global_y = 0;

        // local memory for matrix A
        tile_static float localMemA[TILESIZE][TILESIZE];

        // local memory for matrix B
        tile_static float localMemB[TILESIZE][TILESIZE * AMP_WPT];

        // Initialise the accumulation registers
        float acc[AMP_WPT];
        for (int w = 0; w < AMP_WPT; w++)
        {
            acc[w] = 0.0;
        }

        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // each thread in workgroup reads one element of matrix A from global to local memory
            if ( thread_x + global_x < K )
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + K * thread_y + thread_x];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            for (int w = 0; w < AMP_WPT; w++)
            {
                // each thread in workgroup reads one element of matrix B from global to local memory
                if ( thread_y + global_y < K )
                {
                    localMemB[thread_y][thread_x + w * AMP_RTS] = B[b + bOffset + N * thread_y + thread_x + w * AMP_RTS];
                }
                else
                {   // needed on AMD
                    localMemB[thread_y][thread_x + w * AMP_RTS] = 0.0;
                }
            }
            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                for (int w = 0; w < AMP_WPT; w++)
                {
                    acc[w] += localMemA[thread_y][k] * localMemB[k][thread_x + w * AMP_RTS];
                }
            }
            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        for (int w = 0; w < AMP_WPT; w++)
        {
            if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N )
            {
                C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
            }
        }
    });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_K3_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
    Concurrency::extent<2> grdExt((N / AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp = TILESIZE;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE * tile_x * AMP_WPT;

        // step taken by each thread reading B in local workgroup
        int b_stp = TILESIZE * N;

        int global_x = 0;
        int global_y = 0;

        // local memory for matrix A
        tile_static double localMemA[TILESIZE][TILESIZE];

        // local memory for matrix B
        tile_static double localMemB[TILESIZE][TILESIZE * AMP_WPT];

        // Initialise the accumulation registers
        double acc[AMP_WPT];
        for (int w = 0; w < AMP_WPT; w++)
        {
            acc[w] = 0.0;
        }

        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // each thread in workgroup reads one element of matrix A from global to local memory
            if (thread_x + global_x < K)
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + K * thread_y + thread_x];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            for (int w = 0; w < AMP_WPT; w++)
            {
                // each thread in workgroup reads one element of matrix B from global to local memory
                if (thread_y + global_y < K)
                {
                    localMemB[thread_y][thread_x + w * AMP_RTS] = B[b + bOffset + N * thread_y + thread_x + w * AMP_RTS];
                }
                else
                {   // needed on AMD
                    localMemB[thread_y][thread_x + w * AMP_RTS] = 0.0;
                }
            }
            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                for (int w = 0; w < AMP_WPT; w++)
                {
                    acc[w] += localMemA[thread_y][k] * localMemB[k][thread_x + w * AMP_RTS];
                }
            }
            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        for (int w = 0; w < AMP_WPT; w++)
        {
            if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N)
            {
                C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
            }
        }
    });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
    return AMPBLAS_SUCCESS;
}
/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */

ampblasStatus gemm_NoTransA_K4(Concurrency::accelerator_view &accl_view,
                               Concurrency::array_view<float, 1> &A, long aOffset,
                               Concurrency::array_view<float, 1> &B, long bOffset,
                               Concurrency::array_view<float, 1> &C, long cOffset,
                               int M, int N, int K, int lda, int ldb, int ldc,
                               float alpha, float beta)
{
#undef TILESIZE
#define TILESIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + M * K;

        // step taken by each thread reading A
        int a_stp  = TILESIZE * M;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE * tile_x;

        // step taken by each thread reading B in local workgroup
        int b_stp  = TILESIZE * N;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;

        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // local memory for matrix A
            tile_static float localMemA[TILESIZE][TILESIZE];

            // local memory for matrix B
            tile_static float localMemB[TILESIZE][TILESIZE];

            // each thread in workgroup reads one element of matrix A from global to local memory
            if ( thread_x + global_x < K )
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + M * thread_x + thread_y];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            if ( thread_y + global_y < K )
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + N * thread_y + thread_x];
            }
            else
            {
                // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();
            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }
            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if ( tidx.global[0] < N && tidx.global[1] < M )
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x  * M < M*N )
            {
                C[c + cOffset + thread_y + thread_x * M] = sum + beta * C[c + cOffset + thread_y + thread_x * M];
            }
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransA_K4_d(Concurrency::accelerator_view &accl_view,
                                 Concurrency::array_view<double, 1> &A, long aOffset,
                                 Concurrency::array_view<double, 1> &B, long bOffset,
                                 Concurrency::array_view<double, 1> &C, long cOffset,
                                 int M, int N, int K, int lda, int ldb, int ldc,
                                 double alpha, double beta)
{
#undef TILESIZE
#define TILESIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + M * K;

        // step taken by each thread reading A
        int a_stp = TILESIZE * M;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE * tile_x;

        // step taken by each thread reading B in local workgroup
        int b_stp = TILESIZE * N;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;

        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // local memory for matrix A
            tile_static double localMemA[TILESIZE][TILESIZE];

            // local memory for matrix B
            tile_static double localMemB[TILESIZE][TILESIZE];

            // each thread in workgroup reads one element of matrix A from global to local memory
            if (thread_x + global_x < K)
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + M * thread_x + thread_y];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            if (thread_y + global_y < K)
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + N * thread_y + thread_x];
            }
            else
            {
                // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();
            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }
            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if (tidx.global[0] < N && tidx.global[1] < M)
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x  * M < M*N)
            {
                C[c + cOffset + thread_y + thread_x * M] = sum + beta * C[c + cOffset + thread_y + thread_x * M];
            }
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */
ampblasStatus gemm_NoTransB_K5(Concurrency::accelerator_view &accl_view,
                               Concurrency::array_view<float, 1> &A, long aOffset,
                               Concurrency::array_view<float, 1> &B, long bOffset,
                               Concurrency::array_view<float, 1> &C, long cOffset,
                               int M, int N, int K, int lda, int ldb, int ldc,
                               float alpha, float beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp = TILESIZE;

        // first index of first thread reading A in local workgroup
        int b_bgn = K * TILESIZE * tile_x;

        // step taken by each thread reading A
        int b_stp = TILESIZE;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;

        // each work group moves horizontally over matrix A and vertically over matrix B
        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE )
        {

            // local memory for matrix A
            tile_static float localMemA[TILESIZE][TILESIZE];

            // local memory for matrix B
            tile_static float localMemB[TILESIZE][TILESIZE];

            // each thread in workgroup reads one element of matrix A from global to local memory
            if ( thread_y + global_y < K )
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + thread_y + thread_x * K];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            if ( thread_x + global_x < K )
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + thread_x + thread_y * K];
            }
            else
            {
                // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[k][thread_y] * localMemB[thread_x][k];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if ( tidx.global[0] < N && tidx.global[1] < M )
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x * M < M*N )
            {
                C[c + cOffset + thread_y + thread_x * M ] = sum + beta * C[c + cOffset + thread_y + thread_x * M];
            }
        }

    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransB_K5_d(Concurrency::accelerator_view &accl_view,
                                 Concurrency::array_view<double, 1> &A, long aOffset,
                                 Concurrency::array_view<double, 1> &B, long bOffset,
                                 Concurrency::array_view<double, 1> &C, long cOffset,
                                 int M, int N, int K, int lda, int ldb, int ldc,
                                 double alpha, double beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp = TILESIZE;

        // first index of first thread reading A in local workgroup
        int b_bgn = K * TILESIZE * tile_x;

        // step taken by each thread reading A
        int b_stp = TILESIZE;

        // accumulates the result
        double sum = 0.0;

        int global_x = 0;
        int global_y = 0;

        // each work group moves horizontally over matrix A and vertically over matrix B
        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {

            // local memory for matrix A
            tile_static double localMemA[TILESIZE][TILESIZE];

            // local memory for matrix B
            tile_static double localMemB[TILESIZE][TILESIZE];

            // each thread in workgroup reads one element of matrix A from global to local memory
            if (thread_y + global_y < K)
            {
                localMemA[thread_y][thread_x] = alpha * A[a + aOffset + thread_y + thread_x * K];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            if (thread_x + global_x < K)
            {
                localMemB[thread_y][thread_x] = B[b + bOffset + thread_x + thread_y * K];
            }
            else
            {
                // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++)
            {
                sum += localMemA[k][thread_y] * localMemB[thread_x][k];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if (tidx.global[0] < N && tidx.global[1] < M)
        {
            int c = TILESIZE * tile_y + TILESIZE * tile_x * M;
            if (c + cOffset + thread_y + thread_x * M < M*N)
            {
                C[c + cOffset + thread_y + thread_x * M] = sum + beta * C[c + cOffset + thread_y + thread_x * M];
            }
        }

    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE_1D_X x TILESIZE_1D_Y] elements
 * and satisfies
 *
 * TILESIZE_1D_Y % TILESIZE_1D_X == 0
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE_1D_Y == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE_1D_X
 *   local_size[1] := TILESIZE_1D_Y
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE_1D_X*TILESIZE_1D_Y
 */
ampblasStatus gemm_TransAB_K6(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{
#define TILESIZE_1D_Y 32
#define TILESIZE_1D_X 8
#define TILESIZE_X 8
#define TILESIZE_Y 32
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE_1D_X x TILESIZE_1D_Y]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE_Y * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end   = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp  = TILESIZE_Y;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE_X * tile_x;

        // step taken by each thread reading B in local workgroup
        int b_stp  = TILESIZE_Y * N;

        unsigned int idx = 0;

        // accumulates the result
        float sum = 0.0;

        // each work group moves horizontally over matrix A and vertically over matrix B
        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp)
        {
            // local memory for matrix A
            tile_static float localMemA[TILESIZE_Y][TILESIZE_Y];

            // local memory for matrix B
            tile_static float localMemB[TILESIZE_Y][TILESIZE_X];

            // each thread in workgroup reads several element of matrix A from global to local memory due to the buffer not being quadratic
            for( int i = 0; i < TILESIZE_Y/TILESIZE_X; i++ )
            {
                idx = a + K * thread_y + thread_x + i*TILESIZE_X;
                if ( idx < M*K )
                {
                    localMemA[thread_y][thread_x + i*TILESIZE_X] = alpha * A[aOffset + idx];
                }
                else
                {
                    // needed on AMD
                    localMemA[thread_y][thread_x + i*TILESIZE_X] = 0.0;
                }
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            idx =  b + N * thread_y + thread_x;
            if (idx < K*N )
            {
                localMemB[thread_y][thread_x] = B[bOffset + idx];
            }
            else
            {
                // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // compute limit for loop to stop accumulating when the boundary of the matrix is reached
            int limit = TILESIZE_Y;

            if ( K % TILESIZE_Y != 0 && a == a_end )
            {
                limit = K/TILESIZE_Y;
            }

            if ( K < TILESIZE_Y )
            {
                limit = K;
            }

            // multiply matrix A and B using local memory
            for (int k = 0; k < limit; ++k)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if ( tidx.global[0] < N && tidx.global[1] < M )
        {
            int c = TILESIZE_Y * tile_y + TILESIZE_X * tile_x * M;
            if (c + cOffset + M * thread_x + thread_y < M*N )
            {
                C[c + cOffset + M * thread_x + thread_y] = sum + beta * C[c + cOffset + M * thread_x + thread_y];
            }
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_K6_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{
#define TILESIZE_1D_Y 32
#define TILESIZE_1D_X 8
#define TILESIZE_X 8
#define TILESIZE_Y 32
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE_1D_X x TILESIZE_1D_Y]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE_Y * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp = TILESIZE_Y;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE_X * tile_x;

        // step taken by each thread reading B in local workgroup
        int b_stp = TILESIZE_Y * N;

        unsigned int idx = 0;

        // accumulates the result
        float sum = 0.0;

        // each work group moves horizontally over matrix A and vertically over matrix B
        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp)
        {
            // local memory for matrix A
            tile_static double localMemA[TILESIZE_Y][TILESIZE_Y];

            // local memory for matrix B
            tile_static double localMemB[TILESIZE_Y][TILESIZE_X];

            // each thread in workgroup reads several element of matrix A from global to local memory due to the buffer not being quadratic
            for (int i = 0; i < TILESIZE_Y / TILESIZE_X; i++)
            {
                idx = a + K * thread_y + thread_x + i*TILESIZE_X;
                if (idx < M*K)
                {
                    localMemA[thread_y][thread_x + i*TILESIZE_X] = alpha * A[aOffset + idx];
                }
                else
                {
                    // needed on AMD
                    localMemA[thread_y][thread_x + i*TILESIZE_X] = 0.0;
                }
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            idx = b + N * thread_y + thread_x;
            if (idx < K*N)
            {
                localMemB[thread_y][thread_x] = B[bOffset + idx];
            }
            else
            {
                // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // compute limit for loop to stop accumulating when the boundary of the matrix is reached
            int limit = TILESIZE_Y;

            if (K % TILESIZE_Y != 0 && a == a_end)
            {
                limit = K / TILESIZE_Y;
            }

            if (K < TILESIZE_Y)
            {
                limit = K;
            }

            // multiply matrix A and B using local memory
            for (int k = 0; k < limit; ++k)
            {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if (tidx.global[0] < N && tidx.global[1] < M)
        {
            int c = TILESIZE_Y * tile_y + TILESIZE_X * tile_x * M;
            if (c + cOffset + M * thread_x + thread_y < M*N)
            {
                C[c + cOffset + M * thread_x + thread_y] = sum + beta * C[c + cOffset + M * thread_x + thread_y];
            }
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}
/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE_1D_X x TILESIZE_1D_Y] elements
 * and satisfies
 *
 * TILESIZE_1D_X % TILESIZE_1D_Y == 0
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE_1D_Y == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE_1D_X
 *   local_size[1] := TILESIZE_1D_Y
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE_1D_X*TILESIZE_1D_Y
 */

ampblasStatus gemm_TransAB_K7(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{
#define TILESIZE_1D_Y 8
#define TILESIZE_1D_X 32
#define TILESIZE_X 32
#define TILESIZE_Y 8
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE_1D_X x TILESIZE_1D_Y]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE_Y * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end   = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp  = TILESIZE_X;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE_X * tile_x;

        // step taken by each thread reading B in local workgroup
        int b_stp  = TILESIZE_X * N;

        unsigned int idx = 0;

        // accumulates the result
        float sum = 0.0;

        // each work group moves horizontally over matrix A and vertically over matrix B
        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp)  {

            // local memory for matrix A
            tile_static float localMemA[TILESIZE_Y][TILESIZE_X];

            // local memory for matrix A
            tile_static float localMemB[TILESIZE_X][TILESIZE_X];

            // each thread in workgroup reads one element of matrix A from global to local memory
            idx = a + K * thread_y + thread_x;
            if ( idx < M*K ) {
                localMemA[thread_y][thread_x] = alpha*A[idx];
            } else { // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads several element of matrix B from global to local memory due to the buffer not being quadratic
            for( int i = 0; i < TILESIZE_X/TILESIZE_Y; i++ ) {
                idx =  b + N * thread_y + thread_x + i*N*TILESIZE_Y;
                if (idx < K*N ) {
                    localMemB[thread_y + i*TILESIZE_Y][thread_x] = B[idx];
                } else { // needed on AMD
                    localMemB[thread_y + i*TILESIZE_Y][thread_x] = 0.0;
                }
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // compute limit for loop to stop accumulating when the boundary of the matrix is reached
            int limit = TILESIZE_X;

            if ( K % TILESIZE_X != 0 && a == a_end ) {
                limit = K/TILESIZE_X;
            }

            if ( K < TILESIZE_X ) {
                limit = K;
            }

            // multiply matrix A and B using local memory
            for (int k = 0; k < limit; ++k) {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if ( tidx.global[0] < N && tidx.global[1] < M ) {
            int c = TILESIZE_Y * tile_y + TILESIZE_X * tile_x * M;
            if (c + M * thread_x + thread_y < M*N ) {
                C[c + cOffset + M * thread_x + thread_y] = sum + beta*C[c + cOffset + M * thread_x + thread_y];
            }
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_K7_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{
#define TILESIZE_1D_Y 8
#define TILESIZE_1D_X 32
#define TILESIZE_X 32
#define TILESIZE_Y 8
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE_1D_X x TILESIZE_1D_Y]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE_Y * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp = TILESIZE_X;

        // first index of first thread reading B in local workgroup
        int b_bgn = TILESIZE_X * tile_x;

        // step taken by each thread reading B in local workgroup
        int b_stp = TILESIZE_X * N;

        unsigned int idx = 0;

        // accumulates the result
        float sum = 0.0;

        // each work group moves horizontally over matrix A and vertically over matrix B
        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp)  {

            // local memory for matrix A
            tile_static double localMemA[TILESIZE_Y][TILESIZE_X];

            // local memory for matrix A
            tile_static double localMemB[TILESIZE_X][TILESIZE_X];

            // each thread in workgroup reads one element of matrix A from global to local memory
            idx = a + K * thread_y + thread_x;
            if (idx < M*K) {
                localMemA[thread_y][thread_x] = alpha*A[idx];
            }
            else { // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads several element of matrix B from global to local memory due to the buffer not being quadratic
            for (int i = 0; i < TILESIZE_X / TILESIZE_Y; i++) {
                idx = b + N * thread_y + thread_x + i*N*TILESIZE_Y;
                if (idx < K*N) {
                    localMemB[thread_y + i*TILESIZE_Y][thread_x] = B[idx];
                }
                else { // needed on AMD
                    localMemB[thread_y + i*TILESIZE_Y][thread_x] = 0.0;
                }
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // compute limit for loop to stop accumulating when the boundary of the matrix is reached
            int limit = TILESIZE_X;

            if (K % TILESIZE_X != 0 && a == a_end) {
                limit = K / TILESIZE_X;
            }

            if (K < TILESIZE_X) {
                limit = K;
            }

            // multiply matrix A and B using local memory
            for (int k = 0; k < limit; ++k) {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if (tidx.global[0] < N && tidx.global[1] < M) {
            int c = TILESIZE_Y * tile_y + TILESIZE_X * tile_x * M;
            if (c + M * thread_x + thread_y < M*N) {
                C[c + cOffset + M * thread_x + thread_y] = sum + beta*C[c + cOffset + M * thread_x + thread_y];
            }
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}
/*
 * Matrix-Matrix-Multiplication for the case when K == 1 using
 * two local memory buffers of size TILESIZE_1D_X and
 * TILESIZE_1D_Y]
 *
 * Dimensions:
 *   Matrix A is [Mx1] and A is transposed
 *   Matrix B is [1xN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE_1D_Y == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE_1D_X
 *   local_size[1] := TILESIZE_1D_Y
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE_1D_X*TILESIZE_1D_Y
 */

ampblasStatus gemm_TransAB_K8(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{

#define TILESIZE_1D_Y 1
#define TILESIZE_1D_X 256
#define TILESIZE_X 16
#define TILESIZE_Y 16
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];


        unsigned int idx = 0;

        // local memory for matrix A
        tile_static float localMemA[TILESIZE_1D_Y];

        // local memory for matrix B
        tile_static float localMemB[TILESIZE_1D_X];

        // each thread in workgroup reads one element of matrix A from global to local memory
        idx = tidx.global[1];
        if ( thread_x == 0 ) {
            if ( idx < M ) {
                localMemA[thread_y]= alpha*A[aOffset + idx];
            } else { // needed on AMD
                localMemA[thread_y] = 0.0;
            }
        }

        // each thread in workgroup reads one element of matrix B from global to local memory
        idx = tidx.global[0];
        if ( thread_y == 0 ) {
            if ( idx < N ) {
                localMemB[thread_x] = B[bOffset + idx];
            } else { // needed on AMD
                localMemB[thread_x] = 0.0;
            }
        }

        // Synchronize the reads of A and B
        tidx.barrier.wait();

        // multiply matrix A and matrix B and write all results back to global memory
        if ( tidx.global[0] < N && tidx.global[1] < M ) {
            idx = tidx.global[1] + tidx.global[0] * M;
            C[cOffset + idx] = localMemA[thread_y] * localMemB[thread_x] + beta * C[cOffset + idx];
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}

ampblasStatus gemm_TransAB_K8_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{

#define TILESIZE_1D_Y 1
#define TILESIZE_1D_X 256
#define TILESIZE_X 16
#define TILESIZE_Y 16
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];


        unsigned int idx = 0;

        // local memory for matrix A
        tile_static double localMemA[TILESIZE_1D_Y];

        // local memory for matrix B
        tile_static double localMemB[TILESIZE_1D_X];

        // each thread in workgroup reads one element of matrix A from global to local memory
        idx = tidx.global[1];
        if (thread_x == 0) {
            if (idx < M) {
                localMemA[thread_y] = alpha*A[aOffset + idx];
            }
            else { // needed on AMD
                localMemA[thread_y] = 0.0;
            }
        }

        // each thread in workgroup reads one element of matrix B from global to local memory
        idx = tidx.global[0];
        if (thread_y == 0) {
            if (idx < N) {
                localMemB[thread_x] = B[bOffset + idx];
            }
            else { // needed on AMD
                localMemB[thread_x] = 0.0;
            }
        }

        // Synchronize the reads of A and B
        tidx.barrier.wait();

        // multiply matrix A and matrix B and write all results back to global memory
        if (tidx.global[0] < N && tidx.global[1] < M) {
            idx = tidx.global[1] + tidx.global[0] * M;
            C[cOffset + idx] = localMemA[thread_y] * localMemB[thread_x] + beta * C[cOffset + idx];
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}

/*
 * Matrix-Matrix-Multiplication for the case when K == 1 without using a single local variable
 *
 * Dimensions:
 *   Matrix A is [Mx1] and A is transposed
 *   Matrix B is [1xN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE_1D_Y == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE_1D_X
 *   local_size[1] := TILESIZE_1D_Y
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE_1D_X*TILESIZE_1D_Y
 */

ampblasStatus gemm_TransAB_K9(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{
#define TILESIZE_1D_Y 1
#define TILESIZE_1D_X 256
#define TILESIZE_X 16
#define TILESIZE_Y 16
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // global index of each thread
        int gx = tidx.global[0];
        int gy = tidx.global[1];

        float localVarA = alpha*A[aOffset + gy];

        // Synchronize the reads of A
        tidx.barrier.wait();

        // multiply matrix A and matrix B and write all results back to global memory
        if ( gx < N && gy < M ) {
            C[cOffset + gy + gx * M] = localVarA*B[bOffset + gx] + beta*C[cOffset + gy + gx * M];
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_K9_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{
#define TILESIZE_1D_Y 1
#define TILESIZE_1D_X 256
#define TILESIZE_X 16
#define TILESIZE_Y 16
    Concurrency::extent<2> grdExt((N + (TILESIZE_1D_X - 1)) & ~(TILESIZE_1D_X - 1), (M + (TILESIZE_1D_Y - 1)) & ~(TILESIZE_1D_Y - 1));
    Concurrency::tiled_extent<TILESIZE_1D_X, TILESIZE_1D_Y> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE_1D_X, TILESIZE_1D_Y> tidx) restrict(amp)
    {
        // global index of each thread
        int gx = tidx.global[0];
        int gy = tidx.global[1];

        float localVarA = alpha*A[aOffset + gy];

        // Synchronize the reads of A
        tidx.barrier.wait();

        // multiply matrix A and matrix B and write all results back to global memory
        if (gx < N && gy < M) {
            C[cOffset + gy + gx * M] = localVarA*B[bOffset + gx] + beta*C[cOffset + gy + gx * M];
        }
    });
#undef TILESIZE_1D_Y
#undef TILESIZE_1D_X
#undef TILESIZE_X
#undef TILESIZE_Y
    return AMPBLAS_SUCCESS;
}
/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */

ampblasStatus gemm_TransAB_K10(Concurrency::accelerator_view &accl_view,
                               Concurrency::array_view<float, 1> &A, long aOffset,
                               Concurrency::array_view<float, 1> &B, long bOffset,
                               Concurrency::array_view<float, 1> &C, long cOffset,
                               int M, int N, int K, int lda, int ldb, int ldc,
                               float alpha, float beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int global_idx_i = tidx.global[0];

        // local registers
        float privateMemA[256];

        for( int k = 0; k < 256; k++ )
        {
            privateMemA[k] = alpha*A[aOffset +global_idx_i*K + k];
        }

        tile_static float localMemB[256];

        for( int idx_n = 0; idx_n < N; idx_n++ )
        {
            for( int i = 0; i < t_ext[0]; i++ )
            {
                int idx_k = i * tidx.tile_dim0 + tidx.local[0];
                int idx_B = idx_k*N + idx_n;
                localMemB[bOffset + idx_k ] = B[bOffset + idx_B ];
            }

            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            float sum = 0.0;

            for (int k = 0; k < K; k++) {
                sum += privateMemA[k] * localMemB[k];
            }
            C[cOffset + global_idx_i + idx_n * M ] = sum + beta * C[cOffset + global_idx_i + idx_n * M];
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_K10_d(Concurrency::accelerator_view &accl_view,
                                 Concurrency::array_view<double, 1> &A, long aOffset,
                                 Concurrency::array_view<double, 1> &B, long bOffset,
                                 Concurrency::array_view<double, 1> &C, long cOffset,
                                 int M, int N, int K, int lda, int ldb, int ldc,
                                 double alpha, double beta)
{
#define TILESIZE 8
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int global_idx_i = tidx.global[0];

        // local registers
        double privateMemA[256];

        for (int k = 0; k < 256; k++)
        {
            privateMemA[k] = alpha*A[aOffset + global_idx_i*K + k];
        }

        tile_static double localMemB[256];

        for (int idx_n = 0; idx_n < N; idx_n++)
        {
            for (int i = 0; i < t_ext[0]; i++)
            {
                int idx_k = i * tidx.tile_dim0 + tidx.local[0];
                int idx_B = idx_k*N + idx_n;
                localMemB[bOffset + idx_k] = B[bOffset + idx_B];
            }

            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            double sum = 0.0;

            for (int k = 0; k < K; k++) {
                sum += privateMemA[k] * localMemB[k];
            }
            C[cOffset + global_idx_i + idx_n * M] = sum + beta * C[cOffset + global_idx_i + idx_n * M];
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}

/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */

ampblasStatus gemm_TransAB_11(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{
#define TILESIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int global_idx_i = tidx.global[0];

        float privateMemA[256];
        for( int k = 0; k < 256; k++ ) {
            privateMemA[k] = alpha*A[aOffset + global_idx_i*K + k];
        }

        for( int idx_n = 0; idx_n < N; idx_n++ ) {

            // multiply matrix A and B using local memory
            float sum = 0.0;

            for (int k = 0; k < K; k++)
            {
                int idx_k = global_idx_i;
                int idx_B = idx_k*N + k;
                sum += privateMemA[k] * B[ bOffset + idx_B];
            }
            C[cOffset + global_idx_i + idx_n * M ] = sum + beta * C[cOffset + global_idx_i  + idx_n * M];
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_11_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{
#define TILESIZE 16
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        int global_idx_i = tidx.global[0];

        double privateMemA[256];
        for (int k = 0; k < 256; k++) {
            privateMemA[k] = alpha*A[aOffset + global_idx_i*K + k];
        }

        for (int idx_n = 0; idx_n < N; idx_n++) {

            // multiply matrix A and B using local memory
            float sum = 0.0;

            for (int k = 0; k < K; k++)
            {
                int idx_k = global_idx_i;
                int idx_B = idx_k*N + k;
                sum += privateMemA[k] * B[bOffset + idx_B];
            }
            C[cOffset + global_idx_i + idx_n * M] = sum + beta * C[cOffset + global_idx_i + idx_n * M];
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
/*
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [TILESIZE x TILESIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % TILESIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % TILESIZE == 0 && global_size[1] >= M
 *
 * Local Index Space
 *   local_size[0] := TILESIZE
 *   local_size[1] := TILESIZE
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := TILESIZE*TILESIZE
 */

ampblasStatus gemm_TransAB_12(Concurrency::accelerator_view &accl_view,
                              Concurrency::array_view<float, 1> &A, long aOffset,
                              Concurrency::array_view<float, 1> &B, long bOffset,
                              Concurrency::array_view<float, 1> &C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              float alpha, float beta)
{
#define TILESIZE 8
#define GN N/32 //(N has to be in power of 32)
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_y = tidx.tile[1];

        // local index of each thread inside tile
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // global coordinates for each elemnt in C
        int x = tidx.global[0];
        int y = tidx.global[1];

        int group_size_n  = N / GN;
        int group_n       = x / group_size_n;

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end   = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp  = TILESIZE;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;
        int addr;
        // local memory for matrix A
        tile_static float localMemA[TILESIZE][TILESIZE];

        // local memory for matrix B
        tile_static float localMemB[TILESIZE][TILESIZE];

        for (int a = a_bgn; a <= a_end; a += a_stp, global_x += TILESIZE, global_y += TILESIZE)  {

            // each thread in workgroup reads one element of matrix A from global to local memory
            addr = a + K * thread_y + thread_x;

            if ( (thread_x + global_x) < K && addr < M*K ) {
                localMemA[thread_y][thread_x] = alpha*A[aOffset + addr];
            } else { // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            addr = group_n*(group_size_n*K) + ( x % group_size_n ) + (thread_y+global_y)*group_size_n;
            if ( thread_y + global_y < K  && addr < K*N ) {
                localMemB[thread_y][thread_x] = B[bOffset + addr];
            } else { // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++) {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];

            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if ( x < N && y < M ) {

            addr = group_n*group_size_n*M + y*group_size_n + ( x % group_size_n );
            if (addr < M*N ) {
                C[cOffset + addr] = sum + beta*C[cOffset + addr];
            }
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_TransAB_12_d(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<double, 1> &A, long aOffset,
                                Concurrency::array_view<double, 1> &B, long bOffset,
                                Concurrency::array_view<double, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                double alpha, double beta)
{
#define TILESIZE 8
#define GN N/32 //(N has to be in power of 32)
    Concurrency::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_y = tidx.tile[1];

        // local index of each thread inside tile
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];

        // global coordinates for each elemnt in C
        int x = tidx.global[0];
        int y = tidx.global[1];

        int group_size_n = N / GN;
        int group_n = x / group_size_n;

        // first index of first thread reading A in local workgroup
        int a_bgn = K * TILESIZE * tile_y;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + K - 1;

        // step taken by each thread reading A
        int a_stp = TILESIZE;

        // accumulates the result
        float sum = 0.0;

        int global_x = 0;
        int global_y = 0;
        int addr;
        // local memory for matrix A
        tile_static double localMemA[TILESIZE][TILESIZE];

        // local memory for matrix B
        tile_static double localMemB[TILESIZE][TILESIZE];

        for (int a = a_bgn; a <= a_end; a += a_stp, global_x += TILESIZE, global_y += TILESIZE)  {

            // each thread in workgroup reads one element of matrix A from global to local memory
            addr = a + K * thread_y + thread_x;

            if ((thread_x + global_x) < K && addr < M*K) {
                localMemA[thread_y][thread_x] = alpha*A[aOffset + addr];
            }
            else { // needed on AMD
                localMemA[thread_y][thread_x] = 0.0;
            }

            // each thread in workgroup reads one element of matrix B from global to local memory
            addr = group_n*(group_size_n*K) + (x % group_size_n) + (thread_y + global_y)*group_size_n;
            if (thread_y + global_y < K  && addr < K*N) {
                localMemB[thread_y][thread_x] = B[bOffset + addr];
            }
            else { // needed on AMD
                localMemB[thread_y][thread_x] = 0.0;
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();

            // multiply matrix A and B using local memory
            for (int k = 0; k < TILESIZE; k++) {
                sum += localMemA[thread_y][k] * localMemB[k][thread_x];

            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory
        if (x < N && y < M) {

            addr = group_n*group_size_n*M + y*group_size_n + (x % group_size_n);
            if (addr < M*N) {
                C[cOffset + addr] = sum + beta*C[cOffset + addr];
            }
        }
    });
#undef TILESIZE
    return AMPBLAS_SUCCESS;
}
ampblasStatus gemm_NoTransAB_K3(Concurrency::accelerator_view &accl_view,
                                Concurrency::array_view<float, 1> &A, long aOffset,
                                Concurrency::array_view<float, 1> &B, long bOffset,
                                Concurrency::array_view<float, 1> &C, long cOffset,
                                int M, int N, int K, int lda, int ldb, int ldc,
                                float alpha, float beta)
{
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
    Concurrency::extent<2> grdExt((N/AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
    Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
    {
        // coordinates for each tile of [TILESIZE x TILESIZE]
        int tile_x = tidx.tile[0];
        int tile_y = tidx.tile[1];

        // local index of each thread
        int thread_x = tidx.local[0];
        int thread_y = tidx.local[1];


        // first index of first thread reading A in local workgroup
        int a_bgn = TILESIZE * tile_y ;

        // last index to first thread reading A in local workgroup
        int a_end = a_bgn + M * K;

        // step taken by each thread reading A
        int a_stp  = TILESIZE * M;

        // first index of first thread reading A in local workgroup
        int b_bgn = K * TILESIZE * tile_x * AMP_WPT;

        // step taken by each thread reading A
        int b_stp = TILESIZE;

        // accumulates the result

        int global_x = 0;
        int global_y = 0;

        float acc[AMP_WPT];
        for (int w = 0; w < AMP_WPT; w++)
        {
            acc[w] = 0.0;
        }

        for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
        {
            // local memory for matrix A
            tile_static float localMemA[TILESIZE][TILESIZE];

            // local memory for matrix B
            tile_static float localMemB[TILESIZE][TILESIZE * AMP_WPT];

            // Initialise the accumulation registers


            // each thread in workgroup reads one element of matrix A from global to local memory

            if ( thread_x + global_x < K )
            {
                localMemA[thread_y][thread_x ] = alpha * A[a + aOffset + M * thread_x + thread_y ];
            }
            else
            {
                // needed on AMD
                localMemA[thread_y][thread_x ] = 0.0;
            }


            // each thread in workgroup reads one element of matrix B from global to local memory
            for (int w = 0; w < AMP_WPT; w++)
            {
                if ( thread_y + global_y < K )
                {
                    localMemB[thread_y][thread_x + w * AMP_RTS] = B[b + bOffset + K * (thread_x + w * AMP_RTS) + thread_y];
                }
                else
                {   // needed on AMD
                    localMemB[thread_y][thread_x + w * AMP_RTS] = 0.0;
                }
            }

            // Synchronize the reads of A and B
            tidx.barrier.wait();
            for (int k = 0; k < TILESIZE; k++)
            {
                for (int w = 0; w < AMP_WPT; w++)
                {
                    acc[w] += localMemA[thread_y][k ] * localMemB[k][thread_x + w * AMP_RTS ];
                }
            }

            // Synchronize all sub-results
            tidx.barrier.wait();
        }

        // write all results back to global memory

        for (int w = 0; w < AMP_WPT; w++)
        {
            if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N )
            {
                C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
            }
        }
    });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
	return AMPBLAS_SUCCESS;
	}
    ampblasStatus gemm_NoTransAB_K3_d(Concurrency::accelerator_view &accl_view,
                                      Concurrency::array_view<double, 1> &A, long aOffset,
                                      Concurrency::array_view<double, 1> &B, long bOffset,
                                      Concurrency::array_view<double, 1> &C, long cOffset,
                                      int M, int N, int K, int lda, int ldb, int ldc,
                                      double alpha, double beta)
    {
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
        Concurrency::extent<2> grdExt((N / AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            // coordinates for each tile of [TILESIZE x TILESIZE]
            int tile_x = tidx.tile[0];
            int tile_y = tidx.tile[1];

            // local index of each thread
            int thread_x = tidx.local[0];
            int thread_y = tidx.local[1];


            // first index of first thread reading A in local workgroup
            int a_bgn = TILESIZE * tile_y;

            // last index to first thread reading A in local workgroup
            int a_end = a_bgn + M * K;

            // step taken by each thread reading A
            int a_stp = TILESIZE * M;

            // first index of first thread reading A in local workgroup
            int b_bgn = K * TILESIZE * tile_x * AMP_WPT;

            // step taken by each thread reading A
            int b_stp = TILESIZE;

            // accumulates the result

            int global_x = 0;
            int global_y = 0;

            float acc[AMP_WPT];
            for (int w = 0; w < AMP_WPT; w++)
            {
                acc[w] = 0.0;
            }

            for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
            {
                // local memory for matrix A
                tile_static double localMemA[TILESIZE][TILESIZE];

                // local memory for matrix B
                tile_static double localMemB[TILESIZE][TILESIZE * AMP_WPT];

                // Initialise the accumulation registers


                // each thread in workgroup reads one element of matrix A from global to local memory

                if (thread_x + global_x < K)
                {
                    localMemA[thread_y][thread_x] = alpha * A[a + aOffset + M * thread_x + thread_y];
                }
                else
                {
                    // needed on AMD
                    localMemA[thread_y][thread_x] = 0.0;
                }


                // each thread in workgroup reads one element of matrix B from global to local memory
                for (int w = 0; w < AMP_WPT; w++)
                {
                    if (thread_y + global_y < K)
                    {
                        localMemB[thread_y][thread_x + w * AMP_RTS] = B[b + bOffset + K * (thread_x + w * AMP_RTS) + thread_y];
                    }
                    else
                    {   // needed on AMD
                        localMemB[thread_y][thread_x + w * AMP_RTS] = 0.0;
                    }
                }

                // Synchronize the reads of A and B
                tidx.barrier.wait();
                for (int k = 0; k < TILESIZE; k++)
                {
                    for (int w = 0; w < AMP_WPT; w++)
                    {
                        acc[w] += localMemA[thread_y][k] * localMemB[k][thread_x + w * AMP_RTS];
                    }
                }

                // Synchronize all sub-results
                tidx.barrier.wait();
            }

            // write all results back to global memory

            for (int w = 0; w < AMP_WPT; w++)
            {
                if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N)
                {
                    C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
                }
            }
        });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
        return AMPBLAS_SUCCESS;
    }
    ampblasStatus gemm_NoTransA_K3(Concurrency::accelerator_view &accl_view,
                                   Concurrency::array_view<float, 1> &A, long aOffset,
                                   Concurrency::array_view<float, 1> &B, long bOffset,
                                   Concurrency::array_view<float, 1> &C, long cOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta)

    {
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
        Concurrency::extent<2> grdExt((N/AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
        Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
        Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
        {
            // coordinates for each tile of [TILESIZE x TILESIZE]
            int tile_x = tidx.tile[0];
            int tile_y = tidx.tile[1];

            // local index of each thread
            int thread_x = tidx.local[0];
            int thread_y = tidx.local[1];

            // first index of first thread reading A in local workgroup
            int a_bgn = TILESIZE * tile_y;

            // last index to first thread reading A in local workgroup
            int a_end = a_bgn + M * K;

            // step taken by each thread reading A
            int a_stp  = TILESIZE * M;

            // first index of first thread reading B in local workgroup
            int b_bgn = TILESIZE * tile_x * AMP_WPT;

            // step taken by each thread reading B in local workgroup
            int b_stp  = TILESIZE * N;

            // accumulates the result

            int global_x = 0;
            int global_y = 0;
            float acc[AMP_WPT];
            for (int w = 0; w < AMP_WPT; w++)
            {
                acc[w] = 0.0;
            }

            for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
            {
                // local memory for matrix A
                tile_static float localMemA[TILESIZE][TILESIZE];

                // local memory for matrix B
                tile_static float localMemB[TILESIZE][TILESIZE * AMP_WPT];

                // each thread in workgroup reads one element of matrix A from global to local memory
                if ( thread_x + global_x < K )
                {
                    localMemA[thread_y][thread_x] = alpha * A[a + aOffset + M * thread_x + thread_y];
                }
                else
                {
                    // needed on AMD
                    localMemA[thread_y][thread_x] = 0.0;
                }

                // each thread in workgroup reads one element of matrix B from global to local memory
                for (int w = 0; w < AMP_WPT; w++)
                {
                    if ( thread_y + global_y < K )
                    {
                        localMemB[thread_y][thread_x + w * AMP_RTS] = B[b + bOffset + N * thread_y + thread_x + w * AMP_RTS];
                    }
                    else
                    {
                        // needed on AMD
                        localMemB[thread_y][thread_x + w * AMP_RTS] = 0.0;
                    }
                }

                // Synchronize the reads of A and B
                tidx.barrier.wait();
                // multiply matrix A and B using local memory
                for (int k = 0; k < TILESIZE; k++)
                {
                    for (int w = 0; w < AMP_WPT; w++)
                    {
                        acc[w] += localMemA[thread_y][k ] * localMemB[k][thread_x + w * AMP_RTS ];
                    }
                }

                // Synchronize all sub-results
                tidx.barrier.wait();
            }

            // write all results back to global memory

            for (int w = 0; w < AMP_WPT; w++)
            {
                if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N )
                {
                    C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
                }
            }
        });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
		return AMPBLAS_SUCCESS;
		}
        ampblasStatus gemm_NoTransA_K3_d(Concurrency::accelerator_view &accl_view,
                                         Concurrency::array_view<double, 1> &A, long aOffset,
                                         Concurrency::array_view<double, 1> &B, long bOffset,
                                         Concurrency::array_view<double, 1> &C, long cOffset,
                                         int M, int N, int K, int lda, int ldb, int ldc,
                                         double alpha, double beta)

        {
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
            Concurrency::extent<2> grdExt((N / AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
            Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
            Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
            {
                // coordinates for each tile of [TILESIZE x TILESIZE]
                int tile_x = tidx.tile[0];
                int tile_y = tidx.tile[1];

                // local index of each thread
                int thread_x = tidx.local[0];
                int thread_y = tidx.local[1];

                // first index of first thread reading A in local workgroup
                int a_bgn = TILESIZE * tile_y;

                // last index to first thread reading A in local workgroup
                int a_end = a_bgn + M * K;

                // step taken by each thread reading A
                int a_stp = TILESIZE * M;

                // first index of first thread reading B in local workgroup
                int b_bgn = TILESIZE * tile_x * AMP_WPT;

                // step taken by each thread reading B in local workgroup
                int b_stp = TILESIZE * N;

                // accumulates the result

                int global_x = 0;
                int global_y = 0;
                double acc[AMP_WPT];
                for (int w = 0; w < AMP_WPT; w++)
                {
                    acc[w] = 0.0;
                }

                for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
                {
                    // local memory for matrix A
                    tile_static double localMemA[TILESIZE][TILESIZE];

                    // local memory for matrix B
                    tile_static double localMemB[TILESIZE][TILESIZE * AMP_WPT];

                    // each thread in workgroup reads one element of matrix A from global to local memory
                    if (thread_x + global_x < K)
                    {
                        localMemA[thread_y][thread_x] = alpha * A[a + aOffset + M * thread_x + thread_y];
                    }
                    else
                    {
                        // needed on AMD
                        localMemA[thread_y][thread_x] = 0.0;
                    }

                    // each thread in workgroup reads one element of matrix B from global to local memory
                    for (int w = 0; w < AMP_WPT; w++)
                    {
                        if (thread_y + global_y < K)
                        {
                            localMemB[thread_y][thread_x + w * AMP_RTS] = B[b + bOffset + N * thread_y + thread_x + w * AMP_RTS];
                        }
                        else
                        {
                            // needed on AMD
                            localMemB[thread_y][thread_x + w * AMP_RTS] = 0.0;
                        }
                    }

                    // Synchronize the reads of A and B
                    tidx.barrier.wait();
                    // multiply matrix A and B using local memory
                    for (int k = 0; k < TILESIZE; k++)
                    {
                        for (int w = 0; w < AMP_WPT; w++)
                        {
                            acc[w] += localMemA[thread_y][k] * localMemB[k][thread_x + w * AMP_RTS];
                        }
                    }

                    // Synchronize all sub-results
                    tidx.barrier.wait();
                }

                // write all results back to global memory

                for (int w = 0; w < AMP_WPT; w++)
                {
                    if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N)
                    {
                        C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
                    }
                }
            });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
            return AMPBLAS_SUCCESS;
        }

        ampblasStatus gemm_NoTransB_K3(Concurrency::accelerator_view &accl_view,
                                       Concurrency::array_view<float, 1> &A, long aOffset,
                                       Concurrency::array_view<float, 1> &B, long bOffset,
                                       Concurrency::array_view<float, 1> &C, long cOffset,
                                       int M, int N, int K, int lda, int ldb, int ldc,
                                       float alpha, float beta)

        {
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
            Concurrency::extent<2> grdExt((N/AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
            Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
            Concurrency::parallel_for_each(accl_view, t_ext, [=] (Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
            {
                // coordinates for each tile of [TILESIZE x TILESIZE]
                int tile_x = tidx.tile[0];
                int tile_y = tidx.tile[1];

                // local index of each thread
                int thread_x = tidx.local[0];
                int thread_y = tidx.local[1];

                // first index of first thread reading A in local workgroup
                int a_bgn = K * TILESIZE * tile_y;

                // last index to first thread reading A in local workgroup
                int a_end = a_bgn + K - 1;

                // step taken by each thread reading A
                int a_stp = TILESIZE;

                // first index of first thread reading A in local workgroup
                int b_bgn = K * TILESIZE * tile_x * AMP_WPT;

                // step taken by each thread reading A
                int b_stp = TILESIZE;

                // accumulates the result
                int global_x = 0;
                int global_y = 0;
                float acc[AMP_WPT];
                for (int w = 0; w < AMP_WPT; w++)
                {
                    acc[w] = 0.0;
                }
                // each work group moves horizontally over matrix A and vertically over matrix B
                for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE )
                {

                    // local memory for matrix A
                    tile_static float localMemA[TILESIZE][TILESIZE];

                    // local memory for matrix B
                    tile_static float localMemB[TILESIZE * AMP_WPT][TILESIZE];

                    // each thread in workgroup reads one element of matrix A from global to local memory
                    if ( thread_y + global_y < K )
                    {
                        localMemA[thread_y][thread_x] = alpha * A[a + aOffset + thread_y + thread_x * K];
                    }
                    else
                    {
                        // needed on AMD
                        localMemA[thread_y][thread_x] = 0.0;
                    }

                    // each thread in workgroup reads one element of matrix B from global to local memory
                    for (int w = 0; w < AMP_WPT; w++)
                    {
                        if ( thread_x + global_x < K )
                        {
                            localMemB[thread_y + w * AMP_RTS][thread_x] = B[b + bOffset + thread_x + (thread_y  + w * AMP_RTS) * K];
                        }
                        else
                        {
                            // needed on AMD
                            localMemB[thread_y + w * AMP_RTS][thread_x] = 0.0;
                        }
                    }
                    // Synchronize the reads of A and B
                    tidx.barrier.wait();

                    // multiply matrix A and B using local memory
                    for (int k = 0; k < TILESIZE; k++)
                    {
                        for (int w = 0; w < AMP_WPT; w++)
                        {
                            acc[w] += localMemA[k ][thread_y] * localMemB[thread_x + w * AMP_RTS ][k];
                        }
                    }

                    // Synchronize all sub-results
                    tidx.barrier.wait();
                }

                // write all results back to global memory
                for (int w = 0; w < AMP_WPT; w++)
                {
                    if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N )
                    {
                        C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
                    }
                }
            });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
			return AMPBLAS_SUCCESS;
			}
            ampblasStatus gemm_NoTransB_K3_d(Concurrency::accelerator_view &accl_view,
                                             Concurrency::array_view<double, 1> &A, long aOffset,
                                             Concurrency::array_view<double, 1> &B, long bOffset,
                                             Concurrency::array_view<double, 1> &C, long cOffset,
                                             int M, int N, int K, int lda, int ldb, int ldc,
                                             double alpha, double beta)

            {
#undef  TILESIZE
#define TILESIZE 16
#define AMP_WPT 4
#define AMP_RTS 16
                Concurrency::extent<2> grdExt((N / AMP_WPT + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
                Concurrency::tiled_extent<TILESIZE, TILESIZE> t_ext(grdExt);
                Concurrency::parallel_for_each(accl_view, t_ext, [=](Concurrency::tiled_index<TILESIZE, TILESIZE> tidx) restrict(amp)
                {
                    // coordinates for each tile of [TILESIZE x TILESIZE]
                    int tile_x = tidx.tile[0];
                    int tile_y = tidx.tile[1];

                    // local index of each thread
                    int thread_x = tidx.local[0];
                    int thread_y = tidx.local[1];

                    // first index of first thread reading A in local workgroup
                    int a_bgn = K * TILESIZE * tile_y;

                    // last index to first thread reading A in local workgroup
                    int a_end = a_bgn + K - 1;

                    // step taken by each thread reading A
                    int a_stp = TILESIZE;

                    // first index of first thread reading A in local workgroup
                    int b_bgn = K * TILESIZE * tile_x * AMP_WPT;

                    // step taken by each thread reading A
                    int b_stp = TILESIZE;

                    // accumulates the result
                    int global_x = 0;
                    int global_y = 0;
                    double acc[AMP_WPT];
                    for (int w = 0; w < AMP_WPT; w++)
                    {
                        acc[w] = 0.0;
                    }
                    // each work group moves horizontally over matrix A and vertically over matrix B
                    for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += TILESIZE, global_y += TILESIZE)
                    {

                        // local memory for matrix A
                        tile_static double localMemA[TILESIZE][TILESIZE];

                        // local memory for matrix B
                        tile_static double localMemB[TILESIZE * AMP_WPT][TILESIZE];

                        // each thread in workgroup reads one element of matrix A from global to local memory
                        if (thread_y + global_y < K)
                        {
                            localMemA[thread_y][thread_x] = alpha * A[a + aOffset + thread_y + thread_x * K];
                        }
                        else
                        {
                            // needed on AMD
                            localMemA[thread_y][thread_x] = 0.0;
                        }

                        // each thread in workgroup reads one element of matrix B from global to local memory
                        for (int w = 0; w < AMP_WPT; w++)
                        {
                            if (thread_x + global_x < K)
                            {
                                localMemB[thread_y + w * AMP_RTS][thread_x] = B[b + bOffset + thread_x + (thread_y + w * AMP_RTS) * K];
                            }
                            else
                            {
                                // needed on AMD
                                localMemB[thread_y + w * AMP_RTS][thread_x] = 0.0;
                            }
                        }
                        // Synchronize the reads of A and B
                        tidx.barrier.wait();

                        // multiply matrix A and B using local memory
                        for (int k = 0; k < TILESIZE; k++)
                        {
                            for (int w = 0; w < AMP_WPT; w++)
                            {
                                acc[w] += localMemA[k][thread_y] * localMemB[thread_x + w * AMP_RTS][k];
                            }
                        }

                        // Synchronize all sub-results
                        tidx.barrier.wait();
                    }

                    // write all results back to global memory
                    for (int w = 0; w < AMP_WPT; w++)
                    {
                        if (tile_y * TILESIZE + thread_y < M && (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) < N)
                        {
                            C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M] = acc[w] + beta * C[cOffset + tile_y * TILESIZE + thread_y + (tile_x * TILESIZE * AMP_WPT + thread_x + w * AMP_RTS) * M];
                        }
                    }
                });
#undef TILESIZE
#undef AMP_RTS
#undef AMP_WPT
                return AMPBLAS_SUCCESS;
            }



            ampblasStatus gemm_NoTransAB(Concurrency::accelerator_view &accl_view,
                                         Concurrency::array_view<float, 1> &A, long aOffset,
                                         Concurrency::array_view<float, 1> &B, long bOffset,
                                         Concurrency::array_view<float, 1> &C, long cOffset,
                                         int M, int N, int K, int lda, int ldb, int ldc,
                                         float alpha, float beta)
            {
                if (M < 600 && N < 600 && K < 10)
                {
                    return gemm_NoTransAB_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 600 && ((N < 600 && K < 1800)|| (N < 1800 && K < 10) || (N < 6000 && (K < 10||(K>= 600 && K < 1800))))) || (M < 2000 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))) || (M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && M == K))
                {
                    return gemm_NoTransAB_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 1800 && N < 1800 && K < 1800)
                {
                    return gemm_NoTransAB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 10000 && N < 600 && K < 10)
                {
                    return gemm_NoTransAB_STEP_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else {
                    return gemm_NoTransAB_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
            }
            ampblasStatus gemm_NoTransAB_d(Concurrency::accelerator_view &accl_view,
                                           Concurrency::array_view<double, 1> &A, long aOffset,
                                           Concurrency::array_view<double, 1> &B, long bOffset,
                                           Concurrency::array_view<double, 1> &C, long cOffset,
                                           int M, int N, int K, int lda, int ldb, int ldc,
                                           double alpha, double beta)
            {
                if (M < 600 && N < 600 && K < 10)
                {
                    return gemm_NoTransAB_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 600 && ((N < 600 && K < 1800) || (N < 1800 && K < 10) || (N < 6000 && (K < 10 || (K >= 600 && K < 1800))))) || (M < 2000 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))) || (M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && M == K))
                {
                    return gemm_NoTransAB_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 1800 && N < 1800 && K < 1800)
                {
                    return gemm_NoTransAB_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 10000 && N < 600 && K < 10)
                {
                    return gemm_NoTransAB_STEP_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else {
                    return gemm_NoTransAB_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
            }


            ampblasStatus gemm_NoTransA(Concurrency::accelerator_view &accl_view,
                                        Concurrency::array_view<float, 1> &A, long aOffset,
                                        Concurrency::array_view<float, 1> &B, long bOffset,
                                        Concurrency::array_view<float, 1> &C, long cOffset,
                                        int M, int N, int K, int lda, int ldb, int ldc,
                                        float alpha, float beta)
            {
                if ((M < 10 && N < 1800 && K < 600) || (M < 600 && ((N < 600 && K < 1800) || (N < 6000 && K < 10))) || (M < 6000 && N < 600 && K < 10))
                {
                    return gemm_NoTransA_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 600 && N < 1800 && K < 1800) || (M < 1800 && N < 1800 && K < 600))
                {
                    return gemm_NoTransA_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 10 && N < 1800 && K < 1800) || (M < 600 && (N < 10  || (N >= 1800 && N < 6000)) && K < 1800) || (M < 1800 && ((N < 10 && K < 1800)|| (N < 600 && K >=10 && K < 600))) || (M < 10000 && N < 600 && M == K))
                {
                    return gemm_NoTransA_STEP_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 1800 && N < 1800 && K < 1800)
                {
                    return gemm_NoTransA_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && K < 10))
                {
                    return gemm_NoTransA_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else {
                    return gemm_NoTransA_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
            }

            ampblasStatus gemm_NoTransA_d(Concurrency::accelerator_view &accl_view,
                                          Concurrency::array_view<double, 1> &A, long aOffset,
                                          Concurrency::array_view<double, 1> &B, long bOffset,
                                          Concurrency::array_view<double, 1> &C, long cOffset,
                                          int M, int N, int K, int lda, int ldb, int ldc,
                                          double alpha, double beta)
            {
                if ((M < 10 && N < 1800 && K < 600) || (M < 600 && ((N < 600 && K < 1800) || (N < 6000 && K < 10))) || (M < 6000 && N < 600 && K < 10))
                {
                    return gemm_NoTransA_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 600 && N < 1800 && K < 1800) || (M < 1800 && N < 1800 && K < 600))
                {
                    return gemm_NoTransA_MICRO_NBK_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 10 && N < 1800 && K < 1800) || (M < 600 && (N < 10 || (N >= 1800 && N < 6000)) && K < 1800) || (M < 1800 && ((N < 10 && K < 1800) || (N < 600 && K >= 10 && K < 600))) || (M < 10000 && N < 600 && M == K))
                {
                    return gemm_NoTransA_STEP_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 1800 && N < 1800 && K < 1800)
                {
                    return gemm_NoTransA_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M < 6000 && N < 1800 && K < 10) || (M < 10000 && N < 600 && K < 10))
                {
                    return gemm_NoTransA_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else {
                    return gemm_NoTransA_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
            }





            ampblasStatus gemm_NoTransB(Concurrency::accelerator_view &accl_view,
                                        Concurrency::array_view<float, 1> &A, long aOffset,
                                        Concurrency::array_view<float, 1> &B, long bOffset,
                                        Concurrency::array_view<float, 1> &C, long cOffset,
                                        int M, int N, int K, int lda, int ldb, int ldc,
                                        float alpha, float beta)
            {
                if (((M >=10 && M < 600) && N < 600 && K < 10) || (M >=6000 && M < 10000 && N < 600 && K < 10))
                {
                    return gemm_NoTransB_STEP_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M > 10 && M < 600) && N < 10 && K < 10)
                {
                    return gemm_NoTransB_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((((M < 10 && N < 1800) || (M >=600 && M < 1800 && N < 10) || (M < 600 && N < 1800)) && K < 1800) || (M >=1800 && M < 6000 && N < 1800 && K < 10) || (M < 600 && N >=1800 && N < 6000 && K < 10) || (M < 1800  && N < 600 && K <600) || (M >= 6000 && M < 10000 && N < 600 && (K < 10 || M == K )) || (M < 600 && N >=1800 && N < 6000 && K < 1800))
                {
                    return gemm_NoTransB_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 1800 && N < 1800 && K < 600)
                {
                    return  gemm_NoTransB_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else
                {
                    return  gemm_NoTransB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }

            }
            ampblasStatus gemm_NoTransB_d(Concurrency::accelerator_view &accl_view,
                                          Concurrency::array_view<double, 1> &A, long aOffset,
                                          Concurrency::array_view<double, 1> &B, long bOffset,
                                          Concurrency::array_view<double, 1> &C, long cOffset,
                                          int M, int N, int K, int lda, int ldb, int ldc,
                                          double alpha, double beta)
            {
                if (((M >= 10 && M < 600) && N < 600 && K < 10) || (M >= 6000 && M < 10000 && N < 600 && K < 10))
                {
                    return gemm_NoTransB_STEP_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((M > 10 && M < 600) && N < 10 && K < 10)
                {
                    return gemm_NoTransB_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if ((((M < 10 && N < 1800) || (M >= 600 && M < 1800 && N < 10) || (M < 600 && N < 1800)) && K < 1800) || (M >= 1800 && M < 6000 && N < 1800 && K < 10) || (M < 600 && N >= 1800 && N < 6000 && K < 10) || (M < 1800 && N < 600 && K <600) || (M >= 6000 && M < 10000 && N < 600 && (K < 10 || M == K)) || (M < 600 && N >= 1800 && N < 6000 && K < 1800))
                {
                    return gemm_NoTransB_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else if (M < 1800 && N < 1800 && K < 600)
                {
                    return  gemm_NoTransB_MICRO_NBK_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else
                {
                    return  gemm_NoTransB_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }

            }
            ampblasStatus gemm_TransAB(Concurrency::accelerator_view &accl_view,
                                       Concurrency::array_view<float, 1> &A, long aOffset,
                                       Concurrency::array_view<float, 1> &B, long bOffset,
                                       Concurrency::array_view<float, 1> &C, long cOffset,
                                       int M, int N, int K, int lda, int ldb, int ldc,
                                       float alpha, float beta)
            {
                if ((M < 600 && N < 600 && K < 10) || (M < 1800 && N < 600 && K < 600))
                {
                    return gemm_TransAB_STEP_NBK_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                if ((M < 600 && N < 600 && K < 1800) || (M < 1800 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))))
                {
                    return gemm_TransAB_STEP_NBK_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else
                {
                    return gemm_TransAB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
            }
            ampblasStatus gemm_TransAB_d(Concurrency::accelerator_view &accl_view,
                                         Concurrency::array_view<double, 1> &A, long aOffset,
                                         Concurrency::array_view<double, 1> &B, long bOffset,
                                         Concurrency::array_view<double, 1> &C, long cOffset,
                                         int M, int N, int K, int lda, int ldb, int ldc,
                                         double alpha, double beta)
            {
                if ((M < 600 && N < 600 && K < 10) || (M < 1800 && N < 600 && K < 600))
                {
                    return gemm_TransAB_STEP_NBK_TS8XSS8_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                if ((M < 600 && N < 600 && K < 1800) || (M < 1800 && ((N < 600 && K < 1800) || (N < 1800 && K < 10))))
                {
                    return gemm_TransAB_STEP_NBK_TS16XSS16_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
                else
                {
                    return gemm_TransAB_MICRO_TS16XMTS2_d(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
                }
            }






