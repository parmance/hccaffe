/*
*
*  FILENAME : ampblas.h
*  This file is the top level header file which includes the Ampblaslilbrary class
*  for performing five blas operations ( saxpy, sger, sgemv, sgemm, cgemm )
*
*/

#ifndef AMPBLAS_LIB_H
#define AMPBLAS_LIB_H

#include<iostream>

#include "amp.h"
#include "amp_math.h"
using namespace concurrency;

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

#endif