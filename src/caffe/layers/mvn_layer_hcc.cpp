#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#ifdef HCC_BACKEND
namespace caffe {
  template <typename Dtype>
  void MVNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*> &bottom,
                                    const vector<Blob<Dtype>*> &top) {
    const Dtype *bottom_data = bottom[0]->gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    int num;
    if (this->layer_param_.mvn_param().across_channels()) {
      num = bottom[0]->num();
    } else {
      num = bottom[0]->num() * bottom[0]->channels();
    }

    int dim = bottom[0]->count() / num;

    if (this->layer_param_.mvn_param().normalize_variance()) {
      // put the squares of bottom into temp_
      caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2),
                     temp_.mutable_gpu_data());

      // computes variance using var(X) = E(X^2) - (EX)^2
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, const_cast<Dtype*>(bottom_data), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()),
                            0, 0., const_cast<Dtype*>(mean_.mutable_gpu_data()), 0);  // EX
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim,
                            const_cast<Dtype*>(temp_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0,  0.,
                            variance_.mutable_gpu_data(), 0);  // E(X^2)
      caffe_gpu_powx(mean_.count(), mean_.gpu_data(), Dtype(2),
                     temp_.mutable_gpu_data());  // (EX)^2
      caffe_gpu_sub(mean_.count(), variance_.gpu_data(), temp_.gpu_data(),
                    variance_.mutable_gpu_data());  // variance

      Dtype eps = 1e-10;

      // do mean and variance normalization
      // subtract mean
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            num, dim, 1, -1.,
                            const_cast<Dtype*>(mean_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0, 0.,
                            temp_.mutable_gpu_data(), 0);

      caffe_gpu_add(temp_.count(), bottom_data,
                    temp_.gpu_data(), top_data);

      // normalize variance
      caffe_gpu_powx(variance_.count(), variance_.gpu_data(),
                     Dtype(0.5),
                     variance_.mutable_gpu_data());

      caffe_gpu_add_scalar(variance_.count(), eps,
                           variance_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            const_cast<Dtype*>(variance_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0, 0.,
                            temp_.mutable_gpu_data(), 0);

      caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, const_cast<Dtype*>(bottom_data), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()),
                            0, 0., mean_.mutable_gpu_data(), 0);  // EX

      // subtract mean
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim,
                            1, -1.,
                            const_cast<Dtype*>(mean_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0, 0.,
                            temp_.mutable_gpu_data(), 0);

      caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(), top_data);
    }
  }

  template <typename Dtype>
  void MVNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*> &top,
                                     const vector<bool> &propagate_down,
                                     const vector<Blob<Dtype>*> &bottom) {
    const Dtype *top_diff = top[0]->gpu_diff();
    const Dtype *top_data = top[0]->gpu_data();
    const Dtype *bottom_data = bottom[0]->gpu_data();
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();

    int num;
    if (this->layer_param_.mvn_param().across_channels()) {
      num = bottom[0]->num();
    } else {
      num = bottom[0]->num() * bottom[0]->channels();
    }

    int dim = bottom[0]->count() / num;

    Dtype eps = 1e-10;

    if (this->layer_param_.mvn_param().normalize_variance()) {
      caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., const_cast<Dtype*>(bottom_diff), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0,
                            0., mean_.mutable_gpu_data(), 0);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            const_cast<Dtype*>(mean_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0, 0.,
                            bottom_diff, 0);
      caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., const_cast<Dtype*>(top_diff), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()),
                            0, 0., mean_.mutable_gpu_data(), 0);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            const_cast<Dtype*>(mean_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0, 1.,
                            bottom_diff, 0);

      caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
                      bottom_diff);

      // put the squares of bottom into temp_
      caffe_gpu_powx(temp_.count(), bottom_data, Dtype(2),
                     temp_.mutable_gpu_data());

      // computes variance using var(X) = E(X^2) - (EX)^2
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num,
                            dim, 1. / dim, const_cast<Dtype*>(bottom_data), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()),
                            0, 0., mean_.mutable_gpu_data(), 0);  // EX
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim,
                            const_cast<Dtype*>(temp_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0, 0.,
                            variance_.mutable_gpu_data(), 0);  // E(X^2)
      caffe_gpu_powx(mean_.count(), mean_.gpu_data(), Dtype(2),
                     temp_.mutable_gpu_data());  // (EX)^2
      caffe_gpu_sub(mean_.count(), variance_.gpu_data(), temp_.gpu_data(),
                    variance_.mutable_gpu_data());  // variance

      // normalize variance
      caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
                     variance_.mutable_gpu_data());

      caffe_gpu_add_scalar(variance_.count(), eps,
                           variance_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            const_cast<Dtype*>(variance_.gpu_data()), 0,
                            const_cast<Dtype*>(sum_multiplier_.gpu_data()), 0, 0.,
                            temp_.mutable_gpu_data(), 0);

      caffe_gpu_div(temp_.count(), bottom_diff,
                    temp_.gpu_data(), bottom_diff);
    } else {
      caffe_copy(temp_.count(), top_diff, bottom_diff);
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(MVNLayer);


}  // namespace caffe

#endif  // HCC_BACKEND
