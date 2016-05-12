#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);
  } else {
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
#ifdef USE_CPPAMP
#if multi_process
  amp_setup();
#endif
#endif
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY
#ifdef USE_CPPAMP
extern void* CreateAmpBuffer(size_t size, size_t element_size);
template <typename Dtype>
  size_t BaseConvolutionLayer<Dtype>::subtop_mem_size = 0;
template <typename Dtype>
  size_t BaseConvolutionLayer<Dtype>::trans_mem_size =  0;
template <typename Dtype>  void* BaseConvolutionLayer<Dtype>::subTopMem = NULL;
template <typename Dtype>  void* BaseConvolutionLayer<Dtype>::transMem = NULL;


template <typename Dtype>
void Alloc_public_tmp_mem(size_t subtop_size, size_t trans_size) {
  if (subtop_size > BaseConvolutionLayer < Dtype > ::subtop_mem_size) {
    ConvolutionLayer<Dtype>::subtop_mem_size = subtop_size;
    caffe_amp_free(ConvolutionLayer<Dtype>::subTopMem,
      sizeof(Dtype), false);
    ConvolutionLayer<Dtype>::subTopMem = CreateAmpBuffer(subtop_size,
      sizeof(Dtype));
  }
  if (trans_size > ConvolutionLayer < Dtype > ::trans_mem_size) {
    ConvolutionLayer<Dtype>::trans_mem_size =  trans_size;
    caffe_amp_free(ConvolutionLayer<Dtype>::transMem, sizeof(Dtype), false);
    ConvolutionLayer<Dtype>::transMem = CreateAmpBuffer(trans_size,
      sizeof(Dtype));
  }
  /*
  int *temp = NULL;
  temp = new int[ConvolutionLayer<Dtype>::subtop_mem_size/sizeof(int)];
  caffe_memset(ConvolutionLayer<Dtype>::subtop_mem_size, 0, temp);
  caffe_amp_H2D(static_cast<void*>(temp), ConvolutionLayer<Dtype>::subTopMem,
        sizeof(Dtype), false);
  delete []temp;
  temp = new int[ConvolutionLayer<Dtype>::trans_mem_size/sizeof(int)];
  caffe_memset(ConvolutionLayer<Dtype>::trans_mem_size, 0, temp);
  caffe_amp_H2D(static_cast<void*>(temp), ConvolutionLayer<Dtype>::transMem,
        sizeof(Dtype), false);
  delete []temp;*/
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::amp_setup() {
  M_ = num_output_ / group_;
  K_ = conv_in_channels_ * kernel_w_ * kernel_h_ / group_;
  N_ = height_out_ * width_out_;
  size_t subtop_size = (size_t)((M_ * group_) * N_
    * global_packing_N * sizeof(Dtype));
  size_t trans_size = (size_t)((K_ * group_ ) *
    N_ * global_packing_N * sizeof(Dtype));
  Alloc_public_tmp_mem<Dtype>(subtop_size, trans_size);
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const int offset, const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., const_cast<Dtype*>(bias), 0,
      const_cast<Dtype*>(bias_multiplier_.gpu_data()), 0, (Dtype)1., output, offset);
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input, const int offset) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_,
      1., const_cast<Dtype*>(input), offset, const_cast<Dtype*>(bias_multiplier_.gpu_data()), 0, 1., bias, 0);
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias_opt(Dtype* bias,
    const Dtype* input) {
      caffe_gpu_gemv2<Dtype>(CblasNoTrans, num_output_, N_,
          1., const_cast<Dtype*>(input), top_offset_, N_,
          const_cast<Dtype*>(bias_multiplier_.gpu_data()),
          (size_t)0, 1., 1, bias, (size_t)0, 1);
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(int N1, int N2,
      const Dtype* input, const int offset_input, const Dtype* weights,
      Dtype* output, const int offset_output, bool skip_im2col) {
  const Dtype* col_buff = input;
  int offset_col = offset_input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, offset_input,
          col_buffer_.mutable_gpu_data(), 0);
    }
    col_buff = col_buffer_.gpu_data();
    offset_col = 0;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., const_cast<Dtype*>(weights) , weight_offset_ * g, const_cast<Dtype*>(col_buff),
        offset_col+col_offset_ * g, (Dtype)0., output,
        offset_output+output_offset_ * g);
  }
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm_opt(const Dtype* input,
    const Dtype* weight, Dtype* output, bool skip_im2col) {
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu_opt(input);
    }
  }
    for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_ * opt_num2,
        K_, 1., const_cast<Dtype*>(weight), weight_offset_ * g, static_cast<Dtype*>(transMem),
        col_offset_ * g, 0.,  static_cast<Dtype*>(subTopMem),
        top_offset_opt * g);
    }
  transform_gpu<Dtype>(subTopMem, output, top_offset_, N_, M_*group_, opt_num2);
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias_opt(Dtype* output,
    const Dtype* bias) {
  for (int z = 0; z < opt_num2; z++)
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      N_, 1, 1., const_cast<Dtype*>(bias), 0, const_cast<Dtype*>(bias_multiplier_.gpu_data()), 0,
      1., output, top_offset_ + num_output_ * N_ * z);
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm_opt(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  if (!is_1x1_) {
    conv_im2col_gpu_opt(input);
  }
    opttrans<Dtype>(const_cast<Dtype*>(output), top_offset_, 1, M_ * group_,
      N_, static_cast<Dtype*>(subTopMem), 0, opt_num2);

  for (int g = 0; g < group_; ++g) {
       caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_,
       N_ * opt_num2, 1., static_cast<Dtype*>(subTopMem),
       top_offset_opt * g, static_cast<Dtype*>(transMem),
       col_offset_ * g, 1., weights,
       weight_offset_ * g);
    }
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm_opt(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  if (is_1x1_) {
    int count = height_ * width_ * conv_in_channels_ * opt_num2;
    caffe_amp_copy<Dtype>(count, static_cast<void*>(input),
          static_cast<void*>(transMem), 0, 0);
  }
  for (int g = 0; g < group_; ++g) {
       caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_ * opt_num2, M_,
          (Dtype)1., const_cast<Dtype*>(weights),  weight_offset_ * g,
          static_cast<Dtype*>(subTopMem), top_offset_opt * g,
          (Dtype)0., static_cast<Dtype*>(transMem), col_offset_ * g);
      }

  if (!is_1x1_) {
      conv_col2im_gpu_opt(input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(int N1, int N2,
    const Dtype* output, const int offset_output,
    const Dtype* weights, Dtype* input, const int offset_input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();  int offset_col = 0;
  if (is_1x1_) {
    col_buff = input;
    offset_col = offset_input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., const_cast<Dtype*>(weights), weight_offset_ * g, const_cast<Dtype*>(output),
        offset_output+output_offset_ * g,
        (Dtype)0., col_buff, offset_col+col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input, offset_col, offset_input);
  }
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(int N1, int N2,
    const Dtype* input, const int offset_input, const Dtype* output,
    const int offset_output, Dtype* weights) {
  const Dtype* col_buff = input;
  int col_offset = offset_input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, offset_input,
        col_buffer_.mutable_gpu_data(), 0);
    col_buff = col_buffer_.gpu_data();
    col_offset = 0;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        conv_out_channels_ / group_, kernel_dim_ / group_,
        conv_out_spatial_dim_, (Dtype)1., const_cast<Dtype*>(output),
        offset_output+output_offset_ * g,
        const_cast<Dtype*>(col_buff) ,  col_offset+col_offset_ * g,
        (Dtype)1., weights , weight_offset_ * g);
  }
}
#else
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_emv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}
#endif  // USE_CPPAMP
#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
