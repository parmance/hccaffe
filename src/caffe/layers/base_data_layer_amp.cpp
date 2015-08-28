#include <boost/type_traits/is_same.hpp>
#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {
#ifdef USE_CPPAMP
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_amp_H2D((void*)prefetch_data_.cpu_data(),
    (void*)top[0]->mutable_gpu_data(),
      sizeof(Dtype), boost::is_same<Dtype, int>::value);
  if (this->output_labels_) {
    caffe_amp_H2D((void*)prefetch_label_.cpu_data(),
        (void*)top[1]->mutable_gpu_data(),
        sizeof(Dtype),
        boost::is_same<Dtype, int>::value);
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
#endif
}  // namespace caffe
