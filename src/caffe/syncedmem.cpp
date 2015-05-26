#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifndef USE_CPPAMP
SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
#endif
}



#else
SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

}

inline void SyncedMemory::to_cpu() {
  //std::cerr<<"to_cpu..........\n";
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    //std::cerr<<"uninitialized........\n";
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    //std::cerr<<"at gpu........\n";
    break;
  case HEAD_AT_CPU:

    //std::cerr<<"at cpu........\n";
  case SYNCED:

    //std::cerr<<"synced........\n";
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  //std::cerr<<"to_gpu..........\n";


#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_GPU;
    //std::cerr<<"uninitialized........\n";
    break;
  case HEAD_AT_CPU:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
    }
    head_ = SYNCED;

    //std::cerr<<"at cpu........\n";
    break;
  case HEAD_AT_GPU:

    //std::cerr<<"at gpu........\n";
  case SYNCED:

    //std::cerr<<"at synced........\n";
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  //std::cerr<<"cpu_data..........\n";

  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  //std::cerr<<"set_cpu_data..........\n";
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {

  //std::cerr<<"gpu_data..........\n";
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)cpu_ptr_;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  //std::cerr<<"mutable_cpu_data..........\n";
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
 //std::cerr<<"mutable_gpu_data..........\n";
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return cpu_ptr_;
#else
  NO_GPU;
#endif
}


#endif



}  // namespace caffe

