#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#ifdef USE_CPPAMP
#define TEST_SIZE 12
#else
#define TEST_SIZE 10
#endif
namespace caffe {

class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization) {
  SyncedMemory mem(TEST_SIZE);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), TEST_SIZE);
  SyncedMemory* p_mem = new SyncedMemory(TEST_SIZE * sizeof(float));
  EXPECT_EQ(p_mem->size(), TEST_SIZE * sizeof(float));
  delete p_mem;
}
#ifndef CPU_ONLY  // GPU test

TEST_F(SyncedMemoryTest, TestAllocationCPUGPU) {
  SyncedMemory mem(TEST_SIZE);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.gpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
  EXPECT_TRUE(mem.mutable_gpu_data());
}

#endif

TEST_F(SyncedMemoryTest, TestAllocationCPU) {
  SyncedMemory mem(TEST_SIZE);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
}

#ifndef CPU_ONLY  // GPU test

TEST_F(SyncedMemoryTest, TestAllocationGPU) {
  SyncedMemory mem(TEST_SIZE);
  EXPECT_TRUE(mem.gpu_data());
  EXPECT_TRUE(mem.mutable_gpu_data());
}

#endif

TEST_F(SyncedMemoryTest, TestCPUWrite) {
  SyncedMemory mem(TEST_SIZE);
  void* cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  caffe_memset(mem.size(), 1, cpu_data);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 1);
  }
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  caffe_memset(mem.size(), 2, cpu_data);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 2);
  }
}

#ifndef CPU_ONLY  // GPU test

TEST_F(SyncedMemoryTest, TestGPURead) {
  SyncedMemory mem(TEST_SIZE);
  void* cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  caffe_memset(mem.size(), 1, cpu_data);
  const void* gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  // check if values are the same
  char* recovered_value = new char[TEST_SIZE];
#ifndef USE_CPPAMP
  caffe_gpu_memcpy(TEST_SIZE, gpu_data, recovered_value);
#else
  caffe_amp_D2H(const_cast<void*>(gpu_data),
      static_cast<void*>(recovered_value), sizeof(float), false);
#endif
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(recovered_value))[i], 1);
  }
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  caffe_memset(mem.size(), 2, cpu_data);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 2);
  }
  gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  // check if values are the same
#ifndef USE_CPPAMP
  caffe_gpu_memcpy(TEST_SIZE, gpu_data, recovered_value);
#else
  caffe_amp_D2H(const_cast<void*>(gpu_data),
      static_cast<void*>(recovered_value), sizeof(float), false);
#endif
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(recovered_value))[i], 2);
  }
  delete[] recovered_value;
}

TEST_F(SyncedMemoryTest, TestGPUWrite) {
  SyncedMemory mem(TEST_SIZE);
  void* gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
#ifndef USE_CPPAMP
  caffe_gpu_memset(mem.size(), 1, gpu_data);
#else
  int* temp = new int[TEST_SIZE/sizeof(int)];
  memset(temp, 1, TEST_SIZE);  // NOLINT(caffe/alt_fn)
  caffe_amp_H2D(static_cast<void*>(temp), gpu_data, sizeof(float), false);
#endif
  const void* cpu_data = mem.cpu_data();
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<const char*>(cpu_data))[i], 1);
  }
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);

  gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
#ifndef USE_CPPAMP
  caffe_gpu_memset(mem.size(), 2, gpu_data);
#else
  memset(temp, 2, TEST_SIZE);  // NOLINT(caffe/alt_fn)
  caffe_amp_H2D(static_cast<void*>(temp), gpu_data, sizeof(float), false);
  delete[] temp;
#endif
  cpu_data = mem.cpu_data();
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<const char*>(cpu_data))[i], 2);
  }
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
}

#endif

}  // namespace caffe
