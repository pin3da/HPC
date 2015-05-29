cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_30,code=sm_30)
cuda_add_executable(ntfft nfft.cu)
