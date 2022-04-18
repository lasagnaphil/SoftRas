cmake_minimum_required(VERSION 3.14)
project(SoftRas LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--copy-dt-needed-entries") # For solving libkineto.so -> libcupti.so linking error

file(GLOB SOFTRAS_CUDA_SRCS soft_renderer/cuda/*.cpp soft_renderer/cuda/*.cu)
add_library(SoftRas PUBLIC ${SOFTRAS_CUDA_SRCS})
target_include_directories(SoftRas PUBLIC soft_renderer/cuda)
