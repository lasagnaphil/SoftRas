cmake_minimum_required(VERSION 3.14)

project(nvdiffrast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

file(GLOB SOFTRAS_CUDA_FILES soft_renderer/cuda/*.cpp soft_renderer/cuda/*.cu cpp/src/*.cpp)

add_library(SoftRas STATIC ${SOFTRAS_CUDA_FILES})
target_include_directories(SoftRas PUBLIC cpp/include)
target_link_libraries(SoftRas PUBLIC ${TORCH_LIBRARIES})
target_compile_definitions(SoftRas PUBLIC SOFTRAS_IGNORE_PYBIND)