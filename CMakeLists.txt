cmake_minimum_required(VERSION 3.10)
project(OpenCLGPUDetection)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Add executable
add_executable(opencl_gpu_detection Source/Main.cpp)

# Link OpenCL library
target_link_libraries(opencl_gpu_detection OpenCL::OpenCL)

# Set output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Binaries)

# Ensure the Binaries directory exists
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/Binaries)