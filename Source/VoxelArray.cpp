#include "VoxelArray.h"
#include <iostream>
#include <fstream>
#include <sstream>

VoxelArray::VoxelArray(cl_context context, cl_device_id device, cl_command_queue queue, size_t width, size_t height, size_t depth)
    : context_(context), device_(device), queue_(queue), width_(width), height_(height), depth_(depth) {
    createBuffer();
    createProgram();
    createKernel();
    setKernelArgs();

    // Print Stats
    char device_name[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "Created VoxelArray on GPU " << device_name << std::endl;


}

VoxelArray::~VoxelArray() {
    clReleaseMemObject(buffer_);
    clReleaseKernel(kernel_);
    clReleaseProgram(program_);
}

void VoxelArray::createBuffer() {
    cl_int err;
    size_t size = width_ * height_ * depth_ * sizeof(int);
    buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffer" << std::endl;
        throw std::runtime_error("Buffer creation failed");
    }
}

void VoxelArray::createProgram() {
    std::ifstream kernelFile("Kernels/Kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file" << std::endl;
        throw std::runtime_error("Kernel file open failed");
    }

    std::stringstream buffer;
    buffer << kernelFile.rdbuf();
    std::string kernelSource = buffer.str();

    // std::cout << "Kernel source content:" << std::endl;
    // std::cout << kernelSource << std::endl;

    const char* source = kernelSource.c_str();
    size_t sourceSize = kernelSource.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &source, &sourceSize, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program" << std::endl;
        throw std::runtime_error("Program creation failed");
    }

    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program" << std::endl;
        size_t logSize;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "Build log: " << buildLog.data() << std::endl;
        throw std::runtime_error("Program build failed");
    }
}

void VoxelArray::createKernel() {
    cl_int err;
    kernel_ = clCreateKernel(program_, "increment_voxels", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel. Error code: " << err << std::endl;
        switch (err) {
            case CL_INVALID_PROGRAM:
                std::cerr << "CL_INVALID_PROGRAM: The program is not a valid OpenCL program object." << std::endl;
                break;
            case CL_INVALID_PROGRAM_EXECUTABLE:
                std::cerr << "CL_INVALID_PROGRAM_EXECUTABLE: There is no successfully built executable for program." << std::endl;
                break;
            case CL_INVALID_KERNEL_NAME:
                std::cerr << "CL_INVALID_KERNEL_NAME: The specified kernel name is not found in the program." << std::endl;
                break;
            case CL_INVALID_KERNEL_DEFINITION:
                std::cerr << "CL_INVALID_KERNEL_DEFINITION: The function definition for the kernel is not valid." << std::endl;
                break;
            case CL_INVALID_VALUE:
                std::cerr << "CL_INVALID_VALUE: An invalid value was passed." << std::endl;
                break;
            case CL_OUT_OF_RESOURCES:
                std::cerr << "CL_OUT_OF_RESOURCES: There is a failure to allocate resources required by the OpenCL implementation on the device." << std::endl;
                break;
            case CL_OUT_OF_HOST_MEMORY:
                std::cerr << "CL_OUT_OF_HOST_MEMORY: There is a failure to allocate resources required by the OpenCL implementation on the host." << std::endl;
                break;
            default:
                std::cerr << "Unknown error." << std::endl;
                break;
        }
        throw std::runtime_error("Kernel creation failed");
    }
}
void VoxelArray::setKernelArgs() {
    cl_int err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer_);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel argument 0. Error code: " << err << std::endl;
        throw std::runtime_error("Kernel argument setting failed");
    }

    int width = static_cast<int>(width_);
    int height = static_cast<int>(height_);
    int depth = static_cast<int>(depth_);

    err = clSetKernelArg(kernel_, 1, sizeof(int), &width);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel argument 1. Error code: " << err << std::endl;
        throw std::runtime_error("Kernel argument setting failed");
    }

    err = clSetKernelArg(kernel_, 2, sizeof(int), &height);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel argument 2. Error code: " << err << std::endl;
        throw std::runtime_error("Kernel argument setting failed");
    }

    err = clSetKernelArg(kernel_, 3, sizeof(int), &depth);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel argument 3. Error code: " << err << std::endl;
        throw std::runtime_error("Kernel argument setting failed");
    }
}

void VoxelArray::enqueueKernel() {
    size_t globalWorkSize[3] = { width_, height_, depth_ };
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel_, 3, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel" << std::endl;
        throw std::runtime_error("Kernel enqueue failed");
    }
    clFinish(queue_);
}

void VoxelArray::process() {
    enqueueKernel();
}