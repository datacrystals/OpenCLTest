#pragma once

#include <CL/cl.h>
#include <vector>

class VoxelArray {
public:
    VoxelArray(cl_context context, cl_device_id device, cl_command_queue queue, size_t width, size_t height, size_t depth);
    ~VoxelArray();

    void process();

private:
    cl_context context_;
    cl_device_id device_;
    cl_command_queue queue_;
    size_t width_;
    size_t height_;
    size_t depth_;
    cl_mem buffer_;
    cl_program program_;
    cl_kernel kernel_;

    void createBuffer();
    void createProgram();
    void createKernel();
    void setKernelArgs();
    void enqueueKernel();
};
