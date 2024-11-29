#pragma once

#include <CL/cl.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "VoxelArray.h"

class GPUContext {
public:
    GPUContext(cl_platform_id platform, cl_device_id device);
    ~GPUContext();

    void enqueueTask(size_t width, size_t height, size_t depth);
    void run();

private:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    std::queue<std::tuple<size_t, size_t, size_t>> taskQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCondVar_;
    bool running_;

    void processTask(size_t width, size_t height, size_t depth);
};
