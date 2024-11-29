#include "GPUContext.h"

#include <iostream>
#include <thread>

GPUContext::GPUContext(cl_platform_id platform, cl_device_id device)
    : platform_(platform), device_(device), running_(true), voxelArray_(nullptr) {
    cl_int err;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context" << std::endl;
        throw std::runtime_error("Context creation failed");
    }

    const cl_queue_properties properties[] = {0};
    queue_ = clCreateCommandQueueWithProperties(context_, device_, properties, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue" << std::endl;
        clReleaseContext(context_);
        throw std::runtime_error("Command queue creation failed");
    }
}

GPUContext::~GPUContext() {
    if (voxelArray_) {
        delete voxelArray_;
    }
    clReleaseCommandQueue(queue_);
    clReleaseContext(context_);
}

void GPUContext::terminate() {
    running_ = false;
}

void GPUContext::waitUntilDone() {
    while (!taskQueue_.empty()) {
        std::this_thread::yield();
    }
}

void GPUContext::enqueueTask(size_t width, size_t height, size_t depth) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    taskQueue_.emplace(width, height, depth);
    queueCondVar_.notify_one();
}

void GPUContext::run() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueCondVar_.wait(lock, [this] { return !taskQueue_.empty() || !running_; });

        if (!running_) {
            break;
        }

        auto [width, height, depth] = taskQueue_.front();
        taskQueue_.pop();
        lock.unlock();

        processTask(width, height, depth);
    }
}

void GPUContext::processTask(size_t width, size_t height, size_t depth) {
    if (!voxelArray_) {
        voxelArray_ = new VoxelArray(context_, device_, queue_, width, height, depth);
    }
    voxelArray_->process();
}