#include <iostream>
#include <vector>
#include <thread>
#include <CL/cl.h>

#include "GPUContext.h"

int main() {
    // Get the number of platforms
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);

    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return -1;
    }

    // Get the platform IDs
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    // Vector to hold GPUContext instances and threads
    std::vector<std::unique_ptr<GPUContext>> gpuContexts;
    std::vector<std::thread> threads;

    // Iterate over each platform
    for (cl_platform_id platform : platforms) {
        // Get the number of devices
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

        if (num_devices == 0) {
            std::cout << "No GPUs found on this platform." << std::endl;
            continue;
        }

        // Get the device IDs
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        // Iterate over each device
        for (cl_device_id device : devices) {
            // Get device name
            char device_name[1024];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);

            // Get global memory size (VRAM)
            cl_ulong global_mem_size;
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);

            std::cout << "GPU Found: " << device_name << std::endl;
            std::cout << "  VRAM: " << global_mem_size / (1024 * 1024) << " MB" << std::endl;

            // Create a GPUContext instance
            auto gpuContext = std::make_unique<GPUContext>(platform, device);
            gpuContexts.push_back(std::move(gpuContext));

            // Create a thread for the GPUContext
            threads.emplace_back(&GPUContext::run, gpuContexts.back().get());
        }
    }

    // Enqueue tasks for each GPUContext
    for (auto& gpuContext : gpuContexts) {
        for (int i = 0; i < 50; i++) {
            gpuContext->enqueueTask(512, 512, 512);
        }
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}