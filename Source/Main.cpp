#include <iostream>
#include <vector>
#include <CL/cl.h>

#include "VoxelArray.h"

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

            // Create a context for the device
            cl_int err;
            cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create context for device: " << device_name << std::endl;
                continue;
            }

            // Create a command queue with properties
            const cl_queue_properties properties[] = {0};
            cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create command queue for device: " << device_name << std::endl;
                clReleaseContext(context);
                continue;
            }

            // Create a VoxelArray instance
            size_t width = 512;
            size_t height = 512;
            size_t depth = 512;
            VoxelArray voxelArray(context, device, queue, width, height, depth);

            // Process the voxel array
            voxelArray.process();

            std::cout << "  Voxel array processed on device: " << device_name << std::endl;

            // Release the command queue
            clReleaseCommandQueue(queue);

            // Release the context
            clReleaseContext(context);
        }
    }

    return 0;
}