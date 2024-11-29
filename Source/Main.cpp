#include <iostream>
#include <vector>
#include <CL/cl.h>

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

            std::cout << "GPU Found: " << device_name << std::endl;
        }
    }

    return 0;
}