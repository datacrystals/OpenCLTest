__kernel void increment_voxels(__global int* voxels, int width, int height, int depth) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if (x < width && y < height && z < depth) {
        int index = z * width * height + y * width + x;

        // Perform a computationally intensive operation
        int result = 0;
        for (int i = 0; i < 1000; i++) {
            result += voxels[index] * i;
            result -= voxels[index] / (i + 1);
            result *= voxels[index] + i;
            result /= voxels[index] - i;
        }

        voxels[index] += result;
    }
}