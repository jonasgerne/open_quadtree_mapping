#pragma once
#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>
#include <quadmap/texture_memory.cuh>
#include <ctime>

namespace quadmap
{
//declear function
void generate_gradient(DeviceImage<float> &image, DeviceImage<float2> &gradient_map);
__global__ void gradient_kernel(DeviceImage<float> *image_dev_ptr, DeviceImage<float2> *gradient_dev_ptr);
}