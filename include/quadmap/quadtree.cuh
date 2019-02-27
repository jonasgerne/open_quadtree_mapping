#pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>
#include <quadmap/texture_memory.cuh>

namespace quadmap
{

//kernal
__global__ void quadtree_image_kernal(DeviceImage<int> *quadtree_devptr);
__global__ void quadtree_depth_kernal(DeviceImage<float> *prior_depth_devptr, DeviceImage<int> *quadtree_devptr);

}//namespace
