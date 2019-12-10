#pragma once
#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>
#include <quadmap/texture_memory.cuh>
#include <quadmap/match_parameter.cuh>
#include <quadmap/pixel_cost.cuh>
#include <quadmap/stereo_parameter.cuh>
#include <ctime>

namespace quadmap
{
//function declear here!
void global_upsample(DeviceImage<float> &sparse_depth, DeviceImage<float> &depth);
void local_upsample(DeviceImage<float> &sparse_image, DeviceImage<float> &dense_image);
__global__ void build_weight_row(DeviceImage<float> *row_weight_devptr);
__global__ void build_weight_col(DeviceImage<float> *col_weight_devptr);
__global__ void smooth_row(DeviceImage<float> *sparse_devptr, DeviceImage<float> *row_weight_devptr, DeviceImage<float3> *temp_devptr, DeviceImage<float2> *smooth_devptr);
__global__ void smooth_col(DeviceImage<float2> *row_smooth_devptr, DeviceImage<float> *col_weight_devptr, DeviceImage<float3> *temp_devptr, DeviceImage<float> *smooth_devptr);
__global__ void depth_interpolate(  DeviceImage<float> *featuredepth_devptr,
                                    DeviceImage<float> *depth_devptr);
}