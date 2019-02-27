#pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>
#include <quadmap/texture_memory.cuh>
#include <quadmap/match_parameter.cuh>
#include <quadmap/camera_model/pinhole_camera.cuh>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

namespace quadmap
{

#define initial_a 10 // 10
#define initial_b 10 // 10
#define initial_variance 2500 // 2500
#define MIN_INLIER_RATIO_GOOD 0.6 // 0.6
#define MIN_INLIER_RATIO_BAD 0.45

__device__ __forceinline__ float normpdf(const float &x, const float &mu, const float &sigma_sq)
{
  return (expf(-(x-mu)*(x-mu) / (2.0f*sigma_sq))) * rsqrtf(2.0f*M_PI*sigma_sq);
}
__device__ __forceinline__ bool is_goodpoint(const float4 &point_info)
{
  return (point_info.x /(point_info.x + point_info.y) > MIN_INLIER_RATIO_GOOD);
}
__device__ __forceinline__ bool is_badpoint(const float4 &point_info)
{
  return (point_info.x < 0.001) || (point_info.x /(point_info.x + point_info.y) < MIN_INLIER_RATIO_BAD);
}

__global__ void high_gradient_filter
(DeviceImage<float> *depth_output_devptr,
    DeviceImage<float> *filtered_depth_devptr);

__global__ void fuse_transform(
    DeviceImage<float4> *pre_seeds_devptr,
    DeviceImage<int> *transform_table_devptr,
    SE3<float> last_to_cur,
    PinholeCamera camera);

__global__ void hole_filling(DeviceImage<int> *transform_table_devptr);

__global__ void fuse_currentmap(
    DeviceImage<int> *transform_table_devptr,
    DeviceImage<float> *depth_output_devptr,
    DeviceImage<float4> *former_depth_devptr,
    DeviceImage<float4> *new_depth_devptr);
}//namespace