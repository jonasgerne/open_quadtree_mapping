#pragma once
#include <quadmap/device_image.cuh>
#include <quadmap/stereo_parameter.cuh>
#include <quadmap/DepthSeed.cuh>
#include <quadmap/se3.cuh>
#include <quadmap/texture_memory.cuh>
#include <ctime>

//for camera model
#define fx(camera_para) camera_para.x
#define cx(camera_para) camera_para.y
#define fy(camera_para) camera_para.z
#define cy(camera_para) camera_para.w

namespace quadmap
{
__global__ void initialize_keyframe_kernel(
	DeviceImage<DepthSeed> *new_keyframe_devptr);
__global__ void initialize_keyframe_kernel(
	DeviceImage<DepthSeed> *new_keyframe_devptr,
	DeviceImage<int> *transtable_devptr,
	DeviceImage<float3> *new_info_devptr);
__global__ void propogate_keyframe_kernel(
	DeviceImage<DepthSeed> *old_keyframe_devptr,
	float4 camera_para, SE3<float> old_to_new,
	DeviceImage<int> *transtable_devptr,
	DeviceImage<float3> *new_info_devptr);
__global__ void regulizeDepth_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	bool removeOcclusions);
__global__ void regulizeDepth_FillHoles_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr);
__global__ void update_keyframe_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float> *debug_devptr,
    DeviceImage<float4> *epipolar_devptr,
    float min_inv_depth,
    float max_inv_depth,
    bool fixNearPoint);
__global__ void depth_project_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float> *depth);
__global__ void depth_project_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float2> *depth);
__device__ __forceinline__ float search_point(
  	const int &x,
  	const int &y,
  	const int &width,
  	const int &height,
  	const float2 &epipolar_line,
  	const float &gradient_max,
  	const float &my_gradient,
  	const float &min_idep,
  	const float &max_idep,
  	const float4 &camera_para,
  	const SE3<float> &key_to_income,
  	float &result_idep,
  	float &result_var,
  	float &result_eplength,
    float4& result_epipolar,
    bool fixNearPoint);
__device__ __forceinline__ float subpixle_interpolate(
	const float &pre_cost,
	const float &cost,
	const float &post_cost);
}