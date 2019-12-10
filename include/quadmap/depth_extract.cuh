#pragma once
#include <quadmap/device_image.cuh>
#include <quadmap/stereo_parameter.cuh>
#include <quadmap/pixel_cost.cuh>
#include <quadmap/match_parameter.cuh>
#include <quadmap/texture_memory.cuh>
#include <ctime>

namespace quadmap
{
__global__ void prior_to_cost(
	bool inverse_depth,
	float min_depth,
	float step_depth,
	DeviceImage<float2> *depth_prior_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr);
__global__ void image_to_cost(
	bool inverse_depth,
	float min_depth,
	float step_depth,
	MatchParameter *match_parameter_devptr,
	DeviceImage<int> *age_table_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr);
__global__ void normalize_the_cost(
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr);
__global__ void upsample_naive(
	DeviceImage<float> *coarse_depth_devptr,
	DeviceImage<float> *full_dense_devptr);
}