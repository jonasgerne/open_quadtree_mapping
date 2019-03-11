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
void bp_extract(
            int cost_downsampling,
            bool inverse_depth,
            float min_depth,
            float step_depth,
            DeviceImage<PIXEL_COST> &image_cost_map,
            DeviceImage<float> &depth,
            float P1,
            float P2);
__global__ void cost_distribute(
    DeviceImage<PIXEL_COST> *l0_cost_devptr,
    DeviceImage<PIXEL_COST> *l1_cost_devptr);
__global__ void bp(
    DeviceImage<PIXEL_COST> *data_devptr,
    DeviceImage<PIXEL_COST> *lm_devptr,
    DeviceImage<PIXEL_COST> *rm_devptr,
    DeviceImage<PIXEL_COST> *up_devptr,
    DeviceImage<PIXEL_COST> *dm_devptr,
    bool A_set,
    int i_leverl,
    float P1, float P2);
__global__ void upsample(
    DeviceImage<PIXEL_COST> *l1_message_devptr,
    DeviceImage<PIXEL_COST> *l0_message_devptr);
__global__ void depth_extract(
    int cost_downsampling,
    float min_depth,
    float step_min_depth,
    DeviceImage<PIXEL_COST> *data_devptr,
    DeviceImage<PIXEL_COST> *lm_devptr,
    DeviceImage<PIXEL_COST> *rm_devptr,
    DeviceImage<PIXEL_COST> *up_devptr,
    DeviceImage<PIXEL_COST> *dm_devptr,
    DeviceImage<float> *extracted_depth_devptr);
}