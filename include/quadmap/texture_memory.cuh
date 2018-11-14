#pragma once

#include <cuda_runtime.h>
#include <quadmap/device_image.cuh>
#include <iostream>

namespace quadmap
{
//for image
texture<float, cudaTextureType2D, cudaReadModeElementType> income_image_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> pre_image_tex;
texture<float2, cudaTextureType2D, cudaReadModeElementType> income_gradient_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> keyframe_image_tex;
texture<float2, cudaTextureType2D, cudaReadModeElementType> keyframe_gradient_tex;
texture<int, cudaTextureType2D, cudaReadModeElementType> quadtree_tex;


texture<float, cudaTextureType2D, cudaReadModeElementType> cur_img_tex;
texture<float2, cudaTextureType2D, cudaReadModeElementType> cur_sober_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_l1_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_l2_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> last_img_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> gradient_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> gradient_angle_tex;
texture<int, cudaTextureType2D, cudaReadModeElementType> reference_table_tex;

//for key frame
texture<float, cudaTextureType2D, cudaReadModeElementType> keyframe_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> g_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;

// Pre-computed template statistics
texture<float, cudaTextureType2D, cudaReadModeElementType> sum_templ_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> const_templ_denom_tex;

template<typename ElementType>
inline void bindTexture(
    texture<ElementType, cudaTextureType2D> &tex,
    const DeviceImage<ElementType> &mem,
    cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;

  const cudaError bindStatus = cudaBindTexture2D(
        0,
        tex,
        mem.data,
        mem.getCudaChannelFormatDesc(),
        mem.width,
        mem.height,
        mem.pitch
        );

  //cudaArray* myArray; // declar.
  //                    // ask for memory
  //cudaChannelFormatDesc desc = mem.getCudaChannelFormatDesc();
  //cudaMallocArray(&myArray,
  //    &desc,
  //    mem.width,
  //    mem.height);

  //cudaMemcpyToArray(myArray, // destination: the array
  //    0, 0, // offsets 
  //    mem.data, // pointer uint*
  //    mem.width*mem.height*sizeof(ElementType), // total amount of bytes to be copied 
  //    cudaMemcpyDeviceToDevice);

  //const cudaError bindStatus = cudaBindTextureToArray(tex, myArray, desc);

  //ElementType* debug = new ElementType[mem.width*mem.height * sizeof(ElementType)];
  //cudaMemcpyFromArray(debug, myArray, 0, 0, mem.width*mem.height * sizeof(ElementType), cudaMemcpyDeviceToHost);

  if(bindStatus != cudaSuccess)
  {
    throw CudaException("Unable to bind texture: ", bindStatus);
  }
}

}