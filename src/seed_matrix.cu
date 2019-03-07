#include <quadmap/seed_matrix.cuh>
#include <quadmap/texture_memory.cuh>
// #include <future>
// #include <boost/thread.hpp>

#include "quadmap/seed_init.cuh"
#include "quadmap/lsd_semidense.cuh"
#include "quadmap/quadtree.cuh"
#include "quadmap/depth_extract.cuh"
#include "quadmap/bpextract.cuh"
#include "quadmap/depth_upsample.cuh"
#include "quadmap/depth_fusion.cuh"


#include "seed_init.cu"
// #include "semi_map_update.cu"
#include "lsd_semidense.cu"
// #include "debug_draw.cu"
#include "quadtree.cu"
#include "depth_extract.cu"
#include "bpextract.cu"
#include "depth_upsample.cu"
#include "depth_fusion.cu"

//#define DEBUG_PRINT

quadmap::SeedMatrix::SeedMatrix(
    const size_t &_width,
    const size_t &_height,
    const size_t& _cost_downsampling,
    const PinholeCamera &cam,
    bool doBeliefPropagation,
    bool useQuadtree,
    bool doFusion,
    bool doGlobalUpsampling,
    bool printTimings,
    float P1, float P2,
    float new_keyframe_max_angle,
    float new_keyframe_max_distance,
    float new_reference_max_angle,
    float new_reference_max_distance,
    float min_inlier_ratio_good,
    float min_inlier_ratio_bad)
  : width(_width)
  , height(_height)
  , cost_downsampling(_cost_downsampling)
  , camera(cam)
  , depth_output(_width, _height)
  , debug_image(_width, _height)
  , epipolar_image(_width, _height)
  //income
  , income_image(_width, _height)
  , pre_income_image(_width, _height)
  , income_gradient(_width, _height)
  //copy lsd
  , keyframe_semidense(_width, _height)
  , keyframe_image(_width, _height)
  , keyframe_gradient(_width, _height)
  , semidense_on_income(_width, _height)
  , pixel_age_table(_width, _height)
  , depth_fuse_seeds(_width, _height)
  , initialized(false)
  , doBeliefPropagation(doBeliefPropagation)
  , useQuadtree(useQuadtree)
  , doFusion(doFusion)
  , doGlobalUpsampling(doGlobalUpsampling)
  , printTimings(printTimings)
  , P1(P1)
  , P2(P2)
  , new_keyframe_max_angle(new_keyframe_max_angle)
  , new_keyframe_max_distance(new_keyframe_max_distance)
  , new_reference_max_angle(new_reference_max_angle)
  , new_reference_max_distance(new_reference_max_distance)
  , min_inlier_ratio_good(min_inlier_ratio_good)
  , min_inlier_ratio_bad(min_inlier_ratio_bad)
  , frame_index(0)
{
  cv_output.create(height, width, CV_32FC1);
  cv_debug.create(height, width, CV_32FC1);
  cv_epipolar.create(height, width, CV_32FC4);

  match_parameter.cost_downsampling = cost_downsampling;
  match_parameter.camera_model = camera;
  match_parameter.setDevData();

  keyframe_semidense.zero();
  pixel_age_table.zero();
  depth_fuse_seeds.zero();

  depth_output.zero();
  debug_image.zero();

  //for cpu async
  cudaStreamCreate(&swict_semidense_stream1);
  cudaStreamCreate(&swict_semidense_stream2);
  cudaStreamCreate(&swict_semidense_stream3);
  semidense_hostptr = (DepthSeed*) malloc(width*height*sizeof(DepthSeed));
  semidense_new_hostptr = (DepthSeed*) malloc(width*height*sizeof(DepthSeed));
  income_gradient_hostptr = (float2*) malloc(width*height*sizeof(float2));
}

quadmap::SeedMatrix::~SeedMatrix()
{
  for(int i = 0; i < framelist_host.size(); i++)
  {
    delete framelist_host[i].frame_ptr;
  }
  cudaStreamDestroy(swict_semidense_stream1);
  cudaStreamDestroy(swict_semidense_stream2);
  cudaStreamDestroy(swict_semidense_stream3);
  free(semidense_hostptr);
  free(semidense_new_hostptr);
  free(income_gradient_hostptr);
}

void quadmap::SeedMatrix::set_remap(cv::Mat _remap_1, cv::Mat _remap_2)
{
  remap_1 = _remap_1;
  remap_2 = _remap_2;
#ifdef DEBUG_PRINT
  printf("has success set cuda remap.\n");
#endif
}
bool quadmap::SeedMatrix::input_raw(cv::Mat raw_mat, const SE3<float> T_curr_world)
{
  std::clock_t start = std::clock();  
  input_image = raw_mat;
  cv::remap(input_image, undistorted_image, remap_1, remap_2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  income_undistort = undistorted_image;
  //undistorted_image.convertTo(input_float, CV_32F, 1.0f/255.0f);
  undistorted_image.convertTo(input_float, CV_32F);
#ifdef DEBUG_PRINT
  printf("cuda prepare the image cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
#endif
  return add_frames(input_float, T_curr_world);
}
void quadmap::SeedMatrix::add_income_image(const cv::Mat &input_image, const SE3<float> T_world)
{
  income_image.setDevData(reinterpret_cast<float *>(input_image.data));
  income_transform = T_world;
  generate_gradient(income_image, income_gradient);

  /*cv::Mat grad(input_image.rows, input_image.cols, CV_32FC2);
  income_gradient.getDevData((float2*)grad.data);
  cv::Mat channels[2];
  cv::split(grad, channels);
  cv::Mat norm, color;
  cv::normalize(channels[0], norm, 0, 255, CV_MINMAX, CV_8U);
  cv::applyColorMap(norm, color, cv::COLORMAP_JET);
  cv::imshow("Gradient X", color);
  cv::normalize(channels[1], norm, 0, 255, CV_MINMAX, CV_8U);
  cv::applyColorMap(norm, color, cv::COLORMAP_JET);
  cv::imshow("Gradient Y", color);
  cv::waitKey();*/
}

void quadmap::SeedMatrix::set_income_as_keyframe()
{
  keyframe_image = income_image;
  keyframe_gradient = income_gradient;
  keyframe_transform = income_transform;
  keyframeMat = income_undistort.clone();
}

bool quadmap::SeedMatrix::add_frames(
    cv::Mat &input_image,
    const SE3<float> T_curr_world)
{
  std::clock_t start = std::clock();
  frame_index++;
  pre_income_image = income_image;
  depth_output.zero();
  debug_image.zero();
  semidense_on_income.zero();
  
  //add to the list
  // Compute gradient
  add_income_image(input_image, T_curr_world);

#ifdef DEBUG_PRINT
  printf("till add image cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000); start = std::clock();
#endif

  //for semi-dense update and project
  if(!initialized)
  {
    set_income_as_keyframe();
    // Random depth initialization
    initial_keyframe();
    initialized = true;
    return false;
  }

  // Epipolar depth search and update
  // Project depth from keyframe to frame
  update_keyframe();

  if(need_switchkeyframe())
  {
    // async switch keyframe
    // boost::thread t(&quadmap::SeedMatrix::create_new_keyframe_async,this);
    // t.detach();

    // Propagate depth from last keyframe to new keyframe
    // Initialize LSD seeds
    // Do regularization and hole filling
    create_new_keyframe();
    set_income_as_keyframe();
  }

#ifdef DEBUG_PRINT
  printf("till all semidense cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000); start = std::clock();
#endif

  // If not a dense frame, return update with semidense depth
  if (frame_index % semi2dense_ratio != 0) {
      //return false;
      download_output();
      return true;
  }

  //for full dense
  bool has_depth_output = false;
  if(framelist_host.size() > 1)
  {
    // Compute depth from multi-view-stereo using quadtree and belief propagation
    extract_depth();
    has_depth_output = true;
  }

#ifdef DEBUG_PRINT
  printf("till all full dense cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000); start = std::clock();
#endif

  // If min baseline and angle add the current frame into framelist for mvs
  if(need_add_reference())
    add_reference();

#ifdef DEBUG_PRINT
  printf("till all end cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000); start = std::clock();
#endif

  if(has_depth_output)
  {
    // Fuse
    if(doFusion)
      fuse_output_depth();
    download_output();
  }
  return has_depth_output;
}

bool quadmap::SeedMatrix::need_add_reference()
{
  if(framelist_host.size() == 0)
    return true;

  SE3<float> lastframe_pose = framelist_host.front().transform.inv();
  SE3<float> income_pose = income_transform.inv();
  float3 last_z = make_float3(lastframe_pose.data(0,2),lastframe_pose.data(1,2),lastframe_pose.data(2,2));
  float3 income_z = make_float3(income_pose.data(0,2),income_pose.data(1,2),income_pose.data(2,2));
  float z_cos = dot(last_z, income_z);
  float base_line = length(lastframe_pose.getTranslation()-income_pose.getTranslation());
  return (z_cos < new_reference_max_angle || base_line > new_reference_max_distance);
}

void quadmap::SeedMatrix::add_reference()
{
  printf("Adding reference\n");
  FrameElement newEle;
  newEle.frame_ptr = new DeviceImage<float>(width,height);
  newEle.transform = income_transform;
  *newEle.frame_ptr = income_image;
  framelist_host.push_front(newEle);
  if(framelist_host.size() > KEYFRAME_NUM)
  {
    FrameElement toDelete = framelist_host.back();
    delete toDelete.frame_ptr;
    framelist_host.pop_back();
  }
}

bool quadmap::SeedMatrix::need_switchkeyframe()
{
  SE3<float> keyframe_pose = keyframe_transform.inv();
  SE3<float> income_pose = income_transform.inv();
  float3 keyframe_z = make_float3(keyframe_pose.data(0,2),keyframe_pose.data(1,2),keyframe_pose.data(2,2));
  float3 income_z = make_float3(income_pose.data(0,2),income_pose.data(1,2),income_pose.data(2,2));
  float z_cos = dot(keyframe_z, income_z);
  float base_line = length(keyframe_pose.getTranslation()-income_pose.getTranslation());
  return (z_cos < new_keyframe_max_angle || base_line > new_keyframe_max_distance);
}

void quadmap::SeedMatrix::initial_keyframe()
{
  std::clock_t start = std::clock();

  bindTexture(income_image_tex, income_image);
  bindTexture(income_gradient_tex, income_gradient);
  bindTexture(keyframe_image_tex, keyframe_image);
  bindTexture(keyframe_gradient_tex, keyframe_gradient);

  dim3 image_block;
  dim3 image_grid;
  image_block.x = 8;
  image_block.y = 8;
  image_grid.x = (width + image_block.x - 1) / image_block.x;
  image_grid.y = (height + image_block.y - 1) / image_block.y;
  // Random initialization (LSD-SLAM)
  initialize_keyframe_kernel<<<image_grid, image_block>>>(keyframe_semidense.dev_ptr);
  cudaDeviceSynchronize();

  float4 camera_para = make_float4(camera.fx,camera.cx,camera.fy,camera.cy);
  SE3<float> identity = income_transform * income_transform.inv();

  // Project depth onto itself ???
  depth_project_kernel<<<image_grid, image_block>>>(
  keyframe_semidense.dev_ptr,
  camera_para,
  identity,
  depth_output.dev_ptr);
  cudaDeviceSynchronize();

  cudaUnbindTexture(income_image_tex);
  cudaUnbindTexture(income_gradient_tex);
  cudaUnbindTexture(keyframe_image_tex);
  cudaUnbindTexture(keyframe_gradient_tex);
#ifdef DEBUG_PRINT
  printf("initialize keyframe cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
#endif
}

void quadmap::SeedMatrix::create_new_keyframe()
{
  printf("Adding keyframe\n");
  std::clock_t start = std::clock();

  bindTexture(income_image_tex, income_image);
  bindTexture(income_gradient_tex, income_gradient);
  bindTexture(keyframe_image_tex, keyframe_image);
  bindTexture(keyframe_gradient_tex, keyframe_gradient);

  dim3 image_block;
  dim3 image_grid;
  image_block.x = 8;
  image_block.y = 8;
  image_grid.x = (width + image_block.x - 1) / image_block.x;
  image_grid.y = (height + image_block.y - 1) / image_block.y;

  float4 camera_para = make_float4(camera.fx,camera.cx,camera.fy,camera.cy);
  SE3<float> old_to_new = income_transform * keyframe_transform.inv();
  DeviceImage<int> transform_table(width,height);
  DeviceImage<DepthSeed> new_keyframe(width,height);
  DeviceImage<float3> new_info(width,height);

  new_keyframe.zero();
  transform_table.zero();
  //first propagte the keyframe to new keyframe
  propogate_keyframe_kernel<<<image_grid, image_block>>>(
    keyframe_semidense.dev_ptr,
    camera_para,
    old_to_new,
    transform_table.dev_ptr,
    new_info.dev_ptr);
  cudaDeviceSynchronize();

  // LSD SLAM initialization
  // Intialize mean,var for all pixels with min gradient
  initialize_keyframe_kernel<<<image_grid, image_block>>>(
    new_keyframe.dev_ptr,
    transform_table.dev_ptr,
    new_info.dev_ptr);
  cudaDeviceSynchronize();

  // Regularization from LSD SLAM
  // Check if pixel has min num support pixels in 5x5 neighborhood
  // Smooth over valid neighbors (within depth variance range)
  regulizeDepth_kernel<<<image_grid, image_block>>>(new_keyframe.dev_ptr, true);
  cudaDeviceSynchronize();

  // Hole filling from LSD SLAM
  // If enough valid measurements in 5x5 neighboorhood fill invalid pixel with variance weighted average
  regulizeDepth_FillHoles_kernel<<<image_grid, image_block>>>(new_keyframe.dev_ptr);
  cudaDeviceSynchronize();

  // Another round of regularization
  regulizeDepth_kernel<<<image_grid, image_block>>>(new_keyframe.dev_ptr, false);
  cudaDeviceSynchronize();

  keyframe_semidense = new_keyframe;

  cudaUnbindTexture(income_image_tex);
  cudaUnbindTexture(income_gradient_tex);
  cudaUnbindTexture(keyframe_image_tex);
  cudaUnbindTexture(keyframe_gradient_tex);

  // printf("propagate keyframe cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}

void quadmap::SeedMatrix::create_new_keyframe_async()
{
  std::clock_t start = std::clock();

  //down load the map on cpu
  keyframe_semidense.getDevDataAsync(semidense_hostptr, swict_semidense_stream1);
  income_gradient.getDevDataAsync(income_gradient_hostptr, swict_semidense_stream2);
  memset(semidense_new_hostptr, 0, width * height * sizeof(DepthSeed));

  //project on the new frame
  SE3<float> old_to_new = income_transform * keyframe_transform.inv();
  cudaStreamSynchronize(swict_semidense_stream1);
  cudaStreamSynchronize(swict_semidense_stream2);
#ifdef DEBUG_PRINT
  printf("create_new_keyframe_async: download all information cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);start = std::clock();
#endif
  for(int height_i = 0; height_i < height; height_i ++)
  {
    for(int width_i = 0; width_i < width; width_i ++)
    {
      DepthSeed* original = semidense_hostptr + height_i * width +  width_i;
      if(!original->is_vaild())
        continue;
      float3 point_dir = camera.cam2world(make_float2(width_i, height_i));
      float3 new_point = camera.world2cam_f3( old_to_new * (point_dir / original->smooth_idepth()));
      float new_idepth = 1.0f / new_point.z;
      int new_x = new_point.x + 0.5;
      int new_y = new_point.y + 0.5;
      if(new_x < 2 || new_y < 2 || new_x >= width - 3 || new_y >= height - 3)
        continue;
      //check gradient and color
      float2 new_gradient = income_gradient_hostptr[new_y * width +  new_x];
      float new_gradient_2 = dot(new_gradient, new_gradient);
      if( new_gradient_2 < MIN_GRAIDIENT * MIN_GRAIDIENT)
        continue;
      int diff_color = keyframeMat.at<uchar>(height_i, width_i) - income_undistort.at<uchar>(new_y, new_x);
      if(diff_color * diff_color > (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*new_gradient_2))
        continue;

      float idepth_ratio_4 = new_idepth / original->smooth_idepth();
      idepth_ratio_4 *= idepth_ratio_4;
      idepth_ratio_4 *= idepth_ratio_4;
      float new_var = idepth_ratio_4*original->smooth_variance();

      //occultion
      DepthSeed* destination = semidense_new_hostptr + new_y * width +  new_x;
      if(destination->is_vaild())
      {
        float diff_idepth = destination->idepth() - new_idepth;
        if(diff_idepth * diff_idepth > new_var + destination->variance())
        {
          if(new_idepth > destination->idepth())
            destination->set_invaild();
          else
            continue;
        }
      }

      //fuse
      if(!destination->is_vaild())
      {
        destination->initialize(new_idepth, new_var, original->vaild_counter());
      }
      else
      {
        float w = new_var / (new_var + destination->variance());
        float merged_new_idepth = w*destination->idepth() + (1.0f-w)*new_idepth;
        float merged_new_variance = 1.0f / (1.0f / destination->variance() + 1.0f / new_var);
        int merged_validity = original->vaild_counter() + destination->vaild_counter();
        if(merged_validity > VALIDITY_COUNTER_MAX+(VALIDITY_COUNTER_MAX_VARIABLE))
          merged_validity = VALIDITY_COUNTER_MAX+(VALIDITY_COUNTER_MAX_VARIABLE);
        destination->initialize(merged_new_idepth, merged_new_variance, merged_validity);
      }
    }
  }
#ifdef DEBUG_PRINT
  printf("create_new_keyframe_async: cpu fuse cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);start = std::clock();
#endif
  keyframe_semidense.setDevData(semidense_new_hostptr, swict_semidense_stream1);
#ifdef DEBUG_PRINT
  printf("create_new_keyframe_async: upload semidense cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);start = std::clock();
#endif

  dim3 image_block;
  dim3 image_grid;
  image_block.x = 8;
  image_block.y = 8;
  image_grid.x = (width + image_block.x - 1) / image_block.x;
  image_grid.y = (height + image_block.y - 1) / image_block.y;
  regulizeDepth_kernel<<<image_grid, image_block, 0, swict_semidense_stream1>>>(keyframe_semidense.dev_ptr, true);
  regulizeDepth_FillHoles_kernel<<<image_grid, image_block, 0, swict_semidense_stream1>>>(keyframe_semidense.dev_ptr);
  regulizeDepth_kernel<<<image_grid, image_block, 0, swict_semidense_stream1>>>(keyframe_semidense.dev_ptr, false);
  cudaStreamSynchronize(swict_semidense_stream1);

  set_income_as_keyframe();
#ifdef DEBUG_PRINT
  printf("create_new_keyframe_async: gpu smooth cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);start = std::clock();
#endif
}

void quadmap::SeedMatrix::update_keyframe()
{
  bindTexture(income_image_tex, income_image);
  bindTexture(income_gradient_tex, income_gradient);
  bindTexture(keyframe_image_tex, keyframe_image);
  bindTexture(keyframe_gradient_tex, keyframe_gradient);

 /* cv::Mat grad(input_image.rows, input_image.cols, CV_32FC2);
  keyframe_gradient.getDevData((float2*)grad.data);
  double minVal = 0.0, maxVal = 0.0;
  cv::minMaxIdx(grad, &minVal, &maxVal);
  cv::Mat channels[2];
  cv::split(grad, channels);
  cv::Mat norm, color;
  cv::normalize(channels[0], norm, 0, 255, CV_MINMAX, CV_8U);
  cv::applyColorMap(norm, color, cv::COLORMAP_JET);
  cv::imshow("Gradient X", color);
  cv::normalize(channels[1], norm, 0, 255, CV_MINMAX, CV_8U);
  cv::applyColorMap(norm, color, cv::COLORMAP_JET);
  cv::imshow("Gradient Y", color);
  cv::waitKey();*/

  dim3 image_block;
  dim3 image_grid;
  image_block.x = 8;
  image_block.y = 8;
  image_grid.x = (width + image_block.x - 1) / image_block.x;
  image_grid.y = (height + image_block.y - 1) / image_block.y;

  SE3<float> key_to_income = income_transform * keyframe_transform.inv();
  float4 camera_para = make_float4(camera.fx,camera.cx,camera.fy,camera.cy);
  //std::cout << "key_to_income" << std::endl << key_to_income << std::endl;

  // Epipolar search in new frame and depth update of keyframe
  update_keyframe_kernel<<<image_grid, image_block>>>(
    keyframe_semidense.dev_ptr,
    camera_para,
    key_to_income,
    debug_image.dev_ptr,
    epipolar_image.dev_ptr);

  // cudaDeviceSynchronize();
  // Regularization from LSD-SLAM
  //regulizeDepth_FillHoles_kernel<<<image_grid, image_block>>>(keyframe_semidense.dev_ptr);
  //regulizeDepth_kernel<<<image_grid, image_block>>>(keyframe_semidense.dev_ptr, false);
  // cudaDeviceSynchronize();

  // Project depth from keyframe to new frame
  /*depth_project_kernel<<<image_grid, image_block>>>(
  keyframe_semidense.dev_ptr,
  camera_para,
  key_to_income,
  debug_image.dev_ptr);*/
  SE3<float> identity;
  depth_project_kernel << <image_grid, image_block >> >(
      keyframe_semidense.dev_ptr,
      camera_para,
      identity,
      depth_output.dev_ptr);

  depth_project_kernel<<<image_grid, image_block>>>(
  keyframe_semidense.dev_ptr,
  camera_para,
  key_to_income,
  semidense_on_income.dev_ptr);
  cudaDeviceSynchronize();

  cudaUnbindTexture(income_image_tex);
  cudaUnbindTexture(income_gradient_tex);
  cudaUnbindTexture(keyframe_image_tex);
  cudaUnbindTexture(keyframe_gradient_tex);
#ifdef DEBUG_PRINT
  printf("update keyframe cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
#endif
}

void quadmap::SeedMatrix::extract_depth()
{
  clock_t depth_extract_start = std::clock();

  bindTexture(income_image_tex, income_image);
  bindTexture(income_gradient_tex, income_gradient);

  //prepare the reference data
#ifdef DEBUG_PRINT
  printf("  we have %d of %d frames.\n", framelist_host.size(), KEYFRAME_NUM);
#endif
  for(int i = 0; i < framelist_host.size(); i++)
  {
    FrameElement this_ele = framelist_host[i];
    FrameElement gpu_ele;
    gpu_ele.frame_ptr = this_ele.frame_ptr->dev_ptr;
    gpu_ele.transform = this_ele.transform * income_transform.inv();
    match_parameter.framelist_dev[i] = gpu_ele;
  }
  match_parameter.current_frames = framelist_host.size();
  match_parameter.setDevData(); 

  //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));

  // /*first, generate quad tree*/
  DeviceImage<int> quadtree_index(width,height);
  quadtree_index.zero();

  if (useQuadtree) {
      dim3 quadtree_block;
      dim3 quadtree_grid;
      quadtree_block.x = 16;
      quadtree_block.y = 16;
      quadtree_grid.x = (width + quadtree_block.x - 1) / quadtree_block.x;
      quadtree_grid.y = (height + quadtree_block.y - 1) / quadtree_block.y;

      quadtree_image_kernal << <quadtree_grid, quadtree_block >> > (quadtree_index.dev_ptr);
      cudaDeviceSynchronize();

      //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));

      if(printTimings)
        printf("  quadtree cost %f ms \n", (std::clock() - depth_extract_start) / (double)CLOCKS_PER_SEC * 1000); depth_extract_start = std::clock();
  }

  bindTexture(quadtree_tex, quadtree_index, cudaFilterModePoint);

  DeviceImage<PIXEL_COST> image_cost(width / cost_downsampling, height/ cost_downsampling);
  DeviceImage<int> add_num(width/ cost_downsampling, height/ cost_downsampling);
  image_cost.zero();
  add_num.zero();

  /*add semidense prior into the cost*/
  // dim3 prior2cost_block;
  // dim3 prior2cost_grid;
  // prior2cost_block.x = DEPTH_NUM;
  // prior2cost_grid.x = width;
  // prior2cost_grid.y = height;
  // prior_to_cost<<<prior2cost_grid, prior2cost_block>>>(semidense_on_income.dev_ptr, image_cost.dev_ptr, add_num.dev_ptr);
  // cudaDeviceSynchronize();
  // printf("  prior to cost cost %f ms \n", ( std::clock() - depth_extract_start ) / (double) CLOCKS_PER_SEC * 1000);depth_extract_start = std::clock();

  // Multi View Stereo Cost Volume computation
  clock_t cost_start = std::clock();
  dim3 cost_block;
  dim3 cost_grid;
  cost_block.x = DEPTH_NUM;
  cost_block.y = framelist_host.size();
  cost_grid.x = width / cost_downsampling;
  cost_grid.y = height / cost_downsampling;
  image_to_cost<<<cost_grid, cost_block, 2 * DEPTH_NUM * framelist_host.size() * sizeof(float)>>>(
    match_parameter.dev_ptr,
    pixel_age_table.dev_ptr,
    image_cost.dev_ptr,
    add_num.dev_ptr);

  cudaDeviceSynchronize();
  //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));
  if(printTimings) {
    printf("  cost aggregation cost %f ms \n", ( std::clock() - cost_start ) / (double) CLOCKS_PER_SEC * 1000);
    printf("  total cost aggregation cost %f ms \n", ( std::clock() - depth_extract_start ) / (double) CLOCKS_PER_SEC * 1000);depth_extract_start = std::clock();
  }

  //normalize the cost
  dim3 normalize_block;
  dim3 normalize_grid;
  normalize_block.x = DEPTH_NUM;
  normalize_grid.x = width / cost_downsampling;
  normalize_grid.y = height / cost_downsampling;
  normalize_the_cost<<<normalize_grid,normalize_block>>>(image_cost.dev_ptr, add_num.dev_ptr);
  cudaDeviceSynchronize();

  // Regularize and extract depth
  //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));
  if(printTimings)
    printf("  cost normalize cost %f ms \n", ( std::clock() - depth_extract_start ) / (double) CLOCKS_PER_SEC * 1000);depth_extract_start = std::clock();

  dim3 upsample_block;
  dim3 upsample_grid;
  upsample_block.x = 32;
  upsample_block.y = 32;
  upsample_grid.x = (width + upsample_block.x - 1) / upsample_block.x;
  upsample_grid.y = (height + upsample_block.y - 1) / upsample_block.y;
  //upsample_naive << <upsample_grid, upsample_block >> > (debug_image.dev_ptr, depth_output.dev_ptr, cost_downsampling);

  debug_image.zero();
  depth_output.zero();

  if (!doBeliefPropagation) {
      // naive extract
      naive_extract << <normalize_grid, normalize_block >> > (image_cost.dev_ptr, debug_image.dev_ptr, cost_downsampling);
      cudaDeviceSynchronize();
      //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));
      if(printTimings)
        printf("  naive extract cost %f ms \n", (std::clock() - depth_extract_start) / (double)CLOCKS_PER_SEC * 1000); depth_extract_start = std::clock();

      // quadtree upsampling
      upsample_naive << <upsample_grid, upsample_block >> > (debug_image.dev_ptr, depth_output.dev_ptr, cost_downsampling);
      cudaDeviceSynchronize();
      //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));
      if(printTimings)
        printf("  naive upsample cost %f ms \n", (std::clock() - depth_extract_start) / (double)CLOCKS_PER_SEC * 1000); depth_extract_start = std::clock();
  }
  else {
      // bp extract the depth
      // debug_image.zero();
      bp_extract(cost_downsampling, image_cost, debug_image, P1, P2);
      // hbp(image_cost, feature_depth);
      cudaDeviceSynchronize();
      //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));
      if(printTimings)
        printf("  bp extract cost %f ms \n", (std::clock() - depth_extract_start) / (double)CLOCKS_PER_SEC * 1000); depth_extract_start = std::clock();


      //clock_t upsample_start = std::clock();
      if (!doGlobalUpsampling) {
        upsample_naive << <upsample_grid, upsample_block >> > (debug_image.dev_ptr, depth_output.dev_ptr, cost_downsampling);
        // local_upsample(debug_image, depth_output);
        //printf("  naive upsample cost %f ms \n", (std::clock() - depth_extract_start) / (double)CLOCKS_PER_SEC * 1000); depth_extract_start = std::clock();
      }
      else {
        global_upsample(debug_image, depth_output);

        //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));
        //printf("  global upsample cost %f ms \n", (std::clock() - depth_extract_start) / (double)CLOCKS_PER_SEC * 1000); depth_extract_start = std::clock();
      }
  }
  cudaUnbindTexture(quadtree_tex);
  cudaUnbindTexture(income_image_tex);
  cudaUnbindTexture(income_gradient_tex);
}

void quadmap::SeedMatrix::fuse_output_depth()
{
  bindTexture(pre_image_tex, pre_income_image);
  bindTexture(income_image_tex, income_image);

  DeviceImage<int> transform_table(width, height);
  DeviceImage<float4> new_seed(width, height);
  transform_table.zero();
  // DeviceImage<float> filtered_depth(width, height);
  // filtered_depth = depth_output;

  //tranform the former depth
  dim3 image_block;
  dim3 image_grid;
  image_block.x = 16;
  image_block.y = 16;
  image_grid.x = (width + image_block.x - 1) / image_block.x;
  image_grid.y = (height + image_block.y - 1) / image_block.y;
  // high_gradient_filter<<<image_grid, image_block>>>(filtered_depth.dev_ptr, depth_output.dev_ptr);
  // cudaDeviceSynchronize();

  // Transform depth from last depth map to new frame
  SE3<float> last_to_income = income_transform * this_fuse_worldpose.inv();
  fuse_transform<<<image_grid, image_block>>>(depth_fuse_seeds.dev_ptr, transform_table.dev_ptr, last_to_income,
          camera, min_inlier_ratio_bad);
  // cudaDeviceSynchronize();

  // Fill holes (LSD SLAM)
  hole_filling<<<image_grid, image_block>>>(transform_table.dev_ptr);
  // cudaDeviceSynchronize();

  // Fuse
  fuse_currentmap<<<image_grid, image_block>>>(
  transform_table.dev_ptr,
  depth_output.dev_ptr,
  depth_fuse_seeds.dev_ptr,
  new_seed.dev_ptr,
  min_inlier_ratio_good);
  // cudaDeviceSynchronize();

  // assign
  depth_fuse_seeds = new_seed;
  this_fuse_worldpose = income_transform;

  cudaDeviceSynchronize();
  cudaUnbindTexture(pre_image_tex);
  cudaUnbindTexture(income_image_tex);
}

void quadmap::SeedMatrix::download_output()
{
  std::clock_t start = std::clock();
  depth_output.getDevData(reinterpret_cast<float*>(cv_output.data));
  debug_image.getDevData(reinterpret_cast<float*>(cv_debug.data));
  epipolar_image.getDevData(reinterpret_cast<float4*>(cv_epipolar.data));
#ifdef DEBUG_PRINT
  printf("download depth map cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);start = std::clock();
#endif
}

void quadmap::SeedMatrix::get_result(cv::Mat &depth, cv::Mat &debug, cv::Mat &reference, cv::Mat &epipolar, cv::Mat &keyframe)
{
  depth = cv_output.clone();
  debug = cv_debug.clone();
  reference = income_undistort.clone();
  epipolar = cv_epipolar.clone();
  keyframe = keyframeMat.clone();
}
