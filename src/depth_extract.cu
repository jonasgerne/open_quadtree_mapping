#include <quadmap/depth_extract.cuh>

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

__global__ void prior_to_cost(
	bool inverse_depth,
    float min_depth,
	float step_depth,
	DeviceImage<float2> *depth_prior_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr)
{
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int depth_id = threadIdx.x;
	const int width = depth_prior_devptr->width;
	const int height = depth_prior_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	float2 prior = depth_prior_devptr->atXY(x,y);
	if(prior.x <= 0 || prior.y <= 0)
		return;

	float my_invdepth;
	if (inverse_depth)
		my_invdepth = (step_depth * depth_id + min_depth);
	else
		my_invdepth = 1.0 / (step_depth * depth_id + min_depth);
	float cost = PRIOR_COST_SCALE * (my_invdepth-prior.x) * (my_invdepth-prior.x) / prior.y;
	// float cost = PRIOR_COST_SCALE * (my_depth-prior.x) * (my_depth-prior.x);
	cost = cost < TRUNCATE_COST ? cost : TRUNCATE_COST;
	atomicAdd(cost_devptr->atXY(x/4,y/4).cost_ptr(depth_id), cost);
	atomicAdd(num_devptr->ptr_atXY(x/4,y/4),1);
}
__global__ void image_to_cost(
	bool inverse_depth,
	float min_depth,
	float step_depth,
	MatchParameter *match_parameter_devptr,
	DeviceImage<int> *age_table_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr)
{
    const int cost_downsampling = match_parameter_devptr->cost_downsampling;

    constexpr int patch_radius = 5;

	const int x = blockIdx.x * cost_downsampling;
	const int y = blockIdx.y * cost_downsampling;
	const int depth_id = threadIdx.x;
	const int frame_id = threadIdx.y;
    //const int num_frames = threadsPerBlock.y;
	const int width = age_table_devptr->width;
	const int height = age_table_devptr->height;
	const int frame_num = match_parameter_devptr->current_frames;
	const int my_quadsize = 1 << tex2D(quadtree_tex, x, y);

    // Check out of bounds
    if (x >= width - 1 || y >= height - 1 || x <= 0 || y <= 0)
        return;

	// Skip if not selected by quadtree
    if (((x % my_quadsize) != 0) || ((y % my_quadsize) != 0))
        return;

//	int this_age = age_table_devptr->atXY(x,y);
//
//	if(this_age >= frame_num)
//		this_age = frame_num - 1;

	// Age table not maintained so always zero
	/*if(this_age < KEYFRAME_NUM)
		this_age = KEYFRAME_NUM - 1;*/

    extern __shared__ float s[];
    float* cost = s;
    float* aggregate_num = &s[DEPTH_NUM * frame_num];
	/*__shared__ float cost[DEPTH_NUM][10];
	__shared__ float aggregate_num[DEPTH_NUM][10];*/
	//const int my_frame_id = (float) this_age / 10.0 * (float) frame_id;
    const int my_frame_id = frame_id;

	//read memory
	PinholeCamera camera = match_parameter_devptr->camera_model;
	FrameElement my_reference = match_parameter_devptr->framelist_dev[my_frame_id];

	float my_patch[patch_radius * 2 + 1][patch_radius * 2 + 1];

	for(int j = -patch_radius; j <= patch_radius; j++)
	{
		for(int i = -patch_radius; i <= patch_radius; i++)
		{
			my_patch[i+patch_radius][j+patch_radius] = tex2D(income_image_tex, x + i + 0.5, y + j + 0.5);
		}
	}

	//calculate
	const SE3<float> income_to_ref = my_reference.transform;
	float3 my_dir = normalize(camera.cam2world(make_float2(x, y)));

	float my_depth;
	if (inverse_depth)
		my_depth = 1.0 / (step_depth * depth_id + min_depth); // min_depth is inverted in the function call if invert_depth is true
	else
    	my_depth = (step_depth * depth_id + min_depth);

	float2 project_point = camera.world2cam(income_to_ref*(my_dir*my_depth));
	int point_x = project_point.x + 0.5;
	int point_y = project_point.y + 0.5;
	float u2u = income_to_ref.data(0,0);
	float u2v = income_to_ref.data(1,0);
	float v2u = income_to_ref.data(0,1);
	float v2v = income_to_ref.data(1,1);

    // Check bounds
	if( point_x >= patch_radius && point_x < width - patch_radius && point_y >= patch_radius && point_y < height - patch_radius)
	{
        // Sum cost over 3x3 patch
		float my_cost = 0.0;
		int my_count = 0;
		for(int j = -patch_radius; j <= patch_radius; j++)
		{
			for(int i = -patch_radius; i <= patch_radius; i++)
			{
				int check_x = project_point.x + u2u * i + v2u * j + 0.5;
				int check_y = project_point.y + u2v * i + v2v * j + 0.5;
				if ((check_x >= 0) && (check_x < width) && (check_y >= 0) && (check_y < height)) {
                    my_cost += fabs(my_patch[i + patch_radius][j + patch_radius] -
                                    my_reference.frame_ptr->atXY(check_x, check_y));
                    my_count++;
                }
			}
		}

		cost[depth_id * frame_num + frame_id] = my_cost / (float)my_count;
		aggregate_num[depth_id * frame_num + frame_id] = 1;
	}
	else
	{
		// If not within bounds set cost to zero
//		if((x == 321) && (y == 10))
//			printf("Cost 321/10/%d: Out of bounds\n", depth_id);
		cost[depth_id * frame_num + frame_id] = 0;
		aggregate_num[depth_id * frame_num + frame_id] = 0;
	}
	__syncthreads();

    // Sum reduce over all frames
	for(int r = frame_num / 2; r > 0; r /= 2)
	{
		if(frame_id + r < blockDim.y && frame_id < r)
		{
		  cost[depth_id * frame_num + frame_id] += cost[depth_id * frame_num + frame_id + r];
		  aggregate_num[depth_id * frame_num + frame_id] += aggregate_num[depth_id * frame_num + frame_id + r];
		}
		__syncthreads();
	}

    // Write cost over all frames
	if(frame_id == 0)
	{
		float my_depth_cost = cost[depth_id * frame_num];
		if(aggregate_num[depth_id * frame_num] > 0)
			my_depth_cost = my_depth_cost / (float)aggregate_num[depth_id * frame_num] / 255.0f;
		else
			my_depth_cost = 100;
        // Write cost and counter
		atomicAdd(cost_devptr->atXY(x / cost_downsampling,y / cost_downsampling).cost_ptr(depth_id), my_depth_cost);
		atomicAdd(num_devptr->ptr_atXY(x / cost_downsampling, y / cost_downsampling), 1);

//		if((x == 321) && ( y == 10))
//			printf("Cost 321/10/%d: Cost %f | Count %d\n", depth_id, my_depth_cost, num_devptr->atXY(x / cost_downsampling, y / cost_downsampling));
	}
}

__global__ void normalize_the_cost(
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr)
{
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int depth_id = threadIdx.x;
	const int add_num = num_devptr->atXY(x,y);
	if (add_num<=0)
		return;
	float mycost = cost_devptr->atXY(x,y).get_cost(depth_id);
	mycost /= (float)add_num;
	cost_devptr->atXY(x,y).set_cost(depth_id, mycost);

//	if((x == 321) && ( y == 10))
//		printf("Norm 321/10/%d: Cost %f\n", depth_id, mycost);
}

__global__ void naive_extract(
	float cost_downsampling,
	bool inverse_depth,
	float min_depth,
	float step_depth,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<float> *coarse_depth_devptr)
{
	const int x = blockIdx.x;
	const int y = blockIdx.y;

	const int depth_id = threadIdx.x;
	__shared__ float cost[DEPTH_NUM];
	__shared__ float min_cost[DEPTH_NUM];
	__shared__ int min_id[DEPTH_NUM];
	cost[depth_id] = cost_devptr->atXY(x,y).get_cost(depth_id);
	min_cost[depth_id] = cost[depth_id];
	min_id[depth_id] = depth_id;
	__syncthreads();

    // Find minimum
	for(int i = DEPTH_NUM / 2; i > 0; i /= 2)
	{
		if( depth_id < i && min_cost[depth_id+i] < min_cost[depth_id])
		{
			min_cost[depth_id] = min_cost[depth_id+i];
			min_id[depth_id] = min_id[depth_id+i];
		}
		__syncthreads();
	}

    // Reduce min index
	if (depth_id == 0)
	{
		float disparity = min_id[0];
		if(min_id[0] > 0 && min_id[0] < DEPTH_NUM - 1)
		{
			// Interpolate disparity between neigbouring costs
			float cost_pre = cost[min_id[0] - 1];
			float cost_post = cost[min_id[0] + 1];
			float a = cost_pre - 2.0f * min_cost[0] + cost_post;
			float b = - cost_pre + cost_post;
			float b_a = b/a;
			if (a > 0.0f)
				disparity = (float) min_id[0] - b_a / 2.0f;
			//disparity = (float)min_id[0] - b / (2.0f * a);
		}
		if (inverse_depth)
			coarse_depth_devptr->atXY(x * cost_downsampling, y * cost_downsampling) = 1.0 / (step_depth * disparity + min_depth);
		else
			coarse_depth_devptr->atXY(x * cost_downsampling, y * cost_downsampling) = (step_depth * disparity + min_depth);
	}
}

__global__ void upsample_naive(
	DeviceImage<float> *coarse_depth_devptr,
	DeviceImage<float> *full_dense_devptr,
	const int cost_downsampling)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int width = full_dense_devptr->width;
	const int height = full_dense_devptr->height;
	if(x >= width || y >= height)
		return;
	const int my_quadsize = 1 << tex2D(quadtree_tex, x, y);
	full_dense_devptr->atXY(x,y) = coarse_depth_devptr->atXY((((x/cost_downsampling)*cost_downsampling)/my_quadsize)*my_quadsize, (((y/cost_downsampling)*cost_downsampling)/my_quadsize)*my_quadsize);
}

}