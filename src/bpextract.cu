#include <quadmap/bpextract.cuh>

namespace quadmap
{
//function declear here!
void bp_extract(int cost_downsampling, DeviceImage<PIXEL_COST> &image_cost_map, DeviceImage<float> &depth, float P1, float P2);
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
    int i_leverl);
__global__ void upsample(
    DeviceImage<PIXEL_COST> *l1_message_devptr,
    DeviceImage<PIXEL_COST> *l0_message_devptr);
__global__ void depth_extract(int cost_downsampling,
    DeviceImage<PIXEL_COST> *data_devptr,
    DeviceImage<PIXEL_COST> *lm_devptr,
    DeviceImage<PIXEL_COST> *rm_devptr,
    DeviceImage<PIXEL_COST> *up_devptr,
    DeviceImage<PIXEL_COST> *dm_devptr,
    DeviceImage<float> *extracted_depth_devptr);

//function define here!
//we only optimize the cost at 16x16 and 32x32 level, for finer level, we only optimize at local patch and jump the corser level
//we start the optimize at image_level, and the cost map begins at image_level
void bp_extract(int cost_downsampling, DeviceImage<PIXEL_COST> &image_cost_map, DeviceImage<float> &depth, float P1, float P2)
{
    const int width = image_cost_map.width;
    const int height = image_cost_map.height;
    const int hbp_level = 4;

    // 4 levels: 4x4, 8x8, 16x16, 32x32
    // corresponsible to level 0, 1, 2 ,3 ,4 , 5(this is only used for optimize)
    int hbp_iterate[4] = {4, 10, 10, 10}; // from fine to coarse 3 level

    int h_width[4]; //next four level
    int h_height[4]; //next four level

    h_width[0] = width;
    h_height[0] = height;

    for(int i = 1; i < hbp_level; i++)
    {
        h_width[i] = (h_width[i - 1] + 1) / 2;
        h_height[i] = (h_height[i - 1] + 1) / 2;
    }

    //create the hierarchical cost map
    DeviceImage<PIXEL_COST> *prycost_hostptr[4];
    prycost_hostptr[0] = &image_cost_map;
    for(int i = 1; i < hbp_level; i++)
    {
        prycost_hostptr[i] = new DeviceImage<PIXEL_COST>(h_width[i], h_height[i]);
    }

    // Distribute cost from fine to coarse by summing over 2x2 patches
    dim3 hier_block;
    dim3 hier_grid;
    hier_block.z = DEPTH_NUM;
    for(int i = 1; i < hbp_level; i++)
    {
        hier_grid.x = h_width[i];
        hier_grid.y = h_height[i];
        cost_distribute <<< hier_grid, hier_block>>>(
            prycost_hostptr[i - 1]->dev_ptr,
            prycost_hostptr[i]->dev_ptr);
        cudaDeviceSynchronize();
    }

    //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));

    //loopy bp on each level
    //create the message four dirs
    DeviceImage<PIXEL_COST> *message_hostptr[4];
    message_hostptr[0] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
    message_hostptr[1] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
    message_hostptr[2] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
    message_hostptr[3] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
    // Initialize with zeros
    message_hostptr[0]->zero();
    message_hostptr[1]->zero();
    message_hostptr[2]->zero();
    message_hostptr[3]->zero();

    // Hierachical message computation from coarse to fine
    for(int i_leverl = hbp_level - 1; i_leverl >= 0; i_leverl--)
    {
        // /*if i_leverl is not the coarsest, initialize the message*/
        if( i_leverl < (hbp_level - 1) )
        {
            // New buffers for level
            DeviceImage<PIXEL_COST> *message_next_hostptr[4];
            message_next_hostptr[0] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);
            message_next_hostptr[1] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);
            message_next_hostptr[2] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);
            message_next_hostptr[3] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);

            // Upsample coarse cost to new level, by simply copying from 1 pixel to each pixel in 2x2 patch
            dim3 message_up_block;
            dim3 message_up_grid;
            message_up_block.x = DEPTH_NUM;
            message_up_grid.x = h_width[i_leverl + 1];
            message_up_grid.y = h_height[i_leverl + 1];
            for(int mess_i = 0; mess_i < 4; mess_i++)
                upsample <<< message_up_grid, message_up_block>>>(
                    message_hostptr[mess_i]->dev_ptr,
                    message_next_hostptr[mess_i]->dev_ptr);

            cudaDeviceSynchronize();

            // Clean up last level memory
            for(int mess_i = 0; mess_i < 4; mess_i++)
            {
                delete message_hostptr[mess_i];
                message_hostptr[mess_i] = message_next_hostptr[mess_i];
            }
        }

        // /*loopy bp*/
        dim3 bp_block;
        dim3 bp_grid;
        bp_block.x = 4;
        bp_block.y = 64;
        bp_grid.y = h_height[i_leverl];
        //bp_grid.x = (h_width[i_leverl] + 1) / 2; //every iterate on the A or B set of the whole image
        bp_grid.x = h_width[i_leverl];
        bool A_set = true;
        // Update messages iteratively by summing cost at each costvolume entry with values from messages from 3-neighbors for each direction
        // applying SGM cost regularization strategy
        for(int i_iterate = 0; i_iterate < hbp_iterate[i_leverl]; i_iterate++)
        {
            bp <<< bp_grid, bp_block>>>(
                prycost_hostptr[i_leverl]->dev_ptr,
                message_hostptr[0]->dev_ptr,
                message_hostptr[1]->dev_ptr,
                message_hostptr[2]->dev_ptr,
                message_hostptr[3]->dev_ptr,
                A_set,
                i_leverl + 2,
                P1, P2);
            A_set = !A_set;
            cudaDeviceSynchronize();
        }
    }

    //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));

    // Extract depth by finding min cost for sum of pixel cost + 4-neighbor costs
    dim3 depth_extract_block;
    dim3 depth_extract_grid;
    depth_extract_block.x = DEPTH_NUM;
    depth_extract_grid.x = width;
    depth_extract_grid.y = height;
    depth_extract <<< depth_extract_grid, depth_extract_block>>>(
        cost_downsampling,
        prycost_hostptr[0]->dev_ptr,
        message_hostptr[0]->dev_ptr,
        message_hostptr[1]->dev_ptr,
        message_hostptr[2]->dev_ptr,
        message_hostptr[3]->dev_ptr,
        depth.dev_ptr);

    cudaDeviceSynchronize();
    //printf("CUDA Status %s\n", cudaGetErrorString(cudaGetLastError()));

    for(int i = 1; i < hbp_level; i++)
    {
        delete prycost_hostptr[i];
    }
    delete message_hostptr[0];
    delete message_hostptr[1];
    delete message_hostptr[2];
    delete message_hostptr[3];
}

__global__ void cost_distribute(DeviceImage<PIXEL_COST> *l0_cost_devptr,
                                DeviceImage<PIXEL_COST> *l1_cost_devptr)
{
    const int width = l1_cost_devptr->width;
    const int height = l1_cost_devptr->height;
    const int l0_width = l0_cost_devptr->width;
    const int l0_height = l0_cost_devptr->height;

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int depth_id = threadIdx.z;

    if(x >= width || y >= height)
        return;

    float cost_sum(0.0f);

    // For every pixel in l1 set cost to sum over 2x2 pixels in l0
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            if( (2 * x + i) < l0_width && (2 * y + j) < l0_height)
            {
                cost_sum += (l0_cost_devptr->atXY((2 * x + i), (2 * y + j))).get_cost(depth_id);
            }
        }
    }

    (l1_cost_devptr->atXY(x, y)).set_cost(depth_id, cost_sum);
}

__global__ void bp(
    DeviceImage<PIXEL_COST> *data_devptr,
    DeviceImage<PIXEL_COST> *lm_devptr,
    DeviceImage<PIXEL_COST> *rm_devptr,
    DeviceImage<PIXEL_COST> *up_devptr,
    DeviceImage<PIXEL_COST> *dm_devptr,
    bool A_set,
    int i_leverl,
    float P1,
    float P2)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int dir = threadIdx.x; // direction 0-3
    int depth_id = threadIdx.y; // cost index

    // Check if pixel is defined on current level
    if(i_leverl <= 4)
    {
        int size = 1 << i_leverl;
        int pixel_level = tex2D(quadtree_tex, x * size, y * size);
        if(pixel_level > i_leverl)
            return;
    }

    // Alternate between even and odd pixels in x direction
    /*if(A_set)
        x = x * 2 + y % 2;
    else
        x = x * 2 + (y + 1) % 2 ;*/

    const int width = data_devptr->width;
    const int height = data_devptr->height;
    if(x >= width || y >= height)
        return;

    bool on_left, on_right, on_up, on_down;
    on_left = on_right = on_up = on_down = false;

    if(x == 0)
        on_left = true;
    if(x == width - 1)
        on_right = true;
    if(y == 0)
        on_up = true;
    if(y == height - 1)
        on_down = true;

    __shared__ float neighbor_cost[4][DEPTH_NUM];
    __shared__ float neighbor_cost_min[4][DEPTH_NUM];
    __shared__ float raw_cost[4][DEPTH_NUM];

    // Collect cost messages from all neighbors except the receiving one
    neighbor_cost[dir][depth_id] = (data_devptr->atXY(x, y)).get_cost(depth_id);

    if(dir != 0 && !on_up) // to up
    {
        neighbor_cost[dir][depth_id] += (dm_devptr->atXY(x, y - 1)).get_cost(depth_id);
    }
    if(dir != 1 && !on_down) // to down
    {
        neighbor_cost[dir][depth_id] += (up_devptr->atXY(x, y + 1)).get_cost(depth_id);
    }
    if(dir != 2 && !on_left) // to left
    {
        neighbor_cost[dir][depth_id] += (rm_devptr->atXY(x - 1, y)).get_cost(depth_id);
    }
    if(dir != 3 && !on_right) // to right
    {
        neighbor_cost[dir][depth_id] += (lm_devptr->atXY(x + 1, y)).get_cost(depth_id);
    }
    neighbor_cost_min[dir][depth_id] = neighbor_cost[dir][depth_id];
    __syncthreads();

    // find minimum cost for each direction
    for(int i = DEPTH_NUM / 2; i > 0; i = i / 2)
    {
        if(depth_id < i && neighbor_cost_min[dir][depth_id + i] < neighbor_cost_min[dir][depth_id])
        {
            neighbor_cost_min[dir][depth_id] = neighbor_cost_min[dir][depth_id + i];
        }
        __syncthreads();
    }

    // find min cost for every message using SGM penalties
    float min_cost = neighbor_cost[dir][depth_id];
    // Add terms for previous/next depth step, if cost difference than threshold P1 then take that cost
    if(depth_id > 0)
        min_cost = fminf(min_cost, neighbor_cost[dir][depth_id - 1] + P1);
    if(depth_id < DEPTH_NUM - 1)
        min_cost = fminf(min_cost, neighbor_cost[dir][depth_id + 1] + P1);
    // Minimum cost over all possible depth if difference with minimum cost smaller than threshld P2 take that cost
    min_cost = fminf(min_cost, neighbor_cost_min[dir][0] + P2);

    raw_cost[dir][depth_id] = min_cost;
    __syncthreads();

    // Find new min cost
    for(int i = DEPTH_NUM / 2; i > 0; i = i / 2)
    {
        if(depth_id < i)
        {
            raw_cost[dir][depth_id] += raw_cost[dir][depth_id + i];
        }
        __syncthreads();
    }

    // Normalize min cost by subtracting min cost divided by num depths
    min_cost = min_cost - raw_cost[dir][0] / (float) DEPTH_NUM;

    // Copy final message for direction
    if(dir == 0) //up
        (up_devptr->atXY(x, y)).set_cost(depth_id, min_cost);
    else if(dir == 1) //to down
        (dm_devptr->atXY(x, y)).set_cost(depth_id, min_cost);
    else if(dir == 2) // to left
        (lm_devptr->atXY(x, y)).set_cost(depth_id, min_cost);
    else // to right
        (rm_devptr->atXY(x, y)).set_cost(depth_id, min_cost);
}

__global__ void depth_extract(int cost_downsampling,
    DeviceImage<PIXEL_COST> *data_devptr,
    DeviceImage<PIXEL_COST> *lm_devptr,
    DeviceImage<PIXEL_COST> *rm_devptr,
    DeviceImage<PIXEL_COST> *up_devptr,
    DeviceImage<PIXEL_COST> *dm_devptr,
    DeviceImage<float> *extracted_depth_devptr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int width = data_devptr->width;
    int height = data_devptr->height;
    int depth_id = threadIdx.x;
    int pixel_level = tex2D(quadtree_tex, x * 4, y * 4);
    int level_size = 1 << pixel_level;
    if(x * 4 % level_size != 0 || y * 4 % level_size != 0)
        return;

    __shared__ float cost[DEPTH_NUM];
    __shared__ float min_cost[DEPTH_NUM];
    __shared__ int min_id[DEPTH_NUM];
    // Total cost is sum of node and 4-neighbours
    cost[depth_id] = data_devptr->atXY(x, y).get_cost(depth_id);
    if(x != 0)
        cost[depth_id] += rm_devptr->atXY(x - 1, y).get_cost(depth_id);
    if(x != width - 1)
        cost[depth_id] += lm_devptr->atXY(x + 1, y).get_cost(depth_id);
    if(y != 0)
        cost[depth_id] += dm_devptr->atXY(x, y - 1).get_cost(depth_id);
    if(y != height - 1)
        cost[depth_id] += up_devptr->atXY(x, y + 1).get_cost(depth_id);
    min_cost[depth_id] = cost[depth_id];
    min_id[depth_id] = depth_id;

    __syncthreads();
    // Find minimum cost and index
    for(int i = DEPTH_NUM / 2; i > 0; i /= 2)
    {
        if(depth_id < i && min_cost[depth_id + i] < min_cost[depth_id])
        {
            min_cost[depth_id] = min_cost[depth_id + i];
            min_id[depth_id] = min_id[depth_id + i];
        }
        __syncthreads();
    }

    // Reduce with one thread
    if(depth_id == 0)
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
            if(isfinite(b_a))
	            disparity = (float) min_id[0] - b_a / 2.0f;
            //disparity = (float)min_id[0] - b / (2.0f * a);
        }
#ifdef USE_INVERSE_DEPTH
        extracted_depth_devptr->atXY(x * cost_downsampling, y * cost_downsampling) = 1.0 / (STEP_INV_DEPTH * disparity + MIN_INV_DEPTH);
#else
        extracted_depth_devptr->atXY(x * cost_downsampling, y * cost_downsampling) = (STEP_DEPTH * disparity + MIN_DEP);
#endif
    }
}

__global__ void upsample(
    DeviceImage<PIXEL_COST> *l1_message_devptr,
    DeviceImage<PIXEL_COST> *l0_message_devptr)
{
    const int depth_id = threadIdx.x;
    const int x = blockIdx.x; // in l1 image
    const int y = blockIdx.y; // in l1 image
    const int l0_width = l0_message_devptr->width;
    const int l0_height = l0_message_devptr->height;

    // Copy cost from coarse to every pixel in 2x2 patch in fine level
    float value = (l1_message_devptr->atXY(x, y)).get_cost(depth_id);
    for(int j = 0; j < 2; j++)
    {
        for(int i = 0; i < 2; i++)
        {
            int x_up = x * 2 + i;
            int y_up = y * 2 + j;
            if(x_up < l0_width && y_up < l0_height)
                (l0_message_devptr->atXY(x_up, y_up)).set_cost(depth_id, value);
        }
    }
}
}