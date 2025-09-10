#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	tiles_touched[idx] = 0;

	uint3 rect_min, rect_max;
	getRect(points_xyz + 3 * idx, radii[idx], rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z) == 0)
		return;

	tiles_touched[idx] = (rect_max.z - rect_min.z) * (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
__global__ void renderCUDA(
	const int N,
	const float* __restrict__ pts,
	const int* __restrict__ points_int,
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ opacities,
	float* __restrict__ out_bin_logits)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
	    return;

	const int* point_int = points_int + idx * 3;
	const int voxel_idx = point_int[0] * grid.y * grid.z + point_int[1] * grid.z + point_int[2];
	const float3 point = {pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2]};

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[voxel_idx];

	// Initialize helper variables
	float bin_logit = 1.0;

	for (int i = range.x; i < range.y; i++)
	{
		int gs_idx = point_list[i];
		float3 cov1 = { cov3D[gs_idx * 6 + 0], cov3D[gs_idx * 6 + 1], cov3D[gs_idx * 6 + 2] };  // on the diagonal
		float3 cov2 = { cov3D[gs_idx * 6 + 3], cov3D[gs_idx * 6 + 4], cov3D[gs_idx * 6 + 5] };  // not on the diagonal

		/* start formular (4) */
		float3 d = { means3D[gs_idx * 3] - point.x, means3D[gs_idx * 3 + 1] - point.y, means3D[gs_idx * 3 + 2] - point.z };
		float power = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
		power = -0.5f * power - (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
		power = exp(power);  // > 0.
		/* end formular (4) */

		bin_logit = (1 - power * opacities[gs_idx]) * bin_logit;  // probability of not occupied by gaussian G_i
	}

	// Iterate over batches until all done or range is complete
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	out_bin_logits[idx] = 1 - bin_logit;  // formulate (5)
}


void FORWARD::render(
	const int N,
	const float* pts,
	const int* points_int,
	const dim3 grid,
	const uint2* ranges,
	const uint32_t* point_list,
	const float* means3D,
	const float* cov3D,
	const float* opacities,
	float* out_bin_logits)
{
	renderCUDA << <(N + 255) / 256, 256 >> > (
		N, 
		pts,
		points_int,
		grid,
		ranges,
		point_list,
		means3D,
		cov3D,
		opacities,
		out_bin_logits);
}


void FORWARD::preprocess(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		points_xyz,
		radii,
		grid,
		tiles_touched
	);
}