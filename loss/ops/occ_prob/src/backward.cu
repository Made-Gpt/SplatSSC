#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;

	int voxel_idx = points_xyz[3 * idx] * grid.y * grid.z + points_xyz[3 * idx + 1] * grid.z + points_xyz[3 * idx + 2];
	voxel2pts[voxel_idx] = idx;
}


__global__ void renderCUDA(
	const int P,
	const uint32_t* __restrict__ offsets,
	const uint32_t* __restrict__ point_list_keys_unsorted,
	const int* __restrict__ voxel2pts,
	const float* __restrict__ pts,
	const float* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ opacities,
	const float* __restrict__ bin_logits,
	const float* __restrict__ bin_logits_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ cov3D_grad,
	float* __restrict__ opas_grad)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
	    return;

	uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
	uint32_t end = offsets[idx];

	const float3 means = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	const float3 cov1 = {cov3D[6 * idx], cov3D[6 * idx + 1], cov3D[6 * idx + 2]};
	const float3 cov2 = {cov3D[6 * idx + 3], cov3D[6 * idx + 4], cov3D[6 * idx + 5]};
	const float opa = opacities[idx];
	const float epsilon = 1e-6;

	float means_grad[3] = {0};
	float opa_grad = 0;
	float cov_grad[6] = {0};

	for (int i = start; i < end; i++)
	{
		int voxel_idx = point_list_keys_unsorted[i];
		int pts_idx = voxel2pts[voxel_idx];
		if (pts_idx >= 0)
		{
		    // build forward propagation
			float3 d = {means.x - pts[pts_idx * 3], means.y - pts[pts_idx * 3 + 1], means.z - pts[pts_idx * 3 + 2]};
			float power = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
			power = -0.5f * power - (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
			power = exp(power);
			float power_grad = 0.;

			// My Fix
			// power_grad += prob_grad * powf(2 * 3.1415926535, -1.5) * powf(deter, 0.5);
			// power_grad += (1 - bin_logits[pts_idx]) / (1 - power + 1e-9) * bin_logits_grad[pts_idx];
			// power_grad += density_grad[pts_idx];
			// My Fix
			// power_grad += bin_logits_grad[pts_idx] * opa;
            // opa_grad += bin_logits_grad[pts_idx] * power;

            // My Fix
            float curr_occ_prob = bin_logits[pts_idx];  // 1 - \prod_{i=1}^n (1 - p_i a_i)
            float acc_bin_logit = 1.0f - curr_occ_prob;  // accumulated occupancy probability, \prod_{i=1}^n (1 - p_i a_i)
            float other_bin_logit = acc_bin_logit / fmaxf(1.0f - power * opa, epsilon);  // probability of not occupied

            power_grad = bin_logits_grad[pts_idx] * other_bin_logit * opa;
            opa_grad += bin_logits_grad[pts_idx] * other_bin_logit * power;

			means_grad[0] -= power_grad * power * (cov1.x * d.x + cov2.x * d.y + cov2.z * d.z);
			means_grad[1] -= power_grad * power * (cov2.x * d.x + cov1.y * d.y + cov2.y * d.z);
			means_grad[2] -= power_grad * power * (cov2.z * d.x + cov2.y * d.y + cov1.z * d.z);

            cov_grad[0] += power_grad * power * (-0.5f * d.x * d.x);
            cov_grad[1] += power_grad * power * (-0.5f * d.y * d.y);
            cov_grad[2] += power_grad * power * (-0.5f * d.z * d.z);
            cov_grad[3] += power_grad * power * (-d.x * d.y);
            cov_grad[4] += power_grad * power * (-d.y * d.z);
            cov_grad[5] += power_grad * power * (-d.x * d.z);
		}
	}

	means3D_grad[idx * 3] = means_grad[0];
	means3D_grad[idx * 3 + 1] = means_grad[1];
	means3D_grad[idx * 3 + 2] = means_grad[2];
	opas_grad[idx] = opa_grad;

	for (int ch = 0; ch < 6; ch++)
	{
		cov3D_grad[idx * 6 + ch] = cov_grad[ch];
	}
}


void BACKWARD::render(
	const int P,
	const uint32_t* offsets,
	const uint32_t* point_list_keys_unsorted,
	const int* voxel2pts,
	const float* pts,
	const float* means3D,
	const float* cov3D,
	const float* opacities,
	const float* bin_logits,
	const float* bin_logits_grad,
	float* means3D_grad,
	float* cov3D_grad,
	float* opas_grad)
{
	renderCUDA << <(P + 255) / 256, 256 >> > (
		P,
		offsets,
		point_list_keys_unsorted,
		voxel2pts,
		pts,
		means3D,
		cov3D,
		opacities,
		bin_logits,
		bin_logits_grad,
		means3D_grad,
		cov3D_grad,
		opas_grad);
}

void BACKWARD::preprocess(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	preprocessCUDA << <(N + 255) / 256, 256 >> > (
		N,
		points_xyz,
		grid,
		voxel2pts
	);
}

