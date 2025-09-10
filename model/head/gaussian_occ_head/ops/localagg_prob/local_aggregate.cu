/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "src/config.h"
#include "src/aggregator.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateCUDA(
	const torch::Tensor& pts,            // n, 3
	const torch::Tensor& points_int,
	const torch::Tensor& means3D,        // g, 3
	const torch::Tensor& means3D_int,
	const torch::Tensor& opas,
	const torch::Tensor& semantics,      // g, c
	const torch::Tensor& radii,          // g
	const torch::Tensor& cov3D,          // g, 6
	const float opas_thresh,
	const int H, int W, int D)
{
  
	const int P = means3D.size(0);
	const int N = pts.size(0);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_logits = torch::full({N, NUM_CHANNELS}, 0.0, float_opts);
	torch::Tensor out_bin_logits = torch::full({N}, 0.0, float_opts);
	torch::Tensor out_density = torch::full({N}, 0.0, float_opts);
	torch::Tensor out_probability = torch::full({N}, 0.0, float_opts);
	
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	int rendered;
	rendered = LocalAggregator::Aggregator::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, N,
		opas_thresh,
		pts.contiguous().data_ptr<float>(),
		points_int.contiguous().data_ptr<int>(),
		means3D.contiguous().data_ptr<float>(),
		means3D_int.contiguous().data_ptr<int>(),
		opas.contiguous().data_ptr<float>(),
		semantics.contiguous().data_ptr<float>(),
		cov3D.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>(),
		H, W, D,
		out_logits.contiguous().data_ptr<float>(),
		out_bin_logits.contiguous().data_ptr<float>(),
		out_density.contiguous().data_ptr<float>(),
		out_probability.contiguous().data_ptr<float>());

	return std::make_tuple(rendered, out_logits, out_bin_logits, out_density, out_probability, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateBackwardCUDA(
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int H, int W, int D,
	const int R,
	const float opas_thresh,
	const torch::Tensor& means3D,
	const torch::Tensor& pts,
	const torch::Tensor& points_int,
	const torch::Tensor& cov3D,
	const torch::Tensor& opas,
	const torch::Tensor& semantics,
	const torch::Tensor& logits,
	const torch::Tensor& bin_logits,
	const torch::Tensor& density,
	const torch::Tensor& probability,
	const torch::Tensor& logits_grad,
	const torch::Tensor& bin_logits_grad,
	const torch::Tensor& density_grad) 
{
	const int P = means3D.size(0);
	const int N = pts.size(0);

	torch::Tensor means3D_grad = torch::zeros({P, 3}, means3D.options());
	torch::Tensor opas_grad = torch::zeros({P}, means3D.options());
	torch::Tensor semantics_grad = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor cov3D_grad = torch::zeros({P, 6}, means3D.options());

	torch::Tensor voxel2pts = torch::full({H * W * D}, -1, means3D.options().dtype(torch::kInt32));
  
	LocalAggregator::Aggregator::backward(
		P, R, N,
		H, W, D,
		opas_thresh,
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        points_int.contiguous().data_ptr<int>(),
		voxel2pts.contiguous().data_ptr<int>(),
		pts.contiguous().data_ptr<float>(),
		means3D.contiguous().data_ptr<float>(),
		cov3D.contiguous().data_ptr<float>(),
		opas.contiguous().data_ptr<float>(),
		semantics.contiguous().data_ptr<float>(),
		logits.contiguous().data_ptr<float>(),
		bin_logits.contiguous().data_ptr<float>(),
		density.contiguous().data_ptr<float>(),
		probability.contiguous().data_ptr<float>(),
		logits_grad.contiguous().data_ptr<float>(),
		bin_logits_grad.contiguous().data_ptr<float>(),
		density_grad.contiguous().data_ptr<float>(),
		means3D_grad.contiguous().data_ptr<float>(),
		opas_grad.contiguous().data_ptr<float>(),
		semantics_grad.contiguous().data_ptr<float>(),
		cov3D_grad.contiguous().data_ptr<float>());

	return std::make_tuple(means3D_grad, opas_grad, semantics_grad, cov3D_grad);
}
