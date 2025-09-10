#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="occ_prob",
    packages=['occ_prob'],
    ext_modules=[
        CUDAExtension(
            name="occ_prob._C",
            sources=[
                "src/aggregator_impl.cu",
                "src/forward.cu",
                "src/backward.cu",
                "occ_aggregate.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": [
                    "-Xcompiler", "-fno-gnu-unique",
                    "-Xcudafe", "--diag_suppress=186"  # 添加抑制警告选项
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
