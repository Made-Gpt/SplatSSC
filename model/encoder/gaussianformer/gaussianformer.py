# Copyright (c) Horizon Robotics. All rights reserved.
import time
import torch, torch.nn as nn

from mmengine import MODELS
from mmengine.model import BaseModule
from scipy.signal import ellip

from model.serialization import CustomSerialization
from typing import List, Optional, Union

try:
    from model.encoder.gaussianformer.ops.pointops import DeformableAggregationFunction as DAF
    print(
        f"{'== DAF IMPORTED ==':=^36}\n :: succeeded to import DAF to GaussianFormer! {'== DAF IMPORTED ==':=^36}\n")
except:
    DAF = None
    print(
        f"{'!! DAF IMPORTED !!':!^36}\n :: failed to import DAF to GaussianFormer... {'!! DAF IMPORTED !!':!^36}\n")


@MODELS.register_module()
class SparseGaussianFormer(CustomSerialization):
    def __init__(
        self,
        anchor_encoder,
        coding_param: dict,
        num_ref_pts: int,
        norm_layer: dict,
        ffn: dict,
        refine_layer: dict,
        deformable_model: dict,
        num_decoder: int = 6,
        spconv_layer: dict = None,
        mid_refine_layer: dict = None,
        spatial_attention_layer: dict = None,
        operation_order: Optional[List[str]] = None,
        supervision: bool = False,
        lut_range: list = None,
        grid_size: float = 0.08,
    ):
        super().__init__(**coding_param)

        self.num_decoder = num_decoder
        self.supervision = supervision
        self.num_ref_pts = num_ref_pts
        self.encoder_count = {}
        self.register_buffer('lut_range', torch.tensor(lut_range, dtype=torch.float), False)

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg):
            if cfg is None:
                return None
            return MODELS.build(cfg)

        self.anchor_encoder = build(anchor_encoder)
        self.op_config_map = {
            "ffn": ffn,
            "norm": norm_layer,
            "spconv": spconv_layer,
            "refine": refine_layer,
            "mid_refine":mid_refine_layer,
            "deformable": deformable_model,
            "spatial_attention": spatial_attention_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )

        self.grid_size = grid_size

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        ELUT: list,
        query_point: dict,  # reused_point: dict,
        feature_maps: Union[torch.Tensor, List],  # mlvl_img_feats
        metas: dict,
    ):
        curr_device = query_point['anchor'].device
        self._lut_init(curr_device, ELUT)

        # batch = self.curr_point['anchor'].shape[1]
        anchor_cache = query_point['anchor']  # output
        inference_feat_cache = query_point['feat']
        anchor_embed = self.anchor_encoder(anchor_cache)

        loop_time = 0
        predictions = []
        identity = inference_feat_cache
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                start_spconv_toc = time.time()
                inference_feat_cache = self.layers[i](
                    inference_feat_cache,
                    anchor_cache,
                    metas)

                identity = inference_feat_cache
                self.encoder_count[op] = time.time() - start_spconv_toc
                if self.supervision:
                    print(f"::: spconv time: {self.encoder_count[op]:.5f}s, feat shape: {inference_feat_cache.shape}")
            elif op == "norm" or op == "ffn":
                inference_feat_cache = self.layers[i](inference_feat_cache)
            elif op == "identity":
                identity = inference_feat_cache
            elif op == "add":
                inference_feat_cache = inference_feat_cache + identity
            elif op == "deformable":
                # assert feature_queue is None and meta_queue is None and self.depth_module is None
                start_deformable_toc = time.time()
                inference_feat_cache = self.layers[i](
                    inference_feat_cache,
                    anchor_cache,
                    anchor_embed,
                    feature_maps,
                    metas,
                    loop_time,
                )
                identity = inference_feat_cache
                self.encoder_count[op] = time.time() - start_deformable_toc
                loop_time += 1
                if self.supervision:
                    print(f"::: deformable time: {self.encoder_count[op]:.5f}s, feat shape: {inference_feat_cache.shape}")
            elif op == "refine":
                start_refine_toc = time.time()
                anchor_cache = self.layers[i](
                    inference_feat_cache,
                    anchor_cache,
                    anchor_embed,
                    metas,
                )
                predictions.append({
                    'anchor': anchor_cache,  # in camera coordinate
                })

                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor_cache)
                self.encoder_count[op] = time.time() - start_refine_toc
                if self.supervision:
                    print(f"::: refine time: {self.encoder_count[op]:.5f}s, feat shape: {inference_feat_cache.shape}")
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = anchor_cache

        # output: the latest output in decoder block (in camera coordinate);
        # prediction: the list of all refinement output in world coordinate;
        return output, predictions

