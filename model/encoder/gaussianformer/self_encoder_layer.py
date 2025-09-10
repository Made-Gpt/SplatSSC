import time
import torch, torch.nn as nn

from mmengine import MODELS
from mmengine.model import BaseModule
from model.serialization import CustomSerialization
from typing import List, Optional, Union


@MODELS.register_module()
class SparseSelfEncoderNew(CustomSerialization):
    def __init__(
            self,
            anchor_encoder,
            coding_param: dict,
            num_ref_pts: int,
            norm_layer: dict,
            num_decoder: int = 6,
            spconv_layer: dict = None,
            operation_order: Optional[List[str]] = None,
            supervision: bool = False,
    ):
        super().__init__(**coding_param)
        self.num_decoder = num_decoder
        self.supervision = supervision
        self.num_ref_pts = num_ref_pts

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg):
            if cfg is not None:
                return MODELS.build(cfg)
            return None

        self.anchor_encoder = build(anchor_encoder)
        self.op_config_map = {
            "norm": norm_layer,
            "spconv": spconv_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )

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
            anchor_cache,
            inference_feat_cache,
            metas: dict,
    ):
        self._lut_init(anchor_cache.device, ELUT)

        identity = inference_feat_cache
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                start_spconv_toc = time.time()
                inference_feat_cache = self.layers[i](
                    inference_feat_cache,
                    anchor_cache,
                    metas)
                spconv_toc = time.time() - start_spconv_toc
                if self.supervision:
                    print(f"::: spconv time: {spconv_toc:.5f}s, feat shape: {inference_feat_cache.shape}")
            elif op == "norm" or op == "ffn":
                inference_feat_cache = self.layers[i](inference_feat_cache)
            elif op == "identity":
                identity = inference_feat_cache
            elif op == "add":
                inference_feat_cache = inference_feat_cache + identity
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return anchor_cache, inference_feat_cache
