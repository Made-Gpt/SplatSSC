# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from model.serialization.point_transformer.serialization.z_order import KeyLUT
from mmengine.model import BaseModule


class CustomZOrder:
    def __init__(self): 
        self.init = False 
 
    def _call_init(self, ELUT=None, DLUT=None):
        """ ELUT = [EX, EY, EZ], DLUT = [DX, DY, DZ] """
        self.EX, self.EY, self.EZ = ELUT
        # self.DX, self.DY, self.DZ = DLUT
        self.init = True

    def custom_xyz2key(self, x, y, z, depth=16):
        """ Optimized position encoding using precomputed lookup tables """
        assert self.init, "Call '__init__()' first"
        x, y, z = x.long(), y.long(), z.long()

        mask = 255 if depth > 8 else (1 << depth) - 1
        key = self.EX[x & mask] | self.EY[y & mask] | self.EZ[z & mask]  # code in lower 8 bit

        if depth > 8:
            mask = (1 << (depth - 8)) - 1
            key16 = self.EX[(x >> 8) & mask] | self.EY[(y >> 8) & mask] | self.EZ[(z >> 8) & mask]
            key = key16 << 24 | key

        return key

    def custom_key2xyz(self, key: torch.Tensor, depth: int = 16):
        r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
        and the batch index based on pre-computed look up tables.
        Args:
          key (torch.Tensor): The shuffled key.
          depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
        """
        assert self.init, "Call '_call_init()' first"
        x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)
        b = key >> 48
        key = key & ((1 << 48) - 1)
        n = (depth + 2) // 3

        for i in range(n):
            k = key >> (i * 9) & 511
            x = x | (self.DX[k] << (i * 3))
            y = y | (self.DY[k] << (i * 3))
            z = z | (self.DZ[k] << (i * 3))

        return x, y, z, b


# avoid repeat init
_custom_z_order = CustomZOrder()


class CustomCoding(BaseModule):
    def __init__(self, lut_size=None):
        super(CustomCoding, self).__init__()
        if lut_size is not None:
            self.lut_size = lut_size
        else:
            self.lut_size = ['r256', 'r256', 'r256']
        self.init = False

    @torch.no_grad()
    def _call_init(self, device_type='cpu', ELUT: list=None, DLUT:list=None):
        """ ELUT = [EX, EY, EZ], DLUT = [DX, DY, DZ] """
        # if ELUT is not None and DLUT is not None:
        if ELUT is not None:
            self.EX, self.EY, self.EZ = ELUT
            # self.DX, self.DY, self.DZ = DLUT
            self.init = True
        else:
            self._key_lut = KeyLUT(self.lut_size)
            self.EX, self.EY, self.EZ = self.init_z_order_encode_lut(device_type)
            # self.DX, self.DY, self.DZ = self.init_z_order_decode_lut(device_type)
            self.init = True
        _custom_z_order._call_init(
            [self.EX, self.EY, self.EZ],
            # [self.DX, self.DY, self.DZ],
        )

    @torch.no_grad()
    def init_z_order_encode_lut(self, type='cpu'):
        EX, EY, EZ = self._key_lut.encode_lut(type)  # 'type' is device
        return EX, EY, EZ

    @torch.no_grad()
    def init_z_order_decode_lut(self, type='cpu'):
        DX, DY, DZ = self._key_lut.decode_lut(type)  # 'type' is device
        return DX, DY, DZ

    @torch.no_grad()
    def custom_z_order_encode(self, grid_coord: torch.Tensor, depth: int = 16):
        x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
        # we block the support to batch, maintain batched code in Point class
        code = _custom_z_order.custom_xyz2key(x, y, z, depth=depth)
        return code

