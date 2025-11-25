# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union


class KeyLUT:
    VALID_SIZES = {'r32', 'r64', 'r128', 'r256', 'r512'}

    def __init__(self, xyz_size=None):
        r32 = torch.arange(32, dtype=torch.int64)  # 0-31
        r64 = torch.arange(64, dtype=torch.int64)  # 0-63
        r128 = torch.arange(128, dtype=torch.int64)  # 0-127
        r256 = torch.arange(256, dtype=torch.int64)  # 0-255
        r512 = torch.arange(512, dtype=torch.int64)  # 0-511
        zero = torch.zeros(256, dtype=torch.int64)  # same shape with r256
        device = torch.device("cpu")

        if xyz_size is None:
            xyz_size = ['r256', 'r256', 'r256']

        key_configs = {'r32': r32, 'r64': r64, 'r128': r128, 'r256': r256, 'r512': r512}
        x_range = key_configs[xyz_size[0]]
        x_zero = torch.zeros_like(x_range)
        y_range = key_configs[xyz_size[1]]
        y_zero = torch.zeros_like(y_range)
        z_range = key_configs[xyz_size[2]]
        z_zero = torch.zeros_like(z_range)

        ''' store LUT format for encoder and decoder '''
        self._encode = {
            device: (
                self.xyz2key(x_range, x_zero, x_zero, 8),  # Produces a LUT (EX) mapping every 8-bit x value to its corresponding binary key
                self.xyz2key(y_zero, y_range, y_zero, 8),  # Produces a LUT (EY) mapping every 8-bit y value to its corresponding binary key
                self.xyz2key(z_zero, z_zero, z_range, 8),  # Produces a LUT (EZ) mapping every 8-bit z value to its corresponding binary key
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}  # decode 3D location from key

    def encode_lut(self, device=torch.device("cpu")):
        """ ensure that LUT encoder is valid on the target device """
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        """ ensure that LUT decoder is valid on the target device """
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        """ Vectorized Z-order encoding without loops """
        key = torch.zeros_like(x)
        for i in range(depth):
            '''
                generate a mask and shift left by 1 bit in each loop
                eg. i=0: mask=0001, 
                    i=1: mask=0010, 
                    i=2: mask=0001,
                    (for 4 bit)
                '''
            mask = 1 << i
            # function of '&' is like a mask, eg. '1101' & '1000' = '1000'
            # function of '|' is like cat, eg. '1101' & '1000' = '1101'
            key = (
                    key
                    | ((x & mask) << (3 * i + 2))  # *4
                    | ((y & mask) << (3 * i + 1))  # *2
                    | ((z & mask) << (3 * i + 0))  # *1
            )
        return key

    def key2xyz(self, key, depth):
        """ transfer unique key to the 3D coordinate """
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()


def init_encoder_lut(type):
    EX, EY, EZ = _key_lut.encode_lut(type)  # 'type' is device
    return EX, EY, EZ


def init_decoder_lut(type):
    DX, DY, DZ = _key_lut.decode_lut(type)  # 'type' is device
    return DX, DY, DZ


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      z (torch.Tensor): The z coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be
          smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    EX, EY, EZ = init_encoder_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key


def key2xyz(key: torch.Tensor, depth: int = 16):
    r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    DX, DY, DZ = _key_lut.decode_lut(key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 48
    key = key & ((1 << 48) - 1)

    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 9) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b


"""
    def xyz2key(self, x, y, z, depth):
        ''' transfer 3D coordinate to the unique key '''
        indices = torch.arange(depth, dtype=torch.int64)

        x_bits = ((x[:, None] >> indices) & 1) << (2 * indices + 2)
        y_bits = ((y[:, None] >> indices) & 1) << (2 * indices + 1)
        z_bits = ((z[:, None] >> indices) & 1) << (2 * indices)

        return (x_bits | y_bits | z_bits).sum(dim=1)
"""
