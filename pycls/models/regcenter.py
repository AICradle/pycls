#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RegNet models."""

import numpy as np
from pycls.core.config import cfg
from pycls.models.anycenter import AnyCenter

def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def generate_regcenter(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


class RegCenter(AnyCenter):
    """RegNet model."""

    @staticmethod
    def get_params():
        """Convert RegNet to AnyNet parameter format."""
        # Generates per stage ws, ds, gs, bs, and ss from RegNet parameters
        w_a, w_0, w_m, d = cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH
        ws, ds = generate_regcenter(w_a, w_0, w_m, d)[0:2]
        ss = [cfg.REGNET.STRIDE for _ in ws]
        bs = [cfg.REGNET.BOT_MUL for _ in ws]
        gs = [cfg.REGNET.GROUP_W for _ in ws]
        ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
        # Get AnyNet arguments defining the RegNet
        return {
            "stem_type": cfg.REGNET.STEM_TYPE,
            "stem_w": cfg.REGNET.STEM_W,
            "block_type": cfg.REGNET.BLOCK_TYPE,
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "heads": {"hm": cfg.ANYCENTER.HM_HEAD, "wh": cfg.ANYCENTER.WH_HEAD, "reg": cfg.ANYCENTER.REG_HEAD},
            "se_r": cfg.REGNET.SE_R if cfg.REGNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self):
        params = RegCenter.get_params()
        super(RegCenter, self).__init__(params)

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        params = RegCenter.get_params() if not params else params
        return AnyCenter.complexity(cx, params)
