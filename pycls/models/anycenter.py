from pycls.core.config import cfg
from pycls.models.blocks import (
    activation,
    conv2d,
    conv2d_cx,
    init_weights,
)
from pycls.models.anynet import get_block_fun, get_stem_fun, AnyStage
import torch.nn as nn

class Head(nn.Module):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in, channels, head_classes):
        super(Head, self).__init__()

        self.conv = conv2d(w_in, channels, 3)
        self.af = activation()
        self.conv_fc = conv2d(channels, head_classes, 1)

    def forward(self, x):
        x = self.af(self.conv(x))
        x = self.conv_fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, channels, head_classes):
        cx = conv2d_cx(cx, w_in, channels, 3)
        cx = conv2d_cx(cx, channels, head_classes, 1)

        return cx


class CenterHead(nn.Module):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in, heads):
        super(CenterHead, self).__init__()
        self.head_modules = nn.ModuleDict()
        for head, head_classes in heads.items():
            self.head_modules.add_module(head, Head(w_in, 256, head_classes))


    def forward(self, x):
        y = {}

        for head, head_module in self.head_modules.items():
            y[head] = head_module(x)

        return y

    @staticmethod
    def complexity(cx, w_in, heads):
        for head, head_classes in heads.items():
            cx = Head.complexity(cx, w_in, 256, head_classes)

        return cx

class AnyCenter(nn.Module):
    """AnyNet model."""

    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "heads": {"hm": cfg.ANYCENTER.HM_HEAD, "wh": cfg.ANYCENTER.WH_HEAD, "reg": cfg.ANYCENTER.REG_HEAD},
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super(AnyCenter, self).__init__()
        p = AnyCenter.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            stage = AnyStage(prev_w, w, s, d, block_fun, params)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = CenterHead(prev_w, p["heads"])
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = AnyCenter.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        cx = stem_fun.complexity(cx, 3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        for d, w, s, b, g in zip(*[p[k] for k in keys]):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, params)
            prev_w = w
        cx = CenterHead.complexity(cx, prev_w, p["heads"])
        return cx
