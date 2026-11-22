"""int8 conversion of the student network, the integer reference model, and
the C emitter for the gateway.

The arithmetic here is the specification of the gateway's engine
(hardware/gateway_node/kn_edge.cpp). Every integer operation in `int_forward`
has a line-for-line twin in C, and export_edge.py ships a self-test vector so
the gateway can prove at boot that it reproduces this file bit for bit.

Number format (TFLite-style, all tensors int8):
  * activations: per-tensor affine, real = scale * (q - zero_point)
  * weights:     per-output-channel symmetric, real = w_scale[c] * q, q in [-127, 127]
  * biases:      int32 in units of in_scale * w_scale[c]
  * rescaling:   y = (acc * m0 + 2^(sh-1)) >> sh   with m0 in [2^30, 2^31)
  * the classifier's output stays int32; logit[c] = acc[c] * in_scale * w_scale[c]

Only the ops MobileNetV2 needs exist: conv (any k, groups=1), depthwise conv,
ReLU / ReLU6 folded into the clamp, residual add, global average pool and the
final fully connected layer. Anything else is rejected at extraction time
rather than silently mis-converted.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT_VERSION = 1

OP_CONV = 1      # groups == 1, any kernel; 1x1/stride 1/no pad takes a fast path on the device
OP_DW = 2        # depthwise, channel multiplier 1
OP_ADD = 3
OP_GAP = 4
OP_FC = 5

ACT_NONE, ACT_RELU, ACT_RELU6 = 0, 1, 2
_OP_NAMES = {OP_CONV: "conv", OP_DW: "dwconv", OP_ADD: "add", OP_GAP: "avgpool", OP_FC: "fc"}


# ---------------------------------------------------------------------------
# Graph extraction (timm EfficientNet family: mobilenetv2_*)
# ---------------------------------------------------------------------------

@dataclass
class Tensor:
    id: int
    h: int
    w: int
    c: int
    dtype: str = "int8"  # the FC output is int32

    @property
    def nbytes(self) -> int:
        return self.h * self.w * self.c * (4 if self.dtype == "int32" else 1)


@dataclass
class Op:
    kind: int
    inputs: list[int]
    output: int
    weight: torch.Tensor | None = None   # float, BN folded. conv: (Cout, Cin/g, kh, kw); fc: (N, C)
    bias: torch.Tensor | None = None
    stride: int = 1
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, left, bottom, right
    act: int = ACT_NONE
    name: str = ""


@dataclass
class Graph:
    tensors: list[Tensor] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    input_id: int = 0
    output_id: int = -1

    def new_tensor(self, h: int, w: int, c: int, dtype: str = "int8") -> int:
        self.tensors.append(Tensor(len(self.tensors), h, w, c, dtype))
        return len(self.tensors) - 1

    def t(self, tid: int) -> Tensor:
        return self.tensors[tid]


def _act_code(module: nn.Module | None, where: str) -> int:
    if module is None or isinstance(module, nn.Identity):
        return ACT_NONE
    if isinstance(module, nn.ReLU6):
        return ACT_RELU6
    if isinstance(module, nn.ReLU):
        return ACT_RELU
    if isinstance(module, nn.Hardtanh) and module.min_val == 0.0 and module.max_val == 6.0:
        return ACT_RELU6
    raise NotImplementedError(f"{where}: activation {type(module).__name__} is not supported by the gateway engine")


def _require_identity(module: nn.Module | None, what: str) -> None:
    if module is not None and not isinstance(module, nn.Identity):
        raise NotImplementedError(f"{what} ({type(module).__name__}) is not supported by the gateway engine")


def _conv_padding(conv: nn.Conv2d, h: int, w: int) -> tuple[int, int, int, int]:
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    if type(conv).__name__ == "Conv2dSame":
        # TensorFlow "SAME": the extra pixel goes bottom/right.
        def same(size, k, s):
            total = max((math.ceil(size / s) - 1) * s + k - size, 0)
            return total // 2, total - total // 2
        t, b = same(h, kh, sh)
        l, r = same(w, kw, sw)
        return t, l, b, r
    if isinstance(conv.padding, str):
        raise NotImplementedError(f"string padding {conv.padding!r} is not supported")
    ph, pw = conv.padding
    return ph, pw, ph, pw


def _fold_bn(conv: nn.Conv2d, bn: nn.Module | None) -> tuple[torch.Tensor, torch.Tensor]:
    w = conv.weight.detach().double().clone()
    b = conv.bias.detach().double().clone() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    if bn is None or isinstance(bn, nn.Identity):
        return w, b
    if not isinstance(bn, nn.BatchNorm2d):
        raise NotImplementedError(f"norm layer {type(bn).__name__} is not supported")
    gamma = bn.weight.detach().double() if bn.weight is not None else torch.ones_like(b)
    beta = bn.bias.detach().double() if bn.bias is not None else torch.zeros_like(b)
    mean = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    k = gamma / torch.sqrt(var + bn.eps)
    return w * k.view(-1, 1, 1, 1), (b - mean) * k + beta


def _conv_bn(g: Graph, conv: nn.Module, bn: nn.Module | None, x: int, name: str) -> int:
    if not isinstance(conv, nn.Conv2d):
        raise NotImplementedError(f"{name}: {type(conv).__name__} is not supported (needs a plain Conv2d)")
    if conv.dilation != (1, 1):
        raise NotImplementedError(f"{name}: dilated convolution is not supported")
    if conv.stride[0] != conv.stride[1]:
        raise NotImplementedError(f"{name}: non-square stride is not supported")
    tin = g.t(x)
    if conv.in_channels != tin.c:
        raise RuntimeError(f"{name}: expects {conv.in_channels} channels, graph has {tin.c}")
    pad = _conv_padding(conv, tin.h, tin.w)
    kh, kw = conv.kernel_size
    s = conv.stride[0]
    oh = (tin.h + pad[0] + pad[2] - kh) // s + 1
    ow = (tin.w + pad[1] + pad[3] - kw) // s + 1
    if conv.groups == 1:
        kind = OP_CONV
    elif conv.groups == conv.in_channels == conv.out_channels:
        kind = OP_DW
    else:
        raise NotImplementedError(f"{name}: grouped convolution (groups={conv.groups}) is not supported")
    w, b = _fold_bn(conv, bn)
    act = _act_code(getattr(bn, "act", None), name)
    _require_identity(getattr(bn, "drop", None), f"{name}: dropout inside the norm layer")
    out = g.new_tensor(oh, ow, conv.out_channels)
    g.ops.append(Op(kind, [x], out, w, b, s, pad, act, name))
    return out


def _add(g: Graph, a: int, b: int, name: str) -> int:
    ta, tb = g.t(a), g.t(b)
    if (ta.h, ta.w, ta.c) != (tb.h, tb.w, tb.c):
        raise RuntimeError(f"{name}: residual shapes differ")
    out = g.new_tensor(ta.h, ta.w, ta.c)
    g.ops.append(Op(OP_ADD, [a, b], out, name=name))
    return out


def _has_skip(block: nn.Module) -> bool:
    return bool(getattr(block, "has_skip", getattr(block, "has_residual", False)))


def extract_graph(model: nn.Module, img_size: int) -> Graph:
    """Walk a timm EfficientNet-family model (MobileNetV2) into the op list."""
    model = model.eval()
    for attr in ("conv_stem", "bn1", "blocks", "conv_head", "bn2", "global_pool", "classifier"):
        if not hasattr(model, attr):
            raise NotImplementedError(f"model has no .{attr} - only timm mobilenetv2_* students are supported")

    g = Graph()
    x = g.new_tensor(img_size, img_size, 3)
    g.input_id = x
    x = _conv_bn(g, model.conv_stem, model.bn1, x, "stem")

    for si, stage in enumerate(model.blocks):
        for bi, block in enumerate(stage):
            name = f"blocks.{si}.{bi}"
            kind = type(block).__name__
            _require_identity(getattr(block, "se", None), f"{name}: squeeze-excite")
            _require_identity(getattr(block, "aa", None), f"{name}: anti-aliasing")
            shortcut = x
            if kind == "DepthwiseSeparableConv":
                x = _conv_bn(g, block.conv_dw, block.bn1, x, f"{name}.dw")
                x = _conv_bn(g, block.conv_pw, block.bn2, x, f"{name}.pw")
            elif kind == "InvertedResidual":
                x = _conv_bn(g, block.conv_pw, block.bn1, x, f"{name}.expand")
                x = _conv_bn(g, block.conv_dw, block.bn2, x, f"{name}.dw")
                x = _conv_bn(g, block.conv_pwl, block.bn3, x, f"{name}.project")
            else:
                raise NotImplementedError(f"{name}: block type {kind} is not supported by the gateway engine")
            if _has_skip(block):
                x = _add(g, x, shortcut, f"{name}.add")

    if not isinstance(model.conv_head, nn.Identity):
        x = _conv_bn(g, model.conv_head, model.bn2, x, "head")

    pool = model.global_pool
    if getattr(pool, "pool_type", "avg") not in ("avg", "") or not getattr(pool, "flatten", True):
        raise NotImplementedError("only global average pooling is supported")
    t = g.t(x)
    gap = g.new_tensor(1, 1, t.c)
    g.ops.append(Op(OP_GAP, [x], gap, name="avgpool"))

    fc = model.classifier
    if not isinstance(fc, nn.Linear):
        raise NotImplementedError("the classifier must be a single nn.Linear")
    out = g.new_tensor(1, 1, fc.out_features, dtype="int32")
    bias = fc.bias.detach().double() if fc.bias is not None else torch.zeros(fc.out_features, dtype=torch.float64)
    g.ops.append(Op(OP_FC, [gap], out, fc.weight.detach().double().clone(), bias, name="classifier"))
    g.output_id = out
    return g


# ---------------------------------------------------------------------------
# Float execution of the graph (verification + calibration)
# ---------------------------------------------------------------------------

def _apply_act(x: torch.Tensor, act: int) -> torch.Tensor:
    if act == ACT_RELU6:
        return x.clamp(0.0, 6.0)
    if act == ACT_RELU:
        return x.clamp_min(0.0)
    return x


def float_forward(g: Graph, x: torch.Tensor, keep: bool = False, observer=None) -> torch.Tensor | dict[int, torch.Tensor]:
    """x: (N, 3, H, W) normalized float. Returns logits, or every tensor with keep=True."""
    vals: dict[int, torch.Tensor] = {g.input_id: x}
    last_use = _last_use(g)
    if observer is not None:
        observer(g.input_id, x)
    for i, op in enumerate(g.ops):
        a = vals[op.inputs[0]]
        if op.kind in (OP_CONV, OP_DW):
            w = op.weight.to(a.dtype).to(a.device)
            b = op.bias.to(a.dtype).to(a.device)
            pt, pl, pb, pr = op.pad
            y = F.conv2d(F.pad(a, (pl, pr, pt, pb)), w, b, op.stride, groups=1 if op.kind == OP_CONV else a.shape[1])
            y = _apply_act(y, op.act)
        elif op.kind == OP_ADD:
            y = a + vals[op.inputs[1]]
        elif op.kind == OP_GAP:
            y = a.mean(dim=(2, 3), keepdim=True)
        elif op.kind == OP_FC:
            y = F.linear(a.flatten(1), op.weight.to(a.dtype).to(a.device), op.bias.to(a.dtype).to(a.device))
        else:
            raise AssertionError(op.kind)
        vals[op.output] = y
        if observer is not None:
            observer(op.output, y)
        if not keep:
            for tid in op.inputs:
                if last_use.get(tid) == i:
                    vals.pop(tid, None)
    return vals if keep else vals[g.output_id]


def _last_use(g: Graph) -> dict[int, int]:
    last: dict[int, int] = {}
    for i, op in enumerate(g.ops):
        for tid in op.inputs:
            last[tid] = i
    return last


def verify_graph(g: Graph, model: nn.Module, img_size: int, n: int = 2, seed: int = 0) -> float:
    """Max |graph - model| over random inputs, relative to the logit range."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, img_size, img_size, generator=gen)
    with torch.no_grad():
        ref = model.eval().double()(x.double())
        got = float_forward(g, x.double())
    model.float()
    return float((ref - got).abs().max() / ref.abs().max().clamp_min(1e-9))


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

@dataclass
class QParams:
    scale: float
    zp: int


def quantize_multiplier(m: float) -> tuple[int, int]:
    """Real multiplier -> (m0, shift) with m ~= m0 / 2^shift, m0 in [2^30, 2^31)."""
    if m <= 0.0:
        return 0, 1
    frac, exp = math.frexp(m)          # m = frac * 2^exp, frac in [0.5, 1)
    m0 = int(round(frac * (1 << 31)))
    if m0 == (1 << 31):
        m0 //= 2
        exp += 1
    shift = 31 - exp
    if shift > 62:                     # absurdly small - output is just the zero point
        return 0, 1
    if shift < 1:
        raise ValueError(f"rescale multiplier {m} is too large")
    return m0, shift


def qparams_from_range(lo: float, hi: float) -> QParams:
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    scale = (hi - lo) / 255.0
    zp = int(round(-128 - lo / scale))
    return QParams(scale, max(-128, min(127, zp)))


@dataclass
class QLayer:
    kind: int
    name: str
    inp: int
    inp2: int
    out: int
    act: int
    stride: int = 1
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)
    kh: int = 1
    kw: int = 1
    w_q: np.ndarray | None = None        # torch layout, int8 values
    w_scale: np.ndarray | None = None    # per output channel
    bias_q: np.ndarray | None = None     # int32
    m0: np.ndarray | None = None         # int32 per channel (or [1])
    shift: np.ndarray | None = None      # per channel (or [1])
    add_ma: int = 0
    add_mb: int = 0
    add_shift: int = 0
    act_min: int = -128
    act_max: int = 127


@dataclass
class QModel:
    graph: Graph
    qp: dict[int, QParams]
    layers: list[QLayer]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input_lut: np.ndarray            # (3, 256) int8
    logit_scale: np.ndarray          # (num_classes,) float32

    @property
    def img_size(self) -> int:
        return self.graph.t(self.graph.input_id).h

    @property
    def num_classes(self) -> int:
        return self.graph.t(self.graph.output_id).c


def input_qparams(mean, std) -> QParams:
    lo = min((0.0 - m) / s for m, s in zip(mean, std))
    hi = max((1.0 - m) / s for m, s in zip(mean, std))
    return qparams_from_range(lo, hi)


def make_input_lut(mean, std, qp: QParams) -> np.ndarray:
    v = np.arange(256, dtype=np.float64) / 255.0
    lut = np.empty((3, 256), dtype=np.int8)
    for c in range(3):
        real = (v - mean[c]) / std[c]
        q = np.floor(real / qp.scale + 0.5) + qp.zp      # round half up, same as the C side never needs (table)
        lut[c] = np.clip(q, -128, 127).astype(np.int8)
    return lut


class RangeObserver:
    """Per-tensor min/max over calibration batches.

    method="minmax"       absolute extremes (TFLite's representative-dataset behaviour)
    method="mean_minmax"  average of per-batch extremes (less sensitive to one odd image)
    """

    def __init__(self, method: str = "minmax"):
        if method not in ("minmax", "mean_minmax"):
            raise ValueError(method)
        self.method = method
        self.lo: dict[int, list[float]] = {}
        self.hi: dict[int, list[float]] = {}

    def __call__(self, tid: int, value: torch.Tensor) -> None:
        self.lo.setdefault(tid, []).append(float(value.min()))
        self.hi.setdefault(tid, []).append(float(value.max()))

    def range(self, tid: int) -> tuple[float, float]:
        lo, hi = self.lo[tid], self.hi[tid]
        if self.method == "minmax":
            return min(lo), max(hi)
        return float(np.mean(lo)), float(np.mean(hi))


def calibrate(g: Graph, batches: Iterable[torch.Tensor], method: str = "minmax") -> RangeObserver:
    obs = RangeObserver(method)
    with torch.no_grad():
        for x in batches:
            float_forward(g, x.double(), observer=obs)
    if not obs.lo:
        raise RuntimeError("calibration saw no data")
    return obs


def _act_clamp(act: int, qp: QParams) -> tuple[int, int]:
    if act == ACT_NONE:
        return -128, 127
    lo = max(-128, qp.zp)  # real 0
    if act == ACT_RELU:
        return lo, 127
    hi = min(127, qp.zp + int(math.floor(6.0 / qp.scale + 0.5)))
    return lo, hi


def quantize(g: Graph, obs: RangeObserver, mean, std) -> QModel:
    qp: dict[int, QParams] = {g.input_id: input_qparams(mean, std)}
    for op in g.ops:
        if op.kind == OP_FC:
            continue
        lo, hi = obs.range(op.output)
        if op.act == ACT_RELU6:
            lo, hi = 0.0, min(hi, 6.0)
        elif op.act == ACT_RELU:
            lo = 0.0
        qp[op.output] = qparams_from_range(lo, hi)

    layers: list[QLayer] = []
    logit_scale = None
    for op in g.ops:
        s_in = qp[op.inputs[0]]
        if op.kind in (OP_CONV, OP_DW, OP_FC):
            w = op.weight.double()
            cout = w.shape[0]
            absmax = w.reshape(cout, -1).abs().max(dim=1).values.numpy()
            w_scale = np.where(absmax > 0, absmax / 127.0, 1.0)
            shape = (-1,) + (1,) * (w.dim() - 1)
            w_q = np.clip(np.floor(w.numpy() / w_scale.reshape(shape) + 0.5), -127, 127).astype(np.int8)
            bias_q = np.floor(op.bias.numpy() / (s_in.scale * w_scale) + 0.5)
            if np.abs(bias_q).max() >= 2**31 - 1:
                raise OverflowError(f"{op.name}: bias does not fit int32")
            bias_q = bias_q.astype(np.int32)
            if op.kind == OP_FC:
                logit_scale = (s_in.scale * w_scale).astype(np.float32)
                layers.append(QLayer(op.kind, op.name, op.inputs[0], -1, op.output, ACT_NONE,
                                     w_q=w_q, w_scale=w_scale, bias_q=bias_q))
                continue
            s_out = qp[op.output]
            mults = [quantize_multiplier(s_in.scale * ws / s_out.scale) for ws in w_scale]
            lo, hi = _act_clamp(op.act, s_out)
            layers.append(QLayer(
                op.kind, op.name, op.inputs[0], -1, op.output, op.act, op.stride, op.pad,
                w.shape[2], w.shape[3], w_q, w_scale, bias_q,
                np.array([m for m, _ in mults], dtype=np.int64), np.array([s for _, s in mults], dtype=np.int64),
                act_min=lo, act_max=hi,
            ))
        elif op.kind == OP_ADD:
            sa, sb, so = s_in, qp[op.inputs[1]], qp[op.output]
            ra, rb = sa.scale / so.scale, sb.scale / so.scale
            sh = 30 - max(0, math.ceil(math.log2(max(ra, rb, 1e-12))))
            sh = max(1, min(30, sh))
            ma, mb = int(round(ra * (1 << sh))), int(round(rb * (1 << sh)))
            if ma >= 2**31 or mb >= 2**31:
                raise OverflowError(f"{op.name}: add multiplier overflow")
            layers.append(QLayer(op.kind, op.name, op.inputs[0], op.inputs[1], op.output, ACT_NONE,
                                 add_ma=ma, add_mb=mb, add_shift=sh))
        elif op.kind == OP_GAP:
            t = g.t(op.inputs[0])
            so = qp[op.output]
            m0, sh = quantize_multiplier(s_in.scale / (t.h * t.w * so.scale))
            layers.append(QLayer(op.kind, op.name, op.inputs[0], -1, op.output, ACT_NONE,
                                 m0=np.array([m0], dtype=np.int64), shift=np.array([sh], dtype=np.int64)))
    return QModel(g, qp, layers, tuple(mean), tuple(std), make_input_lut(mean, std, qp[g.input_id]), logit_scale)


# ---------------------------------------------------------------------------
# Integer reference (the specification of kn_edge.cpp)
# ---------------------------------------------------------------------------

def _requant(acc: torch.Tensor, m0: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
    """(acc * m0 + 2^(sh-1)) >> sh, all int64, broadcasting over channel dim 1."""
    one = torch.ones_like(sh)
    return torch.bitwise_right_shift(acc * m0 + torch.bitwise_left_shift(one, sh - 1), sh)


def int_forward(qm: QModel, x_q: torch.Tensor, keep: bool = False):
    """x_q: (N, H, W, 3) int8 (device layout). Returns int32 classifier accumulators
    (N, num_classes) as int64, or (acc, {tensor_id: int8 NCHW tensor}) with keep=True."""
    g = qm.graph
    dev = x_q.device
    vals: dict[int, torch.Tensor] = {g.input_id: x_q.permute(0, 3, 1, 2).to(torch.int64)}
    for layer in qm.layers:
        a = vals[layer.inp]
        zin = qm.qp[layer.inp].zp
        if layer.kind in (OP_CONV, OP_DW):
            zout = qm.qp[layer.out].zp
            xs = (a - zin).double()
            pt, pl, pb, pr = layer.pad
            xs = F.pad(xs, (pl, pr, pt, pb))       # zero in (x - zp) space == zero point in q space
            w = torch.from_numpy(layer.w_q.astype(np.float64)).to(dev)
            acc = F.conv2d(xs, w, None, layer.stride, groups=1 if layer.kind == OP_CONV else xs.shape[1])
            acc = torch.round(acc).to(torch.int64) + torch.from_numpy(layer.bias_q.astype(np.int64)).to(dev).view(1, -1, 1, 1)
            m0 = torch.from_numpy(layer.m0).to(dev).view(1, -1, 1, 1)
            sh = torch.from_numpy(layer.shift).to(dev).view(1, -1, 1, 1)
            y = (_requant(acc, m0, sh) + zout).clamp(layer.act_min, layer.act_max)
        elif layer.kind == OP_ADD:
            b = vals[layer.inp2]
            za, zb, zo = zin, qm.qp[layer.inp2].zp, qm.qp[layer.out].zp
            rnd = 1 << (layer.add_shift - 1)
            y = torch.bitwise_right_shift((a - za) * layer.add_ma + (b - zb) * layer.add_mb + rnd, layer.add_shift)
            y = (y + zo).clamp(-128, 127)
        elif layer.kind == OP_GAP:
            zo = qm.qp[layer.out].zp
            s = (a - zin).sum(dim=(2, 3), keepdim=True)
            y = (_requant(s, torch.tensor(int(layer.m0[0]), device=dev), torch.tensor(int(layer.shift[0]), device=dev))
                 + zo).clamp(-128, 127)
        elif layer.kind == OP_FC:
            xs = (a.flatten(1) - zin).double()
            w = torch.from_numpy(layer.w_q.astype(np.float64)).to(dev)
            y = torch.round(xs @ w.t()).to(torch.int64) + torch.from_numpy(layer.bias_q.astype(np.int64)).to(dev)
        else:
            raise AssertionError(layer.kind)
        vals[layer.out] = y
    out = vals[g.output_id]
    return (out, vals) if keep else out


def logits_from_acc(qm: QModel, acc: torch.Tensor) -> torch.Tensor:
    return acc.double() * torch.from_numpy(qm.logit_scale.astype(np.float64)).to(acc.device)


def dequantize_input(qm: QModel, x_q: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) int8 -> (N, 3, H, W) float, what the float model 'sees' on the device."""
    qp = qm.qp[qm.graph.input_id]
    return ((x_q.to(torch.float32) - qp.zp) * qp.scale).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Self-test vector (same PRNG on the gateway)
# ---------------------------------------------------------------------------

SELFTEST_SEED = 0x4B4E4544  # "KNED"


def xorshift_input(n: int, seed: int = SELFTEST_SEED) -> np.ndarray:
    out = np.empty(n, dtype=np.int8)
    s = seed & 0xFFFFFFFF
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        v = s >> 24
        out[i] = v - 256 if v >= 128 else v      # (int8_t)(s >> 24) on the device
    return out


def fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _fnv1a_np(arr: np.ndarray) -> int:
    # Vectorised is not possible for FNV, but the largest tensor is a few
    # hundred KB, so a tight loop over bytes is fine.
    return fnv1a(arr.tobytes())


def selftest_vector(qm: QModel) -> tuple[np.ndarray, list[int], np.ndarray]:
    s = qm.img_size
    x = xorshift_input(s * s * 3).reshape(1, s, s, 3)
    acc, vals = int_forward(qm, torch.from_numpy(x), keep=True)
    sums = []
    for layer in qm.layers:
        v = vals[layer.out]
        if layer.kind == OP_FC:
            arr = v.to(torch.int32).numpy().astype("<i4")
        else:
            arr = v.permute(0, 2, 3, 1).to(torch.int8).numpy()   # HWC, as the device stores it
        sums.append(_fnv1a_np(arr))
    return x, sums, acc[0].to(torch.int64).numpy()


# ---------------------------------------------------------------------------
# Arena planning
# ---------------------------------------------------------------------------

def plan_arena(qm: QModel, align: int = 16) -> tuple[dict[int, int], int]:
    g = qm.graph
    n_ops = len(qm.layers)
    born = {g.input_id: -1}
    dies = {g.input_id: -1}
    for i, layer in enumerate(qm.layers):
        born[layer.out] = i
        dies.setdefault(layer.out, i)
        for tid in (layer.inp, layer.inp2):
            if tid >= 0:
                dies[tid] = max(dies.get(tid, i), i)
    dies[g.output_id] = n_ops
    order = sorted(born, key=lambda t: -g.t(t).nbytes)
    placed: list[tuple[int, int, int, int]] = []   # off, size, born, dies
    offsets: dict[int, int] = {}
    for tid in order:
        size = (g.t(tid).nbytes + align - 1) // align * align
        b, d = born[tid], dies[tid]
        clashes = sorted((o, o + sz) for o, sz, pb, pd in placed if not (d < pb or pd < b))
        off = 0
        for lo, hi in clashes:
            if off + size <= lo:
                break
            off = max(off, (hi + align - 1) // align * align)
        offsets[tid] = off
        placed.append((off, size, b, d))
    total = max(o + s for o, s, _, _ in placed)
    return offsets, total


# ---------------------------------------------------------------------------
# C emitter
# ---------------------------------------------------------------------------

def _c_array(values: np.ndarray, per_line: int = 24) -> str:
    flat = values.reshape(-1).tolist()
    lines = []
    for i in range(0, len(flat), per_line):
        lines.append("  " + ",".join(str(int(v)) for v in flat[i:i + per_line]) + ",")
    return "\n".join(lines) if lines else "  0,"


def _c_float_array(values: np.ndarray, per_line: int = 8) -> str:
    flat = values.reshape(-1).tolist()
    lines = []
    for i in range(0, len(flat), per_line):
        lines.append("  " + ",".join(f"{float(v):.9e}f" for v in flat[i:i + per_line]) + ",")
    return "\n".join(lines)


def _device_weight_layout(layer: QLayer) -> np.ndarray:
    w = layer.w_q
    if layer.kind == OP_CONV:
        return np.ascontiguousarray(w.transpose(0, 2, 3, 1))     # [Cout][kh][kw][Cin]
    if layer.kind == OP_DW:
        return np.ascontiguousarray(w[:, 0].transpose(1, 2, 0))  # [kh][kw][C]
    if layer.kind == OP_FC:
        return np.ascontiguousarray(w)                           # [N][C]
    raise AssertionError(layer.kind)


def check_labels(labels: list[str]) -> None:
    for lab in labels:
        if not lab or len(lab) > 63 or any(ch in lab for ch in '"\\') or not all(0x20 <= ord(ch) < 0x7F for ch in lab):
            raise ValueError(
                f"class name {lab!r} cannot go into the gateway's JSON as-is (printable ASCII, "
                f"no quotes/backslashes, at most 63 chars)"
            )


def model_id(qm: QModel, student: str) -> str:
    h = hashlib.sha256()
    for layer in qm.layers:
        for arr in (layer.w_q, layer.bias_q, layer.m0, layer.shift):
            if arr is not None:
                h.update(np.ascontiguousarray(arr).tobytes())
    short = student.replace("mobilenetv2_", "mnv2_")
    return f"{short}-{qm.img_size}-{h.hexdigest()[:8]}"


def emit_c(qm: QModel, labels: list[str], out_dir: Path, *, student: str,
           test_top1: float = 0.0, test_top3: float = 0.0) -> dict:
    """Write kn_edge_model.h and kn_edge_model.cpp. Returns a summary dict."""
    check_labels(labels)
    g = qm.graph
    if len(labels) != qm.num_classes:
        raise ValueError("label count does not match the classifier")
    offsets, arena = plan_arena(qm)
    _, layer_sums, logits_ref = selftest_vector(qm)
    mid = model_id(qm, student)
    if test_top1 > 0 or test_top3 > 0:
        acc_line = (f"int8 test-set accuracy measured by the exporter: top-1 {test_top1 * 100:.2f}%, "
                    f"top-3 {test_top3 * 100:.2f}%.")
    else:
        acc_line = "int8 accuracy not measured (trained without a labelled test set)."

    w_blob: list[np.ndarray] = []
    w_off: list[int] = []
    pos = 0
    b_blob, m_blob, s_blob = [], [], []
    b_off, m_off = [], []
    bpos = mpos = 0
    max_c = 0
    for layer in qm.layers:
        tin, tout = g.t(layer.inp), g.t(layer.out)
        max_c = max(max_c, tin.c, tout.c)
        if layer.w_q is not None:
            arr = _device_weight_layout(layer).astype(np.int8).reshape(-1)
            pad = (-len(arr)) % 4
            w_off.append(pos)
            w_blob.append(np.concatenate([arr, np.zeros(pad, dtype=np.int8)]))
            pos += len(arr) + pad
            bias = layer.bias_q.astype(np.int64)
            if layer.kind == OP_CONV and layer.kh == 1 and layer.kw == 1 and layer.stride == 1 \
                    and layer.pad == (0, 0, 0, 0):
                # Fast path on the device: acc = bias' + sum(x * w) with
                # bias' = bias - zp_in * sum(w), so no per-MAC zero-point subtract.
                zin = qm.qp[layer.inp].zp
                bias = bias - zin * layer.w_q.reshape(layer.w_q.shape[0], -1).astype(np.int64).sum(axis=1)
            if np.abs(bias).max() >= 2**31:
                raise OverflowError(f"{layer.name}: folded bias does not fit int32")
            b_off.append(bpos)
            b_blob.append(bias.astype(np.int32))
            bpos += len(bias)
        else:
            w_off.append(-1)
            b_off.append(-1)
        if layer.m0 is not None:
            m_off.append(mpos)
            m_blob.append(layer.m0.astype(np.int64))
            s_blob.append(layer.shift.astype(np.int64))
            mpos += len(layer.m0)
        else:
            m_off.append(-1)

    w_all = np.concatenate(w_blob) if w_blob else np.zeros(4, np.int8)
    b_all = np.concatenate(b_blob).astype(np.int32)
    m_all = np.concatenate(m_blob)
    s_all = np.concatenate(s_blob)
    if m_all.max() >= 2**31 or s_all.min() < 1 or s_all.max() > 62:
        raise OverflowError("multiplier table out of range")

    layer_lines = []
    for i, layer in enumerate(qm.layers):
        tin, tout = g.t(layer.inp), g.t(layer.out)
        zin = qm.qp[layer.inp].zp
        zin2 = qm.qp[layer.inp2].zp if layer.inp2 >= 0 else 0
        zout = qm.qp[layer.out].zp if layer.kind != OP_FC else 0
        pt, pl, _, _ = layer.pad
        w_ptr = f"kn_em_w + {w_off[i]}" if w_off[i] >= 0 else "nullptr"
        b_ptr = f"kn_em_b + {b_off[i]}" if b_off[i] >= 0 else "nullptr"
        m_ptr = f"kn_em_m + {m_off[i]}" if m_off[i] >= 0 else "nullptr"
        s_ptr = f"kn_em_s + {m_off[i]}" if m_off[i] >= 0 else "nullptr"
        in2 = offsets[layer.inp2] if layer.inp2 >= 0 else 0
        layer_lines.append(
            "  {" + ", ".join(str(v) for v in (
                layer.kind, layer.act, layer.kh, layer.kw, layer.stride, pt, pl, 0,
                tin.h, tin.w, tin.c, tout.h, tout.w, tout.c,
                offsets[layer.inp], in2, offsets[layer.out],
                zin, zin2, zout, layer.act_min, layer.act_max, layer.add_shift,
                layer.add_ma, layer.add_mb,
            )) + f", {w_ptr}, {b_ptr}, {m_ptr}, {s_ptr}}},  // {i:2d} {_OP_NAMES[layer.kind]} {layer.name}"
        )

    label_lines = ",\n".join(f'  "{lab}"' for lab in labels)
    in_qp = qm.qp[g.input_id]
    header = f"""// GENERATED by ml/edge/export_edge.py - do not edit by hand.
// On-device disease model for the KrishiNova gateway: {mid}
#pragma once
#include "kn_edge.h"

#define KN_EDGE_MODEL_ID "{mid}"

extern const kn_edge_model_t kn_edge_model;
"""
    src = f"""// GENERATED by ml/edge/export_edge.py - do not edit by hand.
// Model {mid}: {student} at {qm.img_size}x{qm.img_size}, {qm.num_classes} classes,
// {len(qm.layers)} layers, {len(w_all)} weight bytes, arena {arena} bytes.
// {acc_line}
#include "kn_edge_model.h"

static_assert(KN_EDGE_FORMAT_VERSION == {FORMAT_VERSION},
              "kn_edge_model.cpp was generated for a different kn_edge engine - copy the kn_edge.* files "
              "from the same commit as the exporter, or re-run export_edge.py");

alignas(16) static const int8_t kn_em_w[] = {{
{_c_array(w_all, 32)}
}};

static const int32_t kn_em_b[] = {{
{_c_array(b_all, 12)}
}};

static const int32_t kn_em_m[] = {{
{_c_array(m_all, 8)}
}};

static const int8_t kn_em_s[] = {{
{_c_array(s_all, 32)}
}};

static const int8_t kn_em_lut[3][256] = {{
  {{{",".join(str(int(v)) for v in qm.input_lut[0])}}},
  {{{",".join(str(int(v)) for v in qm.input_lut[1])}}},
  {{{",".join(str(int(v)) for v in qm.input_lut[2])}}},
}};

static const float kn_em_logit_scale[] = {{
{_c_float_array(qm.logit_scale)}
}};

static const char *const kn_em_labels[] = {{
{label_lines}
}};

static const uint32_t kn_em_selftest_fnv[] = {{
{_c_array(np.array(layer_sums, dtype=np.uint64), 6)}
}};

static const int32_t kn_em_selftest_logits[] = {{
{_c_array(logits_ref.astype(np.int64), 10)}
}};

// op, act, kh, kw, stride, pad_t, pad_l, _, in_h, in_w, in_c, out_h, out_w, out_c,
// in_off, in2_off, out_off, in_zp, in2_zp, out_zp, act_min, act_max, add_shift,
// add_ma, add_mb, w, bias, mult, shift
static const kn_edge_layer_t kn_em_layers[] = {{
{chr(10).join(layer_lines)}
}};

const kn_edge_model_t kn_edge_model = {{
  KN_EDGE_MODEL_ID,
  {FORMAT_VERSION},
  {qm.img_size}, {qm.img_size},
  {qm.num_classes},
  {len(qm.layers)},
  {arena},
  {offsets[g.input_id]},
  {offsets[g.output_id]},
  {max_c},
  kn_em_lut,
  kn_em_logit_scale,
  kn_em_layers,
  kn_em_labels,
  {SELFTEST_SEED}u,
  kn_em_selftest_fnv,
  kn_em_selftest_logits,
  {test_top1:.4f}f,
  {test_top3:.4f}f,
  {in_qp.zp},
}};
"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kn_edge_model.h").write_text(header, encoding="utf-8", newline="\n")
    (out_dir / "kn_edge_model.cpp").write_text(src, encoding="utf-8", newline="\n")
    return {
        "model_id": mid,
        "layers": len(qm.layers),
        "weight_bytes": int(len(w_all)),
        "arena_bytes": int(arena),
        "max_channels": int(max_c),
        "macs": int(count_macs(qm)),
        "out_dir": str(out_dir),
    }


def count_macs(qm: QModel) -> int:
    g = qm.graph
    total = 0
    for layer in qm.layers:
        tin, tout = g.t(layer.inp), g.t(layer.out)
        if layer.kind == OP_CONV:
            total += tout.h * tout.w * tout.c * layer.kh * layer.kw * tin.c
        elif layer.kind == OP_DW:
            total += tout.h * tout.w * tout.c * layer.kh * layer.kw
        elif layer.kind == OP_FC:
            total += tin.c * tout.c
    return total
