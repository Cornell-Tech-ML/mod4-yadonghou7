from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height: int = height // kh
    new_width: int = width // kw
    res = (
        input.permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, channel, width, new_height, kh)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )
    return res, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    batch, channel, _, _ = input.shape
    t, _, _ = tile(input, kernel)
    res = t.mean(4).view(batch, channel, t.shape[2], t.shape[3])
    return res


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        max_red = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, max_red)
        return max_red

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        (input, max_red) = ctx.saved_values
        return (grad_output * (max_red == input)), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    m = max(input, dim)
    t = (input - m).exp()
    s = t.sum(dim)
    return t / s


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    m = max(input, dim)
    e = (input - m).exp()
    s = e.sum(dim=dim)
    return (input - m) - s.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    batch, channel, _, _ = input.shape
    tiled_input, _, _ = tile(input, kernel)
    pooled = max(tiled_input, 4)
    result = pooled.view(batch, channel, pooled.shape[2], pooled.shape[3])
    return result


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    if not ignore:
        random_drop = rand(input.shape) > rate
        return input * random_drop
    else:
        return input