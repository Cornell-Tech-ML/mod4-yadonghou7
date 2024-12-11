from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input_tensor: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling.

    Args:
    ----
        input_tensor: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tuple of (reshaped tensor, new_height, new_width)

    """
    batch, channel, height, width = input_tensor.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height: int = height // kh
    new_width: int = width // kw
    res = (
        input_tensor.permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, channel, width, new_height, kh)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )
    return res, new_height, new_width


def avgpool2d(input_tensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute 2D average pooling.

    Args:
    ----
        input_tensor: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, _, _ = input_tensor.shape
    t, _, _ = tile(input_tensor, kernel)
    res = t.mean(4).view(batch, channel, t.shape[2], t.shape[3])
    return res


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input_tensor: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input_tensor: input tensor
        dim: dimension to apply argmax

    Returns:
    -------
        Tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input_tensor, dim)
    return out == input_tensor


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction."""
        max_red = max_reduce(input_tensor, int(dim.item()))
        ctx.save_for_backward(input_tensor, max_red)
        return max_red

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)."""
        (input_tensor, max_red) = ctx.saved_values
        return (grad_output * (max_red == input_tensor)), 0.0


def max(input_tensor: Tensor, dim: int) -> Tensor:
    """Compute maximum value along a dimension.

    Args:
    ----
        input_tensor: input tensor
        dim: dimension to reduce

    Returns:
    -------
        Tensor with max values

    """
    return Max.apply(input_tensor, input_tensor._ensure_tensor(dim))


def softmax(input_tensor: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input_tensor: input tensor
        dim: dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    m = max(input_tensor, dim)
    t = (input_tensor - m).exp()
    s = t.sum(dim)
    return t / s


def logsoftmax(input_tensor: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input_tensor: input tensor
        dim: dimension to apply log-softmax

    Returns:
    -------
        log of softmax tensor

    """
    m = max(input_tensor, dim)
    e = (input_tensor - m).exp()
    s = e.sum(dim=dim)
    return (input_tensor - m) - s.log()


def maxpool2d(input_tensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input_tensor: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, _, _ = input_tensor.shape
    tiled_input, _, _ = tile(input_tensor, kernel)
    pooled = max(tiled_input, 4)
    result = pooled.view(batch, channel, pooled.shape[2], pooled.shape[3])
    return result


def dropout(input_tensor: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input_tensor: input tensor
        rate: probability [0, 1) of dropping out each position
        ignore: skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with random positions dropped out

    """
    if not ignore:
        random_drop = rand(input_tensor.shape) > rate
        return input_tensor * random_drop
    else:
        return input_tensor
