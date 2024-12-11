from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just in time compile with NUMBA.

    Args:
    ----
        fn: Function to compile
        kwargs: Compilation options

    Returns:
    -------
        Compiled function

    """
    return _njit(inline="always", **kwargs)(fn)


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input_tensor: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.
    Given input tensor of...
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    for idx in prange(out_size):
        out_index: Index = [0] * 3
        data_index: Index = [0] * 3
        kernel_index: Index = [0] * 3

        to_index(idx, out_shape, out_index)
        cur_batch, cur_out_channels, cur_width = out_index
        val = 0.0
        for index1 in range(in_channels):
            for index2 in range(kw):
                if not reverse:
                    if (cur_width + index2) < width:
                        kernel_index[0], kernel_index[1], kernel_index[2] = (
                            cur_out_channels,
                            index1,
                            index2,
                        )
                        data_index[0], data_index[1], data_index[2] = (
                            cur_batch,
                            index1,
                            cur_width + index2,
                        )
                        val += (
                            input_tensor[index_to_position(data_index, s1)]
                            * weight[index_to_position(kernel_index, s2)]
                        )
                else:
                    if (cur_width - index2) >= 0:
                        kernel_index[0], kernel_index[1], kernel_index[2] = (
                            cur_out_channels,
                            index1,
                            index2,
                        )
                        data_index[0], data_index[1], data_index[2] = (
                            cur_batch,
                            index1,
                            cur_width - index2,
                        )
                        val += (
                            input_tensor[index_to_position(data_index, s1)]
                            * weight[index_to_position(kernel_index, s2)]
                        )

        out[index_to_position(out_index, out_strides)] = val


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input_tensor : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input_tensor, weight)
        batch, in_channels, w = input_tensor.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input_tensor.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input_tensor.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient for 1D convolution.

        Args:
        ----
            ctx: Context with saved tensors
            grad_output: Gradient with respect to output

        Returns:
        -------
            Tuple of gradients for input and weight

        """
        input_tensor, weight = ctx.saved_values
        batch, in_channels, w = input_tensor.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input_tensor.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input_tensor.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input_tensor: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.
    Given input tensor of
       `batch, in_channels, height, width`
    ...
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides

    for p in prange(out_size):
        out_index: Index = np.zeros(4, np.int32)
        weight_index: Index = np.zeros(4, np.int32)
        in_index: Index = np.zeros(4, np.int32)
        to_index(p, out_shape, out_index)
        cur_batch, cur_out_channels, cur_height, cur_width = out_index
        val = 0.0

        for index in range(in_channels):
            for h in range(kh):
                for w in range(kw):
                    if not reverse:
                        (
                            weight_index[0],
                            weight_index[1],
                            weight_index[2],
                            weight_index[3],
                        ) = (cur_out_channels, index, h, w)
                        in_index[0], in_index[1], in_index[2], in_index[3] = (
                            cur_batch,
                            index,
                            cur_height + h,
                            cur_width + w,
                        )
                        if cur_height + h < height and cur_width + w < width:
                            val += (
                                weight[index_to_position(weight_index, s2)]
                                * input_tensor[index_to_position(in_index, s1)]
                            )
                    else:
                        (
                            weight_index[0],
                            weight_index[1],
                            weight_index[2],
                            weight_index[3],
                        ) = (cur_out_channels, index, h, w)
                        in_index[0], in_index[1], in_index[2], in_index[3] = (
                            cur_batch,
                            index,
                            cur_height - h,
                            cur_width - w,
                        )
                        if cur_height - h >= 0 and cur_width - w >= 0:
                            val += (
                                weight[index_to_position(weight_index, s2)]
                                * input_tensor[index_to_position(in_index, s1)]
                            )
        out[index_to_position(out_index, out_strides)] = val


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input_tensor : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input_tensor, weight)
        batch, in_channels, h, w = input_tensor.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input_tensor.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input_tensor.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient for 2D convolution.

        Args:
        ----
            ctx: Context with saved tensors
            grad_output: Gradient with respect to output

        Returns:
        -------
            Tuple of gradients for input and weight

        """
        input_tensor, weight = ctx.saved_values
        batch, in_channels, h, w = input_tensor.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input_tensor.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input_tensor.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
