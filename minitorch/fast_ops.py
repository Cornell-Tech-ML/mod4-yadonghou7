from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function with Numba, inlining it always."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if list(out_shape) == list(in_shape) and list(out_strides) == list(in_strides):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(len(out)):
                in_i = np.empty(MAX_DIMS, np.int32)
                out_i = np.empty(MAX_DIMS, np.int32)
                temp_i = i
                to_index(temp_i, out_shape, out_i)
                broadcast_index(out_i, out_shape, in_shape, in_i)
                in_pos = index_to_position(in_i, in_strides)
                out_pos = index_to_position(out_i, out_strides)
                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        """Element-wise binary operation on two tensors with broadcasting.

        Args:
        ----
            out: Output storage for the result tensor.
            out_shape: Shape of the output tensor.
            out_strides: Strides of the output tensor.
            a_storage: Storage for input tensor A.
            a_shape: Shape of input tensor A.
            a_strides: Strides of input tensor A.
            b_storage: Storage for input tensor B.
            b_shape: Shape of input tensor B.
            b_strides: Strides of input tensor B.
            fn: Binary function to apply to corresponding elements.

        Returns:
        -------
            None. The result is written to `out`.

        """
        MAX_DIMS = len(out_shape)

        # If shapes and strides are identical, avoid indexing
        if (list(out_shape) == list(a_shape) == list(b_shape)) and (
            list(out_strides) == list(a_strides) == list(b_strides)
        ):
            for i in prange(len(out)):  # Parallel main loop
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(len(out)):  # Parallel loop with broadcasting
                a_i = np.empty(MAX_DIMS, np.int32)
                b_i = np.empty(MAX_DIMS, np.int32)
                out_i = np.empty(MAX_DIMS, np.int32)
                temp_i = i
                to_index(temp_i, out_shape, out_i)
                broadcast_index(out_i, out_shape, a_shape, a_i)
                broadcast_index(out_i, out_shape, b_shape, b_i)
                a_pos = index_to_position(a_i, a_strides)
                b_pos = index_to_position(b_i, b_strides)
                out_pos = index_to_position(out_i, out_strides)
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        """Reduce a tensor along a specified dimension using a reduction function.

        Args:
        ----
            out: Output storage for the reduced tensor.
            out_shape: Shape of the output tensor.
            out_strides: Strides of the output tensor.
            a_storage: Storage for the input tensor.
            a_shape: Shape of the input tensor.
            a_strides: Strides of the input tensor.
            reduce_dim: The dimension along which to reduce.
            fn: Reduction function to apply.

        Returns:
        -------
            None. The result is written to `out`.

        """
        MAX_DIMS = len(a_shape)
        reduce_size = a_shape[
            reduce_dim
        ]  # Number of elements in the reduction dimension
        reduce_stride = a_strides[reduce_dim]  # Stride for the reduction dimension

        for i in prange(len(out)):  # Parallel loop over output elements
            # Convert linear index `i` to multi-dimensional index for output tensor
            out_i = np.empty(MAX_DIMS, np.int32)
            temp_i = i
            to_index(temp_i, out_shape, out_i)

            # Calculate linear positions in output and input storage
            out_pos = index_to_position(out_i, out_strides)
            in_pos = index_to_position(out_i, a_strides)

            # Initialize the reduction with a neutral value (e.g., 0 for sum, -inf for max)
            cur = out[out_pos]

            # Perform the reduction along the specified dimension
            for _ in range(reduce_size):
                cur = fn(cur, a_storage[in_pos])
                in_pos += (
                    reduce_stride  # Move to the next element in the reduction dimension
                )

            # Write the reduced value to the output storage
            out[out_pos] = cur

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    reduce_size = a_shape[-1]  # Shared dimension
    a_inner_stride = a_strides[-1]  # Stride for inner dimension of A
    b_inner_stride = b_strides[-2]  # Stride for inner dimension of B

    # Outer loops (parallelized over batches and output elements)
    for batch in prange(out_shape[0]):  # Iterate over batches
        for i in range(out_shape[1]):  # Iterate over rows of the result
            for j in range(out_shape[2]):  # Iterate over columns of the result
                # Initialize accumulation for the dot product
                cur = 0.0

                # Calculate starting positions for A and B
                a_pos = batch * a_batch_stride + i * a_strides[-2]
                b_pos = batch * b_batch_stride + j * b_strides[-1]

                # Inner loop: Perform dot product
                for k in range(reduce_size):
                    cur += a_storage[a_pos] * b_storage[b_pos]
                    a_pos += a_inner_stride  # Move along the row of A
                    b_pos += b_inner_stride  # Move along the column of B

                # Write the result to the output tensor
                out_pos = (
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )
                out[out_pos] = cur


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
