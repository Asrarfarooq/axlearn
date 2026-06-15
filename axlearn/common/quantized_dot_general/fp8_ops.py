# Copyright © 2025 Apple Inc.
"""Ops for FP8 training. Doesn't support gradient accumulation yet."""

from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
from flax.linen import fp8_ops
from jax._src.typing import DTypeLike
from jax.custom_derivatives import custom_vjp

from axlearn.common.utils import Tensor


def _quantize(
    x: Tensor,
    scale: Tensor,
    amax_history: Optional[Tensor],
    *,
    dtype: DTypeLike,
    preferred_element_type: DTypeLike,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    dtype_max = fp8_ops.get_fp8_max(dtype, jnp.float32)

    # This branch handles unbalanced batch dimension, like (X, Y) @ (B, X, Y), e.g in
    # FusedQKVLinear. In general, it's not mathematically possible to compute the gradient when
    # each batch of lhs/rhs has different scaling factor and one of them is missing a batch dim.
    # In these cases, only use one scaling factor.
    is_non_balanced_batch = len(scale.shape) > 0
    if is_non_balanced_batch:
        full_scale = scale
        full_amax_history = amax_history
        if amax_history is not None:
            assert scale.shape[0] == amax_history.shape[0]
            amax_history = amax_history[0]
        scale = scale[0]

    if amax_history is None:
        amax = jnp.max(jnp.abs(x)).astype(scale.dtype)
        new_history = None
    else:
        amax = jnp.max(amax_history, axis=0)
        new_history = fp8_ops.compute_amax_history(x, amax_history)
    new_scale = fp8_ops.compute_scale(amax, scale, dtype_max)
    q_x = fp8_ops.quantize(x, dtype, new_scale, preferred_element_type)

    if is_non_balanced_batch:
        new_scale = full_scale.at[0].set(new_scale)
        if new_history is not None:
            new_history = full_amax_history.at[0].set(new_history)

    return q_x, new_scale, new_history


def _dequantize(x: Tensor, scale: Tensor, *, dq_dtype: DTypeLike):
    if len(scale.shape) > 0:
        scale = scale[0]
    return x.astype(dq_dtype) * jnp.broadcast_to(scale.astype(dq_dtype), x.shape)


# TODO: Try to reduce positional arguments
# pylint: disable-next=too-many-positional-arguments
def _q_dot_dq_impl(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
    is_training: bool,
) -> Union[Tensor, tuple[Tensor, tuple[Tensor, ...]]]:
    """See `q_dot_dq_in_batch`.

    Also returns the residuals for custom_vjp backward if `is_training` is True.
    """
    q_lhs, lhs_scale, lhs_amax_history = _quantize(
        lhs,
        lhs_scale,
        lhs_amax_history,
        dtype=jnp.float8_e4m3fn,
        preferred_element_type=preferred_element_type,
    )
    q_rhs, rhs_scale, rhs_amax_history = _quantize(
        rhs,
        rhs_scale,
        rhs_amax_history,
        dtype=jnp.float8_e4m3fn,
        preferred_element_type=preferred_element_type,
    )

    out = jax.lax.dot_general(
        q_lhs,
        q_rhs,
        dimension_numbers,
        preferred_element_type=preferred_element_type,
        precision=precision,
    )

    out = _dequantize(out, lhs_scale * rhs_scale, dq_dtype=preferred_element_type)
    if is_training:
        res = (
            lhs,
            rhs,
            q_lhs,
            q_rhs,
            lhs_scale,
            rhs_scale,
            out_grad_scale,
            lhs_amax_history,
            rhs_amax_history,
            out_grad_amax_history,
        )
        return out, res
    else:
        return out


# pylint: disable=unused-argument
@partial(custom_vjp, nondiff_argnums=(8, 9, 10))
# TODO: Try to reduce positional arguments
# pylint: disable-next=too-many-positional-arguments
def q_dot_q(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike = None,
) -> Tensor:
    """Computes lhs @ rhs in FP8 using either in-batch scaling or delayed scaling.

    If the amax histories are None, in-batch scaling is used. Otherwise, delayed scaling is used.

    lhs and rhs are divided by scales computed using the amax values before performing matmul in
    fp8 precision. The scales passed into this function are previous scales, used when the newly
    computed scales are zero or inf.

    Args:
        lhs: Left-hand side tensor of matmul.
        lhs: Right-hand side tensor of matmul.
        lhs_scale: The previous scale of lhs.
        rhs_scale: The previous scale of rhs.
        out_grad_scale: The previous scale of output gradient.
        lhs_amax_history: Amax history of lhs.
        rhs_amax_history: Amax history of rhs.
        out_grad_amax_history: Amax history of output gradient.
        dimension_numbers: See `lax.dot_general`.
        precision: Precision of the dot.
        preferred_element_type: See `lax.dot_general`.

    Returns:
        The result of lhs @ rhs after dequantization.
    """
    return _q_dot_dq_impl(**locals(), is_training=False)


# TODO: Try to reduce positional arguments
# pylint: disable-next=too-many-positional-arguments
def _q_dot_dq_fwd(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
):
    """See `q_dot_dq_in_batch`."""
    return _q_dot_dq_impl(
        **locals(),
        is_training=True,
    )


# pylint: enable=unused-argument


def _q_dot_dq_bwd(
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
    res: tuple[Tensor, ...],
    g: Tensor,
) -> tuple[Tensor, ...]:
    (
        lhs,
        rhs,
        q_lhs,
        q_rhs,
        new_lhs_scale,
        new_rhs_scale,
        out_grad_scale,
        lhs_amax_history,
        rhs_amax_history,
        out_grad_amax_history,
    ) = res

    q_g, new_out_grad_scale, out_grad_amax_history = _quantize(
        g,
        out_grad_scale,
        out_grad_amax_history,
        dtype=jnp.float8_e5m2,
        preferred_element_type=preferred_element_type,
    )

    grad_lhs = fp8_ops.dot_general_transpose_lhs(
        q_g,
        lhs,
        q_rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    grad_lhs = _dequantize(
        grad_lhs, new_rhs_scale * new_out_grad_scale, dq_dtype=preferred_element_type
    )

    grad_rhs = fp8_ops.dot_general_transpose_rhs(
        q_g,
        q_lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    grad_rhs = _dequantize(
        grad_rhs, new_lhs_scale * new_out_grad_scale, dq_dtype=preferred_element_type
    )

    return (
        grad_lhs,
        grad_rhs,
        new_lhs_scale,
        new_rhs_scale,
        new_out_grad_scale,
        lhs_amax_history,
        rhs_amax_history,
        out_grad_amax_history,
    )


q_dot_q.defvjp(_q_dot_dq_fwd, _q_dot_dq_bwd)


# ==============================================================================
# Optimized TPU Pallas FP8 Kernel Integration
# ==============================================================================

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools


def _get_max_min(target_dtype):
    if target_dtype == jnp.int4 or target_dtype == jnp.int8:
        return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min
    else:
        return jnp.finfo(target_dtype).max.astype(jnp.bfloat16), jnp.finfo(
            target_dtype
        ).min.astype(jnp.bfloat16)


def _quantize_block(data, axis, target_dtype, use_mxfp8: bool):
    abs_max = jnp.max(
        jnp.abs(data),
        axis=axis,
        keepdims=True,
    )
    dtype_max, dtype_min = _get_max_min(target_dtype)
    scale = abs_max / dtype_max
    if use_mxfp8:
        scale = jnp.exp2(jnp.ceil(jnp.log2(scale)))
    if target_dtype == jnp.int4 or target_dtype == jnp.int8:
        data_q = jnp.round(data / scale).astype(target_dtype)
    else:
        data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    return data_q, scale


def _get_mxu_column_size() -> int:
    return pltpu.get_tpu_info().mxu_column_size if pltpu.is_tpu_device() else 128


@functools.partial(
    jax.jit,
    static_argnames=[
        "block_m",
        "block_n",
        "block_k",
        "sc_size",
        "use_mxfp8",
        "dtype_lhs",
        "dtype_out",
        "use_bf16_acc",
        "n_lane_multiplier",
        "rhs_transposed",
        "is_prequantized",
    ],
)
def quantized_matmul_kernel(
    lhs: jax.Array,
    rhs: jax.Array,
    w_scales: jax.Array | None = None,
    a_scales: jax.Array | None = None,
    *,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 256,
    sc_size: int = 512,
    use_mxfp8: bool = False,
    dtype_lhs: jnp.dtype = jnp.float8_e4m3fn,
    dtype_out: jnp.dtype = jnp.float32,
    use_bf16_acc: bool = True,
    n_lane_multiplier: int = 1,
    rhs_transposed: bool = False,
    is_prequantized: bool = False,
) -> jax.Array:
    m, k_dim = lhs.shape
    if rhs_transposed:
        n, k_dim_rhs = rhs.shape
    else:
        k_dim_rhs, n = rhs.shape

    assert k_dim == k_dim_rhs, "Contracting dimensions must match"
    assert m % block_m == 0, f"M ({m}) must be divisible by block_m ({block_m})"
    assert n % block_n == 0, f"N ({n}) must be divisible by block_n ({block_n})"
    assert (
        k_dim % block_k == 0
    ), f"K ({k_dim}) must be divisible by block_k ({block_k})"
    assert block_k % sc_size == 0, "Block K must be divisible by sub-channel size"
    steps_k = block_k // sc_size
    compute_tile_n = _get_mxu_column_size() * n_lane_multiplier
    steps_n = block_n // compute_tile_n

    def _kernel(lhs_ref, rhs_ref, w_scales_ref, a_scales_ref, out_ref, acc_scratch):
        pid_k = pl.program_id(2)
        is_first_step = pid_k == 0
        is_last_step = pid_k == (k_dim // block_k - 1)

        @pl.when(is_first_step)
        def _init():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        acc_dtype = jnp.bfloat16 if use_bf16_acc else jnp.float32

        if is_prequantized:
            for i in range(steps_k):
                k_start, k_end = i * sc_size, (i + 1) * sc_size
                lhs_q = lhs_ref[:, k_start:k_end]
                rhs_q_full = (
                    rhs_ref[:, k_start:k_end]
                    if rhs_transposed
                    else rhs_ref[k_start:k_end, :]
                )
                for j in range(steps_n):
                    n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n
                    rhs_q_slice = (
                        rhs_q_full[n_start:n_end, :]
                        if rhs_transposed
                        else rhs_q_full[:, n_start:n_end]
                    )
                    dot_res = jax.lax.dot_general(
                        lhs_q,
                        rhs_q_slice,
                        (((1,), (1,)), ((), ()))
                        if rhs_transposed
                        else (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    # Accumulate directly to prevent VMEM overflow
                    acc_scratch[0:block_m, n_start:n_end] += dot_res.astype(acc_dtype)
        else:
            lhs_q_list = []
            lhs_scale_list = []
            rhs_q_list = []
            rhs_scale_list = []

            for i in range(steps_k):
                k_start, k_end = i * sc_size, (i + 1) * sc_size
                lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
                if a_scales_ref is None:
                    l_q, l_s = _quantize_block(lhs_sub, 1, dtype_lhs, use_mxfp8)
                else:
                    l_q = (lhs_sub / a_scales_ref[0]).astype(dtype_lhs)
                    l_s = a_scales_ref[0]
                lhs_q_list.append(l_q)
                lhs_scale_list.append(l_s.astype(acc_dtype))
                if w_scales_ref is None:
                    rhs_sub = (
                        rhs_ref[:, k_start:k_end]
                        if rhs_transposed
                        else rhs_ref[k_start:k_end, :]
                    ).astype(jnp.float32)
                    r_q, r_s = _quantize_block(
                        rhs_sub, 1 if rhs_transposed else 0, dtype_lhs, use_mxfp8
                    )
                    rhs_q_list.append(r_q)
                    rhs_scale_list.append(
                        (r_s.T if rhs_transposed else r_s).astype(acc_dtype)
                    )
                else:
                    rhs_q_list.append(
                        rhs_ref[:, k_start:k_end]
                        if rhs_transposed
                        else rhs_ref[k_start:k_end, :]
                    )
                    rhs_scale_list.append(w_scales_ref[i, :, :].astype(acc_dtype))

            accumulators = [
                jnp.zeros((block_m, compute_tile_n), dtype=acc_dtype)
                for _ in range(steps_n)
            ]
            for i in range(steps_k):
                lhs_q = lhs_q_list[i]
                lhs_scale = lhs_scale_list[i]
                rhs_q_full = rhs_q_list[i]
                rhs_scale_full = rhs_scale_list[i]

                for j in range(steps_n):
                    n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n
                    rhs_q_slice = (
                        rhs_q_full[n_start:n_end, :]
                        if rhs_transposed
                        else rhs_q_full[:, n_start:n_end]
                    )
                    rhs_scale_slice = rhs_scale_full[:, n_start:n_end]
                    if dtype_lhs == jnp.int4 or dtype_lhs == jnp.int8:
                        preferred_element_type = jnp.int32
                    else:
                        preferred_element_type = jnp.float32
                    dot_res = jax.lax.dot_general(
                        lhs_q,
                        rhs_q_slice,
                        (((1,), (1,)), ((), ()))
                        if rhs_transposed
                        else (((1,), (0,)), ((), ())),
                        preferred_element_type=preferred_element_type,
                    )
                    res = dot_res.astype(acc_dtype)
                    res = res * lhs_scale
                    res = res * rhs_scale_slice
                    
                    # Accumulate directly to prevent VMEM overflow
                    acc_scratch[0:block_m, n_start:n_end] += res

        @pl.when(is_last_step)
        def _write():
            out_ref[...] = acc_scratch[...].astype(out_ref.dtype)

    grid = (m // block_m, n // block_n, k_dim // block_k)

    block_spec_lhs = pl.BlockSpec(
        (block_m, block_k), lambda i, j, k: (i, k), memory_space=pltpu.VMEM
    )
    if rhs_transposed:
        block_spec_rhs = pl.BlockSpec(
            (block_n, block_k), lambda i, j, k: (j, k), memory_space=pltpu.VMEM
        )
    else:
        block_spec_rhs = pl.BlockSpec(
            (block_k, block_n), lambda i, j, k: (k, j), memory_space=pltpu.VMEM
        )
    block_spec_w_scales = None
    if w_scales is not None:
        block_spec_w_scales = pl.BlockSpec(
            (steps_k, 1, block_n),
            lambda _, j, k: (k, 0, j),
            memory_space=pltpu.VMEM,
        )
        
    block_spec_a_scales = None
    if a_scales is not None:
        block_spec_a_scales = pl.BlockSpec(
            (1,), lambda *args: (0,), memory_space=pltpu.VMEM
        )

    block_spec_out = pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j))

    scratch_shape = pltpu.VMEM(
        (block_m, block_n), jnp.bfloat16 if use_bf16_acc else jnp.float32
    )

    in_specs = [block_spec_lhs, block_spec_rhs]
    if w_scales is not None:
        in_specs.append(block_spec_w_scales)
    else:
        in_specs.append(None)
    if a_scales is not None:
        in_specs.append(block_spec_a_scales)
    else:
        in_specs.append(None)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype_out),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=block_spec_out,
            grid=grid,
            scratch_shapes=[scratch_shape],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(lhs, rhs, w_scales, a_scales)


def pallas_matmul_2d_wrapper(
    lhs, rhs, dimension_numbers, preferred_element_type=None, is_prequantized: bool = False, a_scales: Optional[jax.Array] = None
):
    from axlearn.common.utils import get_current_abstract_or_physical_mesh
    from jax import shard_map
    from jax.sharding import NamedSharding, PartitionSpec

    def _get_sharding_spec(x):
        if hasattr(x, "sharding") and x.sharding is not None:
            if hasattr(x.sharding, "spec"):
                return x.sharding.spec
        if hasattr(x, "aval") and x.aval is not None:
            if hasattr(x.aval, "sharding") and x.aval.sharding is not None:
                if hasattr(x.aval.sharding, "spec"):
                    return x.aval.sharding.spec
        return None

    mesh = get_current_abstract_or_physical_mesh()
    lhs_spec = _get_sharding_spec(lhs)
    rhs_spec = _get_sharding_spec(rhs)

    if mesh is not None and (lhs_spec is not None or rhs_spec is not None):
        lhs_spec_norm = lhs_spec or PartitionSpec(*(None,) * lhs.ndim)
        rhs_spec_norm = rhs_spec or PartitionSpec(*(None,) * rhs.ndim)

        (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = (
            dimension_numbers[0],
            dimension_numbers[1] if len(dimension_numbers) > 1 else ((), ()),
        )
        if isinstance(lhs_contracting, int):
            lhs_contracting = (lhs_contracting,)
        else:
            lhs_contracting = tuple(lhs_contracting)
        if isinstance(rhs_contracting, int):
            rhs_contracting = (rhs_contracting,)
        else:
            rhs_contracting = tuple(rhs_contracting)
        if isinstance(lhs_batch, int):
            lhs_batch = (lhs_batch,)
        else:
            lhs_batch = tuple(lhs_batch)
        if isinstance(rhs_batch, int):
            rhs_batch = (rhs_batch,)
        else:
            rhs_batch = tuple(rhs_batch)

        contracting_mesh_axes = set()
        for ax in lhs_contracting:
            if lhs_spec_norm is not None and ax < len(lhs_spec_norm) and lhs_spec_norm[ax] is not None:
                if isinstance(lhs_spec_norm[ax], tuple):
                    contracting_mesh_axes.update(lhs_spec_norm[ax])
                else:
                    contracting_mesh_axes.add(lhs_spec_norm[ax])
        for ax in rhs_contracting:
            if rhs_spec_norm is not None and ax < len(rhs_spec_norm) and rhs_spec_norm[ax] is not None:
                if isinstance(rhs_spec_norm[ax], tuple):
                    contracting_mesh_axes.update(rhs_spec_norm[ax])
                else:
                    contracting_mesh_axes.add(rhs_spec_norm[ax])

        lhs_non_contracting = [i for i in range(lhs.ndim) if i not in lhs_contracting and i not in lhs_batch]
        rhs_non_contracting = [i for i in range(rhs.ndim) if i not in rhs_contracting and i not in rhs_batch]

        out_spec_list = []
        for ax in lhs_batch:
            out_spec_list.append(lhs_spec_norm[ax] if ax < len(lhs_spec_norm) else None)
        for ax in lhs_non_contracting:
            out_spec_list.append(lhs_spec_norm[ax] if ax < len(lhs_spec_norm) else None)
        for ax in rhs_non_contracting:
            out_spec_list.append(rhs_spec_norm[ax] if ax < len(rhs_spec_norm) else None)
        out_spec = PartitionSpec(*out_spec_list)

        def local_matmul_fn(l_local, r_local):
            res_local = _pallas_matmul_2d_local_impl(
                l_local, r_local, dimension_numbers, preferred_element_type, is_prequantized, a_scales=a_scales
            )
            if contracting_mesh_axes:
                res_local = jax.lax.psum(res_local, axis_name=tuple(contracting_mesh_axes))
            return res_local

        return shard_map(
            local_matmul_fn,
            mesh=mesh,
            in_specs=(lhs_spec_norm, rhs_spec_norm),
            out_specs=out_spec,
            check_vma=False,
        )(lhs, rhs)
    else:
        return _pallas_matmul_2d_local_impl(
            lhs, rhs, dimension_numbers, preferred_element_type, is_prequantized, a_scales=a_scales
        )


def _pallas_matmul_2d_local_impl(
    lhs, rhs, dimension_numbers, preferred_element_type=None, is_prequantized: bool = False, a_scales=None
):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = (
        dimension_numbers[0],
        dimension_numbers[1] if len(dimension_numbers) > 1 else ((), ()),
    )
    if isinstance(lhs_contracting, int):
        lhs_contracting = (lhs_contracting,)
    else:
        lhs_contracting = tuple(lhs_contracting)
    if isinstance(rhs_contracting, int):
        rhs_contracting = (rhs_contracting,)
    else:
        rhs_contracting = tuple(rhs_contracting)
    if isinstance(lhs_batch, int):
        lhs_batch = (lhs_batch,)
    else:
        lhs_batch = tuple(lhs_batch)
    if isinstance(rhs_batch, int):
        rhs_batch = (rhs_batch,)
    else:
        rhs_batch = tuple(rhs_batch)

    if len(lhs_batch) > 0:
        # Move batch dimensions to the front
        lhs_ndim = lhs.ndim
        lhs_non_batch = [i for i in range(lhs_ndim) if i not in lhs_batch]
        lhs_transposed = jnp.transpose(lhs, lhs_batch + tuple(lhs_non_batch))

        rhs_ndim = rhs.ndim
        rhs_non_batch = [i for i in range(rhs_ndim) if i not in rhs_batch]
        rhs_transposed = jnp.transpose(rhs, rhs_batch + tuple(rhs_non_batch))

        num_batch_dims = len(lhs_batch)

        def matmul_slice(l_slice, r_slice):
            slice_lhs_contracting = tuple(lhs_non_batch.index(ax) for ax in lhs_contracting)
            slice_rhs_contracting = tuple(rhs_non_batch.index(ax) for ax in rhs_contracting)
            return pallas_matmul_2d_wrapper(
                l_slice,
                r_slice,
                dimension_numbers=((slice_lhs_contracting, slice_rhs_contracting), ((), ())),
                preferred_element_type=preferred_element_type,
                is_prequantized=is_prequantized,
            )

        fn = matmul_slice
        for _ in range(num_batch_dims):
            fn = jax.vmap(fn, in_axes=(0, 0))

        return fn(lhs_transposed, rhs_transposed)


    # 2. Reshape lhs to (M, K)
    lhs_ndim = lhs.ndim
    lhs_shape = lhs.shape
    K = 1
    for ax in lhs_contracting:
        K *= lhs_shape[ax]

    lhs_non_contracting = [i for i in range(lhs_ndim) if i not in lhs_contracting]
    lhs_axes = lhs_non_contracting + list(lhs_contracting)
    lhs_transposed = jnp.transpose(lhs, lhs_axes)
    M = lhs_transposed.size // K
    lhs_2d = lhs_transposed.reshape(M, K)

    # 3. Reshape rhs to (N, K)
    rhs_ndim = rhs.ndim
    rhs_shape = rhs.shape
    rhs_non_contracting = [i for i in range(rhs_ndim) if i not in rhs_contracting]
    rhs_axes = rhs_non_contracting + list(rhs_contracting)
    rhs_transposed_t = jnp.transpose(rhs, rhs_axes)
    N = rhs_transposed_t.size // K
    rhs_2d = rhs_transposed_t.reshape(N, K)
    rhs_is_transposed = True

    # Select block sizes dynamically based on dimensions (prefer 1024 for Trillium MXU saturation)
    def _get_block(dim):
        for b in [1024, 512, 256, 128, 64, 32]:
            if dim % b == 0:
                return b
        return dim
        
    block_m = _get_block(M)
    block_n = _get_block(N)
    block_k = _get_block(K)
    sc_size = block_k

    # If any block is smaller than 128, Pallas might not map well to MXU or might crash, 
    # but we prevent the 0-grid crash. For very awkward shapes, fallback to XLA.
    if M % block_m != 0 or N % block_n != 0 or K % block_k != 0 or block_m < 32 or block_n < 32 or block_k < 32:
        # Fallback to standard jax.lax.dot_general for unsupported dimensions
        return jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=preferred_element_type)

    if is_prequantized:
        lhs_input = lhs_2d
        rhs_input = rhs_2d
    else:
        lhs_input = lhs_2d.astype(jnp.bfloat16)
        rhs_input = rhs_2d.astype(jnp.bfloat16)

    # Execute custom Pallas kernel
    out_2d = quantized_matmul_kernel(
        lhs_input,
        rhs_input,
        None, # w_scales
        a_scales, # Passed from kwargs
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        sc_size=sc_size,
        dtype_lhs=lhs.dtype if is_prequantized else jnp.float8_e4m3fn,
        dtype_out=preferred_element_type or jnp.bfloat16,
        use_bf16_acc=True, # Use BF16 to save 2MB of VMEM!
        n_lane_multiplier=1,
        rhs_transposed=rhs_is_transposed,
        is_prequantized=is_prequantized,
    )

    # Reshape back to expected N-dimensional shape
    lhs_non_contracting_shape = tuple(lhs_shape[i] for i in range(lhs_ndim) if i not in lhs_contracting)
    rhs_non_contracting_shape = tuple(rhs_shape[i] for i in range(rhs_ndim) if i not in rhs_contracting)
    out_shape = lhs_non_contracting_shape + rhs_non_contracting_shape
    return out_2d.reshape(out_shape)


def _pallas_q_dot_dq_impl(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
    is_training: bool,
) -> Union[Tensor, tuple[Tensor, tuple[Tensor, ...]]]:
    q_rhs, rhs_scale, rhs_amax_history = _quantize(
        rhs,
        rhs_scale,
        rhs_amax_history,
        dtype=jnp.float8_e4m3fn,
        preferred_element_type=preferred_element_type,
    )

    # We do NOT run _quantize on lhs for forward matmul, but we need q_lhs for backward!
    # Update lhs scale and amax in high precision and get q_lhs for backward.
    q_lhs, new_lhs_scale, new_lhs_amax_history = _quantize(
        lhs,
        lhs_scale,
        lhs_amax_history,
        dtype=jnp.float8_e4m3fn,
        preferred_element_type=preferred_element_type,
    )

    # Run Pallas matmul in forward pass on the mixed inputs!
    # Pallas will take BF16 lhs and FP8 rhs, and do the FP8 cast inside the MXU loop!
    out = pallas_matmul_2d_wrapper(
        lhs, # Pass BF16 activation!
        q_rhs, # Pass FP8 weights
        dimension_numbers,
        preferred_element_type,
        is_prequantized=False,
        a_scales=jnp.reshape(new_lhs_scale, (1,)),
    )

    # Dequantize back to high precision in a fast elementwise post-pass
    out = _dequantize(out, new_lhs_scale * rhs_scale, dq_dtype=preferred_element_type)

    if is_training:
        res = (
            lhs,
            rhs,
            q_lhs,
            q_rhs,
            lhs_scale,
            rhs_scale,
            out_grad_scale,
            lhs_amax_history,
            rhs_amax_history,
            out_grad_amax_history,
        )
        return out, res
    else:
        return out


@partial(custom_vjp, nondiff_argnums=(8, 9, 10))
def pallas_q_dot_q(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike = None,
) -> Tensor:
    return _pallas_q_dot_dq_impl(**locals(), is_training=False)


def _pallas_q_dot_dq_fwd(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
):
    return _pallas_q_dot_dq_impl(
        **locals(),
        is_training=True,
    )


pallas_q_dot_q.defvjp(_pallas_q_dot_dq_fwd, _q_dot_dq_bwd)
