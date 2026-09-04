# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode-latency microbenchmark for the fused GDN v3 kernel, standalone
(single device, no model / no vLLM), at Qwen3.6-27B's Gated DeltaNet shapes.

Measures `wrapper.fused_conv1d_gdn` directly for seq_len=1-per-request decode
across a batch-size sweep. `conv_state` is built in the kernel's native
[num_blocks, kernel_size - 1, 1, dim] fp32 layout (see `kv_cache_manager.py`'s
`_mamba_state_dtype`/mamba cache allocation) so no reshape or dtype cast
happens in the timed loop.

Since `conv_state`/`recurrent_state` are donated args (`donate_argnames` on
`fused_conv1d_gdn`), each call consumes its input buffers; the loop chains
the previous call's output states into the next call's input, exactly as a
real decode loop would, rather than reusing one buffer (which XLA would
reject on the second call).

Run on a TPU host:
    python scripts/benchmarking/kernels/benchmark_gdn_v3_decode.py \
        --batch-sizes 1,8,32,64,128,256

Pass --profile-dir to also capture an xprof/JAX profiler trace per batch
size (written to <profile-dir>/batch<N>), viewable with:
    tensorboard --logdir <profile-dir>
or loaded directly with `jax.profiler.ProfileData.from_file` on the
`*.xplane.pb` file under the batch's `plugins/profile/.../` subdir — this
gives a line-level (source-mapped) op breakdown of the kernel itself,
without any of the noise from profiling the full model/vLLM.

Pass --shard-map to go through the real `gdn_attention.run_jax_gdn_attention`
(the `jax.shard_map`-wrapped entry point production actually calls) instead
of calling `wrapper.fused_conv1d_gdn` directly on pre-divided shapes. This
still needs no vLLM / no model weights — just a JAX mesh sized by --tp/--dp
on the local TPU devices — so it stays cheap to iterate on, but it exercises
the shard_map entry/exit boundary itself, which --tp alone (dividing n_kq/n_v
by hand) does not.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.gdn.v3 import wrapper
from tpu_inference.layers.common import gdn_attention
from tpu_inference.layers.common.sharding import ShardingAxisName

# Qwen3.6-27B Gated DeltaNet shapes (linear_attention layers):
# linear_num_key_heads=16, linear_num_value_heads=48,
# linear_key_head_dim=linear_value_head_dim=128, linear_conv_kernel_dim=4.
N_KQ = 16
N_V = 48
D_K = 128
D_V = 128
KERNEL_SIZE = 4


def build_decode_inputs(batch_size: int, num_blocks: int, context_len: int,
                        dtype: jnp.dtype, n_kq: int, n_v: int, d_k: int,
                        d_v: int, kernel_size: int):
    """One decode step (seq_len=1) for `batch_size` independent sequences
    out of a fixed-size `num_blocks`-slot cache pool (sized for the serving
    engine's max concurrent requests, not for this step's active batch —
    matching production, where the whole persistent conv_state/recurrent_state
    cache is touched by eager ops regardless of how many slots are active
    this step), each already `context_len` tokens deep (so
    has_initial_state=True, the steady-state decode case)."""
    assert batch_size < num_blocks, (
        "num_blocks must exceed batch_size: slot 0 is the reserved null "
        "block and each active sequence needs its own slot")
    dim = n_kq * d_k * 2 + n_v * d_v
    rngs = iter(jax.random.split(jax.random.key(0), 8))

    qkv = jax.random.normal(next(rngs), (batch_size, dim), dtype=dtype)
    b = jax.random.normal(next(rngs), (batch_size, n_v), dtype=dtype)
    a = jax.random.normal(next(rngs), (batch_size, n_v), dtype=dtype)

    # Native kernel layout and dtype: no reshape or cast needed at the call
    # boundary. conv_state is fp32 — the compact per-row VMEM layout only
    # supports 32-bit dtypes (see kv_cache_manager.py's `_mamba_state_dtype`
    # and wrapper.py's `assert conv_state.dtype == jnp.float32`).
    conv_state = jnp.zeros((num_blocks, kernel_size - 1, 1, dim),
                           dtype=jnp.float32)
    recurrent_state = jnp.zeros((num_blocks, n_v, d_k, d_v), dtype=jnp.float32)

    conv_weight = jax.random.normal(next(rngs), (dim, 1, kernel_size),
                                    dtype=dtype)
    conv_bias = jax.random.normal(next(rngs), (dim, ), dtype=dtype)
    a_log = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32)

    query_start_loc = jnp.arange(batch_size + 1, dtype=jnp.int32)
    state_indices = jnp.arange(1, batch_size + 1, dtype=jnp.int32)
    # No prefix caching here, so read from the same slot we write to.
    read_state_indices = state_indices
    distribution = jnp.array([batch_size, batch_size, batch_size],
                             dtype=jnp.int32)
    # query_len=1, seq_lens=context_len+1 => context_len tokens of existing
    # state => has_initial_state=True (the steady-state decode case).
    seq_lens = jnp.full((batch_size, ), context_len + 1, dtype=jnp.int32)

    static = dict(n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
                 kernel_size=kernel_size)
    return dict(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        read_state_indices=read_state_indices,
    ), static


def run_decode_steps(inputs: dict, static: dict, num_steps: int):
    """Runs `num_steps` decode calls, chaining conv_state/recurrent_state
    from each call's output into the next call's input (both are donated).
    Returns the final states so the caller can block on them."""
    conv_state = inputs["conv_state"]
    recurrent_state = inputs["recurrent_state"]
    out = None
    for _ in range(num_steps):
        (conv_state, recurrent_state), out = wrapper.fused_conv1d_gdn(
            inputs["qkv"],
            inputs["b"],
            inputs["a"],
            conv_state,
            recurrent_state,
            inputs["conv_weight"],
            inputs["conv_bias"],
            inputs["a_log"],
            inputs["dt_bias"],
            inputs["query_start_loc"],
            inputs["state_indices"],
            inputs["distribution"],
            inputs["seq_lens"],
            inputs["read_state_indices"],
            **static,
        )
    return conv_state, recurrent_state, out


def benchmark_batch_size(batch_size: int, num_blocks: int, context_len: int,
                         dtype: jnp.dtype, warmup: int, reps: int, n_kq: int,
                         n_v: int, d_k: int, d_v: int, kernel_size: int,
                         profile_dir: str | None = None) -> float:
    inputs, static = build_decode_inputs(batch_size, num_blocks, context_len,
                                         dtype, n_kq, n_v, d_k, d_v,
                                         kernel_size)

    # Warmup: also triggers compilation for this batch size's shapes.
    conv_state, recurrent_state, out = run_decode_steps(
        inputs, static, warmup)
    jax.block_until_ready((conv_state, recurrent_state, out))
    inputs["conv_state"] = conv_state
    inputs["recurrent_state"] = recurrent_state

    if profile_dir is not None:
        batch_dir = os.path.join(profile_dir, f"batch{batch_size}")
        with jax.profiler.trace(batch_dir):
            start = time.perf_counter()
            conv_state, recurrent_state, out = run_decode_steps(
                inputs, static, reps)
            jax.block_until_ready((conv_state, recurrent_state, out))
            elapsed_s = time.perf_counter() - start
    else:
        start = time.perf_counter()
        conv_state, recurrent_state, out = run_decode_steps(
            inputs, static, reps)
        jax.block_until_ready((conv_state, recurrent_state, out))
        elapsed_s = time.perf_counter() - start

    return elapsed_s / reps * 1e3  # ms/call


def build_mesh(dp: int, tp: int) -> jax.sharding.Mesh:
    devices = jax.devices()
    assert len(devices) >= dp * tp, (
        f"need {dp * tp} devices for dp={dp} x tp={tp}, have {len(devices)}")
    return jax.make_mesh((dp, tp),
                         (ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD))


def build_sharded_decode_inputs(batch_size: int, num_blocks: int,
                                context_len: int, dtype: jnp.dtype,
                                n_kq: int, n_v: int, d_k: int, d_v: int,
                                kernel_size: int, mesh: jax.sharding.Mesh):
    """Same decode step as `build_decode_inputs`, but with *global* (unsharded)
    n_kq/n_v — `run_jax_gdn_attention` divides by TP internally, exactly as
    production does — and every array placed with the same `PartitionSpec`s
    `gdn_attention.run_jax_gdn_attention` uses, so the shard_map boundary
    actually sees sharded inputs instead of single-device ones.

    Skips the Q/K/V head-interleave reorder production applies before
    sharding (`reorder_concatenated_tensor_for_sharding` in
    gdn_attention_op.py) — that only changes *which* heads land on which
    chip, not the per-chip shapes, so it doesn't matter for a latency-only
    benchmark.
    """
    inputs, _ = build_decode_inputs(batch_size, num_blocks, context_len,
                                    dtype, n_kq, n_v, d_k, d_v, kernel_size)

    def put(x, spec):
        return jax.device_put(x, NamedSharding(mesh, spec))

    data, head = ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD
    inputs["qkv"] = put(inputs["qkv"], P(data, head))
    inputs["b"] = put(inputs["b"], P(data, head))
    inputs["a"] = put(inputs["a"], P(data, head))
    inputs["conv_state"] = put(inputs["conv_state"], P(data, None, None, head))
    inputs["recurrent_state"] = put(inputs["recurrent_state"],
                                    P(data, head, None, None))
    inputs["conv_weight"] = put(inputs["conv_weight"], P(head, None, None))
    inputs["conv_bias"] = put(inputs["conv_bias"], P(head))
    inputs["a_log"] = put(inputs["a_log"], P(head))
    inputs["dt_bias"] = put(inputs["dt_bias"], P(head))
    inputs["query_start_loc"] = put(inputs["query_start_loc"], P(data))
    inputs["state_indices"] = put(inputs["state_indices"], P(data))
    inputs["distribution"] = put(inputs["distribution"], P(data))
    inputs["seq_lens"] = put(inputs["seq_lens"], P(data))
    inputs["read_state_indices"] = put(inputs["read_state_indices"], P(data))

    static = dict(n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
                 kernel_size=kernel_size, mesh=mesh)
    return inputs, static


def run_decode_steps_sharded(inputs: dict, static: dict, num_steps: int):
    """Same as `run_decode_steps`, calling through the real shard_map entry
    point (`gdn_attention.run_jax_gdn_attention`) instead of
    `wrapper.fused_conv1d_gdn` directly."""
    conv_state = inputs["conv_state"]
    recurrent_state = inputs["recurrent_state"]
    out = None
    for _ in range(num_steps):
        (conv_state, recurrent_state), out = gdn_attention.run_jax_gdn_attention(
            inputs["qkv"],
            inputs["b"],
            inputs["a"],
            conv_state,
            recurrent_state,
            inputs["conv_weight"],
            inputs["conv_bias"],
            inputs["a_log"],
            inputs["dt_bias"],
            inputs["state_indices"],
            inputs["query_start_loc"],
            inputs["distribution"],
            inputs["seq_lens"],
            read_state_indices=inputs["read_state_indices"],
            **static,
        )
    return conv_state, recurrent_state, out


def benchmark_batch_size_sharded(batch_size: int, num_blocks: int,
                                 context_len: int, dtype: jnp.dtype,
                                 warmup: int, reps: int, n_kq: int, n_v: int,
                                 d_k: int, d_v: int, kernel_size: int,
                                 mesh: jax.sharding.Mesh,
                                 profile_dir: str | None = None) -> float:
    inputs, static = build_sharded_decode_inputs(batch_size, num_blocks,
                                                 context_len, dtype, n_kq,
                                                 n_v, d_k, d_v, kernel_size,
                                                 mesh)

    conv_state, recurrent_state, out = run_decode_steps_sharded(
        inputs, static, warmup)
    jax.block_until_ready((conv_state, recurrent_state, out))
    inputs["conv_state"] = conv_state
    inputs["recurrent_state"] = recurrent_state

    if profile_dir is not None:
        batch_dir = os.path.join(profile_dir, f"batch{batch_size}")
        with jax.profiler.trace(batch_dir):
            start = time.perf_counter()
            conv_state, recurrent_state, out = run_decode_steps_sharded(
                inputs, static, reps)
            jax.block_until_ready((conv_state, recurrent_state, out))
            elapsed_s = time.perf_counter() - start
    else:
        start = time.perf_counter()
        conv_state, recurrent_state, out = run_decode_steps_sharded(
            inputs, static, reps)
        jax.block_until_ready((conv_state, recurrent_state, out))
        elapsed_s = time.perf_counter() - start

    return elapsed_s / reps * 1e3  # ms/call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes",
                        default="1,8,32,64,128,256",
                        help="comma-separated decode batch sizes to sweep")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=256,
        help="conv_state/recurrent_state cache pool size (independent of "
        "--batch-sizes — matches production, where the pool is sized for "
        "max concurrent requests and stays fixed regardless of how many "
        "are active this step; must exceed every swept batch size)")
    parser.add_argument("--context-len",
                        type=int,
                        default=4096,
                        help="tokens of existing state per sequence before "
                        "this decode step (has_initial_state=True path)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--reps",
                        type=int,
                        default=50,
                        help="timed decode calls per batch size (chained)")
    parser.add_argument("--n-kq",
                        type=int,
                        default=N_KQ,
                        help="global (unsharded) query/key head count")
    parser.add_argument("--n-v",
                        type=int,
                        default=N_V,
                        help="global (unsharded) value head count")
    parser.add_argument("--d-k", type=int, default=D_K)
    parser.add_argument("--d-v", type=int, default=D_V)
    parser.add_argument("--kernel-size", type=int, default=KERNEL_SIZE)
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor-parallel degree. `fused_conv1d_gdn` itself is "
        "TP-agnostic — production shards n_kq/n_v by TP *before* calling it "
        "(gdn_attention.py's `n_kq=n_kq // tp_size`), so each chip actually "
        "runs with dim = (n_kq/tp)*d_k*2 + (n_v/tp)*d_v, not the global "
        "dim. This flag applies that same division here so a single-chip "
        "run matches what one chip does inside a real TP deployment.")
    parser.add_argument(
        "--shard-map",
        action="store_true",
        help="go through `gdn_attention.run_jax_gdn_attention` (the real "
        "jax.shard_map entry point) on a --dp x --tp mesh, instead of "
        "calling `wrapper.fused_conv1d_gdn` directly with hand-divided "
        "n_kq/n_v. Needs --dp * --tp local TPU devices; still no vLLM / "
        "no model weights.")
    parser.add_argument("--dp",
                        type=int,
                        default=1,
                        help="data-parallel degree, only used with "
                        "--shard-map")
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="if set, capture a JAX profiler trace per batch size under "
        "<profile-dir>/batch<N> (view with `tensorboard --logdir "
        "<profile-dir>`). Keep --reps small (e.g. 10-20) when profiling "
        "to avoid huge trace files.")
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    assert args.n_kq % args.tp == 0 and args.n_v % args.tp == 0, (
        f"--tp={args.tp} must evenly divide --n-kq={args.n_kq} and "
        f"--n-v={args.n_v}")
    dtype = jnp.dtype(args.dtype)
    # Per-chip head counts under TP — what fused_conv1d_gdn actually runs
    # with in production (see the --tp help text above).
    n_kq_local = args.n_kq // args.tp
    n_v_local = args.n_v // args.tp
    dim = n_kq_local * args.d_k * 2 + n_v_local * args.d_v

    mesh = build_mesh(args.dp, args.tp) if args.shard_map else None

    print(f"device={jax.devices()[0].device_kind}, dtype={dtype.name}, tp="
          f"{args.tp}, dp={args.dp if args.shard_map else 1}, "
          f"shard_map={args.shard_map}, "
          f"n_kq(global/local)={args.n_kq}/{n_kq_local}, "
          f"n_v(global/local)={args.n_v}/{n_v_local}, d_k={args.d_k}, "
          f"d_v={args.d_v}, kernel_size={args.kernel_size}, dim(local)={dim}, "
          f"num_blocks={args.num_blocks}, context_len={args.context_len}")
    print(f"{'batch':>8} | {'ms/step':>10} | {'tokens/s':>10}")

    for batch_size in [int(v) for v in args.batch_sizes.split(",")]:
        if batch_size >= args.num_blocks:
            print(f"{batch_size:>8} | (skip: batch_size >= num_blocks="
                  f"{args.num_blocks})")
            continue
        try:
            if args.shard_map:
                # Global (unsharded) n_kq/n_v here — run_jax_gdn_attention
                # divides by tp internally.
                ms = benchmark_batch_size_sharded(
                    batch_size,
                    args.num_blocks,
                    args.context_len,
                    dtype,
                    args.warmup,
                    args.reps,
                    args.n_kq,
                    args.n_v,
                    args.d_k,
                    args.d_v,
                    args.kernel_size,
                    mesh,
                    args.profile_dir,
                )
            else:
                ms = benchmark_batch_size(
                    batch_size,
                    args.num_blocks,
                    args.context_len,
                    dtype,
                    args.warmup,
                    args.reps,
                    n_kq_local,
                    n_v_local,
                    args.d_k,
                    args.d_v,
                    args.kernel_size,
                    args.profile_dir,
                )
            tokens_per_s = batch_size / (ms / 1e3)
            print(f"{batch_size:>8} | {ms:>10.3f} | {tokens_per_s:>10.0f}")
        except Exception as e:  # noqa: BLE001
            print(f"{batch_size:>8} | FAIL: {' '.join(str(e).split())[:80]}")

    if args.profile_dir:
        print(f"xprof traces written under {args.profile_dir} "
              f"(tensorboard --logdir {args.profile_dir})")


if __name__ == "__main__":
    main()
