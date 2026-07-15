# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.utils import load, load_model

from mio.dflash.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    DFlashDraftModelArgs,
)
from mio.dflash.recurrent_rollback_cache import RecurrentRollbackCache


def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")


def _get_dflash_model_classes(config: dict[str, Any]):
    return DFlashDraftModel, DFlashDraftModelArgs


def _resolve_local_model_path(model_ref: str | Path) -> Path:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise FileNotFoundError(f"Model path does not exist and huggingface_hub is unavailable: {model_ref}") from exc

    snapshot_path = snapshot_download(
        repo_id=str(model_ref),
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer*"],
    )
    return Path(snapshot_path)


def _prepare_prompt_tokens(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))


def sample_tokens(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: Optional[list[int]],
) -> Optional[mx.array]:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)


def sample_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return sample_tokens(logits)
    floor = mx.array(-1e9, dtype=logits.dtype)
    return mx.argmax(mx.where(suppress_token_mask, floor, logits), axis=-1)


def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def _match_acceptance_length(
    drafted_tokens: mx.array,
    posterior_tokens: mx.array,
) -> mx.array:
    if int(drafted_tokens.shape[0]) == 0:
        return mx.array(0, dtype=mx.int32)
    matches = mx.equal(drafted_tokens, posterior_tokens).astype(mx.int32)
    return mx.sum(mx.cumprod(matches, axis=0))


def _concat_hidden_state_chunks(
    hidden_state_chunks: list[list[mx.array]],
) -> list[mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return [
        mx.concatenate([chunk[index] for chunk in hidden_state_chunks], axis=1)
        for index in range(len(hidden_state_chunks[0]))
    ]


def _concat_hidden_state_chunk_dicts(
    hidden_state_chunks: list[dict[int, mx.array]],
    capture_layer_ids: set[int],
) -> dict[int, mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return {
        layer_id: mx.concatenate([chunk[layer_id] for chunk in hidden_state_chunks], axis=1)
        for layer_id in sorted(capture_layer_ids)
    }


def _eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)


def _target_text_wrapper(target_model: Any) -> Any:
    if hasattr(target_model, "model"):
        return target_model
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")


def _target_text_model(target_model: Any) -> Any:
    wrapper = _target_text_wrapper(target_model)
    if hasattr(wrapper, "model"):
        return wrapper.model
    raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")


def detect_target_family(target_model: Any) -> str:
    inner = _target_text_model(target_model)
    has_linear = any(
        hasattr(layer, "linear_attn") or hasattr(layer, "is_linear")
        for layer in inner.layers
    )
    return "hybrid_gdn" if has_linear else "pure_attention"


def _target_embed_tokens(target_model: Any) -> Any:
    return _target_text_model(target_model).embed_tokens


def _lm_head_logits(target_model: Any, hidden_states: mx.array) -> mx.array:
    wrapper = _target_text_wrapper(target_model)
    if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
        return wrapper.model.embed_tokens.as_linear(hidden_states)
    return wrapper.lm_head(hidden_states)


def extract_context_feature_from_dict(
    captured_dict: dict[int, mx.array],
    target_layer_ids: list[int],
) -> mx.array:
    selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
    return mx.concatenate(selected, axis=-1)


def _resolve_verify_len_cap(target_model: Any, block_tokens: int) -> int:
    override_raw = os.environ.get("DFLASH_VERIFY_LEN", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(int(block_tokens), override))
    return int(block_tokens)


def _resolve_dflash_max_ctx() -> int:
    raw = os.environ.get("DFLASH_MAX_CTX", "131072").strip()
    try:
        max_ctx = int(raw)
    except ValueError:
        max_ctx = 131072
    return max(1, max_ctx)


def _resolve_draft_window() -> tuple[int, int]:
    sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64").strip())
    window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024").strip())
    return max(0, sink), max(1, window)


def _effective_draft_window(draft_model: Any, requested_window: int) -> int:
    """Honor the minimum sliding-attention window encoded by the checkpoint."""

    configured_window = int(
        getattr(getattr(draft_model, "args", None), "sliding_window", 0) or 0
    )
    return max(1, int(requested_window), configured_window)


def _project_draft_context(
    draft_model: DFlashDraftModel,
    target_hidden: mx.array,
) -> mx.array:
    """Project captured target features once, before the speculative cycle."""

    projector = getattr(draft_model, "project_target_hidden", None)
    return projector(target_hidden) if callable(projector) else target_hidden


def _forward_draft(
    draft_model: DFlashDraftModel,
    *,
    noise_embedding: mx.array,
    draft_context: mx.array,
    cache: list[Any],
) -> mx.array:
    """Run a draft block without re-projecting the full context each cycle."""

    forward_projected = getattr(draft_model, "forward_projected_context", None)
    if callable(forward_projected):
        return forward_projected(
            noise_embedding=noise_embedding,
            draft_context=draft_context,
            cache=cache,
        )
    return draft_model(
        noise_embedding=noise_embedding,
        target_hidden=draft_context,
        cache=cache,
    )

def _should_quantize_draft(quantize_draft: bool = False) -> bool:
    if quantize_draft:
        return True
    raw = os.environ.get("DFLASH_QUANTIZE_DRAFT", "").strip().lower()
    return raw not in {"", "0", "false", "no"}




def _linear_forward(x: mx.array, weight: mx.array, bias: Optional[mx.array]) -> mx.array:
    out = x @ weight.T
    return out if bias is None else out + bias


_EXACT_SMALL_PROJ_PAD_M = 16


class _ExactSmallProjPad(nn.Module):
    def __init__(self, linear: nn.Module, *, pad_m: int = _EXACT_SMALL_PROJ_PAD_M):
        super().__init__()
        self.linear = linear
        self.pad_m = int(pad_m)
        self._dflash_exact_small_proj_wrapped = True

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    @weight.setter
    def weight(self, value: mx.array) -> None:
        self.linear.weight = value

    @property
    def bias(self) -> Optional[mx.array]:
        return getattr(self.linear, "bias", None)

    @bias.setter
    def bias(self, value: Optional[mx.array]) -> None:
        self.linear.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] < self.pad_m:
            batch_size, seq_len, hidden_dim = x.shape
            pad = mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim), dtype=x.dtype)
            out = self.linear(mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)


def _install_exact_small_proj_hooks(
    linear_attn: Any,
    *,
    pad_m: int = _EXACT_SMALL_PROJ_PAD_M,
) -> None:
    for attr_name in ("in_proj_b", "in_proj_a"):
        proj = getattr(linear_attn, attr_name, None)
        if proj is None or getattr(proj, "_dflash_exact_small_proj_wrapped", False):
            continue
        setattr(linear_attn, attr_name, _ExactSmallProjPad(proj, pad_m=pad_m))


def _attention_num_heads(attn: Any) -> int:
    for attr in ("num_attention_heads", "n_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing attention head count attribute")


def _attention_num_kv_heads(attn: Any) -> int:
    for attr in ("num_key_value_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing KV head count attribute")


def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    q_proj_weight = getattr(q_proj, "weight", None)
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_proj_weight is None or q_norm_weight is None:
        return False
    try:
        num_attention_heads = _attention_num_heads(attn)
    except AttributeError:
        return False
    expected_out_dim = 2 * num_attention_heads * int(q_norm_weight.shape[0])
    return int(q_proj_weight.shape[0]) == expected_out_dim


def _concat_biases(*biases: Optional[mx.array]) -> Optional[mx.array]:
    present = [bias for bias in biases if bias is not None]
    if not present:
        return None
    if len(present) != len(biases):
        raise ValueError("expected either all packed biases or none")
    return mx.concatenate(present, axis=0)


def _exact_match(lhs: mx.array, rhs: mx.array) -> bool:
    mx.eval(lhs, rhs)
    return bool(mx.all(mx.equal(lhs, rhs)).item())


def _max_abs_diff(lhs: mx.array, rhs: mx.array) -> float:
    delta = mx.abs(lhs.astype(mx.float32) - rhs.astype(mx.float32))
    mx.eval(delta)
    return float(mx.max(delta).item())


def _assert_exact(label: str, lhs: mx.array, rhs: mx.array) -> None:
    if _exact_match(lhs, rhs):
        return
    raise AssertionError(f"{label} mismatch (max_abs_diff={_max_abs_diff(lhs, rhs)})")


def _set_linear_from_packed(
    linear: Any,
    packed_weight: mx.array,
    start: int,
    end: int,
    packed_bias: Optional[mx.array],
) -> None:
    linear.weight = packed_weight[start:end]
    if getattr(linear, "bias", None) is not None and packed_bias is not None:
        linear.bias = packed_bias[start:end]


def _install_packed_mlp_hook(mlp: Any) -> None:
    cls = type(mlp)
    if getattr(cls, "_dflash_packed_call_installed", False):
        return

    original_call = cls.__call__

    def packed_call(self, x) -> mx.array:
        packed_weight = getattr(self, "_dflash_gate_up_weight", None)
        if packed_weight is None:
            return original_call(self, x)
        gate_up = _linear_forward(
            x,
            packed_weight,
            getattr(self, "_dflash_gate_up_bias", None),
        )
        gate, up = mx.split(gate_up, [self._dflash_gate_proj_out_dim], axis=-1)
        return self.down_proj(swiglu(gate, up))

    cls.__call__ = packed_call
    cls._dflash_packed_call_installed = True


def _install_packed_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_packed_call_installed", False):
        return

    original_call = cls.__call__

    def packed_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        packed_weight = getattr(self, "_dflash_qkv_weight", None)
        if packed_weight is None:
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        B, L, _ = x.shape
        qkv = _linear_forward(x, packed_weight, getattr(self, "_dflash_qkv_bias", None))
        q_proj_dim = self._dflash_q_proj_out_dim
        k_proj_dim = self._dflash_k_proj_out_dim
        q_proj_output, keys, values = mx.split(
            qkv,
            [q_proj_dim, q_proj_dim + k_proj_dim],
            axis=-1,
        )
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    cls.__call__ = packed_call
    cls._dflash_packed_call_installed = True


def _pack_qwen3next_mlp(mlp: Any, *, layer_index: int, validate: bool) -> dict[str, Any]:
    if getattr(mlp, "_dflash_gate_up_weight", None) is not None:
        return {"layer_index": layer_index, "packed": True, "validated": False}

    gate_weight = mlp.gate_proj.weight
    up_weight = mlp.up_proj.weight
    gate_out_dim = int(gate_weight.shape[0])
    gate_bias = getattr(mlp.gate_proj, "bias", None)
    up_bias = getattr(mlp.up_proj, "bias", None)

    x = mx.random.normal((2, 3, int(gate_weight.shape[1]))).astype(gate_weight.dtype)
    if validate:
        gate_old = mlp.gate_proj(x)
        up_old = mlp.up_proj(x)
        mlp_old = mlp(x)

    packed_weight = mx.concatenate([gate_weight, up_weight], axis=0)
    packed_bias = _concat_biases(gate_bias, up_bias)
    if packed_bias is None:
        mx.eval(packed_weight)
    else:
        mx.eval(packed_weight, packed_bias)

    if validate:
        gate_up = _linear_forward(x, packed_weight, packed_bias)
        gate_new, up_new = mx.split(gate_up, [gate_out_dim], axis=-1)
        _assert_exact(f"mlp[{layer_index}] gate_proj", gate_old, gate_new)
        _assert_exact(f"mlp[{layer_index}] up_proj", up_old, up_new)

    _install_packed_mlp_hook(mlp)
    mlp._dflash_gate_up_weight = packed_weight
    mlp._dflash_gate_up_bias = packed_bias
    mlp._dflash_gate_proj_out_dim = gate_out_dim
    _set_linear_from_packed(mlp.gate_proj, packed_weight, 0, gate_out_dim, packed_bias)
    _set_linear_from_packed(
        mlp.up_proj,
        packed_weight,
        gate_out_dim,
        gate_out_dim + int(up_weight.shape[0]),
        packed_bias,
    )

    if validate:
        mlp_new = mlp(x)
        _assert_exact(f"mlp[{layer_index}] full", mlp_old, mlp_new)

    return {
        "layer_index": layer_index,
        "packed": True,
        "validated": validate,
        "gate_out_dim": gate_out_dim,
        "up_out_dim": int(up_weight.shape[0]),
    }


def _pack_qwen3next_attention(attn: Any, *, layer_index: int, validate: bool) -> dict[str, Any]:
    if getattr(attn, "_dflash_qkv_weight", None) is not None:
        return {"layer_index": layer_index, "packed": True, "validated": False}

    q_weight = attn.q_proj.weight
    k_weight = attn.k_proj.weight
    v_weight = attn.v_proj.weight
    q_out_dim = int(q_weight.shape[0])
    k_out_dim = int(k_weight.shape[0])
    v_out_dim = int(v_weight.shape[0])
    q_bias = getattr(attn.q_proj, "bias", None)
    k_bias = getattr(attn.k_proj, "bias", None)
    v_bias = getattr(attn.v_proj, "bias", None)

    x = mx.random.normal((2, 4, int(q_weight.shape[1]))).astype(q_weight.dtype)
    if validate:
        q_old = attn.q_proj(x)
        k_old = attn.k_proj(x)
        v_old = attn.v_proj(x)
        attn_old = attn(x)

    packed_weight = mx.concatenate([q_weight, k_weight, v_weight], axis=0)
    packed_bias = _concat_biases(q_bias, k_bias, v_bias)
    if packed_bias is None:
        mx.eval(packed_weight)
    else:
        mx.eval(packed_weight, packed_bias)

    if validate:
        qkv = _linear_forward(x, packed_weight, packed_bias)
        q_new, k_new, v_new = mx.split(qkv, [q_out_dim, q_out_dim + k_out_dim], axis=-1)
        _assert_exact(f"attn[{layer_index}] q_proj", q_old, q_new)
        _assert_exact(f"attn[{layer_index}] k_proj", k_old, k_new)
        _assert_exact(f"attn[{layer_index}] v_proj", v_old, v_new)

    _install_packed_attention_hook(attn)
    attn._dflash_qkv_weight = packed_weight
    attn._dflash_qkv_bias = packed_bias
    attn._dflash_q_proj_out_dim = q_out_dim
    attn._dflash_k_proj_out_dim = k_out_dim
    _set_linear_from_packed(attn.q_proj, packed_weight, 0, q_out_dim, packed_bias)
    _set_linear_from_packed(
        attn.k_proj,
        packed_weight,
        q_out_dim,
        q_out_dim + k_out_dim,
        packed_bias,
    )
    _set_linear_from_packed(
        attn.v_proj,
        packed_weight,
        q_out_dim + k_out_dim,
        q_out_dim + k_out_dim + v_out_dim,
        packed_bias,
    )

    if validate:
        attn_new = attn(x)
        _assert_exact(f"attn[{layer_index}] full", attn_old, attn_new)

    return {
        "layer_index": layer_index,
        "packed": True,
        "validated": validate,
        "q_out_dim": q_out_dim,
        "k_out_dim": k_out_dim,
        "v_out_dim": v_out_dim,
    }


def pack_target_model_weights(target_model: Any, *, validate: bool = True) -> dict[str, Any]:
    return pack_target_model_weights_selective(
        target_model,
        validate=validate,
        pack_mlp=True,
        pack_attention=False,
    )


def pack_target_model_weights_selective(
    target_model: Any,
    *,
    validate: bool = True,
    pack_mlp: bool = True,
    pack_attention: bool = False,
) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_pack_info", None) is not None:
        return text_model._dflash_pack_info

    pack_info = {
        "enabled": True,
        "validated": validate,
        "pack_mlp": pack_mlp,
        "pack_attention": pack_attention,
        "packed_mlp_layers": [],
        "packed_attention_layers": [],
    }
    for layer_index, layer in enumerate(text_model.layers):
        mlp = getattr(layer, "mlp", None)
        if pack_mlp and type(mlp).__name__ == "Qwen3NextMLP":
            pack_info["packed_mlp_layers"].append(
                _pack_qwen3next_mlp(mlp, layer_index=layer_index, validate=validate)
            )

        attn = getattr(layer, "self_attn", None)
        if pack_attention and type(attn).__name__ == "Qwen3NextAttention":
            pack_info["packed_attention_layers"].append(
                _pack_qwen3next_attention(attn, layer_index=layer_index, validate=validate)
            )

    text_model._dflash_pack_info = pack_info
    return pack_info


def _install_speculative_linear_cache_hook(linear_attn: Any) -> None:
    cls = type(linear_attn)
    if getattr(cls, "_dflash_speculative_call_installed", False):
        return

    original_call = cls.__call__

    def speculative_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(cache, "_armed", False):
            return original_call(self, inputs, mask=mask, cache=cache)

        from mlx.nn.layers.distributed import sum_gradients

        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z_proj = self.in_proj_z(inputs)
        z = z_proj.reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(B, S, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        state = cache[1]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = gated_delta_mod.compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        if state is None:
            _, _, h_k, d_k = q.shape
            h_v, d_v = v.shape[-2:]
            state = mx.zeros((B, h_v, d_v, d_k), dtype=q.dtype)
        state_in = state

        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and not self.training
        ):
            if getattr(cache, "_armed", False):
                from mio.dflash.kernels import gated_delta_kernel_with_tape

                out, state, innovation_tape = gated_delta_kernel_with_tape(
                    q, k, v, g, beta, state, mask
                )
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )
            else:
                out, state = gated_delta_mod.gated_delta_kernel(q, k, v, g, beta, state, mask)
        else:
            out, state = gated_delta_mod.gated_delta_ops(q, k, v, g, beta, state, mask)
            if getattr(cache, "_armed", False):
                decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
                decayed_state = state_in[:, None, ...] * decay
                kv_mem = (decayed_state * k[..., None, :]).sum(axis=-1)
                innovation_tape = (v - kv_mem) * beta[..., None]
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )

        cache[1] = state
        out = self.norm(out, z)
        out_flat = out.reshape(B, S, -1)
        out = self.out_proj(out_flat)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True


def _split_sdpa_mask(
    mask: Optional[Any],
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Optional[Any]:
    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]


def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    q_len = int(queries.shape[2])
    if q_len <= chunk_size:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    outputs: list[mx.array] = []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        key_end = cached_prefix_len + end
        chunk_mask = _split_sdpa_mask(mask, query_start=start, query_end=end, key_end=key_end)
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)


_HYBRID_SDPA_EXACT_KV_THRESHOLD = 1024


def _scalar_cache_offset(cache: Any) -> int | None:
    """Return a scalar cache offset, or ``None`` for per-sequence batches.

    MLX-LM's continuous ``BatchGenerator`` stores one offset per active
    sequence. Mio's split-attention optimization is designed for a single
    speculative stream, while the original MLX attention path already handles
    vector offsets correctly. The hook therefore delegates batched caches back
    to the original implementation instead of coercing an array to ``int``.
    """

    if cache is None:
        return 0
    offset = getattr(cache, "offset", 0)
    size = getattr(offset, "size", 1)
    try:
        if int(size) != 1:
            return None
        return int(offset)
    except (TypeError, ValueError):
        return None


def _install_split_full_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_split_full_attention_installed", False):
        return

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not getattr(self, "_dflash_split_sdpa_enabled", False):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        cached_prefix_len = _scalar_cache_offset(cache)
        if cached_prefix_len is None:
            return original_call(self, x, mask=mask, cache=cache)

        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_len)
            keys = self.rope(keys, offset=cached_prefix_len)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        exact_prefix_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_exact_kv_threshold",
                _HYBRID_SDPA_EXACT_KV_THRESHOLD,
            )
        )
        should_split = (
            cache is not None
            and cached_prefix_len >= exact_prefix_threshold
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        should_use_batched_2pass = (
            should_split
            and int(queries.shape[2]) == 16
            and queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if should_use_batched_2pass:
            from mio.dflash.kernels import batched_sdpa_2pass_exact

            output = batched_sdpa_2pass_exact(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
            )
            if output is None:
                output = _split_sdpa_output(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=self.scale,
                    mask=mask,
                    cache=cache,
                    chunk_size=1,
                    cached_prefix_len=cached_prefix_len,
                )
        elif should_split:
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=1,
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = split_call
    cls._dflash_split_full_attention_installed = True


def _install_target_speculative_hooks(target_model: Any) -> None:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_speculative_hooks_installed", False):
        return
    if detect_target_family(target_model) == "pure_attention":
        text_model._dflash_speculative_hooks_installed = True
        return
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            _install_exact_small_proj_hooks(layer.linear_attn)
            _install_speculative_linear_cache_hook(layer.linear_attn)
        elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            _install_split_full_attention_hook(layer.self_attn)
    text_model._dflash_speculative_hooks_installed = True


def configure_full_attention_split(
    target_model: Any,
    *,
    enabled: bool,
    chunk_size: int = 8,
) -> None:
    text_model = _target_text_model(target_model)
    if detect_target_family(target_model) == "pure_attention":
        return
    _install_target_speculative_hooks(target_model)
    for layer in text_model.layers:
        if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = enabled
            layer.self_attn._dflash_split_sdpa_chunk_size = int(chunk_size)
            layer.self_attn._dflash_split_sdpa_exact_kv_threshold = (
                _HYBRID_SDPA_EXACT_KV_THRESHOLD
            )


def _detect_head_dim_for_tq(target_model: Any) -> int:
    text = _target_text_model(target_model)
    for layer in text.layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            hd = getattr(attn, "head_dim", None)
            if isinstance(hd, int):
                return hd
            q_proj = getattr(attn, "q_proj", None)
            n_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_heads", None)
            if q_proj is not None and n_heads:
                return int(q_proj.weight.shape[0]) // int(n_heads)
    raise RuntimeError("Could not detect head_dim for TurboQuant cache")


def make_target_cache(
    target_model: Any,
    *,
    enable_speculative_linear_cache: bool,
    quantize_kv_cache: bool = False,
    tq_bits: Optional[int] = None,
    pq_bits: Optional[int] = None,
) -> list[Any]:
    """Build per-layer target cache list.

    Precedence for full-attention layers:
      - pq_bits in {2, 3, 4}: PolarQuantKVCache (Hadamard + mx.quantize)
      - tq_bits in {2, 3, 4}: TurboQuantKVCacheV2 (patch.apply() required)
      - quantize_kv_cache: mlx_lm QuantizedKVCache (8-bit)
      - else: plain KVCache
    """
    text_model = _target_text_model(target_model)
    head_dim = _detect_head_dim_for_tq(target_model)

    # PolarQuant takes precedence over TurboQuant
    PolarQuantKVCacheClass = None
    if pq_bits in (2, 3, 4):
        from mio.polarquant.cache import PolarQuantKVCache
        PolarQuantKVCacheClass = PolarQuantKVCache
        # Only patch SDPA when rotation is active (sub-4-bit)
        if pq_bits < 4:
            from mio.polarquant import patch as pq_patch
            pq_patch.apply()
        try:
            configure_full_attention_split(target_model, enabled=False)
        except Exception:
            pass
    elif tq_bits in (2, 3, 4):
        from mio.turboquant import patch as tq_patch
        from mio.turboquant.cache_v2 import TurboQuantKVCacheV2
        tq_patch.apply()
        try:
            configure_full_attention_split(target_model, enabled=False)
        except Exception:
            pass
    else:
        TurboQuantKVCacheV2 = None

    caches: list[Any] = []
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            if enable_speculative_linear_cache:
                _install_target_speculative_hooks(target_model)
                conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                caches.append(
                    RecurrentRollbackCache(size=2, conv_kernel_size=conv_kernel_size)
                )
            else:
                caches.append(cache_mod.ArraysCache(size=2))
        else:
            if PolarQuantKVCacheClass is not None:
                caches.append(
                    PolarQuantKVCacheClass(
                        head_dim=head_dim,
                        bits=pq_bits,
                        group_size=64,
                    )
                )
            elif tq_bits in (2, 3, 4) and TurboQuantKVCacheV2 is not None:
                caches.append(
                    TurboQuantKVCacheV2(
                        head_dim=head_dim,
                        bits=tq_bits,
                        group_size=64,
                        use_qjl=False,
                    )
                )
            elif quantize_kv_cache:
                caches.append(cache_mod.QuantizedKVCache(group_size=64, bits=8))
            else:
                caches.append(cache_mod.KVCache())
    return caches


def load_target_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    pack_target_weights: bool = False,
    pack_attention_weights: bool = False,
    validate_packing: bool = True,
    split_full_attention_sdpa: bool = True,
    split_full_attention_chunk_size: int = 8,
    quantize_kv_cache: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_family = detect_target_family(model)
    if target_family == "hybrid_gdn":
        _install_target_speculative_hooks(model)
        configure_full_attention_split(
            model,
            enabled=split_full_attention_sdpa and not quantize_kv_cache,
            chunk_size=split_full_attention_chunk_size,
        )
    meta = {
        "resolved_model_ref": resolved_ref,
        "config": config,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "target_family": target_family,
    }
    if pack_target_weights:
        meta["packing"] = pack_target_model_weights_selective(
            model,
            validate=validate_packing,
            pack_mlp=True,
            pack_attention=pack_attention_weights,
        )
    return model, tokenizer, meta


def load_draft_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    quantize_draft: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="draft")
    model_path = _resolve_local_model_path(resolved_ref)
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    quantized = _should_quantize_draft(quantize_draft)
    if quantized:
        nn.quantize(model, bits=4, group_size=64)
    return model, {
        "resolved_model_ref": str(model_ref) if model_ref is not None else str(resolved_ref),
        "config": config,
        "quantize_draft": bool(quantized),
    }


def validate_draft_target_compatibility(
    target_config: dict[str, Any],
    draft_config: dict[str, Any],
) -> dict[str, int]:
    """Fail fast when a DFlash checkpoint cannot match the target model."""

    target_text = target_config.get("text_config") or target_config
    errors: list[str] = []

    target_hidden = int(target_text.get("hidden_size", 0) or 0)
    draft_hidden = int(draft_config.get("hidden_size", 0) or 0)
    target_layers = int(target_text.get("num_hidden_layers", 0) or 0)
    declared_target_layers = int(draft_config.get("num_target_layers", 0) or 0)
    target_vocab = int(target_text.get("vocab_size", 0) or 0)
    draft_vocab = int(draft_config.get("vocab_size", 0) or 0)
    target_context = int(target_text.get("max_position_embeddings", 0) or 0)
    sliding_window = int(draft_config.get("sliding_window", 0) or 0)
    draft_layers = int(draft_config.get("num_hidden_layers", 0) or 0)
    layer_types = tuple(draft_config.get("layer_types") or ())
    dflash_config = draft_config.get("dflash_config") or {}
    target_layer_ids = tuple(int(index) for index in dflash_config.get("target_layer_ids") or ())
    mask_token_id = int(dflash_config.get("mask_token_id", -1) or 0)
    block_size = int(draft_config.get("block_size", 0) or 0)

    if not target_hidden or target_hidden != draft_hidden:
        errors.append(f"hidden_size target={target_hidden} draft={draft_hidden}")
    if not target_layers or target_layers != declared_target_layers:
        errors.append(
            f"num_target_layers target={target_layers} draft={declared_target_layers}"
        )
    if not target_vocab or target_vocab != draft_vocab:
        errors.append(f"vocab_size target={target_vocab} draft={draft_vocab}")
    if target_layer_ids and (
        len(target_layer_ids) != draft_layers
        or min(target_layer_ids) < 0
        or max(target_layer_ids) >= target_layers
    ):
        errors.append(
            f"target_layer_ids={list(target_layer_ids)} invalid for {target_layers} target layers"
        )
    if layer_types and len(layer_types) != draft_layers:
        errors.append(
            f"layer_types has {len(layer_types)} entries for {draft_layers} draft layers"
        )
    if sliding_window and target_context and sliding_window > target_context:
        errors.append(
            f"sliding_window={sliding_window} exceeds target context={target_context}"
        )
    if not 0 <= mask_token_id < target_vocab:
        errors.append(f"mask_token_id={mask_token_id} outside target vocab={target_vocab}")
    if block_size <= 1:
        errors.append(f"block_size={block_size} must be greater than one")

    if errors:
        raise ValueError("Incompatible DFlash target/draft: " + "; ".join(errors))
    return {
        "hidden_size": target_hidden,
        "num_target_layers": target_layers,
        "vocab_size": target_vocab,
        "block_size": block_size,
        "sliding_window": sliding_window,
    }


def bind_draft_target_model(
    draft_model: DFlashDraftModel,
    target_model: Any,
) -> None:
    """Bind target-specific embedding semantics required by upstream DFlash."""

    binder = getattr(draft_model, "bind_target_model", None)
    if not callable(binder):
        return

    class _TargetOps:
        @staticmethod
        def text_model(model: Any) -> Any:
            return _target_text_model(model)

    binder(target_model, target_ops=_TargetOps())


def target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: Optional[mx.array] = None,
    cache: Optional[list[Any]] = None,
    input_embeddings: Optional[mx.array] = None,
    capture_layer_ids: Optional[set[int]] = None,
    only_last_logit: bool = False,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    """Target forward with per-layer hidden-state capture.

    If `only_last_logit=True`, the LM head is applied only to the final
    position's hidden state, cutting a vocab-sized matmul per prefill token
    (significant for long prompts). The returned `logits` has shape
    `(B, 1, V)` instead of `(B, T, V)`. Hidden states are fully captured
    regardless of this flag because the draft relies on them.
    """
    inner = _target_text_model(target_model)
    hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(inner.layers)
    capture_all = capture_layer_ids is None
    if capture_all:
        captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
    else:
        capture_layer_ids = set(capture_layer_ids)
        captured = {0: hidden_states} if 0 in capture_layer_ids else {}
    h = hidden_states

    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
            if capture_all:
                captured.append(h)
            elif (layer_index + 1) in capture_layer_ids:
                captured[layer_index + 1] = h
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            h = layer(h, mask, layer_cache)
            if capture_all:
                captured.append(h)
            elif (layer_index + 1) in capture_layer_ids:
                captured[layer_index + 1] = h
    if only_last_logit and int(h.shape[1]) > 1:
        h_last = h[:, -1:, :]
    else:
        h_last = h
    normalized = inner.norm(h_last)
    logits = _lm_head_logits(target_model, normalized)
    return logits, captured


def _prefill_chunk_size_env() -> int:
    """Max tokens per prefill chunk. Default 2048 keeps the attention
    intermediate (heads × chunk × total_ctx × 2 bytes) comfortably below
    the 30 GB Metal single-buffer limit for our largest tiers at 128K."""
    import os
    try:
        return max(256, int(os.environ.get("MIO_PREFILL_CHUNK", "2048")))
    except ValueError:
        return 2048


def chunked_prefill(
    target_model: Any,
    *,
    input_ids: mx.array,
    cache: list[Any],
    capture_layer_ids: Optional[set[int]] = None,
    chunk_size: Optional[int] = None,
    only_last_logit: bool = False,
) -> tuple[mx.array, dict[int, mx.array] | list[mx.array]]:
    """Prefill by feeding input_ids through target_forward_with_hidden_states
    in chunks of `chunk_size`, concatenating hidden states along the sequence
    dim. Returns (logits_from_LAST_chunk, concatenated_hidden_states).

    Required whenever the single-shot attention intermediate could exceed
    Metal's ~30 GB single-buffer cap — e.g., prompts ≳ 8-16K tokens on
    large-moe with full softmax attention.

    `only_last_logit` is applied only to the LAST chunk's forward — the
    intermediate chunks never need the LM-head projection (we don't sample
    from their logits).
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected (B, L), got {input_ids.shape}")
    total = int(input_ids.shape[1])
    if chunk_size is None:
        chunk_size = _prefill_chunk_size_env()
    if total <= chunk_size:
        return target_forward_with_hidden_states(
            target_model, input_ids=input_ids, cache=cache,
            capture_layer_ids=capture_layer_ids,
            only_last_logit=only_last_logit,
        )

    logits = None
    hidden_chunks: list[list[mx.array] | dict[int, mx.array]] = []
    starts = list(range(0, total, chunk_size))
    for i, start in enumerate(starts):
        end = min(start + chunk_size, total)
        piece = input_ids[:, start:end]
        is_last = (i == len(starts) - 1)
        chunk_logits, chunk_hidden = target_forward_with_hidden_states(
            target_model, input_ids=piece, cache=cache,
            capture_layer_ids=capture_layer_ids,
            # Intermediate chunk logits are discarded. Projecting only their
            # last position avoids a vocab-sized matmul for every prompt token.
            only_last_logit=(only_last_logit or not is_last),
        )
        mx.eval(chunk_logits)
        if isinstance(chunk_hidden, dict):
            mx.eval(*chunk_hidden.values())
        else:
            mx.eval(*chunk_hidden)
        logits = chunk_logits
        hidden_chunks.append(chunk_hidden)
    if capture_layer_ids is None:
        merged = _concat_hidden_state_chunks(hidden_chunks)
    else:
        merged = _concat_hidden_state_chunk_dicts(hidden_chunks, capture_layer_ids)
    return logits, merged


def chunked_dflash_prefill(
    target_model: Any,
    draft_model: DFlashDraftModel,
    *,
    input_ids: mx.array,
    cache: list[Any],
    chunk_size: Optional[int] = None,
    only_last_logit: bool = True,
) -> tuple[mx.array, mx.array]:
    """Prefill target caches while storing only projected DFlash features.

    Capturing N target layers normally creates a ``N * hidden_size`` tensor
    for the entire prompt. Projecting each chunk immediately bounds that raw
    feature lifetime to one chunk and keeps the persistent context at one
    ``hidden_size`` per token.
    """

    if input_ids.ndim != 2:
        raise ValueError(f"expected (B, L), got {input_ids.shape}")
    total = int(input_ids.shape[1])
    if total <= 0:
        raise ValueError("DFlash prefill requires at least one token")
    if chunk_size is None:
        chunk_size = _prefill_chunk_size_env()

    capture_layer_ids = {
        int(layer_id) + 1 for layer_id in draft_model.target_layer_ids
    }
    projected_chunks: list[mx.array] = []
    logits: mx.array | None = None
    starts = list(range(0, total, chunk_size))
    for index, start in enumerate(starts):
        end = min(start + chunk_size, total)
        is_last = index == len(starts) - 1
        chunk_logits, captured = target_forward_with_hidden_states(
            target_model,
            input_ids=input_ids[:, start:end],
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            only_last_logit=(only_last_logit or not is_last),
        )
        if not isinstance(captured, dict):
            raise TypeError("selective DFlash capture must return a layer dictionary")
        raw_features = extract_context_feature_from_dict(
            captured,
            list(draft_model.target_layer_ids),
        )
        projected = _project_draft_context(draft_model, raw_features)
        mx.eval(chunk_logits, projected)
        projected_chunks.append(projected)
        logits = chunk_logits

    assert logits is not None
    draft_context = (
        projected_chunks[0]
        if len(projected_chunks) == 1
        else mx.concatenate(projected_chunks, axis=1)
    )
    mx.eval(draft_context)
    return logits, draft_context


def trim_cache_to(cache_entries: list[Any], size: int) -> int:
    if not cache_entries:
        return 0
    current_size = int(getattr(cache_entries[0], "offset", 0) or 0)
    if current_size <= size:
        return 0
    return int(cache_mod.trim_prompt_cache(cache_entries, current_size - size) or 0)


def _arm_target_rollback(cache_entries: list[Any]) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback()


def _arm_target_rollback_with_prefix(
    cache_entries: list[Any],
    *,
    prefix_len: int,
) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback(prefix_len=int(prefix_len))


def _clear_rollback_state(cache_entry: Any) -> None:
    if hasattr(cache_entry, "_armed"):
        cache_entry._armed = False
    if hasattr(cache_entry, "_tape"):
        cache_entry._tape = None
    if hasattr(cache_entry, "_tape_k"):
        cache_entry._tape_k = None
    if hasattr(cache_entry, "_tape_g"):
        cache_entry._tape_g = None
    if hasattr(cache_entry, "_tape_qkv"):
        cache_entry._tape_qkv = None
    if hasattr(cache_entry, "_snapshot"):
        cache_entry._snapshot = None


def _restore_target_cache_after_acceptance(
    cache_entries: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int = 0,
) -> int:
    replay_ns_total = 0
    fully_accepted = drafted_tokens > 0 and acceptance_length == drafted_tokens
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            if fully_accepted:
                _clear_rollback_state(cache_entry)
                continue
            replay_start_ns = time.perf_counter_ns()
            cache_entry.rollback(acceptance_length)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "offset"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                cache_entry.offset = target_len
        elif hasattr(cache_entry, "crop"):
            cache_entry.crop(target_len)
    return replay_ns_total


def _verify_target_block(
    *,
    target_model: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    verify_chunk_tokens: Optional[int],
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    total_tokens = int(verify_ids.shape[1])
    if total_tokens <= 0:
        raise ValueError("verify block must contain at least one token")

    chunk_size = max(1, int(verify_chunk_tokens or total_tokens))
    if chunk_size >= total_tokens:
        verify_logits, verify_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        return verify_logits, verify_hidden_states

    logits_chunks: list[mx.array] = []
    hidden_state_chunks: list[list[mx.array]] | list[dict[int, mx.array]]
    hidden_state_chunks = []
    for offset in range(0, total_tokens, chunk_size):
        verify_chunk = verify_ids[:, offset : offset + chunk_size]
        chunk_logits, chunk_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_chunk,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        logits_chunks.append(chunk_logits)
        hidden_state_chunks.append(chunk_hidden_states)

    if capture_layer_ids is None:
        return mx.concatenate(logits_chunks, axis=1), _concat_hidden_state_chunks(hidden_state_chunks)
    return (
        mx.concatenate(logits_chunks, axis=1),
        _concat_hidden_state_chunk_dicts(hidden_state_chunks, capture_layer_ids),
    )


def generate_baseline_once(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    tq_bits: Optional[int] = None,
    pq_bits: Optional[int] = None,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    stop_token_ids = list(stop_token_ids or [])

    if max_new_tokens <= 0:
        return {
            "elapsed_us": 0.0,
            "prompt_token_count": len(prompt_tokens),
            "generated_token_ids": [],
            "generation_tokens": 0,
        }

    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
        tq_bits=tq_bits,
        pq_bits=pq_bits,
    )
    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    logits, _ = chunked_prefill(
        target_model,
        input_ids=prompt_array,
        cache=cache,
        capture_layer_ids=set(),
        only_last_logit=True,
    )
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    return {
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": len(prompt_tokens),
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
    }


def stream_baseline_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    fallback_reason: Optional[str] = None,
    tq_bits: Optional[int] = None,
    pq_bits: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    prompt_len = len(prompt_tokens)
    stop_token_ids = list(stop_token_ids or [])
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
        tq_bits=tq_bits,
        pq_bits=pq_bits,
    )
    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    logits, _ = chunked_prefill(
        target_model,
        input_ids=prompt_array,
        cache=cache,
        capture_layer_ids=set(),
        only_last_logit=True,
    )
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    yield {
        "event": "prefill",
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": prompt_len,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }

    yield {
        "event": "token",
        "token_id": next_token,
        "generated_tokens": 1,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)
        yield {
            "event": "token",
            "token_id": next_token,
            "generated_tokens": len(generated_tokens),
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "fallback_ar": True,
            "fallback_reason": fallback_reason,
        }

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    yield {
        "event": "summary",
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "accepted_from_draft": 0,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": 0.0,
            "draft_prefill": 0.0,
            "draft_incremental": 0.0,
            "verify": 0.0,
            "replay": 0.0,
            "commit": 0.0,
        },
        "verify_len_cap": None,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }


def generate_dflash_once(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: int = 16,
    verify_chunk_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    tq_bits: Optional[int] = None,
    pq_bits: Optional[int] = None,
    warm_state: Optional[dict] = None,
    return_final_state: bool = False,
    prefill_only: bool = False,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()
    draft_window_size = _effective_draft_window(draft_model, draft_window_size)

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        baseline = generate_baseline_once(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            tq_bits=tq_bits,
            pq_bits=pq_bits,
        )
        baseline.update(
            {
                "accepted_from_draft": 0,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
                "phase_timings_us": {
                    "prefill": baseline["elapsed_us"],
                    "draft": 0.0,
                    "draft_prefill": 0.0,
                    "draft_incremental": 0.0,
                    "verify": 0.0,
                    "replay": 0.0,
                    "commit": 0.0,
                },
                "speculative_linear_cache": False,
                "verify_chunk_tokens": None,
                "verify_len_cap": None,
                "quantize_kv_cache": bool(quantize_kv_cache),
                "fallback_ar": True,
                "fallback_reason": fallback_reason,
            }
        )
        return baseline
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    use_speculative_linear_cache = verify_chunk_tokens is None

    # Warm-prefix optimization: if caller supplies caches pre-populated for the
    # first `warm_offset` tokens of the prompt, we skip prefill for that range.
    warm_offset = 0
    if warm_state is not None:
        target_cache = warm_state["target_cache"]
        draft_cache = warm_state["draft_cache"]
        warm_offset = int(warm_state.get("offset", 0))
    else:
        target_cache = make_target_cache(
            target_model,
            enable_speculative_linear_cache=use_speculative_linear_cache,
            quantize_kv_cache=quantize_kv_cache,
            tq_bits=tq_bits,
            pq_bits=pq_bits,
        )
        draft_cache = [
            ContextOnlyDraftKVCache(
                sink_size=draft_sink_size,
                window_size=draft_window_size,
            )
            for _ in range(len(draft_model.layers))
        ]

    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    # Clip warm_offset to the prompt; must leave at least 1 token to prefill so we
    # have a last-position logit from which to sample the first bonus token.
    if warm_offset >= prompt_len:
        warm_offset = prompt_len - 1
    prefill_tokens = prompt_tokens[warm_offset:]
    prompt_array = mx.array(prefill_tokens, dtype=mx.uint32)[None]

    start_ns = time.perf_counter_ns()
    prefill_start_ns = time.perf_counter_ns()
    # We only use the last position's logits to pick the first bonus token.
    # Passing only_last_logit skips the LM-head projection on intermediate
    # positions — a vocab-sized matmul per prompt token.
    # Use chunked prefill when the prompt could blow past Metal's single-buffer
    # cap (~30 GB). Under the threshold, chunked_prefill takes a single-shot path.
    prefill_logits, draft_context = chunked_dflash_prefill(
        target_model,
        draft_model,
        input_ids=prompt_array,
        cache=target_cache,
        only_last_logit=True,
    )
    prefill_ns = time.perf_counter_ns() - prefill_start_ns

    suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
    # prefill_logits shape is (B, 1, V) when only_last_logit=True.
    staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)
    # Prefill-only snapshots may carry projected draft context while their
    # draft KV cache is still empty. Preserve compatibility with old snapshots
    # that stored raw concatenated target features.
    if warm_state is not None:
        cached_context = warm_state.get("draft_context")
        if cached_context is None and warm_state.get("target_hidden") is not None:
            cached_context = _project_draft_context(
                draft_model,
                warm_state["target_hidden"],
            )
        if cached_context is not None and int(cached_context.shape[1]) > 0:
            draft_context = mx.concatenate([cached_context, draft_context], axis=1)
            mx.eval(draft_context)

    # Prefill-only shortcut: used by PrefixCache to warm caches for a specific prefix.
    if prefill_only:
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        result = {
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": [],
            "generation_tokens": 0,
            "accepted_from_draft": 0,
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": 0.0, "draft_prefill": 0.0, "draft_incremental": 0.0,
                "verify": 0.0, "replay": 0.0, "commit": 0.0,
            },
            "speculative_linear_cache": use_speculative_linear_cache,
            "verify_chunk_tokens": None,
            "verify_len_cap": None,
            "quantize_kv_cache": bool(quantize_kv_cache),
            "tokens_per_cycle": 0.0,
            "peak_memory_gb": (
                float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None
            ),
            "warm_offset": warm_offset,
            "prefill_only": True,
        }
        if return_final_state:
            result["final_state"] = {
                "target_cache": target_cache,
                "draft_cache": draft_cache,
                "draft_context": draft_context,
                "offset": prompt_len,
            }
        return result

    effective_block_tokens = max(1, min(int(block_tokens or 1), int(draft_model.block_size)))
    generated_token_buffer = mx.full((max_new_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
    block_token_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
    generated_token_count = 0
    accepted_from_draft = 0
    cycles_completed = 0
    verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)
    start = prompt_len

    draft_ns_total = 0
    draft_prefill_ns = 0
    draft_incremental_ns = 0
    verify_ns_total = 0
    replay_ns_total = 0
    commit_ns_total = 0
    seen_draft_cycle = False
    acceptance_history: list[int] = []

    while generated_token_count < max_new_tokens:
        draft_cycle_ns = 0
        verify_cycle_ns = 0
        replay_cycle_ns = 0
        remaining = max_new_tokens - generated_token_count
        block_len = max(1, min(effective_block_tokens, remaining))
        block_token_buffer[:block_len] = draft_model.mask_token_id
        block_token_buffer[:1] = staged_first
        block_token_ids = block_token_buffer[:block_len]

        if block_len > 1:
            draft_start_ns = time.perf_counter_ns()
            noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
            draft_hidden = _forward_draft(
                draft_model,
                noise_embedding=noise_embedding,
                draft_context=draft_context,
                cache=draft_cache,
            )
            draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
            mx.async_eval(draft_logits)
            mx.eval(draft_logits)
            drafted = greedy_tokens_with_mask(draft_logits, suppress_token_mask).squeeze(0)
            block_token_ids[1:block_len] = drafted
            draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
            draft_ns_total += draft_cycle_ns
            if not seen_draft_cycle:
                draft_prefill_ns += draft_cycle_ns
                seen_draft_cycle = True
            else:
                draft_incremental_ns += draft_cycle_ns

        verify_token_ids = block_token_ids[: min(block_len, verify_len_cap)]
        verify_ids = verify_token_ids[None]
        if use_speculative_linear_cache:
            _arm_target_rollback_with_prefix(target_cache, prefix_len=start)
        verify_start_ns = time.perf_counter_ns()
        verify_logits, verify_hidden_states = _verify_target_block(
            target_model=target_model,
            verify_ids=verify_ids,
            target_cache=target_cache,
            verify_chunk_tokens=verify_chunk_tokens,
            capture_layer_ids=capture_layer_ids,
        )
        verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
        verify_ns_total += verify_cycle_ns

        posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
        acceptance_len = int(
            _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
        )
        acceptance_history.append(acceptance_len)
        committed_hidden = extract_context_feature_from_dict(
            verify_hidden_states,
            list(draft_model.target_layer_ids),
        )[:, : (1 + acceptance_len), :]
        mx.eval(committed_hidden, posterior)

        commit_count = 1 + acceptance_len
        committed_segment = verify_token_ids[:commit_count]
        generated_token_buffer[generated_token_count : generated_token_count + commit_count] = committed_segment
        generated_token_count += commit_count
        accepted_from_draft += acceptance_len

        commit_start_ns = time.perf_counter_ns()
        start += commit_count
        draft_context = _project_draft_context(draft_model, committed_hidden)
        mx.eval(draft_context)
        replay_cycle_ns = _restore_target_cache_after_acceptance(
            target_cache,
            target_len=start,
            acceptance_length=acceptance_len,
            drafted_tokens=block_len - 1,
        )
        replay_ns_total += replay_cycle_ns
        cycles_completed += 1
        commit_wall_ns = time.perf_counter_ns() - commit_start_ns
        commit_ns_total += commit_wall_ns
        stop_hit = False
        if stop_token_array is not None:
            stop_hit = bool(
                mx.any(
                    mx.equal(
                        committed_segment[:, None],
                        stop_token_array[None, :],
                    )
                ).item()
            )
        if stop_hit:
            break

        staged_first = posterior[acceptance_len : acceptance_len + 1]

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    generated_token_ids = (
        generated_token_buffer[:generated_token_count].tolist()
        if generated_token_count > 0
        else []
    )
    first_20 = acceptance_history[:20]
    last_20 = acceptance_history[-20:]
    result = {
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_token_ids,
        "generation_tokens": len(generated_token_ids),
        "accepted_from_draft": accepted_from_draft,
        "acceptance_ratio": (
            accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
        ),
        "cycles_completed": cycles_completed,
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": draft_ns_total / 1_000.0,
            "draft_prefill": draft_prefill_ns / 1_000.0,
            "draft_incremental": draft_incremental_ns / 1_000.0,
            "verify": verify_ns_total / 1_000.0,
            "replay": replay_ns_total / 1_000.0,
            "commit": commit_ns_total / 1_000.0,
        },
        "speculative_linear_cache": use_speculative_linear_cache,
        "verify_chunk_tokens": int(verify_chunk_tokens) if verify_chunk_tokens else None,
        "verify_len_cap": int(verify_len_cap),
        "quantize_kv_cache": bool(quantize_kv_cache),
        "tokens_per_cycle": (len(generated_token_ids) / cycles_completed) if cycles_completed > 0 else 0.0,
        "acceptance_first_20_avg": (sum(first_20) / len(first_20)) if first_20 else 0.0,
        "acceptance_last_20_avg": (sum(last_20) / len(last_20)) if last_20 else 0.0,
        "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        "warm_offset": warm_offset,
    }
    if return_final_state:
        result["final_state"] = {
            "target_cache": target_cache,
            "draft_cache": draft_cache,
            "offset": prompt_len,  # caches now hold the full prompt
        }
    return result


def stream_dflash_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: int = 16,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    # Conditional suppression: after `relax_suppress_after` generated tokens,
    # remove `relax_suppress_token_ids` from the suppress mask AND add them
    # back to the stop set. Use case: suppress EOS for the first N tokens so
    # the model can't escape via "Now I'll update..." + EOS, then allow EOS
    # so it stops naturally after emitting its tool call (protects DFlash
    # acceptance which drops when EOS predictions are always rejected).
    relax_suppress_after: int = 0,
    relax_suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    tq_bits: Optional[int] = None,
    pq_bits: Optional[int] = None,
    warm_state: Optional[dict] = None,
) -> Iterator[dict[str, Any]]:
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()
    draft_window_size = _effective_draft_window(draft_model, draft_window_size)

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    fallback_reason: Optional[str] = None

    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        yield from stream_baseline_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            fallback_reason=fallback_reason,
            tq_bits=tq_bits,
            pq_bits=pq_bits,
        )
        return
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    # Warm-prefix optimization (mirrors generate_dflash_once).
    warm_offset = 0
    if warm_state is not None:
        target_cache = warm_state["target_cache"]
        draft_cache = warm_state["draft_cache"]
        warm_offset = int(warm_state.get("offset", 0))
    else:
        target_cache = make_target_cache(
            target_model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=quantize_kv_cache,
            tq_bits=tq_bits,
            pq_bits=pq_bits,
        )
        draft_cache = [
            ContextOnlyDraftKVCache(
                sink_size=draft_sink_size,
                window_size=draft_window_size,
            )
            for _ in range(len(draft_model.layers))
        ]

    if warm_offset >= prompt_len:
        warm_offset = prompt_len - 1
    prefill_tokens = prompt_tokens[warm_offset:]
    prompt_array = mx.array(prefill_tokens, dtype=mx.uint32)[None]

    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    start_ns = time.perf_counter_ns()
    prefill_start_ns = time.perf_counter_ns()
    prefill_logits, draft_context = chunked_dflash_prefill(
        target_model,
        draft_model,
        input_ids=prompt_array,
        cache=target_cache,
        only_last_logit=True,
    )
    prefill_ns = time.perf_counter_ns() - prefill_start_ns

    vocab_dim = int(prefill_logits.shape[-1])
    # Build both masks upfront. "relaxed" mask drops `relax_suppress_token_ids`
    # from the initial suppression — we swap to it after N tokens.
    relax_ids = set(int(x) for x in (relax_suppress_token_ids or []))
    initial_mask = build_suppress_token_mask(vocab_dim, suppress_token_ids)
    if relax_suppress_after > 0 and relax_ids:
        relaxed_ids = [t for t in (suppress_token_ids or []) if int(t) not in relax_ids]
        relaxed_mask = build_suppress_token_mask(vocab_dim, relaxed_ids)
    else:
        relaxed_mask = initial_mask
    suppress_token_mask = initial_mask
    staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)
    if warm_state is not None:
        cached_context = warm_state.get("draft_context")
        if cached_context is None and warm_state.get("target_hidden") is not None:
            cached_context = _project_draft_context(
                draft_model,
                warm_state["target_hidden"],
            )
        if cached_context is not None and int(cached_context.shape[1]) > 0:
            draft_context = mx.concatenate([cached_context, draft_context], axis=1)
            mx.eval(draft_context)

    yield {
        "event": "prefill",
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": prompt_len,
        "warm_offset": warm_offset,
    }

    effective_block_tokens = max(1, min(int(block_tokens or 1), int(draft_model.block_size)))
    block_token_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
    generated_token_ids: list[int] = []
    accepted_from_draft = 0
    cycles_completed = 0
    verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)
    start = prompt_len

    draft_ns_total = 0
    draft_prefill_ns = 0
    draft_incremental_ns = 0
    verify_ns_total = 0
    replay_ns_total = 0
    commit_ns_total = 0
    seen_draft_cycle = False

    while len(generated_token_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_token_ids)
        block_len = max(1, min(effective_block_tokens, remaining))
        block_token_buffer[:block_len] = draft_model.mask_token_id
        block_token_buffer[:1] = staged_first
        block_token_ids = block_token_buffer[:block_len]

        if block_len > 1:
            draft_start_ns = time.perf_counter_ns()
            noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
            draft_hidden = _forward_draft(
                draft_model,
                noise_embedding=noise_embedding,
                draft_context=draft_context,
                cache=draft_cache,
            )
            draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])

            mx.async_eval(draft_logits)
            mx.eval(draft_logits)
            drafted = greedy_tokens_with_mask(draft_logits, suppress_token_mask).squeeze(0)
            block_token_ids[1:block_len] = drafted
            draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
            draft_ns_total += draft_cycle_ns
            if not seen_draft_cycle:
                draft_prefill_ns += draft_cycle_ns
                seen_draft_cycle = True
            else:
                draft_incremental_ns += draft_cycle_ns

        verify_token_ids = block_token_ids[: min(block_len, verify_len_cap)]
        verify_ids = verify_token_ids[None]
        _arm_target_rollback_with_prefix(target_cache, prefix_len=start)
        verify_start_ns = time.perf_counter_ns()
        verify_logits, verify_hidden_states = _verify_target_block(
            target_model=target_model,
            verify_ids=verify_ids,
            target_cache=target_cache,
            verify_chunk_tokens=None,
            capture_layer_ids=capture_layer_ids,
        )
        verify_ns_total += time.perf_counter_ns() - verify_start_ns

        posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
        acceptance_len = int(
            _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
        )
        committed_hidden = extract_context_feature_from_dict(
            verify_hidden_states,
            list(draft_model.target_layer_ids),
        )[:, : (1 + acceptance_len), :]
        mx.eval(committed_hidden, posterior)

        commit_count = 1 + acceptance_len
        committed_segment = verify_token_ids[:commit_count]
        commit_start_ns = time.perf_counter_ns()
        start += commit_count
        draft_context = _project_draft_context(draft_model, committed_hidden)
        mx.eval(draft_context)
        replay_ns_total += _restore_target_cache_after_acceptance(
            target_cache,
            target_len=start,
            acceptance_length=acceptance_len,
            drafted_tokens=block_len - 1,
        )
        cycles_completed += 1
        commit_ns_total += time.perf_counter_ns() - commit_start_ns

        accepted_from_draft += acceptance_len
        committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
        for token_id in committed_ids:
            if len(generated_token_ids) >= max_new_tokens:
                break
            generated_token_ids.append(token_id)
            yield {
                "event": "token",
                "token_id": token_id,
                "generated_tokens": len(generated_token_ids),
                "acceptance_ratio": (
                    accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
                ),
                "cycles_completed": cycles_completed,
            }

        # Swap to the relaxed suppress mask once we've generated enough tokens.
        # Also promote the relaxed token ids back to stop tokens so natural
        # EOS after a tool call terminates generation cleanly.
        if (
            relax_suppress_after > 0
            and relax_ids
            and len(generated_token_ids) >= relax_suppress_after
            and suppress_token_mask is initial_mask
        ):
            suppress_token_mask = relaxed_mask
            promoted = [t for t in relax_ids if t not in set(stop_token_ids or [])]
            if promoted:
                stop_token_ids = list(stop_token_ids or []) + list(promoted)
                stop_token_array = mx.array(stop_token_ids, dtype=mx.uint32)

        stop_hit = False
        if stop_token_array is not None:
            stop_hit = bool(
                mx.any(
                    mx.equal(
                        committed_segment[:, None],
                        stop_token_array[None, :],
                    )
                ).item()
            )
        if stop_hit:
            break

        staged_first = posterior[acceptance_len : acceptance_len + 1]

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    yield {
        "event": "summary",
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_token_ids,
        "generation_tokens": len(generated_token_ids),
        "accepted_from_draft": accepted_from_draft,
        "acceptance_ratio": (
            accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
        ),
        "cycles_completed": cycles_completed,
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": draft_ns_total / 1_000.0,
            "draft_prefill": draft_prefill_ns / 1_000.0,
            "draft_incremental": draft_incremental_ns / 1_000.0,
            "verify": verify_ns_total / 1_000.0,
            "replay": replay_ns_total / 1_000.0,
            "commit": commit_ns_total / 1_000.0,
        },
        "verify_len_cap": int(verify_len_cap),
        "warm_offset": warm_offset,
        "final_state": {
            "target_cache": target_cache,
            "draft_cache": draft_cache,
            "offset": prompt_len,
        },
    }
