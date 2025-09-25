# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

import torch
import gc
import math
import functools
from typing import Any, Dict, Optional, Tuple, List, Union
from ._utils import *
from ._utils import __version__, importlib_version
from torch.nn.functional import scaled_dot_product_attention
from transformers import __version__ as transformers_version
from ...unsloth_zoo.utils import Version, _get_dtype
from ...unsloth import DEVICE_TYPE, DEVICE_COUNT
from ...unsloth_zoo.hf_utils import dtype_from_config


transformers_version = Version(transformers_version)
# Transformers moved rotary embeddings out of all attention layers
IS_ATTENTION_REFACTOR = transformers_version > Version("4.47.1")
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except:
    GradientCheckpointingLayer = type(None)

from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.utils.import_utils import _is_package_available

from ..kernels import *
# from ..tokenizer_utils import *

HAS_FLASH_ATTENTION = False
HAS_FLASH_ATTENTION_SOFTCAPPING = False

if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    torch.cuda.get_device_capability = functools.cache(torch.cuda.get_device_capability)

    if major_version >= 8:
        SUPPORTS_BFLOAT16 = True
        if _is_package_available("flash_attn"):
            # Check for CUDA linking errors "undefined symbol: _ZNK3c106SymIntltEl"
            try:
                try:
                    # See https://github.com/unslothai/unsloth/issues/1437
                    from flash_attn.flash_attn_interface import flash_attn_gpu
                except:
                    from flash_attn.flash_attn_interface import flash_attn_cuda
                HAS_FLASH_ATTENTION = True

                # Also check for softcapping
                from flash_attn import __version__ as flash_attn_version
                HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
                if not HAS_FLASH_ATTENTION_SOFTCAPPING:
                    print(
                        "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"\
                        "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"\
                        "To update flash-attn, do the below:\n"\
                        '\npip install --no-deps --no-build-isolation --upgrade "flash-attn>=2.6.3"'
                    )
            except:
                print(
                    "Unsloth: Your Flash Attention 2 installation seems to be broken?\n"\
                    "A possible explanation is you have a new CUDA version which isn't\n"\
                    "yet compatible with FA2? Please file a ticket to Unsloth or FA2.\n"\
                    "We shall now use Xformers instead, which does not have any performance hits!\n"\
                    "We found this negligible impact by benchmarking on 1x A100."
                )

                # Stop Flash Attention from importing!
                import transformers.utils.import_utils
                transformers.utils.import_utils.is_flash_attn_2_available = lambda *args, **kwargs: False
                import transformers.utils
                transformers.utils.is_flash_attn_2_available = lambda *args, **kwargs: False

                HAS_FLASH_ATTENTION = False
            pass
        else:
            HAS_FLASH_ATTENTION = False
    else:
        # Tri Dao's benchmark shows xformers is faster for now.
        HAS_FLASH_ATTENTION = False
    pass
elif DEVICE_TYPE == "hip":
    SUPPORTS_BFLOAT16 = True
    if _is_package_available("flash_attn"):
        # Check for CUDA linking errors "undefined symbol: _ZNK3c106SymIntltEl"
        try:
            try:
                # See https://github.com/unslothai/unsloth/issues/1437
                from flash_attn.flash_attn_interface import flash_attn_gpu
            except:
                from flash_attn.flash_attn_interface import flash_attn_cuda
            HAS_FLASH_ATTENTION = True

            # Also check for softcapping
            from flash_attn import __version__ as flash_attn_version
            HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
            if not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"\
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"\
                    "To update flash-attn, do the below:\n"\
                    '\npip install --no-deps --no-build-isolation --upgrade "flash-attn>=2.6.3"'
                )
        except:
            print(
                "Unsloth: Your Flash Attention 2 installation seems to be broken?\n"\
                "A possible explanation is you have a new CUDA version which isn't\n"\
                "yet compatible with FA2? Please file a ticket to Unsloth or FA2.\n"\
                "We shall now use Xformers instead, which does not have any performance hits!\n"\
                "We found this negligible impact by benchmarking on 1x A100."
            )

            # Stop Flash Attention from importing!
            import transformers.utils.import_utils
            transformers.utils.import_utils.is_flash_attn_2_available = lambda *args, **kwargs: False
            import transformers.utils
            transformers.utils.is_flash_attn_2_available = lambda *args, **kwargs: False

            HAS_FLASH_ATTENTION = False

if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func


# Final patching code
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)

# For Pytorch 2.1.1
try:
    from transformers.models.llama.modeling_llama import (
        LlamaSdpaAttention,
        LlamaFlashAttention2,
    )
except:
    LlamaSdpaAttention   = LlamaAttention
    LlamaFlashAttention2 = LlamaAttention
pass

import re, os, inspect, math, sys
import types
from huggingface_hub.utils import get_token

from triton import __version__ as triton_version
HAS_XFORMERS = xformers is not None
BlockDiagonalCausalMask = xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None

if DEVICE_TYPE == "xpu":
    clean_gpu_cache = torch.xpu.empty_cache
    get_current_device = torch.xpu.current_device
else:
    clean_gpu_cache = torch.cuda.empty_cache
    get_current_device = torch.cuda.current_device
pass

def original_apply_qkv(self, X):
    Q = self.q_proj(X)
    K = self.k_proj(X)
    V = self.v_proj(X)
    return Q, K, V
pass


def original_apply_o(self, X):
    O = self.o_proj(X)
    return O
pass

from math import sqrt as math_sqrt
KV_CACHE_INCREMENT = 512 # KV Cache update size

# SDPA has GQA internally
SDPA_HAS_GQA = "enable_gqa" in scaled_dot_product_attention.__doc__

torch_nn_functional_silu = torch.nn.functional.silu
def fast_swiglu_inference(self, X, temp_gate = None, temp_up = None, gate_multiplier = None, down_multiplier = None):
    # gate = self.gate_proj(X)
    # up   = self.up_proj(X)
    bsz, _, hd = X.shape
    # mlp_size = self.config.intermediate_size
    # temp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda:0")

    gate = fast_linear_forward(self.gate_proj, X, out = temp_gate)

    if gate_multiplier is not None:
        gate *= gate_multiplier

    up   = fast_linear_forward(self.  up_proj, X, out = temp_up)

    gate = torch_nn_functional_silu(gate, inplace = True)
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out = up[:,:,:hd])

    if down_multiplier is not None:
        down *= down_multiplier

    return down
pass

torch_square = torch.square
torch_mean   = torch.mean
def fast_rms_layernorm_inference(self, X, XX = None, XX2 = None, variance = None):
    old_dtype = X.dtype
    if XX is None:
        XX = X.to(torch.float32)
        variance = XX.square().mean(-1, keepdim = True)
    else:
        XX.copy_(X)
        torch_mean(torch_square(XX, out = XX2), -1, keepdim = True, out = variance)
    pass
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if XX is None: X = XX.to(old_dtype)
    else: X.copy_(XX)

    X *= self.weight
    return X
pass

def fast_rms_layernorm_inference_gemma(self, X, out_weight = None):
    XX = X.to(torch.float32)
    variance = XX.square().mean(-1, keepdim = True)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if out_weight is None:
        out_weight = self.weight + 1.0
    else:
        out_weight[:] = self.weight
        out_weight += 1.0
    pass

    XX *= out_weight
    return XX.to(X.dtype)
pass

# Normal layernorm with mean removal
@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def fast_layernorm_compiled(layernorm, X):
    old_dtype = X.dtype
    X = X.float()
    mean = X.mean(-1, keepdim = True)
    Xbar = X - mean
    X = Xbar * torch.rsqrt(Xbar.square().mean(-1, keepdim = True) + \
        layernorm.variance_epsilon) * \
        layernorm.weight.float()
    return X.to(old_dtype)
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L320
def LlamaAttention_fast_forward(
    self,
    hidden_states:       torch.Tensor,
    causal_mask:         Optional[BlockDiagonalCausalMask] = None,
    attention_mask:      Optional[torch.Tensor] = None,
    position_ids:        Optional[torch.LongTensor] = None,
    past_key_value:      Optional[Tuple[torch.Tensor]] = None,
    output_attentions:   bool = False,
    use_cache:           bool = False,
    padding_mask:        Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.config.num_attention_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings:
        cos, sin = position_embeddings
    else:
        # Extend RoPE dynamically to fit in VRA
        rotary_emb = self.rotary_emb
        rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)

        # if position_ids is None:
        #     # Useful for LongRoPE
        #     cos, sin = rotary_emb.get_cached(kv_seq_len, device = Q.device)
        # else:
        #     cos, sin = rotary_emb.get_cached(seq_len = kv_seq_len, device = Q.device)
        cos, sin = rotary_emb.get_cached(kv_seq_len, Q.device.index)

    # Q, K = (
    #     fast_rope_embedding(Q, K, cos, sin)
    #     if position_ids is None
    #     else inplace_rope_embedding(Q, K, cos, sin, position_ids)
    # )
    Q, K = fast_rope_embedding(Q, K, cos, sin)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (not HAS_FLASH_ATTENTION and HAS_XFORMERS and attention_mask is None):
        # Xformers memory efficient attention
        # Also has Flash Attention v2 dispatching
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Group query attention
        if n_groups != 1:
            K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
            V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
            K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            if hidden_states.requires_grad:
                K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)
        pass
        A = xformers_attention(Q, K, V, attn_bias = causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        A = flash_attn_func(Q, K, V, causal = True)
    else:
        # when qlen==vlen and attn_mask is None, we should use causal attention
        Q_len = Q.shape[-2]
        K_len = K.shape[-2]
        if attention_mask is None and Q_len == K_len:
            is_causal = True
        else:
            is_causal = False
        # Grouped query attention
        if SDPA_HAS_GQA:
            # Needs (batch_size, n_heads, seq_len, head_dim)
            # is_casual and attention_mask must not be both set!
            A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = is_causal, enable_gqa = n_groups != 1)
            # Go back to (batch_size, seq_len, n_heads, head_dim)
            A = A.transpose(1, 2)#.contiguous()
        else:
            if n_groups != 1:
                K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
                V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
            pass
            # Must be contiguous or else results are False!
            # https://github.com/pytorch/pytorch/issues/112577
            Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
            # Needs (batch_size, n_heads, seq_len, head_dim)
            # is_casual and attention_mask must not be both set!
            A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = is_causal)
            # Go back to (batch_size, seq_len, n_heads, head_dim)
            A = A.transpose(1, 2).contiguous()
        pass
    pass
    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def LlamaDecoderLayer_fast_forward(
    self,
    hidden_states:       torch.Tensor,
    causal_mask          = None,
    attention_mask:      Optional[torch.Tensor] = None,
    position_ids:        Optional[torch.LongTensor] = None,
    past_key_value:      Optional[Tuple[torch.Tensor]] = None,
    output_attentions:   Optional[bool] = False,
    use_cache:           Optional[bool] = False,
    padding_mask:        Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if use_cache and hasattr(self, "_flag_for_generation"):
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states       = hidden_states,
            causal_mask         = causal_mask,
            attention_mask      = attention_mask,
            position_ids        = position_ids,
            past_key_value      = past_key_value,
            output_attentions   = output_attentions,
            use_cache           = use_cache,
            padding_mask        = padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states = fast_swiglu_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states       = hidden_states,
            causal_mask         = causal_mask,
            attention_mask      = attention_mask,
            position_ids        = position_ids,
            past_key_value      = past_key_value,
            output_attentions   = output_attentions,
            use_cache           = use_cache,
            padding_mask        = padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if use_cache: outputs += (present_key_value,)
    return outputs
pass

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward(
    self,
    input_ids:            torch.LongTensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_values:      Optional[List[torch.FloatTensor]] = None,
    inputs_embeds:        Optional[torch.FloatTensor] = None,
    use_cache:            Optional[bool] = None,
    output_attentions:    Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict:          Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    assert(output_attentions is False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Unsloth: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("Unsloth: You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length

    # Fix out of bounds tokenization
    if hasattr(self, "max_seq_length"):
        if seq_length > self.max_seq_length:
            shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
            logger.warning_once(
                f"Unsloth: Input IDs of shape {shape} with length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"\
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:,:self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:,:self.max_seq_length,:]
        pass
    pass

    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    pass

    # We already handle KV cache position_ids ourselves.
    if False:#(past_key_values_length != 0):
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype  = torch.int32,
            device = f"{DEVICE_TYPE}:0",
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)#.long()
    else:
        position_ids = None
    pass

    if position_ids is not None:
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.repeat((batch_size, 1))
    pass

    # Embed positions
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    inputs_embeds = inputs_embeds.to(_get_dtype(dtype_from_config(self.config)))

    # Normalized from Gemma
    IS_GEMMA   = self.config.model_type.startswith("gemma")
    IS_GEMMA2  = self.config.model_type.startswith("gemma2")
    IS_COHERE  = self.config.model_type.startswith("cohere")
    IS_GRANITE = self.config.model_type.startswith("granite")
    IS_FALCON_H1 = self.config.model_type.startswith("falcon_h1")

    train_embed_tokens = self.embed_tokens.weight.requires_grad

    if IS_GEMMA:
        # Match Gemma exactly by casting to bfloat16 / float16
        # inputs_embeds *= math_sqrt(self.config.hidden_size)
        # Ie 3072**0.5 = 55.5000 in bfloat16, whilst 55.4256 in float32
        # &  2048**0.5 = 45.2500 in bfloat16, whilst 45.2548 in float32
        normalizer = torch.tensor(math_sqrt(self.config.hidden_size), dtype = inputs_embeds.dtype)

        if train_embed_tokens:
            # Careful we must not do an inplace op!
            inputs_embeds = inputs_embeds * normalizer
        else:
            inputs_requires_grad = inputs_embeds.requires_grad
            if not inputs_embeds.is_leaf:
                inputs_embeds = inputs_embeds.detach()
                inputs_requires_grad = True
            elif inputs_requires_grad:
                inputs_embeds.requires_grad_(False)
            pass
            inputs_embeds *= normalizer
            # inputs_embeds *= math_sqrt(self.config.hidden_size)
            if inputs_requires_grad: inputs_embeds.requires_grad_(True)
        pass
    pass

    # Fix up attention mask by setting elements to 0
    # Specifically for DPO
    if getattr(self, "_has_no_labels", False) is True and (attention_mask is not None) and (past_key_values is None) and \
        (not train_embed_tokens) and self.training:
        # Careful for inference the attention_mask is size (1, kv_seq_len)
        # Whilst the input_embeds is size (1, 1, 4096)
        inputs_requires_grad = inputs_embeds.requires_grad
        if not inputs_embeds.is_leaf:
            inputs_embeds = inputs_embeds.detach()
            inputs_requires_grad = True
        elif inputs_requires_grad:
            inputs_embeds.requires_grad_(False)
        pass
        attention_mask = attention_mask[:,:self.max_seq_length] # Must resize!
        inputs_embeds *= attention_mask.unsqueeze(0).transpose(0, 1).transpose(1, 2)
        if inputs_requires_grad: inputs_embeds.requires_grad_(True)
    pass

    # Ignore attention_mask
    if attention_mask is None:
        padding_mask = None
    elif self.training:
        attention_mask = None
        padding_mask = None
    else:
        # if 0 in attention_mask:
        #     padding_mask = attention_mask
        # else:
        padding_mask = None

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window = getattr(self.config, "sliding_window", None),
        )
        # Must NOT convert to bool - weirdly this causes stuff to error out!
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(torch.bool)
    pass

    hidden_states = inputs_embeds
    if IS_GRANITE or IS_FALCON_H1: #granite has embedding multiplier
        hidden_states = self.config.embedding_multiplier * hidden_states

    if past_key_values is None and self.training:
        use_cache = False
        # if use_cache:
        #     logger.warning_once(
        #         "Unsloth: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`"
        #     )
        #     use_cache = False
    pass

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # Gradient checkpointing methods (ie sqrt)
    if hasattr(self, "_gradient_checkpointing_boundaries"):
        boundaries = self._gradient_checkpointing_boundaries
    else:
        boundaries = None
    pass

    # Check checkpointing method
    gradient_checkpointing = False

    if (self.gradient_checkpointing and self.training and not use_cache):
        gradient_checkpointing = True
    pass

    # Gemma2 has alternating SWA and global attn
    use_static_mask  = True
    dynamic_SWA_mask = None
    dynamic_GA_mask  = None
    if IS_GEMMA2:
        if HAS_FLASH_ATTENTION_SOFTCAPPING and attention_mask is None:
            self.SWA_mask = True
            self.GA_mask  = False
        elif attention_mask is not None:
            # Fixes https://github.com/unslothai/unsloth/issues/853
            # Unsloth needs a 2D mask, not a [2, 1, n, n] mask!

            # https://github.com/pytorch/pytorch/issues/103749
            # Need to convert to float and not using bool
            # attention_mask = (1.0 - attention_mask.float()) * torch.finfo(inputs_embeds.dtype).min
            dynamic_SWA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = self.config.sliding_window,
            )
            dynamic_GA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = None,
            )
            use_static_mask = False

        elif not hasattr(self, "SWA_mask"):
            if HAS_FLEX_ATTENTION:
                # Use Flex Attention instead!
                self.SWA_mask = create_flex_attention_sliding_window_mask(self.max_seq_length, self.config.sliding_window)
                self.GA_mask  = create_flex_attention_causal_mask(self.max_seq_length)
            else:
                n = self.max_seq_length # self.config.max_position_embeddings
                # masked_fill is making stuff slower!
                # self. GA_mask = create_boolean_mask(n = n, sliding_window = 0)
                # self.SWA_mask = create_boolean_mask(n = n, sliding_window = self.config.sliding_window)
                from transformers.modeling_attn_mask_utils import AttentionMaskConverter
                self.SWA_mask = AttentionMaskConverter(
                    is_causal = True,
                    sliding_window = self.config.sliding_window,
                )\
                    .to_causal_4d(1, n, n, dtype = inputs_embeds.dtype, device = DEVICE_TYPE,)\
                    .squeeze(0).squeeze(0)

                self.GA_mask = AttentionMaskConverter(
                    is_causal = True,
                )\
                    .to_causal_4d(1, n, n, dtype = inputs_embeds.dtype, device = DEVICE_TYPE,)\
                    .squeeze(0).squeeze(0)
            pass
        pass
    pass

    if (IS_ATTENTION_REFACTOR and (hasattr(self, "rotary_emb") or not hasattr(self.layers[0].self_attn, "rotary_emb"))) or IS_GRANITE:
        # Transformers main has made it mandatory to pass position_embeddings
        # https://github.com/huggingface/transformers/pull/34858
        # Also, transformers 4.45.0 supports granite but with the attention refactor (it always had the refactor)
        # unsloth's check for granite too has "version >= 4.45.0 (rightly so)".
        # so let granite always use the attention refactor implementation.
        position_embeddings = self.rotary_emb.get_cached(self.config.max_position_embeddings, hidden_states.device.index)
    else:
        position_embeddings = None

    # Go through every layer!
    for idx, decoder_layer in enumerate(self.layers):

        if output_hidden_states: all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        mask = causal_mask
        if IS_GEMMA2:
            if (idx % 2 == 0):
                mask = self.SWA_mask if use_static_mask else dynamic_SWA_mask
            else:
                mask = self. GA_mask if use_static_mask else dynamic_GA_mask
        pass

        if gradient_checkpointing and not isinstance(decoder_layer, GradientCheckpointingLayer):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions, padding_mask = padding_mask, position_embeddings = position_embeddings)
                return custom_forward
            pass
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                mask,
                attention_mask,
                position_ids,
                use_reentrant = True,
                preserve_rng_state = False,
            )
            hidden_states = layer_outputs[0]

        else:
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask         = mask,
                attention_mask      = attention_mask,
                position_ids        = position_ids,
                past_key_value      = past_key_value,
                output_attentions   = output_attentions,
                use_cache           = use_cache,
                padding_mask        = padding_mask,
                position_embeddings = position_embeddings,
            )
            hidden_states = layer_outputs[0]
        pass

        if use_cache: next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions: all_self_attns += (layer_outputs[1],)
    pass

    # Final layernorm
    if use_cache:
        if IS_FALCON_H1:
            hidden_states = fast_rms_layernorm_inference(self.final_layernorm, hidden_states)
        else:
            hidden_states = \
                (fast_rms_layernorm_inference_gemma if IS_GEMMA else fast_rms_layernorm_inference)\
                (self.norm, hidden_states)
    elif IS_COHERE:
        hidden_states = self.norm(hidden_states)
    elif IS_FALCON_H1:
        hidden_states = fast_rms_layernorm(self.final_layernorm, hidden_states, gemma = IS_GEMMA)
    else:
        hidden_states = fast_rms_layernorm(self.norm, hidden_states, gemma = IS_GEMMA)
    pass

    if output_hidden_states: all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
pass

# Solves https://github.com/unslothai/unsloth/issues/168
# Static KV Cache was introduced in 4.38.0, causing training to be much slower.
# Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
# https://github.com/huggingface/transformers/pull/27931
# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(torch.nn.Module):
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            dim = getattr(config, "head_dim", None)
            if dim is None: dim = int((config.hidden_size // config.num_attention_heads))
            device = DEVICE_TYPE
            max_position_embeddings = config.max_position_embeddings
        pass

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cos_cached = [None]*DEVICE_COUNT
        self.multi_gpu_sin_cached = [None]*DEVICE_COUNT

        # Build here to make `torch.jit.trace` work.
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(seq_len=self.current_rope_size, device=torch.device(device_idx), dtype=torch.get_default_dtype())

        # dummy so that patch_utils doesn't fail for now
        self.cos_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
        self.sin_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.current_rope_size, device="cpu", dtype=torch.int64).float()

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin
    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        device_index = x.device.index
        return (
            self.multi_gpu_cos_cached[device_index][:seq_len],
            self.multi_gpu_sin_cached[device_index][:seq_len],
        )
    pass

    def get_cached(self, seq_len = None, device_index = None):
        if device_index is None:
            device_index = get_current_device()
        return self.multi_gpu_cos_cached[device_index], self.multi_gpu_sin_cached[device_index]
    pass

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size: return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(self.current_rope_size, device = torch.device(device_idx), dtype = x.dtype)
    pass
pass


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim = dim, max_position_embeddings = max_position_embeddings, base = base, device = device, config = config)
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.current_rope_size, device="cpu", dtype=torch.int64).float()
        t = t / self.scaling_factor

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin
    pass
pass
