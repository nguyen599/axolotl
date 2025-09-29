"""Monkeypatch for Qwen3_Next model to pass position_ids to linear attention."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# For Pytorch 2.1.1
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3SdpaAttention,
        Qwen3FlashAttention2,
    )
except:
    Qwen3SdpaAttention   = Qwen3Attention
    Qwen3FlashAttention2 = Qwen3Attention
pass
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def get_cu_seqlens(position_ids):
    """
    Adapted from transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids.

    https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/modeling_flash_attention_utils.py#L316
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )

    return cu_seq_lens_q


def patch_qwen3_next_decoder_layer():
    """Patch Qwen3NextDecoderLayer to pass position_ids to linear attention."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDecoderLayer,
        )
    except ImportError:
        LOG.warning("Qwen3Next model not found, skipping patch")
        return

    # Store original forward method
    original_decoder_forward = Qwen3NextDecoderLayer.forward

    def patched_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, Tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states

    # Apply the patches
    Qwen3NextDecoderLayer.forward = patched_decoder_forward

    def unpatch():
        """Restore the original forward method"""
        Qwen3NextDecoderLayer.forward = original_decoder_forward

    return unpatch


def patch_qwen3_next_gateddelta_layer():
    """Patch Qwen3NextGatedDeltaNet to parse cu_seqlens and pass to chunk_gated_delta_rule"""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDynamicCache,
            Qwen3NextGatedDeltaNet,
            apply_mask_to_padding_states,
        )
    except ImportError:
        LOG.warning("Qwen3Next model not found, skipping patch")
        return

    # Store original forward method
    original_gated_delta_net_forward = Qwen3NextGatedDeltaNet.forward

    def patched_gated_delta_net_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Qwen3NextDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (
            x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value)
        )

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            cu_seqlens = get_cu_seqlens(position_ids=position_ids)
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        output = self.out_proj(core_attn_out)
        return output

    # Apply the patches
    Qwen3NextGatedDeltaNet.forward = patched_gated_delta_net_forward

    def unpatch():
        """Restore the original forward method"""
        Qwen3NextGatedDeltaNet.forward = original_gated_delta_net_forward

    return unpatch


def patch_qwen3_next_imports():
    """Patch Qwen3Next imports to use try/except instead of is_flash_linear_attention_available."""
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next as qwen3_modeling
    except ImportError:
        LOG.warning("Qwen3Next model not found, skipping import patch")
        return

    # Save original values for unpatch
    original_FusedRMSNormGated = getattr(qwen3_modeling, "FusedRMSNormGated", None)
    original_chunk_gated_delta_rule = getattr(
        qwen3_modeling, "chunk_gated_delta_rule", None
    )
    original_fused_recurrent_gated_delta_rule = getattr(
        qwen3_modeling, "fused_recurrent_gated_delta_rule", None
    )
    original_is_fast_path_available = getattr(
        qwen3_modeling, "is_fast_path_available", False
    )

    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        qwen3_modeling.FusedRMSNormGated = FusedRMSNormGated
        qwen3_modeling.chunk_gated_delta_rule = chunk_gated_delta_rule
        qwen3_modeling.fused_recurrent_gated_delta_rule = (
            fused_recurrent_gated_delta_rule
        )

        # Force is_fast_path_available to be True
        # fla has triton kernels for causal_conv1d
        qwen3_modeling.is_fast_path_available = True
    except ImportError:
        qwen3_modeling.chunk_gated_delta_rule = None
        qwen3_modeling.fused_recurrent_gated_delta_rule = None
        qwen3_modeling.FusedRMSNormGated = None

    def unpatch():
        """Restore the original import values"""
        qwen3_modeling.FusedRMSNormGated = original_FusedRMSNormGated
        qwen3_modeling.chunk_gated_delta_rule = original_chunk_gated_delta_rule
        qwen3_modeling.fused_recurrent_gated_delta_rule = (
            original_fused_recurrent_gated_delta_rule
        )
        qwen3_modeling.is_fast_path_available = original_is_fast_path_available

    return unpatch

def from_pretrained():
    pass

# def new_init(self, config):
#     super().__init__(config)
#     self.model = Qwen3Model(config)
#     self.vocab_size = config.vocab_size
#     self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#     # Initialize weights and apply final processing
#     self.post_init()
#     import deepspeed
#     deepspeed.zero.register_external_parameter(self, self.lm_head.weight)


# class Qwen3ForCausalLMWithDS(Qwen3ForCausalLM):
#     def __init__(self, config):
#         super().__init__(config)
#         import deepspeed
#         deepspeed.zero.register_external_parameter(self, self.lm_head.weight)
#         print('heeeeeeeeeeeeeeexxxxxxxxxxx')

# from typing import Callable, Optional, Union
# from transformers.cache_utils import Cache, DynamicCache
# from transformers.generation import GenerationMixin
# from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
# from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# from transformers.processing_utils import Unpack
# from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
# import deepspeed

# class Qwen3ForCausalLMWithDS(Qwen3PreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]
#     _tp_plan = {"lm_head": "colwise_rep"}
#     _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = Qwen3Model(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()
#         deepspeed.zero.register_external_parameter(self, self.lm_head.weight)
#         print('heeeeeeeeeeeeeeexxxxxxxxxxx')

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         **kwargs: Unpack[TransformersKwargs],
#     ) -> CausalLMOutputWithPast:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

#         >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
#         >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         outputs: BaseModelOutputWithPast = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             **kwargs,
#         )

#         hidden_states = outputs.last_hidden_state
#         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
        # )

def patch_qwen3_model():
    from axolotl.monkeypatch.unsloth.models._utils import patch_linear_scaling
    from axolotl.monkeypatch.unsloth.models.llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaModel_fast_forward, LlamaDecoderLayer_fast_forward, fix_prepare_inputs_for_generation
    from axolotl.monkeypatch.unsloth.models.qwen3 import Qwen3Attention_fast_forward
    init_name, function = patch_linear_scaling(
        model_name         = "Qwen3",
        rope_module        = LlamaRotaryEmbedding,
        scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
        attention_module   = Qwen3Attention,
    )
    if init_name is not None:
        exec(function, globals())
        Qwen3Attention.__init__  = eval(init_name)
    pass
    Qwen3Attention      .forward = Qwen3Attention_fast_forward
    Qwen3SdpaAttention  .forward = Qwen3Attention_fast_forward
    Qwen3FlashAttention2.forward = Qwen3Attention_fast_forward
    Qwen3DecoderLayer   .forward = LlamaDecoderLayer_fast_forward
    Qwen3Model          .forward = LlamaModel_fast_forward
    # Qwen3ForCausalLM    .forward = CausalLM_fast_forward(_LlamaModel_fast_forward_inference(Qwen3Attention_fast_forward_inference))
    # PeftModelForCausalLM.forward = PeftModel_fast_forward
    fix_prepare_inputs_for_generation(Qwen3ForCausalLM)

    # Solves https://github.com/unslothai/unsloth/issues/168
    # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
    # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
    # https://github.com/huggingface/transformers/pull/27931
    # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
    import transformers.models.qwen3.modeling_qwen3
    transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding = LlamaRotaryEmbedding
    # modeling_qwen3 = sys.modules["transformers.models.qwen3.modeling_qwen3"]
    # modeling_qwen3.Qwen3RotaryEmbedding = LlamaRotaryEmbedding

def patch_qwen3_modeling():
    """Apply all Qwen3 model patches."""
    # patch_qwen3_next_imports()
    # patch_qwen3_next_decoder_layer()
    patch_qwen3_model()
    # import transformers.models.qwen3.modeling_qwen3
    # transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM = Qwen3ForCausalLMWithDS

    LOG.info("Applied Qwen3 patch for Unsloth fast forward")