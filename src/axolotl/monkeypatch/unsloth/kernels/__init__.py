# from .cross_entropy_loss import (
#     fast_cross_entropy_loss,
#     post_patch_loss_function,
#     patch_loss_functions,
# )
from .rms_layernorm import (
    fast_rms_layernorm,
    patch_rms_layernorm,
    unpatch_rms_layernorm,
)
# from .layernorm import (
#     fast_layernorm,
#     patch_layernorm,
# )
from .rope_embedding import fast_rope_embedding, inplace_rope_embedding
# from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
from .geglu import (
    geglu_exact_forward_kernel,
    geglu_exact_backward_kernel,
    geglu_approx_forward_kernel,
    geglu_approx_backward_kernel,
)

from .utils import fast_dequantize, fast_gemv, QUANT_STATE, fast_linear_forward, matmul_lora

from .flex_attention import (
    HAS_FLEX_ATTENTION,
    slow_attention_softcapping,
    slow_inference_attention_softcapping,
    create_flex_attention_causal_mask,
    create_flex_attention_sliding_window_mask,
)