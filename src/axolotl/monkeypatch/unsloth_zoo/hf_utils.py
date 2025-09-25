from transformers import PretrainedConfig
HAS_TORCH_DTYPE = "torch_dtype" in PretrainedConfig.__doc__

def dtype_from_config(config):
    check_order = ['dtype', 'torch_dtype']
    if HAS_TORCH_DTYPE:
        check_order = ['torch_dtype', 'dtype']
    dtype = None
    for dtype_name in check_order:
        if dtype is None:
            dtype = getattr(config, dtype_name, None)
    return dtype