import torch
from typing import List, Optional, Tuple
from quick.awq.modules.linear.quick import WQLinear_QUICK
from quick.awq.modules.linear.gemm import WQLinear_GEMM
from quick.awq.modules.linear.gemv import WQLinear_GEMV
from quick.awq.modules.linear.exllama import WQLinear_Exllama
from quick.awq.modules.linear.exllamav2 import WQLinear_ExllamaV2

def prepare_correct_devices(next_layer, hidden_states, mask):
    hidden_states = hidden_states.to(next_layer.device)

    if mask is not None:
        mask = mask.to(next_layer.device)

    return hidden_states, mask
    
def prepare_cache(blocks, seqlen: int) -> int:
    for block in blocks:
        start_pos = block.attn.start_pos
        will_cache_be_exceeded = start_pos + seqlen > block.attn.max_seq_len

        # Reset and avoid retaining state when processing context
        if seqlen > 1 and (will_cache_be_exceeded or start_pos > 0):
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(start_pos, n=start_pos)
        
        # Slowly roll out old tokens without performance hit if exceeded during decoding 
        elif seqlen == 1 and will_cache_be_exceeded:
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(start_pos, n=100)

def prepare_input_ids(input_ids: torch.Tensor, last_forward_num_tokens: int):
    # NOTE: from transformers 4.35.0, input_ids includes full context during decoding
    num_input_tokens = input_ids.shape[-1]
    num_new_tokens = num_input_tokens

    if num_input_tokens != 1:
        num_new_tokens = num_input_tokens - last_forward_num_tokens
        
        # after context is processed, slice to latest token
        if num_new_tokens == 1:
            input_ids = input_ids[:, -1:]

    return input_ids, last_forward_num_tokens + num_new_tokens

def prepare_attention_mask(seqlen, start_pos, device, type_as: torch.Tensor):
    mask = None
    if seqlen > 1:
        mask = torch.full(
            (1, 1, seqlen, seqlen), float("-inf"), device=device
        )
        mask = torch.triu(mask, diagonal=start_pos+ 1).type_as(type_as)
    
    return mask

def fuse_qkv(module, q_proj, k_proj, v_proj):
    bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

    if isinstance(q_proj, WQLinear_GEMV):
        q_linear = WQLinear_GEMV
    elif isinstance(q_proj, WQLinear_GEMM):
        q_linear = WQLinear_GEMM
    elif isinstance(q_proj, WQLinear_Exllama):
        q_linear = WQLinear_Exllama
    else:
        q_linear = WQLinear_ExllamaV2

    qkv_layer = q_linear(
        q_proj.w_bit,
        q_proj.group_size,
        q_proj.in_features,
        q_proj.out_features + k_proj.out_features + v_proj.out_features,
        q_proj.bias is not None,
        next(iter(module.state_dict().values())).device
    )

    if isinstance(q_proj, WQLinear_GEMV):
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=0)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=0)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=0)
        qkv_layer.split_k_iters = q_proj.split_k_iters
    elif isinstance(q_proj, WQLinear_GEMM):
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
    elif isinstance(q_proj, WQLinear_Exllama):
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
    elif isinstance(q_proj, WQLinear_ExllamaV2):
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
    
    qkv_layer.bias = bias

    return qkv_layer

def fuse_qkv_quick(module, q_proj, k_proj, v_proj):
    q_linear = WQLinear_QUICK

    qkv_layer = q_linear(
        q_proj.w_bit,
        q_proj.group_size,
        q_proj.in_features,
        q_proj.out_features + k_proj.out_features + v_proj.out_features,
        q_proj.bias is not None,
        next(iter(module.state_dict().values())).device
    )
    
    bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

    qkv_layer.qweight = QUICK_cat(q_proj.qweight, k_proj.qweight, v_proj.qweight, options='qweight')
    qkv_layer.qzeros = QUICK_cat(q_proj.qzeros, k_proj.qzeros, v_proj.qzeros, options='qzeros')
    qkv_layer.scales = QUICK_cat(q_proj.scales, k_proj.scales, v_proj.scales, options='scales')
    
    qkv_layer.bias = bias

    return qkv_layer

def QUICK_cat(*input_layers: torch.Tensor, options: str, reshape_dims: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Concatenates multiple input layers after reshaping based on specified options for QUICK.

    Args:
        *input_layers: Variable number of tensor layers to concatenate.
        options: A string indicating how the layers should be reshaped ('qweight', 'qzeros', or 'scales').
        reshape_dims: Optional tuple indicating custom dimensions for reshaping. If None, default settings are used.

    Returns:
        torch.Tensor: The concatenated and reshaped layers.

    Raises:
        ValueError: If the options provided are invalid or if input layers have incompatible shapes.
    """
    # Check if there are at least two layers to concatenate
    if len(input_layers) < 2:
        raise ValueError("At least two input layers are required")

    # Check for shape compatibility
    H, W = input_layers[0].shape
    for layer in input_layers[1:]:
        if layer.shape != input_layers[0].shape:
            raise ValueError("All input layers must have the same shape")

    # Determine reshape dimensions based on options
    if not reshape_dims:
        reshape_dims = {
            'qweight': (H // 2, W * 2),
            'qzeros': (H * 4, W // 4),
            'scales': (H * 4, W // 4)
        }.get(options)

    if reshape_dims is None:
        raise ValueError("Unknown options provided or invalid reshape dimensions")

    # Reshape and concatenate the input layers
    layers_to_cat = [layer.reshape(*reshape_dims) for layer in input_layers]
    output_layer = torch.cat(layers_to_cat, dim=1).reshape(H, -1)

    return output_layer


def get_attention_shapes(attention_shapes, max_seq_len, cache_batch_size, n_heads, n_kv_heads, head_dim):
    if attention_shapes is not None:
        attention_shapes = attention_shapes

    elif n_kv_heads == 0:
        attention_shapes = {
            # following fastertransformer definition
            "cache_v": (cache_batch_size, n_heads, max_seq_len, head_dim,),
            # 8: pack 8 fp16 in FT, if fp32 then use 4
            "cache_k": (cache_batch_size, n_heads, head_dim // 8, max_seq_len, 8,),
            "xqkv_view": (-1, n_heads, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0],
            "xk_slice": lambda xqkv: xqkv[:, :, 1],
            "xv_slice": lambda xqkv: xqkv[:, :, 2],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_heads, head_dim),
            "xv_view": (n_heads, head_dim),
            "xk_reshape": (n_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_heads, head_dim),
            "single_xv_view": (n_heads, head_dim)
        }

    else:
        attention_shapes = {
            # following fastertransformer definition
            "cache_v": (cache_batch_size, n_kv_heads, max_seq_len, head_dim,),
            # 8: pack 8 fp16 in FT, if fp32 then use 4
            "cache_k": (cache_batch_size, n_kv_heads, head_dim // 8, max_seq_len, 8,),
            "xqkv_view": (n_heads + n_kv_heads * 2, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0 : n_heads],
            "xk_slice": lambda xqkv: xqkv[:, :, n_heads : (n_heads + n_kv_heads)],
            "xv_slice": lambda xqkv: xqkv[:, :, -n_kv_heads :],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_kv_heads, head_dim),
            "xv_view": (n_kv_heads, head_dim),
            "xk_reshape": (n_kv_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_kv_heads, head_dim),
            "single_xv_view": (n_kv_heads, head_dim)
        }
    
    return attention_shapes