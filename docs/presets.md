# Presets

KohakuBench ships with a suite of presets that serve as ready-made bench cases and usage examples. All presets return a `BenchmarkCase` that you can pass to `KoBench.benchmark`.

## Attention
`kohakubench.presets.attention.multi_head_attention_case(batch_size=4, seq_len=1024, embed_dim=1024, num_heads=16, device=None, dtype=None)`
- Uses Scaled-Dot Product Attention implementation under the hood.
- Metadata: batch_size, seq_len, embed_dim, num_heads
- Counters: MACs for QKV/proj and matmuls; FLOPs for scale/softmax

## MLP
`kohakubench.presets.mlp.standard_mlp_case(batch_size, seq_len, embed_dim, hidden_dim, activation='gelu', device=None, dtype=None)`
`kohakubench.presets.mlp.swiglu_mlp_case(batch_size, seq_len, embed_dim, hidden_dim, device=None, dtype=None)`
- Metadata: batch_size, seq_len, embed_dim, hidden_dim, activation
- Counters: linear MACs and activation FLOPs

## Convolution
`kohakubench.presets.convolution.conv2d_case(batch_size, in_channels, out_channels, image_size, kernel_size=3, stride=1, padding=None, device=None, dtype=None)`
`kohakubench.presets.convolution.depthwise_conv2d_case(batch_size, channels, image_size, kernel_size=3, stride=1, padding=None, pointwise=True, device=None, dtype=None)`
- Metadata: batch_size, channels, out_dim, kernel_size, pointwise
- Counters: conv2d MACs and elementwise FLOPs

## Norms
`kohakubench.presets.norms.layer_norm_case(batch_size, seq_len, embed_dim, device=None, dtype=None)`
`kohakubench.presets.norms.group_norm_case(batch_size, channels, height, width, groups=32, device=None, dtype=None)`
`kohakubench.presets.norms.rms_norm_case(batch_size, seq_len, embed_dim, eps=1e-6, device=None, dtype=None)`
- Counters: elementwise FLOPs per element

## Transformer Block
`kohakubench.presets.transformer.transformer_block_case(batch_size, seq_len, embed_dim, num_heads, mlp_ratio=4, device=None, dtype=None)`
- Uses SDP attention + MLP (GELU)
- Counters: QKV/proj/MLP MACs, softmax/scale/activation FLOPs

## Stable Diffusion ResBlock
`kohakubench.presets.stable_diffusion.stable_diffusion_resblock_case(batch_size, channels, height, width, groups=32, device=None, dtype=None)`
- GroupNorm + SiLU + Conv stack with residual
- Counters: conv MACs, norm + activation FLOPs
