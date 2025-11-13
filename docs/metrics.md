# Metrics

## Terms
- MAC: multiply-accumulate operation
- MACs: total multiply-accumulate count
- MACS: multiply-accumulates per second
- FLOP: floating-point operation
- FLOPs: total floating-point operation count
- FLOPS: floating-point operations per second

## Built-in counters
- `matmul_macs(batch, m, n, k)`
- `linear_macs(batch_tokens, in_features, out_features)`
- `conv2d_macs(batch, in_channels, out_channels, out_h, out_w, kernel_h, kernel_w, groups=1)`
- `elementwise_flops(num_elements, ops_per_element=1.0)`
- `activation_flops(num_elements, approx_ops=4.0)`
- `total_ops(iterable)`

## Notes
- Counters are analytical and shape-driven; kernel fusion or library-specific optimizations may alter the real hardware path. Use counters as an upper-bound estimate.
- KoBench computes MACS/FLOPS rates from counts and measured latency whenever counters are provided.
