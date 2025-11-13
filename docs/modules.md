# Modules

## ScaledDotProductAttention
A minimal, fast attention block built on `torch.nn.functional.scaled_dot_product_attention`.

### Signature
```python
ScaledDotProductAttention(embed_dim: int, num_heads: int, bias: bool = False)
```

### Input/Output
- Input: `(batch, seq_len, embed_dim)`
- Output: `(batch, seq_len, embed_dim)`

### Example
```python
import torch
from kohakubench.modules import ScaledDotProductAttention

attn = ScaledDotProductAttention(1024, 16).cuda().eval()
x = torch.randn(2, 1024, 1024, device='cuda')
y = attn(x)
```
