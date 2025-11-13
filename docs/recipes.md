# Recipes

This section collects practical patterns for using KoBench.

## Programmatic presets

```python
from kohakubench.core import KoBench
from kohakubench.presets import attention, mlp
from kohakubench.report import print_table

bench = KoBench(device='cuda:0', dtype='float16')
cases = [
    attention.multi_head_attention_case(batch_size=2, seq_len=1024, embed_dim=1024, num_heads=16,
                                        device=bench.device, dtype=bench.dtype),
    mlp.swiglu_mlp_case(batch_size=2, seq_len=512, embed_dim=2048, hidden_dim=8192,
                        device=bench.device, dtype=bench.dtype),
]
print_table(bench.benchmark_many(cases))
```

## Custom counters

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench()
conv = torch.nn.Conv2d(128, 256, 3, padding=1, bias=False).to(bench.device, bench.dtype).eval()
x = torch.randn(4, 128, 64, 64, device=bench.device, dtype=bench.dtype)

meta = {'batch': 4, 'in_ch': 128, 'out_ch': 256, 'h': 64, 'w': 64, 'k': 3}

def mac_counter(m):
    return m['batch'] * m['out_ch'] * m['h'] * m['w'] * (m['k']*m['k']*m['in_ch'])

def flop_counter(m):
    return m['batch'] * m['out_ch'] * m['h'] * m['w']  # approximate elementwise ops

case = BenchmarkCase('conv2d', conv, args=(x,), metadata=meta,
                     mac_counter=mac_counter, flop_counter=flop_counter)
print(bench.benchmark(case).to_dict())
```

## Backward (training) steps

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(inference_mode=False)
model = torch.nn.Linear(1024, 1024).to(bench.device, bench.dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)
y = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)

@torch.enable_grad()
def step(x, y):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    return loss

print(bench.benchmark(BenchmarkCase('train_step', step, args=(x, y))).to_dict())
```

## Exporting results

```python
results = bench.benchmark_many(cases)
rows = [r.to_dict() for r in results]
# Write rows to CSV/JSON for dashboards
```
