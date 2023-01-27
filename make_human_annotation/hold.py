import torch
num_gpus=8
vars = [torch.randn(42,32,32,32).to(f"cuda:{i}") for i in range(num_gpus)]

while True:
    for a in vars:
        a * a
