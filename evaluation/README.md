# Wikitext103

## 1. Wikitext103-1024

| Models | MAUVE (gpt2-large,c=1.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)  | 70.53 | 0.4356 | 0.3855 | 0.355 | 0.2237 | -0.74 |
| gpt2 (nucleus sampling p=0.95)  | 80.53 | 0.051 | 0.0133 | 0.005 | 0.9322 | -3.65 |
| neurlab gpt2 (greedy search)  | 75.25 | 0.346 | 0.2795 | 0.2398 | 0.3583 | -1.34 |
| neurlab gpt2 (nucleus sampling p=0.95)  | 81.39 | 0.0544 | 0.0149 | 0.005 | 0.9268 | -3.72 |
| knnlm (greedy search full) | 0.8032 | 0.0681 | 0.0213 | 0.0111 | 0.9019 | -4.00 |
| knnlm (nucleus sampling p=0.95 full) | 81.02 | 0.0332 | 0.0062 | 0.0017 | 0.9592 | -4.65 |
| retro (greedy search) | 64.93 | 0.4465 | 0.3963 | 0.366 | 0.2119 | -0.74 |
| retro (nucleus sampling p=0.95) | 71.98 | 0.0621 | 0.0193 | 0.0086 | 0.9119 | -3.63 |
| copyisallyouneed (greedy search) | 76.04 | 0.2814 | 0.238 | 0.214 | 0.4303 | -1.73 |
| copyisallyouneed (nucleus sampling p=0.95) | 79.54 | 0.0731 | 0.0266 | 0.0128 | 0.8907 | -2.91 |
