# Evaluation Commands

```bash
# run the evaluation in cuda=0 device, the results file is defined in the run.sh file
./run.sh 0
```

# Evaluation Results

## 1. Wikitext103-1024

__roberta-large + without length truncation + prefix__
__roberta-large + without length truncation + no prefix__
__roberta-large + without length truncation + prefix + reference >= 100__

| Models | MAUVE (gpt2-large,c=1.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence | MAUVE (roberta-large, c=1.0, EOS token) | BLEURT (bleurt-large-512) |
| - | - | - | - | - | - | - | - | - |
| gpt2 (greedy search)  | 70.53 | 0.4356 | 0.3855 | 0.355 | 0.2237 | -0.74 | 60.63| |
| gpt2 (nucleus sampling p=0.95)  | 80.53 | 0.051 | 0.0133 | 0.005 | 0.9322 | -3.65 | 66.06 | |
| neurlab gpt2 (greedy search)  | 75.25 | 0.346 | 0.2795 | 0.2398 | 0.3583 | -1.34 | 60.17 | |
| neurlab gpt2 (nucleus sampling p=0.95)  | 81.39 | 0.0544 | 0.0149 | 0.005 | 0.9268 | -3.72 | 64.37 | |
| knnlm (greedy search full) | 80.32 | 0.0681 | 0.0213 | 0.0111 | 0.9019 | -4.00 |64.65 | |
| knnlm (nucleus sampling p=0.95 full) | 81.02 | 0.0332 | 0.0062 | 0.0017 | 0.9592 | -4.65 | 62.11 | |
| retro (greedy search) | 64.93 | 0.4465 | 0.3963 | 0.366 | 0.2119 | -0.74 |59.62 | |
| retro (nucleus sampling p=0.95) | 71.98 | 0.0621 | 0.0193 | 0.0086 | 0.9119 | -3.63 | 67.13 | |
| copyisallyouneed (greedy search) | 76.04 | 0.2814 | 0.238 | 0.214 | 0.4303 | -1.73 |74.54| |
| copyisallyouneed (nucleus sampling p=0.95) | 79.54 | 0.0731 | 0.0266 | 0.0128 | 0.8907 | -2.91 | 78.96| |


## 2. LawMT

| Models | MAUVE (gpt2-large,c=1.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence | MAUVE (roberta-large,c=1.0) |
| - | - | - | - | - | - | - | - |
| gpt2 (greedy search)  | 53.43 | 0.1433 | 0.1017 | 0.0818 | 0.7066 |-0.64 | 54.01|
| gpt2 (nucleus sampling p=0.95)  | 53.42 | 0.0395 | 0.0149 | 0.0078 | 0.9388 | -3.61 | 52.97|
| neurlab gpt2 (greedy search)  | 55.15  | 0.1375 | 0.0954 | 0.0726 | 0.7236 | -1.20 | 52.90 |
| neurlab gpt2 (nucleus sampling p=0.95)  | 54.59 | 0.0367 | 0.0128 | 0.0057 | 0.9456 | -3.92 | 53.22|
| knnlm (greedy search full) | 53.67 | 0.04 | 0.015 | 0.0085 | 0.9376 | -4.64 |53.41 |
| knnlm (nucleus sampling p=0.95 full) | 53.39 | 0.0283 | 0.0097 | 0.0047 | 0.9577 | -5.05 | 53.18|
| retro (greedy search) | 52.85 | 0.139 | 0.0996 | 0.0823 | 0.7114 | -0.67 | 51.73|
| retro (nucleus sampling p=0.95) | 52.60 | 0.0347 | 0.0123 | 0.0057 | 0.948 | -4.18 | 52.74 |
| copyisallyouneed (greedy search) | 53.25 | 0.0806 | 0.0482 | 0.0365 | 0.8432 | -1.51 | 56.66|
| copyisallyouneed (nucleus sampling p=0.95) | 53.39 | 0.0501 | 0.0176 | 0.0081 | 0.9256 | -2.78 | 54.32|

## 3. EN-Wiki

| Models | MAUVE (gpt2-large,c=1.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence | MAUVE (roberta,c=1.0) |
| - | - | - | - | - | - | - | - |
| gpt2 (greedy search)  | 79.62 | 0.5075 | 0.4601 | 0.4314 | 0.1512 | -0.81 | 67.62|
| gpt2 (nucleus sampling p=0.95)  | 93.82 | 0.0616 | 0.0154 | 0.0061 | 0.9183 | -3.55 | 63.49|
| neurlab gpt2 (greedy search)  |  83.64 | 0.4254 | 0.359 | 0.3197 | 0.2505 |  -1.29 | 61.19 |
| neurlab gpt2 (nucleus sampling p=0.95)  | 93.79 | 0.0602 | 0.0153 | 0.0053 | 0.9204 |-3.88| 71.03 |
| knnlm (greedy search full) | - | 0.0948 | 0.0367 | 0.0216 | 0.8531 | -3.85 |  68.36 |
| knnlm (nucleus sampling p=0.95 full) | - | 0.036 | 0.0053 | 0.0013 | 0.9576 | -4.68 | 67.08 |
| retro (greedy search) | 76.58 | 0.5493 | 0.5017 | 0.4714 | 0.1187 |  -0.80 | 66.55|
| retro (nucleus sampling p=0.95) | 93.34 | 0.0809 | 0.0275 | 0.0136 | 0.8817 | -3.54 |71.91 |
| copyisallyouneed (greedy search) | 87.81 | 0.3611 | 0.3063 | 0.2732 | 0.3221 | -1.65 | 92.29|
| copyisallyouneed (nucleus sampling p=0.95) |93.16 | 0.1476 | 0.077 | 0.0457 |0.7508 | -2.58 | 87.64|


