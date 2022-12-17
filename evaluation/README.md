# Evaluation Commands

```bash
# run the evaluation in cuda=0 device, the results file is defined in the run.sh file
./run.sh 0
```

# Evaluation Results

## 1. Wikitext103-1024

| Models | MAUVE (gpt2-large,c=1.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)  | 70.53 | 0.4356 | 0.3855 | 0.355 | 0.2237 | -0.74 |
| gpt2 (nucleus sampling p=0.95)  | 80.53 | 0.051 | 0.0133 | 0.005 | 0.9322 | -3.65 |
| neurlab gpt2 (greedy search)  | 75.25 | 0.346 | 0.2795 | 0.2398 | 0.3583 | -1.34 |
| neurlab gpt2 (nucleus sampling p=0.95)  | 81.39 | 0.0544 | 0.0149 | 0.005 | 0.9268 | -3.72 |
| knnlm (greedy search full) | 80.32 | 0.0681 | 0.0213 | 0.0111 | 0.9019 | -4.00 |
| knnlm (nucleus sampling p=0.95 full) | 81.02 | 0.0332 | 0.0062 | 0.0017 | 0.9592 | -4.65 |
| retro (greedy search) | 64.93 | 0.4465 | 0.3963 | 0.366 | 0.2119 | -0.74 |
| retro (nucleus sampling p=0.95) | 71.98 | 0.0621 | 0.0193 | 0.0086 | 0.9119 | -3.63 |
| copyisallyouneed (greedy search) | 76.04 | 0.2814 | 0.238 | 0.214 | 0.4303 | -1.73 |
| copyisallyouneed (nucleus sampling p=0.95) | 79.54 | 0.0731 | 0.0266 | 0.0128 | 0.8907 | -2.91 |

## 2. LawMT

| Models | MAUVE (gpt2-large,c=1.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)  | 53.43 | 0.1433 | 0.1017 | 0.0818 | 0.7066 |-0.64 |
| gpt2 (nucleus sampling p=0.95)  | 53.42 | 0.0395 | 0.0149 | 0.0078 | 0.9388 | -3.61 |
| neurlab gpt2 (greedy search)  | 55.15  | 0.1375 | 0.0954 | 0.0726 | 0.7236 | -1.20 |
| neurlab gpt2 (nucleus sampling p=0.95)  | 54.59 | 0.0367 | 0.0128 | 0.0057 | 0.9456 | -3.92 |
| knnlm (greedy search full) | 53.67 | 0.04 | 0.015 | 0.0085 | 0.9376 | -4.64 |
| knnlm (nucleus sampling p=0.95 full) | 53.39 | 0.0283 | 0.0097 | 0.0047 | 0.9577 | -5.05 |
| retro (greedy search) | 52.85 | 0.139 | 0.0996 | 0.0823 | 0.7114 | -0.67 |
| retro (nucleus sampling p=0.95) | 52.60 | 0.0347 | 0.0123 | 0.0057 | 0.948 | -4.18 |
| copyisallyouneed (greedy search) | 53.25 | 0.0806 | 0.0482 | 0.0365 | 0.8432 | -1.51 |
| copyisallyouneed (nucleus sampling p=0.95) | 53.39 | 0.0501 | 0.0176 | 0.0081 | 0.9256 | -2.78 |

## 3. EN-Wiki

| Models | MAUVE (gpt2-large,c=1.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)  | 79.62 | 0.5075 | 0.4601 | 0.4314 | 0.1512 | -0.81 |
| gpt2 (nucleus sampling p=0.95)  | 93.82 | 0.0616 | 0.0154 | 0.0061 | 0.9183 | -3.55 |
| neurlab gpt2 (greedy search)  |  83.64 | 0.4254 | 0.359 | 0.3197 | 0.2505 |  -1.29 |
| neurlab gpt2 (nucleus sampling p=0.95)  | 93.79 | 0.0602 | 0.0153 | 0.0053 | 0.9204 |-3.88|
| knnlm (greedy search full) |  | |  |  |  |  |
| knnlm (nucleus sampling p=0.95 full) |  |  |  |  |  |  |
| retro (greedy search) | 76.58 | 0.5493 | 0.5017 | 0.4714 | 0.1187 |  -0.80 |
| retro (nucleus sampling p=0.95) | 93.34 | 0.0809 | 0.0275 | 0.0136 | 0.8817 | -3.54 |
| copyisallyouneed (greedy search) |  |  |  |  |  |  |
| copyisallyouneed (nucleus sampling p=0.95) | |  |  |  | |  |


