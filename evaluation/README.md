# Evaluation Commands

```bash
# run the evaluation in cuda=0 device, the results file is defined in the run.sh file
./run.sh 0
```

# Evaluation Results

## 1. Wikitext103-1024

| Models | MAUVE (gpt2-large,c=2.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)                       | 19.87 | 0.4356 | 0.3855 | 0.355 | 0.2237 | -0.74 | 
| gpt2 (nucleus sampling p=0.95)             | 25.59 | 0.051 | 0.0133 | 0.005 | 0.9322 | -3.65 | 
| neurlab gpt2 (greedy search)               | 21.51 | 0.346 | 0.2795 | 0.2398 | 0.3583 | -1.34 |
| neurlab gpt2 (nucleus sampling p=0.95)     | 22.04 | 0.0544 | 0.0149 | 0.005 | 0.9268 | -3.72  |
| knnlm (greedy search full)                 | 23.21 | 0.0681 | 0.0213 | 0.0111 | 0.9019 | -4.00  |
| knnlm (nucleus sampling p=0.95 full)       | 24.53 | 0.0332 | 0.0062 | 0.0017 | 0.9592 | -4.65  |
| retro (greedy search)                      | 19.83 | 0.4465 | 0.3963 | 0.366 | 0.2119 | -0.74 |
| retro (nucleus sampling p=0.95)            | 23.51 | 0.0621 | 0.0193 | 0.0086 | 0.9119 | -3.63 |
| copyisallyouneed (greedy search)           | 26.01 | 26.01 | 0.2814 | 0.238 | 0.214 | 0.4303 | -1.73 |
| copyisallyouneed (nucleus sampling p=0.95) | 28.34 | 28.34 | 0.0731 | 0.0266 | 0.0128 | 0.8907 | -2.91 |


## 2. LawMT

| Models | MAUVE (gpt2-large,c=2.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)                       | 20.32 | 0.1433 | 0.1017 | 0.0818 | 0.7066 |-0.64 |
| gpt2 (nucleus sampling p=0.95)             | 25.21 | 0.0395 | 0.0149 | 0.0078 | 0.9388 | -3.61 |
| neurlab gpt2 (greedy search)               | 19.08 | 0.1375 | 0.0954 | 0.0726 | 0.7236 | -1.20  |
| neurlab gpt2 (nucleus sampling p=0.95)     | 22.77 | 0.0367 | 0.0128 | 0.0057 | 0.9456 | -3.92 |
| knnlm (greedy search full)                 | 19.95 | 0.04 | 0.015 | 0.0085 | 0.9376 | -4.64 |
| knnlm (nucleus sampling p=0.95 full)       | 21.22 | 0.0283 | 0.0097 | 0.0047 | 0.9577 | -5.05 |
| retro (greedy search)                      | 18.66 | 0.139 | 0.0996 | 0.0823 | 0.7114 | -0.67 |
| retro (nucleus sampling p=0.95)            | 20.94 | 0.0347 | 0.0123 | 0.0057 | 0.948 | -4.18  |
| copyisallyouneed (greedy search)           | 21.31 | 0.0806 | 0.0482 | 0.0365 | 0.8432 | -1.51 |
| copyisallyouneed (nucleus sampling p=0.95) | 28.14 | 0.0501 | 0.0176 | 0.0081 | 0.9256 | -2.78 |

## 3. EN-Wiki

| Models | MAUVE (gpt2-large,c=2.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - | - |
| gpt2 (greedy search)                         | 40.97 | 0.5075 | 0.4601 | 0.4314 | 0.1512 | -0.81 |
| gpt2 (nucleus sampling p=0.95)               | 64.62 | 0.0616 | 0.0154 | 0.0061 | 0.9183 | -3.55 |
| neurlab gpt2 (greedy search)                 | 43.22 | 0.4254 | 0.359 | 0.3197 | 0.2505 |  -1.29 |
| neurlab gpt2 (nucleus sampling p=0.95)       | 59.21 | 0.0602 | 0.0153 | 0.0053 | 0.9204 |-3.88| 
| knnlm (greedy search full)                   | 65.81 | 0.0948 | 0.0367 | 0.0216 | 0.8531 | -3.85 |
| knnlm (nucleus sampling p=0.95 full)         | 66.35 | 0.036 | 0.0053 | 0.0013 | 0.9576 | -4.68 |
| retro (greedy search)                        | 41.99 | 0.5493 | 0.5017 | 0.4714 | 0.1187 |  -0.80 |
| retro (nucleus sampling p=0.95)              | 63.78 | 0.0809 | 0.0275 | 0.0136 | 0.8817 | -3.54 |
| copyisallyouneed (greedy search)             | 58.84 | 0.3611 | 0.3063 | 0.2732 | 0.3221 | -1.65 |
| copyisallyouneed (nucleus sampling p=0.95)   | 66.98 | 0.1476 | 0.077 | 0.0457 |0.7508 | -2.58 |


