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
| copyisallyouneed (greedy search)           | 26.01 | 0.2814 | 0.238 | 0.214 | 0.4303 | -1.73 |
| copyisallyouneed (nucleus sampling p=0.95) | 28.34 | 0.0731 | 0.0266 | 0.0128 | 0.8907 | -2.91 |


## 2. LawMT

| Models | MAUVE (gpt2-large,c=2.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)                       | 20.32 | 0.1433 | 0.1017 | 0.0818 | 0.7066 |-0.64 |
| gpt2 (nucleus sampling p=0.95)             | 25.21 | 0.0395 | 0.0149 | 0.0078 | 0.9388 | -3.61 |
| gpt2-fine-tuned (greedy search)            | 23.00 | 0.0968 | 0.0633 | 0.0483 | 0.8052 | -0.88 |
| gpt2-fine-tuned (nucleus sampling p=0.95)  | 26.85 | 0.0702 | 0.0217 | 0.009 | 0.9014 | -3.33 |
| neurlab gpt2 (greedy search)               | 19.08 | 0.1375 | 0.0954 | 0.0726 | 0.7236 | -1.20  |
| neurlab gpt2 (nucleus sampling p=0.95)     | 22.77 | 0.0367 | 0.0128 | 0.0057 | 0.9456 | -3.92 |
| knnlm (greedy search full)                 | 25.38 | 0.0511 | 0.0207 | 0.0121 | 0.918 | -3.69 |
| knnlm (nucleus sampling p=0.95 full)       | 25.06 | 0.031 | 0.0102 | 0.0045 | 0.9547 |  -4.81 |
| retro (greedy search)                      | 18.70 | 0.139 | 0.0996 | 0.0823 | 0.7114 | -0.67 |
| retro (nucleus sampling p=0.95)            | 20.35 | 0.0343 | 0.0126 | 0.0057 | 0.9481 | -4.22  |
| copyisallyouneed (greedy search)           | 21.31 | 0.0806 | 0.0482 | 0.0365 | 0.8432 | -1.51 |
| copyisallyouneed (nucleus sampling p=0.95) | 28.14 | 0.0501 | 0.0176 | 0.0081 | 0.9256 | -2.78 |

## 3. EN-Wiki

| Models | MAUVE (gpt2-large,c=2.0) | Rep-2 | Rep-3 | Rep-4 | Diversity | Coherence |
| - | - | - | - | - | - | - |
| gpt2 (greedy search)                         | 40.97 | 0.5075 | 0.4601 | 0.4314 | 0.1512 | -0.81 |
| gpt2 (nucleus sampling p=0.95)               | 64.62 | 0.0616 | 0.0154 | 0.0061 | 0.9183 | -3.55 |
| gpt2-fined (greedy search)                   | 39.46 | 0.4959 | 0.4482 | 0.4199 | 0.1614 | -0.72 |
| gpt2-fined (nucleus sampling p=0.95)         | 65.39 | 0.0458 | 0.0188 | 0.01   | 0.9268 | -2.67 |
| neurlab gpt2 (greedy search)                 | 43.22 | 0.4254 | 0.359  | 0.3197 | 0.2505 | -1.29 |
| neurlab gpt2 (nucleus sampling p=0.95)       | 59.21 | 0.0602 | 0.0153 | 0.0053 | 0.9204 | -3.88 | 
| knnlm (greedy search full)                   | 65.81 | 0.0948 | 0.0367 | 0.0216 | 0.8531 | -3.85 |
| knnlm (nucleus sampling p=0.95 full)         | 66.35 | 0.036  | 0.0053 | 0.0013 | 0.9576 | -4.68 |
| retro (greedy search)                        | 41.99 | 0.5493 | 0.5017 | 0.4714 | 0.1187 | -0.80 |
| retro (nucleus sampling p=0.95)              | 63.78 | 0.0809 | 0.0275 | 0.0136 | 0.8817 | -3.54 |
| copyisallyouneed (greedy search)             | 58.84 | 0.3611 | 0.3063 | 0.2732 | 0.3221 | -1.65 |
| copyisallyouneed (nucleus sampling p=0.95)   | 66.98 | 0.1476 | 0.077  | 0.0457 | 0.7508 | -2.58 |


| Methods | MAUVE (gpt2-large, c=2) | |||||
| -| -|-|-|-|-|-|
| gpt2-fined (nucleus sampling p=0.95)   | 66.84 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index,nucleus sampling p=0.95)   | 66.14 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.1 en_wiki,nucleus sampling p=0.95)   | 68.88 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.2 en_wiki,nucleus sampling p=0.95)   | 69.93 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.3 en_wiki,nucleus sampling p=0.95)   | 69.93 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.4 en_wiki,nucleus sampling p=0.95)   | 67.87 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.5 en_wiki,nucleus sampling p=0.95)   | 67.87 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.6 en_wiki,nucleus sampling p=0.95)   | 68.94 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.7 en_wiki,nucleus sampling p=0.95)   | 66.18 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.8 en_wiki,nucleus sampling p=0.95)   | 66.18 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 0.9 en_wiki,nucleus sampling p=0.95)   | 70.92 |  |   |  |  |  |
| copyisallyouneed (wikitext103 index + 1.0 en_wiki,nucleus sampling p=0.95)   | 70.92 |  |   |  |  |  |





# MAUVE score for the En-Wiki Test set

| Models | MAUVE (gpt2-large, c=2.0) |
| - | - |
| GPT2 w/o FT        | 59.79 |
| GPT2 w/ FT         | 55.72 |
| CoG (0.0x index)   | 65.17 |
| CoG (0.001x index) | 65.62 |
| CoG (0.003x index) | 66.55 |
| CoG (0.01x index)  | 65.56 |
| CoG (0.03x index)  | 68.35 |
| CoG (0.1x index)   | 66.53 |
| CoG (0.3x index)   | 66.84 |
| CoG (1.0x index)   | 67.82 |

# MAUVE score for the WikiText Test set

run 10 times with different random seed, and the average MAUVE scores are recorded

| Models | MAUVE (gpt2-large, c=2.0) |
| - | - |
| GPT2 w/o FT        | 23.43 |
| GPT2 w/ FT         | 21.01 |
| CoG (0.0x index)   | 26.14 |
| CoG (0.001x index) | 25.99 |
| CoG (0.003x index) | 26.11 |
| CoG (0.01x index)  | 25.97 |
| CoG (0.03x index)  | 26.14 |
| CoG (0.1x index)   | 26.45 |
| CoG (0.3x index)   | 26.67 |
| CoG (1.0x index)   | 26.97 |
