# Evaluation Commands

```bash
# run the evaluation in cuda=0 device, the results file is defined in the run.sh file
./run.sh 0
```

# Evaluation Results

## 1. Wikitext103-1024

| Models | MAUVE (roberta-large, c=1.0, EOS token) | MAUVE (roberta-large,c=1.0, EOS token, remove unk and \n) | MAUVE(gpt2-large,c=1.0,last token, remove \n and unk)| MAUVE (deberta-v2-xlarge,c=1.0,start token)| MAUVE (deberta-v2-xlarge,c=1.0,start token, remove \n and unk) | MAUVE(roberta-base,c=1.0,start token, remove \n and unk) | MAUVE (electra-large,c-1, start token) | MAUVE (gpt2-large,c=1.0,last token, remove \n and unk) |
| - | - | - | - | - | - | - | - | - |
| gpt2 (greedy search)                       | 60.63 |  | 0.5| 92.10| 92.10| 64.73| |
| gpt2 (nucleus sampling p=0.95)             | 66.06 |  | 0.5| 85.47| 85.47| 68.24| |
| neurlab gpt2 (greedy search)               | 60.17 |  | 50.24| 88.95|88.95 | 62.48| |
| neurlab gpt2 (nucleus sampling p=0.95)     | 64.37 |  | 50.18| 85.21| 85.21| 71.12| |
| knnlm (greedy search full)                 | 64.65 |  | 0.5| 85.34| 85.34| 60.83| |
| knnlm (nucleus sampling p=0.95 full)       | 62.11 |  | 0.5| 86.02| 86.02| 61.59| |
| retro (greedy search)                      | 59.62 |  | 0.5| 90.84| 90.84| 62.00| |
| retro (nucleus sampling p=0.95)            | 67.13 |  | 0.5| 88.52| 88.52| 76.62| |
| copyisallyouneed (greedy search)           | 74.54 |  | 50.32| 97.33| 97.33| 78.15| |
| copyisallyouneed (nucleus sampling p=0.95) | 78.96 |  | 50.25| 91.38| 91.38| 80.12| |


## 2. LawMT

| Models | MAUVE (roberta-large,c=1.0) | MAUVE (roberta-large,c=1,EOS, remove \n and unk)| MAUVE (roberta-base,c=1.0, start token, remove \n and unk)|
| - | - | - | - |
| gpt2 (greedy search)  | 54.01| 59.79| 52.91|
| gpt2 (nucleus sampling p=0.95)  | 52.97| 55.28| 53.86|
| neurlab gpt2 (greedy search)  | 52.90 | 51.42| 51.76|
| neurlab gpt2 (nucleus sampling p=0.95)  | 53.22| 51.89| 52.43|
| knnlm (greedy search full) | 53.41 | 52.90| 51.27|
| knnlm (nucleus sampling p=0.95 full) | 53.18| 51.52| 50.87|
| retro (greedy search) | 51.73| 59.00| 51.50|
| retro (nucleus sampling p=0.95) | 52.74 | 54.91| 50.77|
| copyisallyouneed (greedy search) | 56.66| 59.79| 52.63|
| copyisallyouneed (nucleus sampling p=0.95) |54.32| 55.28| 53.22|

## 3. EN-Wiki

| Models | MAUVE (roberta,c=1.0) | MAUVE (roberta-large,c=1.0,EOS, remove \n and unk)| MAUVE (roberta-base,c=1.0, EOS, remove \n and unk) |
| - | - | - | - |
| gpt2 (greedy search)  | 67.62| 58.82| 63.95|
| gpt2 (nucleus sampling p=0.95)  | 63.49| 64.99| 72.10|
| neurlab gpt2 (greedy search)  |  61.19 | 61.67| 74.17|
| neurlab gpt2 (nucleus sampling p=0.95)  | 71.03 | 66.06| 69.74|
| knnlm (greedy search full) | 68.36 | 60.49| 71.23|
| knnlm (nucleus sampling p=0.95 full) | 67.08 | 63.92| 66.06|
| retro (greedy search) | 66.55| 79.65| 66.47|
| retro (nucleus sampling p=0.95) |71.91 | 65.82| 85.54|
| copyisallyouneed (greedy search) | 92.29| 82.45| 85.64|
| copyisallyouneed (nucleus sampling p=0.95) |87.64| 77.62| 87.38|


