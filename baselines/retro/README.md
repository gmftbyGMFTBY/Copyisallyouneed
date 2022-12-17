# RETRO-Pytorch-Test

Training dataset comes from `../../data/wikitext103_1024/base_data_128.txt`

## 1. Training the RETRO Baseline

The training args are saved in the `train.py` file.
Noted that, the training file should be saved under the folder, which is named of `documents_path` in `TraininigWrapper` class of `train.py`.

```bash
# running the training scripts: (1) build the index for the training file under `documents_path`
# (2) train the model
./train.sh
```

## 2. Test the RETRO Baseline

Details can be found in the `test.py` file

```bash
./test.sh
```
