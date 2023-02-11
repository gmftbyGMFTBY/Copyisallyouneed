# Source codes for `Copy is all you need`
**Authors**: Tian Lan, Deng Cai, Yan Wang, Heyan Huang, Xian-Ling Mao

**[Contact]** If you have any questions, feel free to contact me via (lantiangmftby at gmail.com).

This repository contains code other related resources of our paper "Copy is All You Need"


<span id='all_catelogue'/>

### Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#prepare dataset'>2. Prepare the Dataset</a>
* <a href='#train the Models'>3. Train the Models</a>
* <a href='#test with prefix'>4. Test the Models</a>
    
****

<span id='introduction'/>

#### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>

The dominant text generation models compose output by selecting words in a fixed vocabulary. In this paper, we formulate text generation as progressively copying text segments (e.g., words or phrases) from an existing text collection. We compute the contextualized representations of meaningful text segments and index them using efficient vector search toolkits. The task of text generation is then decomposed into a series of copy-and-paste operations: at each time step, we seek suitable text spans from existing articles in the text collection rather than selecting from a standalone vocabulary. Experiments on the standard language modeling benchmark (WikiText-103) show that our approach achieves better generation quality by coping from the original training data (0.758 vs. 0.691 MAUVE). We also show that our approach attains additional performance gains by simply scaling up to larger text collections without extra training. Furthermore, our approach allows for effective domain adaptation by simply switching to any domain-specific text collection, again without further training. Finally, we observe that our approach achieves better inference efficiency than standard token-level autoregressive models thanks to the reduction of decoding steps.

<img src="./img/overview.png" width = "1100" height = "400" alt="overview" align=center />

Three benchmarks are used in this paper, and their preprocessing procedures are listed under `data` folder (`wikitext103`, `en_wiki`, `lawmt`).

****

<span id='prepare dataset'/>

#### 2. Prepare the Dataset: <a href='#all_catelogue'>[Back to Top]</a>

The corpus for Wikitext-103, Law-MT, and En-Wiki can be downloaded from [this link](https://pan.baidu.com/s/13JmmAZPN_5jLkSbS-V51rg) (with the code `ufhn`).
For `wikitext-103`, `law-mt`, and `en-wiki` datasets, please move their corresponding `base_data_128.txt` and `test.txt` into `data/{dataset_name}_1024`,
and conduct the commands in `data/README.md` to process theses datasets.

****

<span id='train the models'/>

#### 3. Train the Models: <a href='#all_catelogue'>[Back to Top]</a>

##### 1. prepare the environment

```bash
pip install -r requirments.txt
```

##### 2. get into the folder and initialize the workspace

```bash
cd copyisallyouneed;
python prepare_work_space.py
```

Running the `prepare_work_space.py` script will initialize folders under the `root_dir` (defined in `config/base.yaml`): 
* `log`: save the backup of the previous checkpoints
* `ckpt`: save the checkpoints of the trained models
* `rest`: save the tensorboard log files during the training

Before the running, make sure the `root_dir` variable is renamed on your local environemnt.

##### 3. running baselines

The following examples runs on `wikinews` benchmark, replace it with `wikitext` or `story` to test other benchmark.
Noted that the training args and details are listed under the `config/*.yaml`.

1. train the gpt2 baseline

    ```bash
    # distributted train the gpt2 model on wikitext103 dataset
    ./scripts/train.sh wikitext103 gpt2 0,1,2,3,4,5,6,7
    ```

2. train the retro baseline

   Follow the description under the `baseline/retro/README.md` to train the retro baseline
  
3. train the KNN-LM baseline

    Noted that the KNN-LM baseline is built upon the GPT2 baseline. Here, we aim to inference the whole dataset to build the FAISS index for KNN-LM.

    ```bash
    ./scripts/knnlm_inference.sh 0;
    
    # build the FAISS index: more details, such as the faiss index type can be found in `build_index.py`
    python build_index.py
    ```
   
4. train the copyisallyouneed model

    ```bash
    ./scripts/train.sh wikitext103 copyisallyouneed 0,1,2,3,4,5,6,7
    ```
    
##### 4. Test the Models: <a href='#all_catelogue'>[Back to Top]</a>

After the training procedure, the following commands are conducted to generate the results file for automatic and human evaluations.
More details about the inference can be found in these corresponding bash scripts.

1. generate the results for gpt2 baseline

    ```bash
    ./scripts/gpt2_test.sh
    ```

2. generate the results for retro baseline

    Following the details under `baseline/retro/README.md`

3. generate the results for KNN-LM baseline

    ```bash
    ./scripts/knnlm_test.sh
    ```
    
4. generate the results for Copyisallyouneed baseline

    ```bash
    ./scripts/copyisallyouneed_test.sh
    ```
    
Running the above scripts will generate the corresponding results files under the `copyisallyouneed` folder with the clear name.
For the automatic evaluation, move these files to the `evaluation/*` folder to test `MAUVE`, `Diversity`/`Rep-n`, and Coherence.
More details about the automatic evaluation procedure can be found in `evaluation/README.md`.

For the human evaluation, move these files to the `make_human_evaluation/raw_files/` folder, and run this command:

```bash
./run.sh
```

More details about the human evaluation can be found in `make_human_evaluation/README.md`.

