# Process Datasets

After moving the datasets into their corresponding `{dataset_name}_1024` folder, the following commands should be conducted to process the datasets.
Take the `Wikitext-103` corpus as the example:

## 1. Build the FAISS Index 

build the FAISS index on each corpus

get into the work folder:

```bash
cd dpr_wikitext103_1024
```

build the index 

```bash
./build_index.sh 0,1,2,3,4,5,6,7
```

search the top-k documents for each given document

```bash
./dpr_search.sh
```

running this command will generate the results file under `wikitext103_1024` folder

## 2. Run the phrase segmentation algorithm

run the phrase segmentation algorithm to split the document into phrases

```bash
cd wikitext103_1024/phrase_split;
./phrase_split.sh
```
