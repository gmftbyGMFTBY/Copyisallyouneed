# Wikitext103 Datasets

## Description about the Dataset

1. `bast_data.txt`
    recodes the the overall documents saved in wikitext-103 corpus
2. 

## Results

**Note that the following results are based on the 64 pool size**

1. Chunk size 128; recall method BM25:

* find ratio: 39.39% words are copied
* copied n-grams

| n-gram | ratio |
| - | - |
| 2-gram | 65.11 |
| 3-gram | 20.85 |
| 4-gram | 8.07 |
| 5-gram | 3.28 |
| 6-gram | 1.67 |
| 7-gram | 1.02 |

test set phrase copied statistic

* find ratio: 22.51% words are copied (64 docs); 26.43% for 128 docs; 30.4% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram | 73.85 | 71.8 | 70.14 |
| 3-gram | 15.83 | 17.55 | 19.05 |
| 4-gram | 6 | 6.29 | 6.6 |
| 5-gram | 2.43 | 2.47 | 2.45 | 
| 6-gram | 1.13 | 1.19 | 1.11 | 
| 7-gram | 0.75 | 0.7 |0.65 |


test set phrase copied statistic with full doc 

* find ratio: 24.89% words are copied (64 docs); 28.92% for 128 docs; 32.91% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram | 69.94 | 68.48 | 67.04 |
| 3-gram | 17.92 | 19.18 | 20.24 |
| 4-gram | 7.03 | 7.23 | 7.63 |
| 5-gram | 3.02 | 3.01 | 3.05 | 
| 6-gram | 1.43 | 1.39 | 1.38 | 
| 7-gram | 0.67 | 0.7 |0.66 |


2. Chunk size 128; recall method DPR:

* find ratio: 40.99% words are copied;
* copied n-grams

| n-gram | atio | 
| - | - |
| 2-gram | 65.75 |
| 3-gram | 20.7 |
| 4-gram | 2.65 |
| 5-gram | 7.76 |
| 6-gram | 3.15 |
| 7-gram | 1.01 |

test set phrase copied statistic

* find ratio: 23.13% words are copied; 27.41% for 128 docs; 31.76% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram | 74.6 | 72.01 | 69.77 |
| 3-gram | 16.26 | 18.24 |19.8|
| 4-gram | 5.62 | 6.13 |6.57|
| 5-gram | 1.99 | 2.05 |2.35|
| 6-gram | 0.98 | 1.05 |0.98|
| 7-gram | 0.54 | 0.51 |0.52|


test set phrase copied statistic with full doc 

* find ratio: 26.63% words are copied (64 docs); 30.73% for 128 docs; 34.73% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram | 71.54 | 69.63 | 67.71 |
| 3-gram | 17.8 | 19.24 | 20.68 |
| 4-gram | 6.24 | 6.77 | 7.27 |
| 5-gram | 2.45 | 2.51 | 2.65 | 
| 6-gram | 1.22 | 1.18 | 1.08 | 
| 7-gram | 0.74 | 0.66 |0.62 |


3. Chunk size 256; recall method BM25:

* find ratio: 39.26% words are copied
* copied n-grams

| n-gram | ratio |
| - | - |
| 2-gram | 66.86 |
| 3-gram | 20.3 |
| 4-gram | 7.48 |
| 5-gram | 2.95 |
| 6-gram | 1.51 |
| 7-gram | 0.9 |

test set phrase copied statistic

* find ratio: 25.48% words are copied (64 docs); 29.42% for 128 docs; 33.16% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram | 70.14 | 72.74 | 68.39 |
| 3-gram | 19.05 | 16.48 | 20.1 |
| 4-gram | 6.6 | 6.13 | 7.14 |
| 5-gram | 2.45 | 2.51 | 2.44 | 
| 6-gram | 1.11 | 1.27 | 1.17 | 
| 7-gram | 0.65 | 0.87 |0.76 |



4. Chunk size 256; recall method DPR:

* find ratio: 43.44% words are copied
* copied n-grams

| n-gram | ratio |
| - | - |
| 2-gram | 65.65 |
| 3-gram | 21.01 |
| 4-gram | 7.72 |
| 5-gram | 3.08 |
| 6-gram | 1.58 |
| 7-gram | 0.97 |

test set phrase copied statistic

* find ratio: 25.46% words are copied; 29.76% for 128 docs; 34.17% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram | 73.02 | 70.55 | 68.26 |
| 3-gram | 17.06 | 18.99|20.61|
| 4-gram | 5.95 | 6.39 |6.94|
| 5-gram |  2.14 | 2.31|2.46|
| 6-gram |  1.1 |1.08|1.04|
| 7-gram | 0.73 |0.69|0.68|





5. Chunk size 512; recall method BM25:

* find ratio: 39.37% words are copied
* copied n-grams

| n-gram | ratio |
| - | - |
| 2-gram | 66.98 |
| 3-gram | 20.28 |
| 4-gram | 7.43 |
| 5-gram | 2.93 |
| 6-gram | 1.5 |
| 7-gram | 0.89 |

test set phrase copied statistic

* find ratio: 25.79% words are copied; 29.71% for 128 docs; 33.54% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram |72.59 |70.58| 68.38|
| 3-gram |16.71 |18.5|20.04|
| 4-gram |6.07 | 6.49|7.18|
| 5-gram |2.47 |2.43|2.45|
| 6-gram |1.3 |1.18|1.21|
| 7-gram |0.85 |0.82|0.74|




6. Chunk size 512; recall method DPR:

* find ratio: 43.68% words are copied
* copied n-grams

| n-gram | ratio |
| - | - |
| 2-gram | 65.7 |
| 3-gram | 20.99 |
| 4-gram | 7.7 |
| 5-gram | 3.08 |
| 6-gram | 1.57 |
| 7-gram | 0.96 |

test set phrase copied statistic

* find ratio: 25.66% words are copied; 29.96% for 128 docs; 34.35% for 256 docs
* copied n-grams

| n-gram | 64 docs ratio | 128 docs ratio | 256 docs ratio |
| - | - | - | - |
| 2-gram |72.95 |70.4| 68.07|
| 3-gram |17.07 |19.03|20.68|
| 4-gram |5.98 |6.47|7.03|
| 5-gram |2.15 |2.31|2.48|
| 6-gram |1.13 |1.1|1.07|
| 7-gram |0.72 |0.69|0.67|


