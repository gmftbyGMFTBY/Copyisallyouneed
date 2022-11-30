# TODO

Data preparation
1. chunk size: 128, 256, 512
2. recall method: DPR, BM25

Evaluate the propotion of the 2-gram, 3-gram, 4-gram, 5-gram, ..., and the copied ratio, choose the combination of chunk_size and recall_method that could achieve the longest and highest copied phrases.

# 补充实验

如果我们把train set数据重新切分构造train set 和test set，这个gap还是存在(40.41% on train set以及25.89% on test set))
怀疑是因为一个document切分成多个chunk之后，这些chunk被准确召回了所以引入的相关数据更多，但是test set上没有这一步操作，所有test set的prefix都是32，且都存train set中召回

# 进一步补充实验

我们去掉了train set中多余的chunk，只保留一条chunk，看看train set上的copy phrase召回率有多高
确实是这个问题(64 docs: train set召回率是19.05%, test set召回率是11.95%, full doc test set召回率是11.95%)
确实是这个问题(128 docs: train set召回率是22.97%, test set召回率是14.83%, full doc test set召回率是14.91%)
确实是这个问题(256 docs: train set召回率是22.97%, test set召回率是14.83%, full doc test set召回率是18.24%)

召回64 docs: train set召回率是40.06%, test set召回率是25.11%, full doc test set召回率是28.54%)

# 进进一步实验
分析train set和test set gap的影响

从train set分割train set和test set，然后再只保留一条chunk，看看train set 和 test set上的copy phrase比例
64 docs: train set召回率是41.05%, test set召回率是30.3%, full doc test set召回率是%)

