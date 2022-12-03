## Training

To run hierarchial method: python3 train_hierarchial.py (or run in background: nohup python3 -u train_hierarchial.py > output.txt &)

To run sparse method: python3 train_sparse.py (or run in background)

To run truncation method: python3 train_trun_sum.py (or run in background)

To run summarization method: first summarize the text and then adjust summarizer part in train_trun_sum.py

## Hyperparameters

can be changed in the para part of each file (train_hierarchial.py, train_sparse.py, train_trun_sum.py)

train_hierarchial.py: total_len (maximal input tokens of each doc), chunk_lens (segment length), overlap_lens: number of overlapping tokens.

train_sparse.py: sparse_max_lens (maximal input tokens of each doc), attention_windows (attention window of Longformer), block_sizes (block size of Bigbird).

