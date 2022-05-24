## File structure

**Models**

base.py: keep the first 510 tokens of long document

mean_pooling.py:  mean-pooled result from the output of each segment of the long document

lstm.py: output of each segment of the long document as input to lstm to get classification

**Test results lr = 2e-5**

base.txt

mean_pooling.txt

lstm.txt

**Visualizations**

base.png

mean_pooling.png

lstm.png

## memory problem solved

### The reason of too much memory usage:

In base model, the input ids and attention mask of BERT are [batch_size, 512] and [batch_size, 512] respectively because only the first 510 tokens of each document is kept.

But after segmentation, the input ids and attention mask of BERT are [number of segments of documents in one batch * 512] and [number of segments of documents in one batch * 512] respectively, then cuda will calculate the output for each segment in parallel. It can lead to too much memory usage.

For example, when the document becomes very large, e.g. if can be divided into 200 segments, the input size of BERT model would be  [200, 512] even when batch_size is set to 1. The output of the 200 segments will be calculated in parallel. This is where the problem occurs.

With experiment, the maximal number of segments as input to BERT when batch_size = 1 is found to be 101 (device cuda with MEM 80G).

### Solutions

Two solutions have been tried.

1. remove the n longest document
 
In the experiment, the 1000 longest documents of 20newsgroups training set and 100 longest documents of 20newsgroups validation set after tokenization have been removed. It successfully solved the problem.

However, the problem would be how to choose the number of removed longest documents.  If batch_size is set to 1, we can choose by the maximal number of segments per document. But computation would be slow for small batch size. If we set batch size higher, the total number of segments for every batch need to be pre-computed.

2. for-loop the BERT model

We can split the input the BERT model and then let cuda calculate in sequence.

Code like this:

```
max_num = 32
input_ids = torch.split(input_ids, max_num)
attention_mask = torch.split(attention_mask, max_num)
pooled_output_list=[]
for i in range(len(input_ids)):
    _, pooled_output = self.bert(
              input_ids = input_ids[i],
              attention_mask = attention_mask[i],
              return_dict=False
            )
    pooled_output_list.append(pooled_output)
pooled_output = torch.cat(pooled_output_list)
```

However, the memory usage problem still occurs. It seems that cuda will calculate in parallel even with for loop? It is a bit confusing for me after several tries.

## Difficulties

I am still confused about how to do the benchmarks more efficiently and record the results more nicely, because there are many hyperparameters that can be tried like learning rate, the maximal length of segments, the number of overlapping tokens, window size in sparse attention ... as well as different models. 

In addition, because the GPU is shared by all students, the memory usage problem may occur at some timepoint when other students have shared too much memory at that specific timepoint. Then the training needs to be started from the first epoch. One possible solution is to save the checkpoints of each epoch. Considering the large size of BERT model, I don't know whether it's necessary. 


