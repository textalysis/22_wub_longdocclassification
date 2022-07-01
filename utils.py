import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from summarizer import Summarizer
from transformers import BertTokenizer, LongformerTokenizer, BigBirdTokenizer,RobertaTokenizer
from datasets import load_dataset
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
nltk.download('punkt')
from docDataset import docDataset
from longdocDataset import longdocDataset
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler, Dataset)
import ast

CACHE_DIR = 'transformers-cache'

def get_dataset(dataset):
    if dataset == "20newsgroups":
        # remove = ("headers", "footers", "quotes") not remove here to keep the document long
        newsgroups = fetch_20newsgroups(subset='train', shuffle=True,
                                        random_state=238)
        data_1, data_2, label_1, label_2 = train_test_split(newsgroups.data, newsgroups.target,
                                                                        test_size=0.1, random_state=42)
        data_train = {'data': data_1, 'target': label_1}
        data_val = {'data': data_2, 'target': label_2}
        data_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=238) #['data] ['target']
       
    """
    will not use the dataset from huggingface because need to do preprocessing
    besides there are [] labels
    use the preprocessed data instead
    """
    if dataset == "ECtHR":
       train = open("data/ECtHR/train.json", "r")
       train = train.readlines()
       train = [ast.literal_eval(i) for i in train]  # str -> dict
       data_1 =  [i["text"] for i in train]
       label_1 =  [i["labels"] for i in train]
       dev = open("data/ECtHR/dev.json", "r")
       dev = dev.readlines()
       dev = [ast.literal_eval(i) for i in dev]
       data_2 = [i["text"] for i in dev]
       label_2 = [i["labels"] for i in dev]
       test = open("data/ECtHR/test.json", "r")
       test = test.readlines()
       test = [ast.literal_eval(i) for i in test]
       data_3 =  [i["text"] for i in test]
       label_3 = [i["labels"] for i in test]
       data_train = {'data': data_1, 'target': label_1}
       data_val = {'data': data_2, 'target': label_2}
       data_test = {'data': data_3, 'target': label_3}

    if dataset == "Hyperpartisan":

       
    if dataset == ""    


    return data_train, data_val, data_test


def bert_summarizer(docs):
    bert_summarizer = Summarizer()
    summarized_docs = [bert_summarizer(doc) for doc in docs]
    return summarized_docs


def text_rank(docs):
    summarized_docs = []
    for doc in docs:
        parser = PlaintextParser.from_string(doc, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, 30)
        text_summary = ""
        for sentence in summary:
            text_summary += str(sentence)
        summarized_docs.append(text_summary)
    return summarized_docs


def tokenize(tokenizer):
    if tokenizer=='BERT':
        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=CACHE_DIR)
    elif tokenizer=='longformer':
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir=CACHE_DIR)
    else:
        tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

    return tokenizer


def my_collate1(batches):
    # return batches
    # avoid shape error
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


def create_data_loader(toke_type, newsgroups, tokenizer, max_len, batch_size, approach="all", chunk_len=512, overlap_len=50, total_len=512*20):
    if toke_type == "short":
        ds = docDataset(
        docs=newsgroups['data'],
        targets=newsgroups['target'],
        tokenizer=tokenizer,
        max_len=max_len)

        dataloader = DataLoader(
                ds,
                batch_size=batch_size,
            )

    else:
        if approach == "all":
           ds = longdocDataset(docs=newsgroups['data'],
           targets=newsgroups['target'],
           tokenizer=tokenizer,
           max_len=max_len,
           approach=approach,
           chunk_len=chunk_len,
           overlap_len=overlap_len,
           total_len=total_len)
   
           dataloader = DataLoader(
              ds,
              batch_size=batch_size,
              collate_fn=my_collate1
                      )
        else:
           ds = longdocDataset(docs=newsgroups['data'],
           targets=newsgroups['target'],
           tokenizer=tokenizer,
           max_len=max_len,
           approach=approach,
           chunk_len=chunk_len,
           overlap_len=overlap_len,
           total_len=total_len)

           dataloader = DataLoader(
              ds,
              batch_size=batch_size)

    return dataloader





