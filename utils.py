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
import os
#import json

CACHE_DIR = 'transformers-cache'


# map ECtHR labels to 0-num_classes
def map_to_classes(label):
    for i, x in enumerate(label):
        for j, a in enumerate(x):
            if a == 'P1-1':
                label[i][j] = a.replace('P1-1', '0')
            if a == '2':
                label[i][j] = a.replace('2', '1')
            if  a == '3':
                label[i][j] = a.replace('3', '2')
            if a == '5':
                label[i][j] = a.replace('5', '3')
            if a == '6':
                label[i][j] = a.replace('6', '4')
            if a == '8':
                label[i][j] = a.replace('8', '5')
            if a == '9':
                label[i][j] = a.replace('9', '6')
            if a == '10':
                label[i][j] = a.replace('10', '7')
            if a == '11':
                label[i][j] = a.replace('11', '8')
            if a == '14':
                label[i][j] = a.replace('14', '9')


# map multi-label str to int
def label_to_int(label):
    for i, x in enumerate(label):
        for j, a in enumerate(x):
            label[i][j] = int(label[i][j])


# preprocess to one hot to avoid batch loader error
# keep label size the same, 10 is the number of classes
def one_hot_labels(integer_encoded):
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(10)]
        for i in value:
            letter[i] = 1    
        onehot_encoded.append(letter)
    return onehot_encoded


def get_dataset(dataset):
    # from sklearn
    if dataset == "20newsgroups":
        # remove = ("headers", "footers", "quotes") not remove here to keep the document long
        newsgroups = fetch_20newsgroups(subset='train', shuffle=True, random_state=238)
        data_1, data_2, label_1, label_2 = train_test_split(newsgroups.data, newsgroups.target, test_size=0.1, random_state=42)
        data_train = {'data': data_1, 'target': label_1}
        data_val = {'data': data_2, 'target': label_2}
        data_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=238)  # ['data], ['target']
       
    """
    will not use the dataset from huggingface because need to do preprocessing
    besides there are [] labels
    use the preprocessed data instead
    """
    if dataset == "ECtHR":
        train = open(os.path.join("data/ECtHR", "train.txt"), "r")
        train = train.readlines()
        train = [ast.literal_eval(i) for i in train]  # str -> dict
        data_1 = [i["text"] for i in train]
        label_1 = [i["labels"] for i in train]
        map_to_classes(label_1)  # label to 0-9
        label_to_int(label_1)  # label to int
        label_1 = one_hot_labels(label_1)  # label to one hot
        dev = open(os.path.join("data/ECtHR", "dev.txt"), "r")
        dev = dev.readlines()
        dev = [ast.literal_eval(i) for i in dev]
        data_2 = [i["text"] for i in dev]
        label_2 = [i["labels"] for i in dev]
        map_to_classes(label_2)
        label_to_int(label_2)
        label_2 = one_hot_labels(label_2)
        test = open(os.path.join("data/ECtHR", "test.txt"), "r")
        test = test.readlines()
        test = [ast.literal_eval(i) for i in test]
        data_3 = [i["text"] for i in test]
        label_3 = [i["labels"] for i in test]
        map_to_classes(label_3)
        label_to_int(label_3)
        label_3 = one_hot_labels(label_3)
        data_train = {'data': data_1, 'target': label_1}
        data_val = {'data': data_2, 'target': label_2}
        data_test = {'data': data_3, 'target': label_3}

    if dataset == "Hyperpartisan":
        train = open(os.path.join("data/Hyperpartisan", "train.txt"), "r")
        train = train.readlines()
        train = [ast.literal_eval(i) for i in train]  # str -> dict
        data_1 = [i["text"] for i in train]
        label_1 = [i["label"] for i in train]
        dev = open(os.path.join("data/Hyperpartisan", "dev.txt"), "r")
        dev = dev.readlines()
        dev = [ast.literal_eval(i) for i in dev]
        data_2 = [i["text"] for i in dev]
        label_2 = [i["label"] for i in dev]
        test = open(os.path.join("data/Hyperpartisan", "test.txt"), "r")
        test = test.readlines()
        test = [ast.literal_eval(i) for i in test]
        data_3 =  [i["text"] for i in test]
        label_3 = [i["label"] for i in test]
        data_train = {'data': data_1, 'target': label_1}
        data_val = {'data': data_2, 'target': label_2}
        data_test = {'data': data_3, 'target': label_3}

    return data_train, data_val, data_test


def bert_summarizer(docs):
    bert_summarizer = Summarizer()
    summarized_docs = [bert_summarizer(doc, num_sentences=15) for doc in docs]
    #summarized_docs = [bert_summarizer(doc) for doc in docs]
    """
    i = 0
    for doc in docs:
        print("len",len(doc.split(" ")))
        print("summa",len(bert_summarizer(doc).split(" ")))
        print(i)
        print("-----")
        i = i + 1
    #print(len(summarized_docs))
    """
    return summarized_docs


def text_rank(docs):
    summarized_docs = []
    for doc in docs:
        parser = PlaintextParser.from_string(doc, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, 5)
        text_summary = ""
        for sentence in summary:
            text_summary += str(sentence)
        summarized_docs.append(text_summary)
    return summarized_docs


def filter_testset(tokenizer, data_test):
    tokenizer = tokenize('BERT')
    long_docs = []
    long_labels = []
    i = 0
    for data in data_test['data']:
        if(len(tokenizer.tokenize(data)) > 510):
            long_docs.append(data)
            long_labels.append(data_test['target'][i])
        i = i+1
    
    data_test["data"] = long_docs
    data_test["target"] = long_labels 
    
    return data_test

def tokenize(tokenizer):
    if tokenizer=='BERT':
        #  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)
        #  use roberta instead of bert
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=CACHE_DIR)
    elif tokenizer=='longformer':
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir=CACHE_DIR)
    else:
        tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base', cache_dir=CACHE_DIR)

    return tokenizer


def my_collate1(batches):
    # return batches
    # avoid shape error
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


def create_data_loader(toke_type, newsgroups, tokenizer, max_len, batch_size, approach="all", chunk_len=512, overlap_len=50, total_len=4096):
    # only keep first 512 tokens, don't need longdocDataset
    if toke_type == "short":
        ds = docDataset(
            docs=newsgroups['data'],
            targets=newsgroups['target'],
            tokenizer=tokenizer,
            max_len=max_len)

        dataloader = DataLoader(ds,
                batch_size=batch_size
            )

    # for hierarchical method and truncation tail and head_tail
    # tail and head_tail implemented in longdocDataset, but for dataloader don't need to handle shape
    else:
        if approach == "all":
            # hierarchical method
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
            # tail and head_tail
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





