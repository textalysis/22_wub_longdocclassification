#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (TensorDataset, DataLoader,
                             RandomSampler, SequentialSampler)
from torch.optim import AdamW

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
from collections import defaultdict, Counter
#import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers import pipeline
from transformers import LongformerForSequenceClassification, LongformerTokenizer, LongformerConfig

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import classification_report

import time

import os

# ### Data Preparing
# 
# get the 20newsgroups dataset and then split into training set (90%) and validation set(10%)

print('Tokenizer')


# In[ ]:
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

#categories = [ "alt.atheism", "talk.religion.misc", "comp.graphics",]

remove = ("headers", "footers", "quotes")

# newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, 
#                                        random_state=238, remove=remove)
# val_ng = fetch_20newsgroups(subset='test',  categories=categories, shuffle=True, 
#                                        random_state=42, remove=remove)

newsgroups = fetch_20newsgroups(subset='train', shuffle=True, 
                                       random_state=238, remove=remove)
# test_ng = fetch_20newsgroups(subset='test', shuffle=True, 
#                                       random_state=238, remove=remove)

data_train, data_val, label_train, label_val = train_test_split(newsgroups.data, newsgroups.target, test_size=0.1, random_state=42)

"""
label_train = label_train.tolist()
label_val = label_val.tolist()
tokenized_data = [tokenizer.tokenize(data) for data in data_train]
data_length = [len(i) for i in tokenized_data]
del_data_index = sorted(range(len(data_length)), key=lambda x: data_length[x])[-1000:]
for index in sorted(del_data_index, reverse=True):
    del data_train[index]
    del label_train[index]

tokenized_data = [tokenizer.tokenize(data) for data in data_val]
data_length = [len(i) for i in tokenized_data]
del_data_index = sorted(range(len(data_length)), key=lambda x: data_length[x])[-100:]
for index in sorted(del_data_index, reverse=True):
    del data_val[index]
    del label_val[index]
"""
train_ng = {'data': data_train, 'target': label_train}
val_ng = {'data': data_val, 'target': label_val}

#data_train = data_train.pop(66)
#label_train = label_train.tolist().pop(66)

# print('size of training set:', len(train_ng.data))
# print('size of validation set:', len(val_ng.data))
# print('classes:', train_ng.target_names)
print('size of training set:', len(data_train))
print('size of training set:', len(label_train))
print('size of validation set:', len(data_val))
print('size of validation set:', len(label_val))
print('classes:', newsgroups.target_names)

# data_train = train_ng.data
# label_train = train_ng.target
# data_test = test_ng.data
# label_test = test_ng.target


# In[ ]:


#length = [len(l) for l in train_ng.data]
length = [len(l) for l in train_ng['data']]
plt.hist(length, bins=1000)
plt.title("Frequency Histogram")
plt.xlabel("length")
plt.ylabel("Frequency")
plt.xlim((0, 5000))
plt.xticks(np.arange(0, 5000, step=500))
plt.show()


# In[ ]:


#print('Initializing BertTokenizer')

#BERTMODEL='bert-base-uncased'
#CACHE_DIR='transformers-cache'

#tokenizer = BertTokenizer.from_pretrained(BERTMODEL, cache_dir=CACHE_DIR,
#                                          do_lower_case=True)

# In[ ]:


if torch.cuda.is_available():    
    device = torch.device("cuda:0") # specify  devicethe
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[ ]:
class newsDataset(torch.utils.data.Dataset):
    
    def __init__(self, docs, targets, tokenizer, max_len):
        self.docs = docs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, item):
        doc = str(self.docs[item])
        target = self.targets[item]

        encoding = self.tokenizer(
            doc, 
            padding = 'max_length', 
            truncation = True, 
            max_length = self.max_len,
            return_tensors="pt")   # return tensor   

        return {
          'news_text': doc,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

# get the input_ids_list, attention_mask_list of the long document. Each element in the list correspondes to a segment.
# also get target for each segment and the number of segments.
def create_data_loader(newsgroups, tokenizer, max_len, batch_size):
    ds = newsDataset(
        docs=newsgroups['data'],
        #docs=newsgroups.data,
        targets=newsgroups['target'],
        #targets=newsgroups.target,
        tokenizer=tokenizer,
        max_len=max_len
      )

    return DataLoader(
        ds,
        batch_size=batch_size,
      )


# In[ ]:


BATCH_SIZE = 8
#BATCH_SIZE = 83
MAX_LEN = 1024

train_data_loader = create_data_loader(train_ng, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_ng, tokenizer, MAX_LEN, BATCH_SIZE)


# In[ ]:

num_labels = len(set(train_ng['target']))
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',gradient_checkpointing=False,attention_window=512,num_labels=num_labels)
model = model.to(device)


# In[ ]:


# Hyperparameters
EPOCHS = 40
#EPOCHS = 2

optimizer = AdamW(model.parameters(), lr=2e-5)
#optimizer = AdamW(model.parameters(), lr=0.1)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# In[ ]:


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    t0 = time.time()

    for batch in data_loader:
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device) 
        
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        
        outputs = F.softmax(outputs.logits, dim=1)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(float(loss.item()))
        #losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  
        
        torch.cuda.empty_cache()
    
    print(f"time = {time.time()-t0:.2f} secondes")
    t0 = time.time()
        
    return np.mean(losses), float(correct_predictions / n_examples)


# In[ ]:


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device) 
            
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
            outputs = F.softmax(outputs.logits, dim=1)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(float(loss.item()))
            #losses.append(loss.item())
            torch.cuda.empty_cache()
    return np.mean(losses), float(correct_predictions / n_examples)


# In[ ]:


history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_loss, train_acc = train_epoch(model,
                                        train_data_loader,    
                                        loss_fn, 
                                        optimizer, 
                                        device, 
                                        scheduler, 
                                        len(train_ng['data'])
                                        #len(train_ng.data)
                                        )

    
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_loss, val_acc = eval_model(
    model,
    val_data_loader,
    loss_fn, 
    device, 
    len(val_ng['data'])
    #len(val_ng.data)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    
    #print(history)
    
    #if val_acc > best_accuracy:
        #torch.save(model.state_dict(), 'best_model_state.bin')
        #best_accuracy = val_acc


# In[ ]:

fig, ax = plt.subplots()
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_xticks(np.arange(0, 50, 5))

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('longformer')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('longformer.png')
