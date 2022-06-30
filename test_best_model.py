from utils import *
import train
from sklearn.datasets import fetch_20newsgroups
from transformers import get_linear_schedule_with_warmup, AdamW
import torch
import torch.nn as nn
import trainer
from models.ToBERT import ToBERT

tokenizer = tokenize('BERT')
total_len = 4096
max_len = 512
data_val = fetch_20newsgroups(subset='test', shuffle=True, random_state=238)
#model = torch.load("best_models/20newsgroups_ToBERT_512_50_best.bin")
device = torch.device("cuda:0")
model = ToBERT(20)
model.load_state_dict(torch.load("best_models/20newsgroups_ToBERT_512_25_best.bin"))
model = model.to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
test_data_loader = create_data_loader("long", data_val, tokenizer, max_len, 16,"all", 512, 50,4096)
loss, acc, real_values,pred = train.hierarchical_eval_model(model, test_data_loader, loss_fn, device, 7532)
print(loss, acc) 
