from utils import *
from models.BERT import BERT
from models.PoBERT import PoBERT
from models.RoBERT import RoBERT
from models.ToBERT import ToBERT
from models.Bigbird import Bigbird
from models.Longformer import Longformer
import train
#from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AdamW
import torch.nn as nn
import trainer


para = {'datasets': ["20newsgroups"],
        'tokenizers': ["BERT", "longformer", "bigbird"],
        'batch_sizes': [16],
        'learning_rate': 2e-5,
        'epochs': 15,
        'model_names': ["Bigbird","Longformer"],
        'sparse_max_lens': [1024, 2048, 4096],
        'attention_windows': [256, 512],
        'block_sizes': [64, 128]
}


for dataset in para["datasets"]:
    if dataset == "imdb":
        print("importing dataset imdb")
        data_train, data_val = get_dataset("imdb")
        print("imdb dataset imported")
    # dataset == "20newsgroups"
    else:
        data_train, data_val = get_dataset("20newsgroups")

    for batch_size in para['batch_sizes']:
                #for learning_rate in para['learning_rates']:
                    learning_rate = para['learning_rate'] 
                    for model_name in para['model_names'][0:2]:
                        for spase_max_len in para['sparse_max_lens']:
                            if model_name == "Longformer":
                                for attention_window in para['attention_windows']:
                                    tokenizer = tokenize('longformer')
                                    max_len = spase_max_len
                                    train_data_loader = create_data_loader("short", data_train, tokenizer, max_len,
                                                                                 batch_size)
                                    val_data_loader = create_data_loader("short", data_val, tokenizer, max_len,
                                                                               batch_size)
                                    model = Longformer(attention_window=attention_window,num_labels=len(set(data_train['target'])))
                                    device = train.available_device()
                                    model = model.to(device)
                                    optimizer = AdamW(model.parameters(), lr=learning_rate)
                                    total_steps = len(train_data_loader) * para['epochs']
                                    scheduler = get_linear_schedule_with_warmup(
                                        optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps
                                    )
                                    loss_fn = nn.CrossEntropyLoss().to(device)
                                    filename = "{}_{}_{}_{}".format(dataset,learning_rate, model_name, attention_window)
                                    try:
                                        trainer.trainer(para['epochs'], model, train_data_loader,
                                                                 val_data_loader, data_train, data_val, loss_fn,
                                                                 optimizer, device, scheduler, filename)
                                    except Exception as e:
                                        print("Exception")
                                        print(e)
                                       
                            else:
                                for block_size in para['block_sizes']:
                                    tokenizer = tokenize("bigbird")
                                    max_len = spase_max_len
                                    train_data_loader = create_data_loader("short", data_train, tokenizer,
                                                                     max_len, batch_size)
                                    val_data_loader = create_data_loader("short", data_val, tokenizer,
                                                                    max_len, batch_size)
                                    model = Bigbird(block_size=block_size,num_labels=len(set(data_train['target'])))         
                                    device = train.available_device()
                                    model = model.to(device)
                                    optimizer = AdamW(model.parameters(), lr=learning_rate)
                                    total_steps = len(train_data_loader) * para['epochs']
                                    scheduler = get_linear_schedule_with_warmup(
                                        optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps
                                        )
                                    loss_fn = nn.CrossEntropyLoss().to(device)
                                    filename = "{}_{}_{}_{}".format(dataset,
                                                                     learning_rate, model_name, block_size)
                                    try:
                                        trainer.trainer(para['epochs'], model, train_data_loader,
                                        val_data_loader, data_train, data_val, loss_fn,
                                        optimizer, device, scheduler, filename)
                                    except Exception as e:
                                        print("Exception")
                                        print(e)

