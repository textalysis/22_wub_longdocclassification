import utils
from models.BERT import BERT
from models.Bigbird import Bigbird
from models.Longformer import Longformer
from models.PoBERT import PoBERT
from models.RoBERT import RoBERT
from models.ToBERT import ToBERT
import train
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import trainer


para = {'datasets': ["imdb", "20newsgroups"],
        'summarizer': ["none", "bert_summarizer", "text_rank"],
        'tokenizers': ["BERT", "longformer", "bigbird"],
        'batch_sizes': [4, 8, 16],
        'learning_rates': [3e-4, 1e-4, 5e-5],
        'chunk_lens': [512, 256],
        'overlap_lens': [25, 50],
        'total_len': 4096,
        'epochs': 60,
        'batch_size': [8, 16, 32],
        'lr': [3e-4, 1e-4, 5e-5],
        'max_len': 512,
        'models': ["PoBERT_mean", "PoBERT_max", "RoBERT", "ToBERT", "Longformer", "Bigbird", "BERT"],
        'sparse_max_lens': [1024, 2048, 4096],
        'attention_windows':[256, 512],
        'block_sizes': [64, 128],
        'truncations': ["head_tail", "tail", "head"]
}


for dataset in para["datasets"]:
    if dataset == "imdb":
        data_train, data_val = utils.get_dataset("imdb")
    # dataset == "20newsgroups"
    else:
        data_train, data_val = utils.get_dataset("20newsgroups")

    for summarizer in para["summarizer"]:
        if summarizer == "bert_summarizer":
            data_train['data'] = utils.bert_summarizer(data_train['data'])
            data_val['data'] = utils.bert_summarizer(data_val['data'])
        elif summarizer == "text_rank":
            data_train['data'] = utils.text_rank(data_train['data'])
            data_val['data'] = utils.text_rank(data_val['data'])
        else:
            for batch_size in para['batch_sizes']:
                for learning_rate in para['learning_rates']:
                    for model in para['models'][0:4]:
                        tokenizer = utils.tokenize('BERT')
                        max_len = para['max_len']
                        total_len = para['total_len']
                        for chunk_len in para['chunk_lens']:
                            for overlap_len in para['overlap_lens']:
                                train_data_loader = utils.create_data_loader("long", "bert", data_train, tokenizer, max_len, batch_size,
                                                                             approach="all", chunk_len=chunk_len, overlap_len=overlap_len, total_len=total_len)
                                val_data_loader = utils.create_data_loader("long", "bert", data_val, tokenizer, max_len, batch_size,
                                                                           approach="all", chunk_len=chunk_len, overlap_len=overlap_len, total_len=total_len)
                                if model == "PoBERT_mean":
                                    model = PoBERT(len(set(data_train['target'])), pooling_method="mean")
                                elif model == "PoBERT_max":
                                    model = PoBERT(len(set(data_train['target'])), pooling_method="max")
                                elif model == "ToBERT":
                                    model = ToBERT(len(set(data_train['target'])))
                                else:
                                    model = RoBERT(len(set(data_train['target'])))
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
                                filename = "{}_{}_{}_{}_{}_{}_{}_{}.txt".format(dataset, summarizer, batch_size, learning_rate, model, chunk_len, overlap_len, device)
                                trainer.trainer_hierarchical(para['epochs'], model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device, scheduler, filename)


                    for model in para['models'][4:6]:
                        for spase_max_len in para['sparse_max_lens']:
                            if model == "Longformer":
                                for attention_window in para['attention_windows']:
                                    tokenizer = utils.tokenize('longformer')
                                    max_len = spase_max_len
                                    train_data_loader = utils.create_data_loader("short", "longformer", data_train, tokenizer, max_len,
                                                                                 batch_size)
                                    val_data_loader = utils.create_data_loader("short", "longformer", data_val, tokenizer, max_len,
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
                                    filename = "{}_{}_{}_{}_{}_{}_{}.txt".format(dataset, summarizer, batch_size,
                                                                                    learning_rate, model, attention_window, device)
                                    trainer.trainer(para['epochs'], model, train_data_loader,
                                                                 val_data_loader, data_train, data_val, loss_fn,
                                                                 optimizer, device, scheduler, filename)

                            else:
                                for block_size in para['block_sizes']:
                                    tokenizer = utils.tokenize("bigbird")
                                    max_len = spase_max_len
                                    train_data_loader = utils.create_data_loader("short", "bigbird", data_train, tokenizer,
                                                                     max_len, batch_size)
                                    val_data_loader = utils.create_data_loader("short", "bigbird", data_val, tokenizer,
                                                                    max_len, batch_size)
                                    model = Bigbird(block_size=block_size,
                                                       num_labels=len(set(data_train['target'])))
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
                                    filename = "{}_{}_{}_{}_{}_{}_{}.txt".format(dataset, summarizer, batch_size,
                                                                     learning_rate, model, block_size, device)
                                    trainer.trainer(para['epochs'], model, train_data_loader,
                                        val_data_loader, data_train, data_val, loss_fn,
                                        optimizer, device, scheduler, filename)

                    for model in para['models'][6]:
                        for truncation in para['truncations']:
                            tokenizer = utils.tokenize('BERT')
                            max_len = para['max_len']
                            if truncation == "head":
                                train_data_loader = utils.create_data_loader("short", "bert", data_train, tokenizer, max_len, batch_size)
                                val_data_loader = utils.create_data_loader("short", "bert", data_val, tokenizer, max_len, batch_size)
                                model = BERT(len(set(data_train['target'])))
                                device = train.available_device()
                                model = model.to(device)
                                optimizer = AdamW(model.parameters(), lr=learning_rate)
                                total_steps = len(train_data_loader) * para['epochs']
                                scheduler = get_linear_schedule_with_warmup(
                                    optimizer,
                                    num_warmup_steps=0,
                                    num_training_steps=total_steps)
                                loss_fn = nn.CrossEntropyLoss().to(device)
                                filename = "{}_{}_{}_{}_{}_{}_{}.txt".format(dataset, summarizer, batch_size,
                                                                         learning_rate, model, truncation, device)
                                trainer.trainer(para['epochs'], model, train_data_loader,
                                                val_data_loader, data_train, data_val, loss_fn,
                                                optimizer, device, scheduler, filename)

                            else:
                                train_data_loader = utils.create_data_loader("long", "bert", data_train, tokenizer,
                                                                             max_len, batch_size, approach=truncation)
                                val_data_loader = utils.create_data_loader("long", "bert", data_val, tokenizer,
                                                                           max_len, batch_size, approach=truncation)
                                model = BERT(len(set(data_train['target'])))
                                device = train.available_device()
                                model = model.to(device)
                                optimizer = AdamW(model.parameters(), lr=learning_rate)
                                total_steps = len(train_data_loader) * para['epochs']
                                scheduler = get_linear_schedule_with_warmup(
                                    optimizer,
                                    num_warmup_steps=0,
                                    num_training_steps=total_steps)
                                loss_fn = nn.CrossEntropyLoss().to(device)
                                filename = "{}_{}_{}_{}_{}_{}_{}.txt".format(dataset, summarizer, batch_size,
                                                                         learning_rate, model, truncation, device)
                                trainer.trainer_hierarchical(para['epochs'], model, train_data_loader,
                                                val_data_loader, data_train, data_val, loss_fn,
                                                optimizer, device, scheduler, filename)


