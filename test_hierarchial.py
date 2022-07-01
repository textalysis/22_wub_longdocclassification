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


para = {#'datasets': ["Hyperpartisan", "20newsgroups", "ECtHR"],
        'datasets': ["20newsgroup"],
        'summarizer': ["none", "bert_summarizer", "text_rank"],
        'tokenizers': ["BERT", "longformer", "bigbird"],
        'batch_size': [16],
        'learning_rate': [2e-5],
        'chunk_lens': [256,512],
        'overlap_lens': [25, 50],
        'total_len': 4096,
        'epochs': 40,
        'max_len': 512,
        'model_names': ["ToBERT", "Longformer", "Bigbird", "BERT"],
        'sparse_max_lens': [1024, 2048, 4096],
        'attention_windows': [256, 512],
        'block_sizes': [64, 128],
        'truncations': ["head_tail", "tail", "head"]
}


for dataset in para["datasets"]:
    if dataset == "Hyperpartisan":
        data_train, data_val, data_test = get_dataset("Hyperpartisan")
    elif dataset == "20newsgroups":
        data_train, data_val, data_test = get_dataset("20newsgroups")
    else:
        data_train, data_val, data_test = get_dataset("ECtHR")
    
    for batch_size in para['batch_sizes']:
                for learning_rate in para['learning_rates']:
                    for model_name in para['model_names'][0]:
                        tokenizer = tokenize('BERT')
                        max_len = para['max_len']
                        total_len = para['total_len']
                        for chunk_len in para['chunk_lens']:
                            for overlap_len in para['overlap_lens']:
                                print("create data loader")
                                train_data_loader = create_data_loader("long", data_train, tokenizer, max_len, batch_size,
                                                                             approach="all", chunk_len=chunk_len, overlap_len=overlap_len, total_len=total_len)
                                val_data_loader = create_data_loader("long", data_val, tokenizer, max_len, batch_size,
                                                                           approach="all", chunk_len=chunk_len, overlap_len=overlap_len, total_len=total_len)
                                model = ToBERT(len(set(data_train['target'])))
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
                                filename = "{}_{}_{}_{}".format(dataset, model_name, chunk_len, overlap_len)
                                print("training")
                                try:
                                    trainer.trainer_hierarchical(para['epochs'], model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device, scheduler, filename)
                                except Exception as e:
                                    print("Exception")
                                    print(e)
