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

para = {#'datasets': ["Hyperpartisan", "20newsgroups","ECtHR"],
        'datasets': ["20newsgroups"],
        'seeds': [1, 2, 3, 4, 5],
        #'seeds': [5],
        #'summarizer': ["none", "bert_summarizer", "text_rank"],
        'summarizer': ["bert_summarizer", "text_rank"],
        'tokenizers': ["BERT", "longformer", "bigbird"],
        'batch_size': 16,
        'learning_rate': 2e-5,
        'chunk_lens': [256,512],
        'overlap_lens': [25, 50],
        'total_len': 4096,
        'epochs': 40,
        'max_len': 512,
        'model_names': ["ToBERT", "Longformer", "Bigbird", "BERT"],
        'sparse_max_lens': [1024, 2048, 4096],
        'attention_windows': [256, 512],
        'block_sizes': [64, 128],
        'truncations': ["head", "head_tail", "tail"]
}

batch_size = para["batch_size"]
learning_rate = para["learning_rate"]
model_name = para["model_names"][3]
max_len = para["max_len"]
total_len = para["total_len"]

def available_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:2")  # specify  device
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


dataset = "20newsgroups"
summarizer = "text_rank"
data_train, data_val, data_test = get_dataset("20newsgroups")
loss_fn = nn.CrossEntropyLoss()
num_labels = 20
class_type = "single_label"
data_train_sum = [x for i,x in enumerate(data_train['data']) if i not in [606, 4130]]
data_train_sum = text_rank(data_train_sum)
data_train['data'] = data_train_sum[0:606]+[data_train['data'][606]]+data_train_sum[606:4130]+[data_train['data'][4130]]+data_train_sum[4130::]  
data_val_sum = [x for i,x in enumerate(data_val['data']) if i!=1039]
data_val_sum = text_rank(data_val_sum)
data_val['data'] = data_val_sum[0:1039]+[data_val['data'][1039]]+data_val_sum[1039::]
data_test_sum = [x for i,x in enumerate(data_test['data']) if i not in [4392,6229]]
data_test_sum = text_rank(data_test_sum)
data_test['data'] = data_test_sum[0:4392]+[data_test['data'][4392]]+data_test_sum[4392:6229]+[data_test['data'][6229]]+data_test_sum[6229::]   


for seed in para["seeds"]:
        train.seed_everything(seed)
        for truncation in para['truncations']:
            tokenizer = tokenize('BERT')
            if truncation == "head":
                #for summarizer in para["summarizer"]:
                    
                    train_data_loader = create_data_loader("short", data_train, tokenizer, max_len, batch_size)
                    val_data_loader = create_data_loader("short",  data_val, tokenizer, max_len, batch_size)
                    test_data_loader = create_data_loader("short", data_test, tokenizer, max_len, batch_size)
                    model = BERT(num_labels)
                    device = available_device()
                    model = model.to(device)
                    optimizer = AdamW(model.parameters(), lr=learning_rate)
                    total_steps = len(train_data_loader) * para['epochs']
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=total_steps)
                    loss_fn = loss_fn.to(device)
                    filename = "{}_{}_{}_{}_{}".format(dataset, model_name, truncation, summarizer, seed)
                    try:
                        if class_type == "multi_label":
                            trainer.trainer_multi_label(para['epochs'], model, train_data_loader, val_data_loader, data_train, data_val, loss_fn,
                                optimizer, device, scheduler, filename, class_type, test_data_loader, data_test)
                        else:
                            trainer.trainer(para['epochs'], model, train_data_loader, val_data_loader, data_train, data_val, loss_fn,
                                optimizer, device, scheduler, filename, class_type, test_data_loader, data_test)
                    except Exception as e:
                        print("Exception")
                        print(e)
            """
            else:
                train_data_loader = create_data_loader("long", data_train, tokenizer, max_len, batch_size, approach=truncation)
                val_data_loader = create_data_loader("long", data_val, tokenizer, max_len, batch_size, approach=truncation)
                test_data_loader = create_data_loader("long", data_test, tokenizer, max_len, batch_size, approach=truncation)
                model = BERT(num_labels)
                device = available_device()
                model = model.to(device)
                optimizer = AdamW(model.parameters(), lr=learning_rate)
                total_steps = len(train_data_loader) * para['epochs']
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps)
                loss_fn = loss_fn.to(device)
                filename = "{}_{}_{}_{}".format(dataset, model_name, truncation, seed)
                try:
                    if class_type == "multi_label":
                        trainer.trainer_multi_label(para['epochs'], model, train_data_loader, val_data_loader, data_train, data_val, loss_fn,
                            optimizer, device, scheduler, filename, class_type, test_data_loader, data_test)
                    else:
                        trainer.trainer(para['epochs'], model, train_data_loader, val_data_loader, data_train, data_val, loss_fn,
                            optimizer, device, scheduler, filename, class_type, test_data_loader, data_test)
                except Exception as e:
                    print("Exception")
                    print(e)
            """



