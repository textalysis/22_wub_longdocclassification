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
from sklearn.metrics import classification_report


para = {'datasets': ["Hyperpartisan", "20newsgroups", "ECtHR"],
        #'datasets': ["20newsgroups"],
        'seeds': [1,2,3,4,5],
        'summarizer': ["none", "bert_summarizer", "text_rank"],
        'tokenizers': ["BERT", "longformer", "bigbird"],
        'batch_size': 16,
        'learning_rate': 2e-5,
        'chunk_lens': [256,512],
        'overlap_lens': [25, 50],
        'total_len': 4096,
        'epochs': 40,
        'max_len': 512,
        'model_names': ["ToBERT", "Longformer", "Bigbird", "BERT"],
       #'model_names': ["Bigbird"],
        'sparse_max_lens': [1024, 2048, 4096],
        #'sparse_max_lens': [1024],
        'attention_windows': [256, 512],
        #'attention_windows': [512],
        'block_sizes': [64, 128],
        #'block_sizes': [128],
        'truncations': ["head_tail", "tail", "head"]
}

batch_size = para["batch_size"]
learning_rate = para["learning_rate"]
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



for dataset in para["datasets"]:
    if dataset == "Hyperpartisan":
        data_train, data_val, data_test = get_dataset("Hyperpartisan")
        loss_fn = nn.CrossEntropyLoss()
        num_labels = 2
        class_type = "single_label"
    elif dataset == "20newsgroups":
        data_train, data_val, data_test = get_dataset("20newsgroups")
        loss_fn = nn.CrossEntropyLoss()
        num_labels = 20
        class_type = "single_label"
    else:
        data_train, data_val, data_test = get_dataset("ECtHR")
        loss_fn = nn.BCEWithLogitsLoss()
        num_labels = 10
        class_type = "multi_label"
    print("datasets imported")

    for model_name in para['model_names'][1:3]:
        for spase_max_len in para['sparse_max_lens']: 
            if model_name == "Longformer":
                for attention_window in para['attention_windows']:
                    for seed in para["seeds"]:
                        tokenizer = tokenize('longformer')
                        data_test = filter_testset(tokenizer, data_test)
                        max_len = spase_max_len
                        test_data_loader = create_data_loader("short", data_test, tokenizer, max_len, batch_size)
                        model = Longformer(attention_window=attention_window, num_labels=num_labels)
                        device = available_device()
                        filename = "{}_{}_{}_{}_{}".format(dataset, model_name, spase_max_len, attention_window, seed)
                        try:
                            model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
                        except Exception as e:
                            print("Exception")
                            print(e)
                        model = model.to(device)
                        loss_fn = loss_fn.to(device)
                        print('#' * 10)
                        print(filename)
                        if class_type == "multi_label":
                            test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
                                    model,
                                    test_data_loader,
                                    loss_fn,
                                    device,
                                    len(data_test['data']),
                                    class_type
                                )
                            test_report = classification_report(test_real, test_pred, output_dict=True)
                            print(f'test_f1_score {test_report["micro avg"]["f1-score"]}' + "\n")
                        else:
                            test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
                                model,
                                test_data_loader,
                                loss_fn,
                                device,
                                len(data_test['data']),
                                class_type
                            )
                            test_report = classification_report(test_real, test_pred, output_dict=True)
                            print(f'test_accuracy {test_acc} macro_avg {test_report["macro avg"]} weighted_avg {test_report["weighted avg"]}'+"\n")

            else:
                for block_size in para['block_sizes']:
                    for seed in para["seeds"]:
                        tokenizer = tokenize("bigbird")
                        data_test = filter_testset(tokenizer, data_test)
                        max_len = spase_max_len
                        test_data_loader = create_data_loader("short", data_test, tokenizer, max_len, batch_size)
                        model = Bigbird(block_size=block_size,num_labels=num_labels)
                        device = available_device()
                        filename = "{}_{}_{}_{}_{}".format(dataset, model_name, spase_max_len, block_size, seed)
                        try:
                            model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
                        except Exception as e:
                            print("Exception")
                            print(e)
                        model = model.to(device)
                        loss_fn = loss_fn.to(device)
                        print('#' * 10)
                        print(filename)
                        if class_type == "multi_label":
                            test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
                                    model,
                                    test_data_loader,
                                    loss_fn,
                                    device,
                                    len(data_test['data']),
                                    class_type
                                )
                            test_report = classification_report(test_real, test_pred, output_dict=True)
                            print(f'test_f1_score {test_report["micro avg"]["f1-score"]}' + "\n")
                        else:
                            test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
                                model,
                                test_data_loader,
                                loss_fn,
                                device,
                                len(data_test['data']),
                                class_type
                            )
                            test_report = classification_report(test_real, test_pred, output_dict=True)
                            print(f'test_accuracy {test_acc} macro_avg {test_report["macro avg"]} weighted_avg {test_report["weighted avg"]}'+"\n")
