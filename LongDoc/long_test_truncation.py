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


para = {'datasets': ["Hyperpartisan", "20newsgroups","ECtHR"],
        #'datasets': ["ECtHR","Hyperpartisan"],
        'seeds': [1, 2, 3, 4, 5],
        #'seeds': [1,3,5],
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
        device = torch.device("cuda:0")  # specify  device
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


for seed in para["seeds"]:
    train.seed_everything(seed)

    for dataset in para["datasets"]:
        if dataset == "Hyperpartisan":
            data_train, data_val, data_test = get_dataset("Hyperpartisan")
            loss_fn = nn.CrossEntropyLoss()
            num_labels = 2
            class_type = "single_label"
            print(len(data_train["data"]), len(data_train["target"]))
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

        for truncation in para['truncations']:
            tokenizer = tokenize('BERT')
            if truncation == "head":
                #for summarizer in para["summarizer"]:
                    summarizer = para["summarizer"][0]
                    if summarizer == "bert_summarizer":
                        #print(data_train['data'][0])
                        #data_train['data'] = [x for i,x in enumerate(data_train['data']) if i not in [606, 4130]]
                        #data_val['data'] = [x for i,x in enumerate(data_val['data']) if i!=1039]
                        #data_test['data'] = [x for i,x in enumerate(data_test['data']) if i not in [4392,6229]]
                        #document coding problem garbled
                        if dataset == "20newsgroups":
                            data_train_sum = [x for i,x in enumerate(data_train['data']) if i not in [606, 4130]]
                            data_train_sum = bert_summarizer(data_train_sum)
                            data_train['data'] = data_train_sum[0:606]+data_train['data'][606]+data_train_sum[606:4130]+data_train['data'][4130]+data_train_sum[4130::]  
                            data_val_sum = [x for i,x in enumerate(data_val['data']) if i!=1039]
                            data_val_sum = bert_summarizer(data_val_sum)
                            data_val['data'] = data_val_sum[0:1039]+data_val['data'][1039]+data_val_sum[1039::]
                            data_test_sum = [x for i,x in enumerate(data_test['data']) if i not in [4392,6229]]
                            data_test_sum = bert_summarizer(data_test_sum)
                            data_test['data'] = data_test_sum[0:4392]+data_test['data'][4392]+data_test_sum[4392:6229]+data_test['data'][6229]+data_test_sum[6229::]            
                        else:
                            data_train['data'] = bert_summarizer(data_train['data'])
                            data_val['data'] = bert_summarizer(data_val['data'])
                            data_test['data'] = bert_summarizer(data_test['data'])
                    if summarizer == "text_rank":
                        data_train['data'] = text_rank(data_train['data'])
                        data_val['data'] = text_rank(data_val['data'])
                        data_test['data'] = text_rank(data_test['data'])

                    test_data_loader = create_data_loader("short", data_test, tokenizer, max_len, batch_size)
                    model = BERT(num_labels)
                    device = available_device()
                    filename = "{}_{}_{}_{}_{}".format(dataset, model_name, truncation, summarizer, seed)
                    model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
                    model = model.to(device)
                    loss_fn = loss_fn.to(device)
                    print('#' * 10)
                    print(filename)
                    if class_type == "multi_label":
                        # use micro f1 as metric
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
                test_data_loader = create_data_loader("long", data_test, tokenizer, max_len, batch_size, approach=truncation)
                model = BERT(num_labels)
                device = available_device()         
                filename = "{}_{}_{}_{}".format(dataset, model_name, truncation, seed)
                model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
                model = model.to(device)
                loss_fn = loss_fn.to(device)
                print('#' * 10)
                print(filename)                


                if class_type == "multi_label":
                    # use micro f1 as metric
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
