import train
import trainer
import torch.nn as nn
from utils import *
from models.BERT import BERT
from sklearn.metrics import classification_report

def available_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # specify  device
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

device = available_device()
"""
data_train, data_val, data_test = get_dataset("ECtHR")
loss_fn = nn.BCEWithLogitsLoss()
num_labels = 10
class_type = "multi_label"
"""
data_train, data_val, data_test = get_dataset("20newsgroups")
loss_fn = nn.CrossEntropyLoss()
num_labels = 20
class_type = "single_label"

batch_size = 16
max_len = 512
truncation = "tail"
tokenizer = tokenize('BERT')
test_data_loader = create_data_loader("long", data_test, tokenizer, max_len, batch_size, approach=truncation)


model = BERT(num_labels)

model.load_state_dict(torch.load(os.path.join('best_models', "20newsgroups_BERT_tail_3_best.bin")))
model = model.to(device)
test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
                model,
                test_data_loader,
                loss_fn,
                device,
                len(data_test['data']),
                class_type
            )
#test_report = classification_report(test_real, test_pred, output_dict=True)
#test_micro_f1_score = test_report["micro avg"]["f1-score"]
print(test_acc)
