import train
import trainer
import torch.nn as nn
from utils import *
from models.BERT import BERT

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
data_train, data_val, data_test = get_dataset("Hyperpartisan")
loss_fn = nn.CrossEntropyLoss()
num_labels = 2
tokenizer = tokenize('BERT')
batch_size = 16
max_len = 512
truncation = "tail"
test_data_loader = create_data_loader("long", data_test, tokenizer, max_len, batch_size, approach=truncation)
class_type = "single_label"

model = BERT(num_labels)

model.load_state_dict(torch.load(os.path.join('best_models', "Hyperpartisan_BERT_tail_2_best.bin")))
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
print(test_acc)
