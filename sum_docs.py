from utils import *

for dataset in ["20newsgroups","ECtHR","Hyperpartisan"]:
    if dataset == "Hyperpartisan":
        data_train, data_val, data_test = get_dataset("Hyperpartisan")
        data_train = bert_summarizer(data_train['data'])
        #data_val = bert_summarizer(data_val['data'])
        #data_test = bert_summarizer(data_test['data'])
        with open("data/bert_sum/Hyperpartisan/data_train.txt", "a") as f:
            for line in data_train:
                f.write(line+'\n')
