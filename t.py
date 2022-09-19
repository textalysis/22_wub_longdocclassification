import os

summarizer = "bert_summarizer"
dataset = "20newsgroups"
summarizer_path = os.path.join('data', "{}".format(summarizer), "{}".format(dataset))
                  
with open(os.path.join(summarizer_path, "data_train_sum.txt")) as f:
         data_train = f.readlines()

print(len(data_train))
