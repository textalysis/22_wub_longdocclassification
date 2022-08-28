from utils import *

for dataset in ["20newsgroups","ECtHR","Hyperpartisan"]:
    if dataset == "Hyperpartisan":
        data_train, data_val, data_test = get_dataset("ECtHR")
        data_train_sum = bert_summarizer(data_train['data'])
        data_val_sum = bert_summarizer(data_val['data'])
        data_test_sum = bert_summarizer(data_test['data'])
        """
        data_train_sum = [x for i,x in enumerate(data_train['data']) if i not in [606, 4130]]
        data_train_sum = bert_summarizer(data_train_sum)
        data_train_sum = data_train_sum[0:606]+[data_train['data'][606]]+data_train_sum[606:4130]+[data_train['data'][4130]]+data_train_sum[4130::]  
        data_val_sum = [x for i,x in enumerate(data_val['data']) if i!=1039]
        data_val_sum = bert_summarizer(data_val_sum)
        data_val_sum = data_val_sum[0:1039]+[data_val['data'][1039]]+data_val_sum[1039::]
        data_test_sum = [x for i,x in enumerate(data_test['data']) if i not in [4392,6229]]
        data_test_sum = bert_summarizer(data_test_sum)
        data_test_sum = data_test_sum[0:4392]+[data_test['data'][4392]]+data_test_sum[4392:6229]+[data_test['data'][6229]]+data_test_sum[6229::]            
        """                  
        
        with open(os.path.join('data', "data_train_sum.txt"), "a") as f:
            for line in data_train_sum:
                line.replace('\n','')
                f.write(line+'\n')


        with open(os.path.join('data', "data_val_sum.txt"), "a") as f:
            for line in data_val_sum:
                line.replace('\n','')        
                f.write(line+'\n')
        
        with open(os.path.join('data', "data_test_sum.txt"), "a") as f:
            for line in data_test_sum:
                line.replace('\n','')
                f.write(line+'\n')
