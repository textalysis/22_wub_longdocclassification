from models.BERT import BERT
from models.PoBERT import PoBERT
from models.RoBERT import RoBERT
from models.ToBERT import ToBERT
from models.Bigbird import Bigbird
from models.Longformer import Longformer

def count_parameters(model):
    s = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return s

# longformer(1024,10) 148667146
L = Longformer(1024,20)
longformer_para = count_parameters(L)
print("longformer_para",longformer_para)


bert = BERT(20)
bert_para =  count_parameters(bert)
print("bert_para",bert_para)


bigbird =  Bigbird(20,128)
bigbird_para =  count_parameters(bigbird)
print("bigbird_para", bigbird_para)


tobert = ToBERT(20)
tobert_para = count_parameters(tobert)
print("tobert_para",tobert_para)




