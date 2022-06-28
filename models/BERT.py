import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel

class BERT(nn.Module):
    def __init__(self, n_classes):
        super(BERT, self).__init__()
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = RobertaModel.from_pretrained("roberta-base")
        #self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        #output = self.out(self.drop(pooled_output))
        output = self.out(pooled_output)
        return F.softmax(output, dim=1)
