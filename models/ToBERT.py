import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel


class ToBERT(nn.Module):
    def __init__(self, n_classes):
        super(ToBERT, self).__init__()
        #self.bert = BertModel.from_pretrained('bert-base-uncased')  # bert model from huggingface
        self.bert = RobertaModel.from_pretrained("roberta-base")
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=8, batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        #self.drop = nn.Dropout(p=0.3)  # add dropout of 0.3 on top of bert output
        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)  # Linear layer as a classifier


    def forward(self, input_ids, attention_mask, lengt):

        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        chunks_emb = pooled_output.split_with_sizes(lengt)
        # input batch, sequence, hidden -- unsqueeze
        hid = [self.transformer_encoder(x.unsqueeze(0)).squeeze(0) for x in chunks_emb]
        f = torch.stack([torch.mean(t, dim=0) for t in hid])
        dense = self.relu(self.dense(f))
        output = self.out(dense)
        # F.softmax(output, dim=1) 
        return output
