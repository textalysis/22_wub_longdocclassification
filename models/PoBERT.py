import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel

# pooling_method, dropout
class PoBERT(nn.Module):
    def __init__(self, n_classes, pooling_method="mean"):
        super(PoBERT, self).__init__()
        self.pooling_method = pooling_method
        #self.bert = BertModel.from_pretrained('bert-base-uncased')  # bert model from huggingface
        self.bert = RobertaModel.from_pretrained("roberta-base")
        self.drop = nn.Dropout(p=0.3)  # add dropout of 0.3 on top of bert output
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)  # Linear layer as a classifier

    def forward(self, input_ids, attention_mask, lengt):

        # for example, when batch_size=4, four document, doc 1, doc 2, doc 4 are less than 512 tokens, doc 3 about 4*512 tokens
        # input_ids:  tensor([[ doc 1], [doc 2], [doc 3 seg 1], [ doc 3 seg 2], [doc 3 seg 3], [doc 3 seg 4], [doc 4]])

        # pooled output: CLS token
        # pooled_output shape: (number of segments of all documents in one batch * 786)
        # In this small example, 7 * 786
        # here not batch_size * 786 because of segmentation of long document
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        # split according to the number of segments
        # then know which document the pooled_output belongs to
        # chunks_emb (tensor([[doc1]]), tensor([[doc2]]), tensor([[doc3 seg1], [doc3 seg2], [doc3 seg3], [doc4 seg4]]), tensor([[doc4]])
        chunks_emb = pooled_output.split_with_sizes(lengt)

        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x, dim=0) for x in chunks_emb])
        elif self.pooling_method == "max":
            # torch.max return (value, indice)
            # here [0] to get the value
            emb_pool = torch.stack([torch.max(x, dim=0)[0] for x in chunks_emb])

        output = self.out(self.drop(emb_pool))
        # output = self.out(emb_pool)
        return F.softmax(output, dim=1)
