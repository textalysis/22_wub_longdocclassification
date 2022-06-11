import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class RoBERT(nn.Module):
    def __init__(self, n_classes):
        super(RoBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 100, num_layers=1, bidirectional=False)
        self.out = nn.Linear(100, n_classes)

    def forward(self, input_ids, attention_mask, lengt):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        chunks_emb = pooled_output.split_with_sizes(lengt)

        # tensor([len of first doc, len of second doc, ..])
        # tensor([1, 1, 4, 1]
        seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])

        # padding to get same size for each doc
        # if len [1, 1, 4, 1]
        # ([[1][pad][pad][pad]],[[2][pad][pad][[pad]],[[3.1][3.2][3.3][3.4]],[[4][pad][pad][pad]])
        batch_emb_pad = nn.utils.rnn.pad_sequence(chunks_emb, padding_value=-2, batch_first=True)
        #([[1][2][3.1][4]],[[pad][pad][3.2][[pad]],[[pad][pad][3.3][pad]],[[pad][pad][3.4][pad]])
        batch_emb = batch_emb_pad.transpose(0, 1)
        # optimize the computations
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

        packed_output, (h_t, h_c) = self.lstm(lstm_input, )

        h_t = h_t.view(-1, 100)

        output = self.out(h_t)

        return F.softmax(output, dim=1)