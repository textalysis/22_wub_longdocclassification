import torch
import torch.nn as nn
from transformers import LongformerForSequenceClassification


class Longformer(nn.Module):
    def __init__(self, attention_window, num_labels):
        super(Longformer, self).__init__()
        self.longformer = LongformerForSequenceClassification.from_pretrained\
            ('allenai/longformer-base-4096', gradient_checkpointing=False,
             attention_window=attention_window, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        return outputs.logits
