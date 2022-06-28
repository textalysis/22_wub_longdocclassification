import torch
import torch.nn as nn
from transformers import BigBirdForSequenceClassification


class Bigbird(nn.Module):
    def __init__(self, num_labels, block_size):
        super(Bigbird, self).__init__()
        self.bigbird = BigBirdForSequenceClassification.from_pretrained\
        ('google/bigbird-roberta-base', gradient_checkpointing=False, block_size = block_size, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bigbird(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        return outputs.logits
