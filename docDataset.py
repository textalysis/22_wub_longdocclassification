import torch
from torch.utils.data import Dataset

class docDataset(Dataset):

    def __init__(self, docs, targets, tokenizer, max_len, model_type):
        self.docs = docs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_type = model_type

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, item):
        doc = str(self.docs[item])
        target = self.targets[item]

        if self.model_type == "bert":
            encoding = self.tokenizer.encode_plus(
                doc,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                return_token_type_ids=False,
                padding='max_length',
                return_overflowing_tokens=False,
                return_attention_mask=True,
                return_tensors='pt',
            )

        else:
            encoding = self.tokenizer(
                doc,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt")  # return tensor


        return {
            'news_text': doc,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }