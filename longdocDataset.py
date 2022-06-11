import torch


# chunk_len, overlap_len
class longdocDataset(torch.utils.data.Dataset):
    def __init__(self, docs, targets, tokenizer, max_len, approach, chunk_len, overlap_len, total_len):
        self.docs = docs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.approach = approach
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        self.total_len = total_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        doc = str(self.docs[item])
        target = int(self.targets[item])

        # get the encoding of the first 512 tokens and the overflowing tokens if the document has more than the defined max length tokens
        encoding = self.tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_overflowing_tokens=True,
            return_tensors='pt')

        if self.approach == "all":
            long_token = self.long_terms_tokenizer(encoding, target, total_len=self.total_len)

        elif self.approach == "head_tail":
            remain = encoding['overflowing_tokens'].flatten()
            input_ids = encoding['input_ids'].flatten()
            if remain.shape[0] != 0:
                complete_tokens = torch.cat((input_ids, remain))
                complete_tokens = complete_tokens[complete_tokens!=101]
                complete_tokens = complete_tokens[complete_tokens!=102]
                start_token = torch.tensor([101], dtype=torch.long)
                end_token = torch.tensor([102], dtype=torch.long)
                input_ids = torch.cat((start_token,complete_tokens[:int((self.max_len-2)/2)], complete_tokens[-int((self.max_len-2)/2):],end_token))

            long_token = {'input_ids': input_ids,
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)}

        # self.approach == "tail"
        else:
            remain = encoding['overflowing_tokens'].flatten()
            input_ids = encoding['input_ids'].flatten()

            if remain.shape[0] != 0:
                complete_tokens = torch.cat((input_ids, remain))
                complete_tokens = complete_tokens[complete_tokens != 101]
                complete_tokens = complete_tokens[complete_tokens != 102]
                start_token = torch.tensor([101], dtype=torch.long)
                end_token = torch.tensor([102], dtype=torch.long)
                input_ids = torch.cat((start_token, complete_tokens[-(self.max_len - 2):], end_token))

            long_token = {'input_ids': input_ids,
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)
            }

        return long_token

    def long_terms_tokenizer(self, data_tokenize, target, total_len):
        input_ids_list = []
        attention_mask_list = []
        target_list = []

        # get the input_ids and attention mask of the first 512 tokens
        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        # get the input_ids of overflowing tokens
        remain = data_tokenize['overflowing_tokens'].flatten()
        remain = remain[0: (total_len - 512)]
        target = torch.tensor(target, dtype=torch.int)

        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        target_list.append(target)

        # segment the input_ids with overlapping
        # some tricks are used here to segment with overlapping
        if remain.shape[0] != 0:
            idxs = range(len(remain) + self.chunk_len)
            idxs = idxs[(self.chunk_len - self.overlap_len - 2)::(self.chunk_len - self.overlap_len - 2)]
            input_ids_first_overlap = previous_input_ids[-(self.overlap_len + 1):-1]
            start_token = torch.tensor([101], dtype=torch.long)
            end_token = torch.tensor([102], dtype=torch.long)

            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = torch.cat(
                        (input_ids_first_overlap, remain[:idx]))
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx - self.overlap_len):idx]

                previous_idx = idx

                nb_token = len(input_ids) + 2
                attention_mask = torch.ones(self.chunk_len, dtype=torch.long)
                attention_mask[nb_token:self.chunk_len] = 0
                input_ids = torch.cat((start_token, input_ids, end_token))

                if self.chunk_len - nb_token > 0:
                    padding = torch.zeros(
                        self.chunk_len - nb_token, dtype=torch.long)
                    input_ids = torch.cat((input_ids, padding))

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                target_list.append(target)

        # input_ids_list[:60] if len(input_ids_list) > 60 else input_ids_list
        # attention_mask_list[:60] if len(attention_mask_list) > 60 else attention_mask_list

        return ({
            # input_ids_list [tensor seg1, tensor seg2, tensor seg3, ...]
            # torch.tensor(ids, dtype=torch.long)
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            # [tensor seg1, tensor seg2, tensor seg3,...]
            'targets': target_list,
            'len': [torch.tensor(len(target_list), dtype=torch.long)]
        })
