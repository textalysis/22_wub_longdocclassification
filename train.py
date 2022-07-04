import time
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def available_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:2")  # specify  device
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def hierarchical_train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, class_type):
    model = model.train()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []

    t0 = time.time()
    #print(data_loader)
    for batch_idx, batch in enumerate(data_loader):
        # for example, when batch_size=4, four document, doc 1, doc 2, doc 4 are less than 512 tokens, doc 3 about 4*512 tokens
        # the input_ids would be:
        # [tensor([[doc1 input_ids]]),tensor([[doc2 segment1 input_ids], [doc2 seg2], [doc2 seg3],[doc2 seg4]]), tensor([[doc3]]), tensor([[doc4]])]
        input_ids = [data["input_ids"] for data in batch]
        # similar to input_ids
        attention_mask = [data["attention_mask"] for data in batch]
        # [tensor(target), tensor(target), tensor(target), tensor(target)]
        # here [0] to get the first seg label, the other segs are the same
        targets = [data["targets"][0] for data in batch]
        # get the number of segments for each document
        # [tensor([1]), tensor([1]), tensor([4]), tensor([1])]
        lengt = [data['len'] for data in batch]

        # change the shape as input to Bert Model
        # tensor([[ doc 1], [doc 2], [doc 3 seg 1], [ doc 3 seg 2], [doc 3 seg 3], [doc 3 seg 4], [doc 4]])
        input_ids = torch.cat(input_ids)
        attention_mask = torch.cat(attention_mask)
        # tensor([doc1 target, doc2 target, doc3 target, doc4 target], dtype=torch.int32)
        # change list of tensor to tensor of list    [tensor(0), tensor(1)] -> tensor([0,1])
        # targets = torch.stack(targets)
        # [doc1 num of segments, doc2 num of segments,... ]
        # [1, 1, 4, 1]
        lengt = [x.item() for x in lengt]

        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        #targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengt=lengt
        )

        if class_type == "multi_label":
            # outputs.shape [batch, n_classes]
            print(targets)
            # change to tensor then put to device and split 
            num_labels =[]
            for i in targets:
               num_labels.append(len(i))

            targets = torch.cat(targets)
            targets = targets.to(device, dtype=torch.long)
            targets = list(targets.split_with_sizes(num_labels))
            print(targets)
            # get prediction, use np round

            preds = []
            one_hot = torch.round(torch.sigmoid(outputs))  # one hot
            for i, x in enumerate(one_hot): # change one hot to index
                #index = torch.argwhere(x==1).squeeze(1)
                index = (x == 1).nonzero(as_tuple=True)[0]
                preds.extend(index.unsqueeze(0))
            print(preds)
            # iterate the list of tensors, if the multi label equal, plus 1
            

            for i in range(len(targets)):
                if torch.equal(targets[i],preds[i]):
                    correct_predictions = correct_predictions + 1 
            print(correct_predictions)
   
        else:
            # tensor([doc1 target, doc2 target, doc3 target, doc4 target], dtype=torch.int32)
            # change list of tensor to tensor of list    [tensor(0), tensor(1)] -> tensor([0,1])
            targets = torch.stack(targets)
            targets = targets.to(device, dtype=torch.long)
            # preds tensor([0, 1], device='cuda:2')
            _, preds = torch.max(outputs, dim=1) #dim=1 to get the index of the maximal value
            # preds shape [batch_size], predictions shape[number of documents]
            correct_predictions += torch.sum(preds == targets)
        
        loss = loss_fn(outputs, targets)
        predictions.extend(preds)
        real_values.extend(targets)

        #correct_predictions += torch.sum(preds == targets)
        losses.append(float(loss.item()))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
    
    print(f"time = {time.time() - t0:.2f} secondes" + "\n")
    t0 = time.time()

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()

    return np.mean(losses), float(correct_predictions / n_examples), real_values, predictions


def hierarchical_eval_model(model, data_loader, loss_fn, device, n_examples,  class_type):
    model = model.eval()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = [data["input_ids"] for data in batch]
            attention_mask = [data["attention_mask"] for data in batch]
            targets = [data["targets"][0] for data in batch]
            lengt = [data['len'] for data in batch]

            input_ids = torch.cat(input_ids)
            attention_mask = torch.cat(attention_mask)
            targets = torch.stack(targets)
            lengt = [x.item() for x in lengt]

            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengt=lengt
            )
            
            if class_type == "multi_label":
                preds = []
                one_hot = torch.round(torch.sigmoid(outputs))
                for i, x in enumerate(one_hot): 
                    index = torch.argwhere(x==1).squeeze(1)
                    preds.extend(index.unsqueeze(0))
            else:
                _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            predictions.extend(preds)
            real_values.extend(targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(float(loss.item()))
    
    print(" ")    
    print(f"time = {time.time() - t0:.2f} secondes" + "\n")
    t0 = time.time()
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()

    return np.mean(losses), float(correct_predictions / n_examples), real_values, predictions


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, class_type):
    model = model.train()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []

    t0 = time.time()

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
       
        if class_type == "multi_label":
            preds = []
            one_hot = torch.round(torch.sigmoid(outputs))
            for i, x in enumerate(one_hot): 
                index = torch.argwhere(x==1).squeeze(1)
                preds.extend(index.unsqueeze(0))
        else:
            _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        predictions.extend(preds)
        real_values.extend(targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(float(loss.item()))
        # losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
    
    print(f"time = {time.time() - t0:.2f} secondes" + "\n")
    t0 = time.time()

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()

    return np.mean(losses), float(correct_predictions / n_examples), real_values, predictions


def eval_model(model, data_loader, loss_fn, device, n_examples, class_type):
    model = model.eval()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    t0 = time.time()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            if class_type == "multi_label":
                preds = []
                one_hot = torch.round(torch.sigmoid(outputs))
                for i, x in enumerate(one_hot): 
                    index = torch.argwhere(x==1).squeeze(1)
                    preds.extend(index.unsqueeze(0))
            else:
                _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            predictions.extend(preds)
            real_values.extend(targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(float(loss.item()))
            # losses.append(loss.item())
            torch.cuda.empty_cache()
    
    print(" ")
    print(f"time = {time.time() - t0:.2f} secondes" + "\n")
    t0 = time.time()
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()

    return np.mean(losses), float(correct_predictions / n_examples), real_values, predictions
