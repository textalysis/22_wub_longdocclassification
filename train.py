import time
import torch

import numpy as np
import torch.nn as nn


def available_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # specify  device
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def hierarchical_train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []

    t0 = time.time()

    for batch_idx, batch in enumerate(data_loader):
        # for example, when batch_size=4, four document, doc 1, doc 2, doc 4 are less than 512 tokens, doc 3 about 4*512 tokens
        # the input_ids would be:
        # [tensor([[doc1 input_ids]]),tensor([[doc2 segment1 input_ids], [doc2 seg2], [doc2 seg3],[doc2 seg4]]), tensor([[doc3]]), tensor([[doc4]])]
        input_ids = [data["input_ids"] for data in batch]
        # similar to input_ids
        attention_mask = [data["attention_mask"] for data in batch]
        # [tensor(target), tensor(target), tensor(target), tensor(target)]
        # here [0] to reduce the dimension
        targets = [data["targets"][0] for data in batch]
        # get the number of segments for each document
        # [tensor([1]), tensor([1]), tensor([4]), tensor([1])]
        lengt = [data['len'] for data in batch]

        # change the shape as input to Bert Model
        # tensor([[ doc 1], [doc 2], [doc 3 seg 1], [ doc 3 seg 2], [doc 3 seg 3], [doc 3 seg 4], [doc 4]])
        input_ids = torch.cat(input_ids)
        attention_mask = torch.cat(attention_mask)
        # tensor([doc1 target, doc2 target, doc3 target, doc4 target], dtype=torch.int32)
        targets = torch.stack(targets)
        # [doc1 num of segments, doc2 num of segments,... ]
        # [1, 1, 4, 1]
        lengt = [x.item() for x in lengt]

        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengt=lengt
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        predictions.extend(preds)
        real_values.extend(targets)

        correct_predictions += torch.sum(preds == targets)
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


def hierarchical_eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []

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
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            predictions.extend(preds)
            real_values.extend(targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(float(loss.item()))

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()

    return np.mean(losses), float(correct_predictions / n_examples), real_values, predictions


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
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


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            predictions.extend(preds)
            real_values.extend(targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(float(loss.item()))
            # losses.append(loss.item())
            torch.cuda.empty_cache()

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()

    return np.mean(losses), float(correct_predictions / n_examples), real_values, predictions
