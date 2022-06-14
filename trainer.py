import train
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report


def trainer(epochs, model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device, scheduler, filename):
    history = defaultdict(list)
    best_accuracy = 0

    #save_path = 'Test_results'
    #with open(os.path.join(save_path, "{}.txt".format(filename)), "a") as f:
    print('#' * 10)
    print(filename)
    for epoch in range(epochs):
            #f.write('-' * 10 + "\n")
            #f.write(f'Epoch {epoch + 1}/{epochs}' + "\n")
            print('-' * 10)
            print(f'Epoch {epoch + 1}/{epochs}')
            train_loss, train_acc, train_real, train_pred = train.train_epoch(model,
                                                train_data_loader,
                                                loss_fn,
                                                optimizer,
                                                device,
                                                scheduler,
                                                len(data_train['data'])
                                                )

            train_report = classification_report(train_real, train_pred, output_dict=True)
            #f.write(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}' + "\n")
            print(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}')
            val_loss, val_acc, val_real, val_pred = train.eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(data_val['data'])
            )

            val_report = classification_report(val_real, val_pred, output_dict=True)
            #f.write(f'Val loss {val_loss} accuracy {val_acc} macro_avg {val_report["macro avg"]} weighted_avg {val_report["weighted avg"]}' + "\n")
            print(f'Val loss {val_loss} accuracy {val_acc} macro_avg {val_report["macro avg"]} weighted_avg {val_report["weighted avg"]}')
            print(" ")
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), os.path.join('best_models', "{}_best.bin".format(filename)))
                best_accuracy = val_acc
                # epoch index starting from 0
                best_epoch = epoch + 1
                best_report = val_report

    #f.write('-' * 10 + "\n")
    #f.write(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}' + "\n")
    print('-' * 10)
    print(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}')
    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, epochs, 5))

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title(filename)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(os.path.join('Visualizations', "{}.png".format(filename)))


def trainer_hierarchical(epochs, model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device, scheduler, filename):
    history = defaultdict(list)
    best_accuracy = 0

    #save_path = 'Test_results'
    print('#' * 10)
    print(filename)
    #with open(os.path.join(save_path, "{}.txt".format(filename)), "a") as f:

    for epoch in range(epochs):
            #f.write('-' * 10 + "\n")
            #f.write(f'Epoch {epoch + 1}/{epochs}' + "\n")
            print('-' * 10)
            print(f'Epoch {epoch + 1}/{epochs}')

            train_loss, train_acc, train_real, train_pred = train.hierarchical_train_epoch(model,
                                                train_data_loader,
                                                loss_fn,
                                                optimizer,
                                                device,
                                                scheduler,
                                                len(data_train['data'])
                                                )

            train_report = classification_report(train_real, train_pred, output_dict=True)
            #f.write(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}' + "\n")
            print(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}' )
            val_loss, val_acc, val_real, val_pred = train.hierarchical_eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(data_val['data'])
            )

            val_report = classification_report(val_real, val_pred, output_dict=True)
            #f.write(f'Val loss {val_loss} accuracy {val_acc} macro_avg {val_report["macro avg"]} weighted_avg {val_report["weighted avg"]}' + "\n")
            print(f'Val loss {val_loss} accuracy {val_acc} macro_avg {val_report["macro avg"]} weighted_avg {val_report["weighted avg"]}')
            print(" ")
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), os.path.join('best_models', "{}_best.bin".format(filename)))
                best_accuracy = val_acc
                # epoch index starting from 0
                best_epoch = epoch + 1
                best_report = val_report

    #f.write('-' * 10 + "\n")
    #f.write(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}' + "\n")
    print('-' * 10 )
    print(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}')
    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, epochs, 5))

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title(filename)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(os.path.join('Visualizations', "{}.png".format(filename)))
