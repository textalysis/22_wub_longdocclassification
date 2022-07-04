import train
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report


def trainer(epochs, model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device, scheduler, filename, class_type, test_data_loader, data_test):
    history = defaultdict(list)
    best_accuracy = 0
    train_time_list = []
    val_time_list =[]
    #save_path = 'Test_results'
    #with open(os.path.join(save_path, "{}.txt".format(filename)), "a") as f:
    print('#' * 10)
    print(filename)
    for epoch in range(epochs):
            #f.write('-' * 10 + "\n")
            #f.write(f'Epoch {epoch + 1}/{epochs}' + "\n")
            print('-' * 10)
            print(f'Epoch {epoch + 1}/{epochs}')
            train_loss, train_acc, train_real, train_pred, train_time = train.train_epoch(model,
                                                train_data_loader,
                                                loss_fn,
                                                optimizer,
                                                device,
                                                scheduler,
                                                len(data_train['data']),
                                                class_type
                                                )
            train_time_list.append(train_time)
            train_report = classification_report(train_real, train_pred, output_dict=True)
            #f.write(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}' + "\n")
            print(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}')
            #print(train_report["accuracy"])

            val_loss, val_acc, val_real, val_pred,val_time = train.eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(data_val['data']),
                class_type
            )
            val_time_list.append(val_time)
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
    print(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}' + "\n")
    print(f'average train time {np.mean(train_time_list)}'+"\n")
    print(f'average val time {np.mean(val_time_list)}'+"\n")

    model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
    test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
                model,
                test_data_loader,
                loss_fn,
                device,
                len(data_test['data']),
                class_type
            )
    test_report = classification_report(test_real, test_pred, output_dict=True)
    print(f'test_accuracy {test_acc} macro_avg {test_report["macro avg"]} weighted_avg {test_report["weighted avg"]}'+"\n")
    
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


def trainer_hierarchical(epochs, model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device, scheduler, filename, class_type, test_data_loader, data_test):
    history = defaultdict(list)
    best_accuracy = 0
    train_time_list = []
    val_time_list = []
    #save_path = 'Test_results'
    print('#' * 10)
    print(filename)
    #with open(os.path.join(save_path, "{}.txt".format(filename)), "a") as f:

    for epoch in range(epochs):
            #f.write('-' * 10 + "\n")
            #f.write(f'Epoch {epoch + 1}/{epochs}' + "\n")
            print('-' * 10)
            print(f'Epoch {epoch + 1}/{epochs}')

            train_loss, train_acc, train_real, train_pred, train_time = train.hierarchical_train_epoch(model,
                                                train_data_loader,
                                                loss_fn,
                                                optimizer,
                                                device,
                                                scheduler,
                                                len(data_train['data']),
                                                class_type   
                                                )
            train_time_list.append(train_time)
            train_report = classification_report(train_real, train_pred, output_dict=True)
            #f.write(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}' + "\n")
            print(f'Train loss {train_loss} accuracy {train_acc} macro_avg {train_report["macro avg"]} weighted_avg {train_report["weighted avg"]}' )
            print(train_report["accuracy"])
            val_loss, val_acc, val_real, val_pred, val_time = train.hierarchical_eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(data_val['data']),
                class_type
            )
            val_time_list.append(val_time)
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
    print(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}'+ "\n")
    print(f'average train time {np.mean(train_time_list)}'+"\n")
    print(f'average val time {np.mean(val_time_list)}'+"\n")

    model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
    test_loss, test_acc, test_real, test_pred, test_time = train.hierarchical_eval_model(
                model,
                test_data_loader,
                loss_fn,
                device,
                len(data_test['data']),
                class_type
            )
    test_report = classification_report(test_real, test_pred, output_dict=True)
    print(f'test_accuracy {test_acc} macro_avg {test_report["macro avg"]} weighted_avg {test_report["weighted avg"]}'+"\n")

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


def trainer_multi_label(epochs, model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device,
            scheduler, filename, class_type, test_data_loader, data_test):
    history = defaultdict(list)
    best_f1_score = 0
    train_time_list = []
    val_time_list = []
    print('#' * 10)
    print(filename)
    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss, train_acc, train_real, train_pred, train_time = train.train_epoch(model,
                                                                                      train_data_loader,
                                                                                      loss_fn,
                                                                                      optimizer,
                                                                                      device,
                                                                                      scheduler,
                                                                                      len(data_train['data']),
                                                                                      class_type
                                                                                      )
        train_time_list.append(train_time)
        train_report = classification_report(train_real, train_pred, output_dict=True)
        train_micro_f1_score = train_report["micro avg"]["f1-score"]
        print(f'Train loss {train_loss} micro_f1_score {train_micro_f1_score} ')
        val_loss, val_acc, val_real, val_pred, val_time = train.eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(data_val['data']),
            class_type
        )
        val_time_list.append(val_time)
        val_report = classification_report(val_real, val_pred, output_dict=True)
        val_micro_f1_score = val_report["micro avg"]["f1-score"]
        print(f'Val loss {val_loss} micro_f1_score {val_micro_f1_score}')
        print(" ")
        history['train_f1_score'].append(train_micro_f1_score)
        history['train_loss'].append(train_loss)
        history['val_f1_score'].append(val_micro_f1_score)
        history['val_loss'].append(val_loss)

        if val_micro_f1_score > best_f1_score:
            torch.save(model.state_dict(), os.path.join('best_models', "{}_best.bin".format(filename)))
            best_f1_score = val_micro_f1_score
            # epoch index starting from 0
            best_epoch = epoch + 1

    # f.write('-' * 10 + "\n")
    # f.write(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}' + "\n")
    print('-' * 10)
    print(f'best_f1_socre {best_f1_score} best_epoch {best_epoch}' + "\n")
    print(f'average train time {np.mean(train_time_list)}' + "\n")
    print(f'average val time {np.mean(val_time_list)}' + "\n")

    model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
    test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(data_test['data']),
        class_type
    )
    test_report = classification_report(test_real, test_pred, output_dict=True)
    print(f'test_f1_score {test_report["micro avg"]["f1-score"]}' + "\n")

    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, epochs, 5))

    plt.plot(history['train_f1_score'], label='train f1 score')
    plt.plot(history['val f1 score'], label='val f1 score')

    plt.title(filename)
    plt.ylabel('f1 score')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(os.path.join('Visualizations', "{}.png".format(filename)))


def trainer_hierarchical_multi_label(epochs, model, train_data_loader, val_data_loader, data_train, data_val, loss_fn, optimizer, device,
            scheduler, filename, class_type, test_data_loader, data_test):
    history = defaultdict(list)
    best_f1_score = 0
    train_time_list = []
    val_time_list = []
    print('#' * 10)
    print(filename)
    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss, train_acc, train_real, train_pred, train_time = train.hierarchical_train_epoch(model,
                                                                                      train_data_loader,
                                                                                      loss_fn,
                                                                                      optimizer,
                                                                                      device,
                                                                                      scheduler,
                                                                                      len(data_train['data']),
                                                                                      class_type
                                                                                      )
        train_time_list.append(train_time)
        train_report = classification_report(train_real, train_pred, output_dict=True)
        train_micro_f1_score = train_report["micro avg"]["f1-score"]
        print(f'Train loss {train_loss} micro_f1_score {train_micro_f1_score} ')
        val_loss, val_acc, val_real, val_pred, val_time = train.hierarchical_eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(data_val['data']),
            class_type
        )
        val_time_list.append(val_time)
        val_report = classification_report(val_real, val_pred, output_dict=True)
        val_micro_f1_score = val_report["micro avg"]["f1-score"]
        print(f'Val loss {val_loss} micro_f1_score {val_micro_f1_score}')
        print(" ")
        history['train_f1_score'].append(train_micro_f1_score)
        history['train_loss'].append(train_loss)
        history['val_f1_score'].append(val_micro_f1_score)
        history['val_loss'].append(val_loss)

        if val_micro_f1_score > best_f1_score:
            torch.save(model.state_dict(), os.path.join('best_models', "{}_best.bin".format(filename)))
            best_f1_score = val_micro_f1_score
            # epoch index starting from 0
            best_epoch = epoch + 1

    # f.write('-' * 10 + "\n")
    # f.write(f'best_accuracy {best_accuracy} best_epoch {best_epoch} macro_avg {best_report["macro avg"]} weighted_avg {best_report["weighted avg"]}' + "\n")
    print('-' * 10)
    print(f'best_f1_socre {best_f1_score} best_epoch {best_epoch}' + "\n")
    print(f'average train time {np.mean(train_time_list)}' + "\n")
    print(f'average val time {np.mean(val_time_list)}' + "\n")

    model.load_state_dict(torch.load(os.path.join('best_models', "{}_best.bin".format(filename))))
    test_loss, test_acc, test_real, test_pred, test_time = train.eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(data_test['data']),
        class_type
    )
    test_report = classification_report(test_real, test_pred, output_dict=True)
    print(f'test_f1_score {test_report["micro avg"]["f1-score"]}' + "\n")

    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, epochs, 5))

    plt.plot(history['train_f1_score'], label='train f1 score')
    plt.plot(history['val f1 score'], label='val f1 score')

    plt.title(filename)
    plt.ylabel('f1 score')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(os.path.join('Visualizations', "{}.png".format(filename)))
