from globals import *
from .visualisation import *

import torch
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def evaluate(net, test_set, history):
    ''' Evaluates the performance of the
        RNN on the test set.

    Arguments:
        net (nn.Module): RNN net
        test_set (dict): test input and target output
        history (dict): dict used for loss log
    Returns:
        test_loss (float): loss on the test set
    '''
    net.eval()

    test_predict = net(test_set['X'])
    test_loss = loss_func(test_predict, test_set['Y'])
    history['test_loss'].append(test_loss)

    return test_loss.item()


def train(net, train_loader, optimizer, history):
    ''' Evaluates the performance of the
        RNN on the test set.

    Arguments:
        net (nn.Module): RNN net
        train_loader (DataLoader): train input and 
                                   target output
        optimizer: optimizer object (Adam)
        history (dict): dict used for loss log
    Returns:
        test_loss (float): loss on the test set
    '''
    net.eval()

    total_num = 0
    train_loss = 0
    for input, target in train_loader:
        optimizer.zero_grad()
        loss = loss_func(net(input), target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(target)
        total_num += len(target)

    history['train_loss'].append(train_loss / total_num)

    return loss.item()


def train_loop(net, epochs, lr, wd, train_loader, test_set, debug=True):
    ''' Performs the training of the RNN using Adam optimizer.
        Train and evaluation losses are being logged.

    Arguments:
        net (nn.Module): RNN to be trained
        epochs (int): number of epochs we wish to train
        lr (float): max learning rate for Adam optimizer
        wd (float): L2 regularization weight decay
        train_loader (DataLoader): train input and target output
        test_set (dict): test input and target output
        debug (bool): Should we display train progress?
    '''
    history = dict()
    history['train_loss'] = list()
    history['test_loss'] = list()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        train_loss = train(net, train_loader, optimizer, history)

        with torch.no_grad():
            test_loss = evaluate(net, test_set, history)

        if debug and (epoch + 1) % 200 == 0:
            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.8f}",
                  f" |  Test Loss: {test_loss:.8f}")

    if debug:
        show_loss(history)


def train_test_split(subsequences):
    ''' Splits the loaded subsequences into train
        and test set w.r.t. split ratio.

    Arguments:
        subsequences (): input-output pairs
    Returns:
        train_loader (DataLoader): train set inputs and target outputs
        test_set (dict): test set inputs and target outputs
    '''
    # Get the length of the train set
    TRAIN_SIZE = int(config.split_ratio * len(subsequences))
    train_seqs = subsequences[:TRAIN_SIZE]
    test_seqs = subsequences[TRAIN_SIZE:]

    # Divide inputs and target outputs
    trainX, trainY = [torch.Tensor(list(x)).to(device)
                      for x in zip(*train_seqs)]
    testX, testY = [torch.Tensor(list(x)).to(device)
                    for x in zip(*test_seqs)]

    train_set = torch.utils.data.TensorDataset(trainX, trainY)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.bs)

    test_set = dict()
    test_set['X'] = torch.Tensor(testX).to(device)
    test_set['Y'] = torch.Tensor(testY).to(device)

    return train_loader, test_set


def extract_subsequences(sequence, lag=3):
    ''' Extracts subsequences in order to create
        input - output pairs for the train / test data.
        We extract input subsequences of lag length
        ex: sequence[i:i + lab] and set it's target
        output as the next value in the sequence

    Arguments:
        sequence(numpy.ndarray): entire dataset sequence
        lag(int): number of previous values we use as input
    Returns:
        subseqs(list): list of extracted input - output pairs
    '''
    subseqs = list()

    for i in range(len(sequence) - lag - 1):
        input = sequence[i:i + lag]
        output = sequence[i + lag]

        subseqs.append((input, output))

    return subseqs


def load_dataset(dataset_path, show_data=True):
    ''' Loads the dataset stored as the csv file.
        Displays the loaded data if desired.

    Arguments:
        dataset_path(string): path to the dataset file
        show_data(bool): should we show loaded data?
    Returns:
        dataset (numpy.ndarray): loaded dataset
        scaler (MinMaxScaler): normalizes dataset values
    '''
    # Load the dataset as DataFrame
    dataset = pd.read_csv(dataset_path)
    # Dataset contains dates as well as
    # monthly car sale measurement so we
    # extract only sale column as numpy array
    xlabels = dataset.iloc[:, 0].values
    dataset = dataset.iloc[:, 1:].values

    if show_data:
        display_dataset(dataset, xlabels)

    # We normalize the dataset values
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    return dataset, scaler
