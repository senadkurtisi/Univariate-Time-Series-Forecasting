from globals import *

import torch
import numpy as np
import matplotlib.pyplot as plt


x_ticks = list()
tick_positions = list()


def show_evaluation(net, dataset, scaler, debug=True):
    ''' Evaluates performance of the RNN on the entire
        dataset, and shows the prediction as well as
        target values.

    Arguments:
        net (nn.Module): RNN to evaluate
        dataset (numpy.ndarray): dataset
        scaler (MinMaxScaler): used for denormalization
        debug (bool): should we calculate/display eval.
                      MSE/MAE
    '''
    dataset = torch.FloatTensor(dataset).unsqueeze(-1).to(device)
    total_train_size = int(config.split_ratio * len(dataset))

    # Prediction on the entire dataset
    net.eval()
    test_predict = net(dataset)

    # Perform denormalization of the target values and prediction
    test_predict = scaler.inverse_transform(test_predict.cpu().data.numpy())
    dataset = scaler.inverse_transform(dataset.cpu().squeeze(-1).data.numpy())

    # Plotting the original sequence vs. predicted
    plt.figure(figsize=(8, 5))
    plt.axvline(x=total_train_size, c='r')
    plt.plot(dataset)
    plt.plot(test_predict)
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.xlabel('Year-Month')
    plt.ylabel("Number of car sales")
    plt.title('Univariate Time-Series Forecast')
    plt.legend(['Train-Test split', 'Target', 'Prediction'])
    plt.show()

    if debug:
        # Calculating total MSE & MAE
        total_mse = (np.square(test_predict - dataset)).mean()
        total_mae = (np.abs(test_predict - dataset)).mean()
        # Calculating train MSE & MAE
        train_mse = (np.square(test_predict - dataset)
                     )[total_train_size:].mean()
        train_mae = (np.abs(test_predict - dataset))[:total_train_size].mean()
        # Calculating test MSE & MAE
        test_mse = (np.square(test_predict - dataset)
                    )[total_train_size:].mean()
        test_mae = (np.abs(test_predict - dataset))[:total_train_size].mean()

        print(f"Total MSE:  {total_mse:.4f}  |  Total MAE:  {total_mae:.4f}")
        print(f"Train MSE:  {train_mse:.4f}  |  Train MAE:  {train_mae:.4f}")
        print(f"Test MSE:   {test_mse:.4f}  |  Test MAE:   {test_mae:.4f}")


def show_loss(history):
    ''' Display train and evaluation loss

    Arguments:
        history(dict): Contains train and test loss logs
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['test_loss'], label='Evaluation loss')

    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()


def display_dataset(dataset, xlabels):
    ''' Displays the loaded data

    Arguments:
        dataset(numpy.ndarray): loaded data
        xlabels(numpy.ndarray): strings representing
                                 according dates
    '''
    global x_ticks
    global tick_positions
    # We can't show every date in the dataset
    # on the x axis because we couldn't see
    # any label clearly. So we extract every
    # n-th label/tick
    segment = int(len(dataset) / 6)

    for i, date in enumerate(xlabels):
        if i > 0 and (i + 1) % segment == 0:
            x_ticks.append(date)
            tick_positions.append(i)
        elif i == 0:
            x_ticks.append(date)
            tick_positions.append(i)

    # Display loaded data
    plt.figure(figsize=(8, 5))
    plt.plot(dataset)
    plt.title('Monthly car sales')
    plt.xlabel('Year-Month')
    plt.ylabel("Number of car sales")
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.show()
