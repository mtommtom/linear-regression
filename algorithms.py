import numpy as np
import random
import matplotlib.pyplot as plt


def zscore(mean, std_dev, x):
    return (mean - x) / std_dev



def apply_weights_and_bias(inputs, weights, bias):
    '''
    given input varibles weights and bias
    return the output f(x1, x2, ..., xn) = ?
    '''
    _sum = bias
    for i in range(len(inputs)):
        _sum += (inputs[i] * weights[i])
    
    return _sum


def train_test_split(df, train_bias=.75):
    '''
    split the training data and training labels 
    given training sample size bias
    '''
    deli = int(len(df) * train_bias)
    train = df.iloc[0:deli]
    test = df.iloc[(deli + 1):]
    return train, test

def mean_squared_error(weights, bias, df):
    '''
    calculate the mean squared error given a 'best fitline'
    average squared error between all points of dataset
    and best fit function (current function)
    '''
    def squared_error(row):
        output = bias
        for i in range(len(weights)):
            output += weights[i]*row.iloc[i]

        return (output - row.iloc[len(row) - 1])**2

    return df.apply(squared_error, axis=1).mean()


def der_weight_naught_mean_squared_error(weights, bias, df, x):
    '''
    derivitve of mean squared error with respect to a given weight in a set of weights
    x is loc of given weight in weights that is getting derivitve w/ respect to
    '''
    def der_squared_error(row):
        output = bias
        for i in range(len(weights)):
            output += weights[i]*row.iloc[i]

        return -2*row.iloc[x]*(row.iloc[len(row) - 1] - output)

    return df.apply(der_squared_error, axis=1).mean()


def der_bias_mean_squared_error(weights, bias, df):
    '''
    derivitve of mean squared error with respect to the bias
    '''
    def der_squared_error(row):
        output = bias
        for i in range(len(weights)):
            output += weights[i]*row.iloc[i]

        return -2*(row.iloc[len(row) - 1] - output)

    return df.apply(der_squared_error, axis=1).mean()


def stochastic_gradient_descent(_df, learning_rate=.01, min_step_size=.001, epoches=100, weights=[1], bias=0, stochastic_sample_size=.1, error_over_time=[]):
    '''
    apply stochastic gradient decesnt algorithm given hyperparameters:
    learning rate, epoch count, starting weights, startign bias and stochastic gradient mini batch size
    '''

    step_num = 0
    while True:
        # filter a certian df amount
        df = _df.iloc[random.sample(range(0, len(_df)), int(len(_df) * stochastic_sample_size))]
        # calculate weight and bias step size
        bias_step_size = learning_rate * der_bias_mean_squared_error(weights, bias, df)
        weight_step_sizes = [learning_rate * der_weight_naught_mean_squared_error(weights, bias, df, i) for i in range(len(weights))]

        can_stop = True
        # step each weight and bias
        bias -= bias_step_size
        for i in range(len(weight_step_sizes)):
            if abs(weight_step_sizes[i]) > min_step_size:
                can_stop = False
            if abs(bias_step_size) > min_step_size:
                can_stop = False
            weights[i] -= weight_step_sizes[i]
        step_num += 1

        # alert the step size
        loss = mean_squared_error(weights, bias, df)
        print(f'Epoch [{step_num}/{epoches}],   loss: {loss}')
        error_over_time.append(loss)

        if step_num >= epoches or can_stop: # do while loop
            break
    return (weights, bias)