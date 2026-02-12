import numpy as np
import random
import matplotlib.pyplot as plt


def zscore(mean, std_dev, x):
    return (mean - x) / std_dev

def standard_deviation(x):
    n = len(x)
    mean = sum(x) / n

    dif = sum([i - mean for i in x])**2
    return np.sqrt(dif / n)

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

    return (weights, bias) as a tuple of all given weights and biases
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

def matrix_mean_squared_error(theta, x, y):
    return np.sum(np.square((theta.dot(x.T).T - y).T)) / len(x)

def optimized_stochastic_gradient_descent(df, lr=.01, min_step_size=.001, epoches=100, weights=[1], bias=0, stochastic_sample_size=.1, displayProgress: bool=True, error_over_time = []):
    '''
    optimized stochastic gradient descent method by using numpy and linear algebra
    
    :param df: dataframe of all features and label should be last column
    :param lr: learning rate
    :param min_step_size: if step size is less than this end program
    :param epoches: how many repititions of gradient descent 
    :param weights: the starting weights
    :param bias: the starting bias
    :param stochastic_sample_size: percent of each sample group for each epoch
    '''

    # matrix = np.matrix([[df.iloc[i, j] for j in range(len(df.columns))] for name in df.columns])
    matrix = np.matrix([list(df[name]) for name in df.columns]).T
    theta = np.matrix(np.append(np.array(weights), bias))
    # print(theta)
    # print("the above should be one list of weights and bias at end of list")
    epoch_number = 0
    while epoch_number < epoches:
        # sample (according to stochastic sample size) the dataframe
        _matrix = matrix[random.sample(range(0, len(df)), int(len(df) * stochastic_sample_size))]
        
        # establish features (x) and labels (y)
        x = _matrix[:, 0:_matrix.shape[1] - 1]
        x = np.column_stack((x, np.ones(len(x))))
        y = _matrix[:, matrix.shape[1] - 1]
        # print(x)
        # print("the above should be matrix")
        # print(y)
        # print("the above should be just one vector")

        # get the step direction vector (direction to go) and the step vector with correct magnitude
        m = len(df)
        step_direction_vector = (2/m) * x.T.dot(theta.dot(x.T).T - y)
        step_vector = step_direction_vector * lr
        if displayProgress:
            loss = matrix_mean_squared_error(theta, x, y)
            error_over_time.append(loss)
            print(f'Epoch [{epoch_number + 1}/{epoches}],   loss: {loss}')

        # icond = step_direction_vector < min_step_size
        # if icond.any():
        #     break

        # change the weights (theta)
        theta -= step_vector.T
        epoch_number += 1
    
    return theta

