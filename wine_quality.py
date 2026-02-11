'''
determine wine quality given traits
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from algorithms import mean_squared_error, stochastic_gradient_descent, train_test_split, zscore, apply_weights_and_bias

def prepreocess():
    '''
    preprocesss and return data frame depending on 
    if choosen red or white wine
    '''
    file = 'winequality-red.csv'
    df = pd.read_csv(file, sep=';')
    return df

def multiple_linear_regression(df, stochastic_sample_size):
    '''
    run the actual multiple linear regression on red or white wine data
    return a list representing the losses per each epoch
    '''
    train, test = train_test_split(df)
    loss_per_epoch = []

    weights, bias = [0 for i in range(len(df.columns) - 1)], 0.0
    weights, bias = stochastic_gradient_descent(train, learning_rate=.0001, weights=weights, bias=bias, epoches=100, min_step_size=0.0001, stochastic_sample_size=stochastic_sample_size, error_over_time=loss_per_epoch)

    print(weights, bias)
    test_results = mean_squared_error(weights=weights, bias=bias, df=test)
    print('testing results (mse):', test_results)
    return loss_per_epoch, test_results

def graph_background_information(df):
    '''
    given data frame graph:
        - correlation plot
        - correlation plot of alcohol vs quality
    '''
    sns.set_theme()
    
    # correlation plot
    correlations = df.corr()

    fig, ax = plt.subplots()
    cmap = sns.diverging_palette(229, 19, as_cmap=True)
    ax = sns.heatmap(
        data = correlations,
        cmap = cmap,
        linewidths = .5
    )
    ax.set_title('Wine features correlation plot')
    fig.tight_layout()
    fig.savefig('red_wine/correlations.png')

    # alcohol vs quality
    jointplot = sns.jointplot(x='quality', y='alcohol', data=df[['quality', 'alcohol']], kind='reg', ratio=2)
    jointplot.figure.suptitle('quality vs alcohol percent (%)')
    jointplot.figure.tight_layout()
    jointplot.savefig('red_wine/alcohol_vs_quality.png')


def graph_trials(results, name, cost=0, time=0):
    '''
    graph and save trail results for the cost function over trails
    '''
    fig, ax = plt.subplots()
    ax.scatter(list(range(0, len(results))), results)
    ax.set_xlabel('epoches')
    ax.set_ylabel('loss (mse)')
    ax.set_title(f'{name} - epoches vs loss (mse)\ncost: {cost}\ntime: {time}', wrap=True)
    plt.tight_layout()
    fig.savefig(f'results/{name}.png')

def main():
    df = prepreocess()
    graph_background_information(df)

    start = time.time()
    mini_batch_results, cost_mini = multiple_linear_regression(df, .1)
    mini_batch_time = time.time() - start

    # start = time.time()
    # batch_results, cost_batch = multiple_linear_regression(df, 1.0)
    # batch_time = time.time() - start

    graph_trials(mini_batch_results, f'Mini-batch Gradient Descent', cost = round(cost_mini, 2), time = round(mini_batch_time, 2))
    # graph_trials(batch_results, f'Batch Gradient Descent', cost = round(cost_batch, 2), time = round(batch_time, 2))


if __name__ == '__main__':
    main()