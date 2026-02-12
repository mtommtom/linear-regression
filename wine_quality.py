'''
determine wine quality given traits
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import pearsonr, ttest_ind
from algorithms import mean_squared_error, stochastic_gradient_descent, train_test_split, zscore, apply_weights_and_bias, standard_deviation, optimized_stochastic_gradient_descent

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
        - distribution of the qualities
        - correlation plot
        - correlation plot of quality vs alcohol and volatile acidity vs quality
    '''
    sns.set_theme()

    # wine quality distribution
    value_counts = [int(df['quality'].value_counts()[i]) if i in df['quality'].value_counts() else 0 for i in range(0, 11) ]
    ax = sns.barplot(x=range(0, 11), y=value_counts)
    ax.set_title('Distribution of wine quality entries')
    ax.set_xlabel('wine quality (1-10)')
    ax.set_ylabel('count')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig('red_wine/wine_quality_distribution.png')
    
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

    def plot_correlation(df, x, y, xlabel = None, ylabel = None):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        aq_df = df[[x, y]]
        # get p value
        slope = stochastic_gradient_descent(aq_df, learning_rate=.001)[0][0]
        corr, pvalue = pearsonr(aq_df[x], aq_df[y])
        # plot data
        jointplot = sns.jointplot(x=x, y=y, data=aq_df, kind='reg', ratio=2)
        jointplot.figure.suptitle(f'{xlabel} vs {ylabel}\np = {pvalue}')
        jointplot.figure.tight_layout()
        jointplot.savefig(f'red_wine/{x}_vs_{y}.png')

    # alcohol vs quality
    plot_correlation(df, 'alcohol', 'quality', xlabel='alcohol percent')
    # volatile acidity vs quality
    plot_correlation(df, 'volatile acidity', 'quality')


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

    start = time.time()
    batch_results, cost_batch = multiple_linear_regression(df, 1.0)
    batch_time = time.time() - start

    # optimized
    start = time.time()
    o_mini_batch_results, o_cost_mini = optimized_multiple_linear_regression(df, .1)
    o_mini_batch_time = time.time() - start

    start = time.time()
    o_batch_results, o_cost_batch = optimized_multiple_linear_regression(df, 1.0)
    o_batch_time = time.time() - start

    graph_trials(mini_batch_results, f'Mini-batch Gradient Descent', cost = round(cost_mini, 2), time = round(mini_batch_time, 2))
    graph_trials(batch_results, f'Batch Gradient Descent', cost = round(cost_batch, 2), time = round(batch_time, 2))
    graph_trials(o_mini_batch_results, f'(optimized) Mini-batch Gradient Descent', cost = round(o_cost_mini, 2), time = round(o_mini_batch_time, 2))
    graph_trials(o_batch_results, f'(optimized) Batch Gradient Descent', cost = round(o_cost_batch, 2), time = round(o_batch_time, 2))


def optimized_multiple_linear_regression(df, stochastic_sample_size):
    train, test = train_test_split(df)
    loss_per_epoch = []

    weights, bias = [0 for i in range(len(df.columns) - 1)], 0.0
    theta = optimized_stochastic_gradient_descent(train, lr=.0001, weights=weights, bias=bias, epoches=2000, min_step_size=0, error_over_time=loss_per_epoch, stochastic_sample_size=stochastic_sample_size)

    print(theta)
    test_results = mean_squared_error(weights=[theta[0, i] for i in range(theta.shape[1] - 1)], bias=theta[0, len(theta[0]) - 1], df=test)
    print('testing results (mse):', test_results)
    return loss_per_epoch, test_results




if __name__ == '__main__':
    main()