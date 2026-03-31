'''
determine wine quality given traits
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import pearsonr, ttest_ind
from algorithms import mean_squared_error, stochastic_gradient_descent, train_test_split, zscore, apply_weights_and_bias, standard_deviation, optimized_stochastic_gradient_descent, optimized_stochastic_gradient_descent_ridge
import mplcyberpunk

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

def optimized_multiple_linear_regression(df, stochastic_sample_size, lr, epoches=100):
    '''
    run an optimized version of multiple linear regression
    '''
    train, test = train_test_split(df)
    loss_per_epoch = []

    weights, bias = [0 for i in range(len(df.columns) - 1)], 0.0
    theta = optimized_stochastic_gradient_descent(train, lr=lr, weights=weights, bias=bias, epoches=epoches, min_step_size=0, error_over_time=loss_per_epoch, stochastic_sample_size=stochastic_sample_size)

    print(theta)
    test_results = mean_squared_error(weights=[theta[0, i] for i in range(theta.shape[1] - 1)], bias=theta[0, len(theta[0]) - 1], df=test)
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
    # plt.style.use('cyberpunk')
    
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
    # cmap = sns.diverging_palette(229, 19, as_cmap=True)
    ax = sns.heatmap(
        data = correlations,
        vmin = -1.0,
        vmax = 1.0,
        cmap = sns.diverging_palette(230, 20, as_cmap=True),
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

def graph_trials(results, name, cost=0, time=0, x=None, costs=None):
    '''
    graph and save trail results for the cost function over trails
    '''
    if x is None:
        x = range(0, len(results))

    # plt.style.use('cyberpunk')

    fig, ax = plt.subplots()
    ax.plot(x, results, label='loss')
    if costs is not None:
        ax.plot(x, costs, color='red', label='cost')

    ax.set_xlabel('epoches')
    ax.set_ylabel('loss (mse)')
    ax.set_title(f'{name} - epoches vs loss (mse)\ncost: {cost}\ntime: {time}', wrap=True)
    plt.tight_layout()

    # mplcyberpunk.add_glow_effects(gradient_fill=True)
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=.4, gradient_start='bottom')
    fig.savefig(f'results/{name}.png')


def final_optimized_linear_regression(df):
    '''
    example of best linear reression found
    '''
    from algorithms import optimized_stochastic_gradient_descent_ridge, optimized_stochastic_gradient_descent_lasso, optimized_stochastic_gradient_descent_elastic_net
    from cross_validation import n_fold_cross_validation

    df = prepreocess()
    df = df.loc[:, df.columns != 'pH']
    df = df.loc[:, df.columns != 'residual sugar']
    df = df.loc[:, df.columns != 'fixed acidity']

    lr = .1
    stochastic_sample_size = .1
    epoches = 1_000
    n = 5

    train_test_sets = n_fold_cross_validation(n=n, df=df)

    for i in range(len(train_test_sets)):
        train = train_test_sets[i][0]
        test = train_test_sets[i][1]
        itr = i
        # normalize all training data
        parameters = [(train[col].mean(), train[col].std()) for col in train.columns[:-1]]
        for col in train.columns[:-1]:
            mean = train[col].mean()
            std_dev = train[col].std()
            def stand(row, mean, std_dev):
                return (row - mean) / std_dev
            train[col] = train[col].apply(stand, args=(mean, std_dev))
        # normalize all the test
        for i in range(len(train.columns[:-1])):
            col = train.columns[i]
            mean = parameters[i][0]
            std_dev = parameters[i][1]
            def stand(row, mean, std_dev):
                return (row - mean) / std_dev
            test[col] = test[col].apply(stand, args=(mean, std_dev))

        start = time.time()
        final_results = []
        weights, bias = [0 for i in range(len(df.columns) - 1)], 0.0
        theta = optimized_stochastic_gradient_descent_elastic_net(train, lmda=1, lr=lr, weights=weights, bias=bias, epoches=epoches, min_step_size=0, error_over_time=final_results, stochastic_sample_size=stochastic_sample_size, printProgress=False)

        # find mean squared error
        weights = [theta[0, i] for i in range(theta.shape[1] - 1)]
        bias = theta[0, theta.shape[1] - 1]
        test_results = mean_squared_error(df=test, weights=weights, bias=bias)

        final_time = time.time() - start

        # plot results
        final_results = final_results[::10]
        epoch_number = range(0, 1_000, 10)
        graph_trials(final_results, f'{n}-fold ({itr + 1}\\{n})', cost = f'{round(test_results, 2)} final loss: {final_results[len(final_results) - 1]}', time = round(final_time, 2), x=epoch_number)


def main():
    '''
    build all types of regression
    '''
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
    o_mini_batch_results, o_cost_mini = optimized_multiple_linear_regression(df, .1, lr=.001)
    o_mini_batch_time = time.time() - start

    start = time.time()
    o_batch_results, o_cost_batch = optimized_multiple_linear_regression(df, 1.0, lr=.0001)
    o_batch_time = time.time() - start

    graph_trials(mini_batch_results, f'Mini-batch Gradient Descent', cost = round(cost_mini, 2), time = round(mini_batch_time, 2))
    graph_trials(batch_results, f'Batch Gradient Descent', cost = round(cost_batch, 2), time = round(batch_time, 2))
    graph_trials(o_mini_batch_results, f'(optimized) Mini-batch Gradient Descent', cost = round(o_cost_mini, 2), time = round(o_mini_batch_time, 2))
    graph_trials(o_batch_results, f'(optimized) Batch Gradient Descent', cost = round(o_cost_batch, 2), time = round(o_batch_time, 2))

def easy(df):
    '''
    using ml libraries to make a linear regression
    '''
    import numpy as np
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    inner = [list(df[name]) for name in df.columns]
    _matrix = np.array(inner).T
    
    # establish features (x) and labels (y)
    x = _matrix[:, 0:_matrix.shape[1] - 1]
    y = _matrix[:, _matrix.shape[1] - 1]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


    reg = LinearRegression().fit(X_train, y_train)
    res = reg.predict(X_test)
    print(mean_squared_error(y_test, res))


if __name__ == '__main__':
    # main()
    final_optimized_linear_regression(prepreocess())