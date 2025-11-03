# linear-regression

## Description/Overview

The purpose of this project is to get hands on expeirence with different ways of solving the linear regression problem.
The methods used so far are: Batch Gradient Descent, and Mini Batch Gradient decent.
All of these algorithms are implemented by hand without the assist of any libraries like keras or scikit-learn

The data set used comes from UCI which contain red and white wine samples. They include many features one of which is
wine quality which is what the linear regression algorithms will be trained to find.

## Results

All of the results are plotted with matplot lib.

### Batch Gradient Descent

![batch-gradient-descent](./results/Batch%20Gradient%20Descent%20-%20cost:0.7%20-%20time:158.34%20seconds.png)

#### Analysis of results
The loss for each epoch gets consistantly lower each time with a negative acceleration. This is due to the nature of
the batch gradient descent algorithm because it garuentees that the loss will go down each time. Because batch
gradient descent calculates the gradient of the weights by using the mean of all derivitives (mean gradient). While you
can gaurentee a reduction in the loss, it takes a very long time per epoch.

### Mini-batch Gradient Descent

![mini-batch-gradient-descent](./results/Mini-batch%20Gradient%20Descent%20-%20cost:0.69%20-%20time:15.34%20seconds.png)

#### Analysis of results
The loss for each epoch on average, goes down. Unlike batch gradient descent which gaurentees there will be a decrease in
the loss, mini-batch gradient descent "eventually" converges to a gradientnt. This is because of the stochastic nature of
mini-batch gradient descent which randomly samples some of the training data for testing. Disadvantages of this is that it
is possible to converge in unlikely patterns and converging patterns are inconsistant. A major advantage is that it runs
much faster because instead of finding the average gradient over the whole section, you can find it over just a certian
portion.

## Summary of findings

... was the best method for understanding...

### Citations
```
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
Wine Quality [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C56S3T.
```