## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
import matplotlib.pyplot as plt

# Optimization module in scipy
from scipy import optimize

import plotData
import sigmoid
import costFunction
import predict

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# define the submission/grader object for this exercise
#grader = utils.Grader()

# Load data
# The first two columns contains the exam scores and the third column
# contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=',')

examples, features = data.shape

#print(examples, features)

X = data[:, :(features - 1)]
y = data[:, (features - 1)]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']);

plotData.myPlot(X,y)
# add axes labels
pyplot.title("Historic Data - Students' Test scores and University admission status")
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not admitted'])
pyplot.show()

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# Initialize fitting parameters
initial_theta = np.zeros(n+1)

# Compute and display initial cost and gradient
cost, grad = costFunction.computeCostnGrad(initial_theta, X, y)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction.computeCostnGrad(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
# set options for optimize.minimize
options= {'maxiter': 400}

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost
# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
# We use truncated Newton algorithm for optimization which is
# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
res = optimize.minimize(costFunction.computeCostnGrad,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

# Plot Boundary
utils.plotDecisionBoundary(plotData.myPlot, theta, X, y)

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 20 on exam 1 and
#  score 80 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid.gOfz(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict.project(theta, X);

print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')
