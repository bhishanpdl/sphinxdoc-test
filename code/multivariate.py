#!python
# -*- coding: utf-8 -*-#
"""
:Title: Multivariate Linear Regression.

@author: Bhishan Poudel

@date: Sep 22, 2017

@email: bhishanpdl@gmail.com

The cost function is given by

.. math::

  J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2

Minimizing the cost function w.r.t. w gives the solution:

.. math::

  w = np.linalg.lstsq(X1,t)[0]
"""
# Imports
import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly
from numpy.core.umath_tests import inner1d
from numpy.linalg import norm,lstsq,inv

# for univariate multivariate comparison
from univariate import univariate_reg

# checking
#


# Read data matrix X and labels t from text file.
def read_data(infile):
    """Read the datafile.

    Args:
      infile (str): path to datafile

    """
    # data = np.loadtxt(infile)
    data = np.genfromtxt(infile, delimiter=None, dtype=float)
    X = data[:, :-1]
    t = data[:, [-1]]
    return X, t


#----------------------------------------------------------------------#
#  function: train                                                     #
#----------------------------------------------------------------------#
# Here no. of features M = 3 (floor, bedrooms, age)
# Implement normal equations to compute w = [w0, w1, ..., w_M].
def train(X1, t):
    """Train the data and return the weights w.

    Args:

      X1 (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X1.

      t (column): target column vector

    .. note::

       Here the design matrix X1 should have one extra bias term.

    .. warning::

       The operator @ requires python >= 3.5

    """
    # Method 1
    w = np.linalg.inv(X1.T.dot(X1)) .dot(X1.T) .dot(t)
    w = np.array(w).reshape(1, len(w)) # make 1d row array

    # Method 2
    # w = (inv(X1.T @ X1)) @ X1.T @ t
    # w = np.array(w).reshape(1, len(w)) # make 1d row array

    # Method 3
    # w = np.linalg.lstsq(X1,t)[0]
    # w = np.array(w).reshape(1, len(w)) # make 1d row array

    return w


# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
    """Compute the RMSE.

    RMSE is the root mean square error.

    .. math:: RMSE = \sqrt{\sum_{i=1}^{n}  \\frac{(h - t)^2}{n} }

    h is the hypothesis.

    :math:`h = X w^T`

    To find the norm of the residual matrix h-t we may use
    the code::

      # inner1d is the fastest subroutine.
      from numpy.core.umath_tests import inner1d
      np.sqrt(inner1d(h-t,h-t))

      # We can also use another method:
      ht_norm = np.linalg.norm(h - t)

    """

    # Method 1
    h = np.dot(X, w.T) # h = X @ w.T
    rmse = np.sqrt(((h - t) ** 2).mean())

    # Method 2
    # h = np.dot(X, w.T)
    # ht_norm = np.sqrt(inner1d(h-t,h-t))
    # rmse = ht_norm / np.sqrt(len(X))
    # rmse = rmse[0]

    # Method 3
    # norm is square root of sum of squares
    # rmse is norm/ sqrt(n)
    #
    # h = np.dot(X, w.T)
    # ht_norm = np.linalg.norm(h - t)
    # rmse = ht_norm / np.sqrt(len(X))

    # Checking
    # print("t.shape = ", t.shape)
    # print("w.shape = ", w.shape)
    # print("h.shape = ", h.shape)
    # print("X.shape = ", X.shape)
    # print("len(X1) = ", len(X))

    # Checking
    # rmse = 0.0
    # try:
    #     from sklearn.metrics import mean_squared_error
    #     rmse = mean_squared_error(h, t)**0.5
    #     rmse = np.sqrt(np.square(h - t).mean())
    # except:
    #     print('Error: The library sklearn not installed!')


    # Return RMSE
    return rmse

# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
    """Compute the cost function.

    .. math:: J = \\frac{1}{2n} \sum_{i=1}^{n}  \\frac{(h - t)^2}{n}

    """

    # Compute cost
    # N = float(len(t))
    # h = np.dot(X, w.T)   # h = X @ w.T
    # J = np.sum((h - t) ** 2) /2 / N

    # One liner
    J = np.sum((X @ w.T - t) ** 2) /2 / float(len(t))


    return J

def check_results(y_train, x1_train):
    """Multivariate Regression with statsmodels.api

    Args:
      y_train (float): target column vector of floats.
      x1_train (array): features+1 dimensional numpy array

    This fits the multivariate linear regression in four lines::

        import statsmodels.api as sm
        model = sm.OLS(y_train, x1_train)
        result = model.fit()
        print (result.summary())



    """
    try:
        import statsmodels.api as sm
        model = sm.OLS(y_train, x1_train)
        result = model.fit()
        print (result.summary())

    except:
        print('Error: statsmodels libray not found!')




##=======================================================================
## Main Program
##=======================================================================
def main():
    parser = argparse.ArgumentParser('Multivariate Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../data/multivariate',
                        help='Directory for the multivariate houses dataset.')
    FLAGS, unparsed = parser.parse_known_args()

    # Read the training and test data.
    Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
    Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

    #  Append ones to the first column
    X1train  = np.append(np.ones_like(ttrain), Xtrain, axis= 1)
    X1test  = np.append(np.ones_like(ttest), Xtest, axis= 1)

    # debug
    # print(" First column X1train[:, [0]] = \n{}".format(X1train[:, [0]]))
    # print(" First row X1train[0] = \n{}".format(X1train[0]))


    # Train model on training examples.
    w = train(X1train, ttrain)

    # Print model parameters.
    print("#"*50)
    print("Multivariate Regression")
    print('Params Mulitvariate: ', w[0], '\n')


    # Print cost and RMSE on training data.
    # train
    E_rms_train_multi = compute_rmse(X1train, ttrain, w)
    J_train_multi = compute_cost(X1train, ttrain, w)

    # test
    E_rms_test_multi = compute_rmse(X1test, ttest, w)
    J_test_multi = compute_cost(X1test, ttest,w)


    print('E_rms_train Multivariate: %0.2e' % E_rms_train_multi)
    print('J_train Multivariate: %0.2e' % J_train_multi)

    # Print cost and RMSE on test data.
    print("\n")
    print('E_rms_test Multivariate: %0.2e' % E_rms_test_multi)
    print('J_test Multivariate: %0.2e' % J_test_multi)

    #===========================================================
    print("\n")
    print("="*50)
    print("Comparison of Univariate and Multivariate")
    fh_train_uni = '../data/univariate/train.txt'
    fh_test_uni = '../data/univariate/test.txt'

    E_rms_train_uni, J_train_uni, E_rms_test_uni, J_test_uni = univariate_reg(fh_train_uni, fh_test_uni)

    print('Univariate             Multivariate')
    print("E_train = {:.4e}     {:.4e}".format(E_rms_train_uni, E_rms_train_multi))
    print("E_test  = {:.4e}     {:.4e}".format(E_rms_test_uni , E_rms_test_multi))
    print("J_train = {:.4e}     {:.4e}".format(J_train_uni    , J_train_multi))
    print("J_test  = {:.4e}     {:.4e}".format(J_test_uni     , J_test_multi))
    print("-"*50)
    print('Multivariate Params are given below:')
    print([ "{:.2e}".format(x) for x in list(w[0])])
    print("#"*20, "The End", "#"*20)
    print("\n")

    # Check result with statsmodels
    # check_results(ttrain, X1train)

if __name__ == "__main__":
    # Run main function
    main()
