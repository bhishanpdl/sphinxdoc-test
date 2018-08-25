#!python
# -*- coding: utf-8 -*-#
"""
:Title: Univariate Linear Regression.

@author: Bhishan Poudel

@date: Sep 22, 2017

@email: bhishanpdl@gmail.com

The cost function is given by

.. math::

  J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2

Minimizing the cost function w.r.t. w gives two system of liner equations:

.. math::

    w_0N + w_1 \sum_{n=1}^N x_n = \sum_{n=1}^N t_n \\\\\\\\
    w_0 \sum_{n=1}^N x_n + w_1 \sum_{n=1}^N x_n^2 = \sum_{n=1}^N t_nx_n

We solve these normal equations and find the values w0 and w1.
"""
# Imports
import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly

# checking
# import statsmodels.api as sm # sm 0.8.0 gives FutureWarning




def read_data(infile):
    X,t = np.loadtxt(infile,unpack=True)
    return X,t


#
def train(X, t):
    """Implement univariate linear regression to compute w = [w0, w1].

    I solve system of linear equations from lecture 01

    w0 N      + w1 sum_x  = sum_t

    w0 sum_x  + w1 sum_xx = sum_tx

    """

    # Use system of equations
    N = len(t)
    sum_x = sum(X)
    sum_t = sum(t)

    sum_xsp = sum(X*X)
    sum_tx = sum(X*t)

    w1 = (sum_t * sum_x - N * sum_tx) / (sum_x * sum_x - N * sum_xsp)
    w0 = (sum_t - w1 * sum_x) / N

    w = np.array([w0, w1])


    # checking values using statsmodel library
    # w = sm.OLS(t,sm.add_constant(X)).fit().params
    # [-15682.27021631    115.41845202]

    # params w
    # print('y-intercept bias term w0 = {:.2f}'.format(w[0][0]))
    # print('weight term           w1 = {:.2f}'.format(w[1][0]))

    # plt.scatter(X,t)
    # plt.plot(X, X*w[1] + w[0])
    # plt.show()


    return w


def compute_rmse(X,t,w):
    """Compute RMSE on dataset (X, t).

    Note: cost function J is 1/2 of mean squared error.
    RMSE is square root of mean squared error.

    """
    h = X*w[1] + w[0]
    rmse = np.sqrt(np.mean(( h - t )**2) )

    # debug
    # print('w[0] =', w[0])
    # print('w[1] =', w[1])


    # rmse = np.sqrt(((np.dot(X,w.T)- t)**2).mean())

    return rmse


#
def compute_cost(X, t, w):
    """Compute objective function on dataset (X, t)."""
    h = X*w[1] + w[0]
    J = 1/2 *  np.mean(( h - t )**2)
    return J

def univariate_reg(fh_train, fh_test):
    # Read the training and test data.
    Xtrain, ttrain = read_data(fh_train)
    Xtest, ttest = read_data(fh_test)


    # Train model on training examples.
    w = train(Xtrain, ttrain)

    # train
    E_rms_train_uni = compute_rmse(Xtrain, ttrain, w)
    J_train_uni = compute_cost(Xtrain, ttrain, w)

    # test
    E_rms_test_uni = compute_rmse(Xtest, ttest, w)
    J_test_uni = compute_cost(Xtest, ttest, w)

    return E_rms_train_uni, J_train_uni, E_rms_test_uni, J_test_uni


def myplot(fh_train,fh_test,w):
    # matplotlib customization
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    # data
    Xtrain, ttrain = read_data(fh_train)
    Xtest, ttest = read_data(fh_test)
    Xhyptest = Xtest * w[1] + w[0]


    # plot with label, title
    ax.scatter(Xtrain,ttrain,color='b',marker='o', label='Univariate Train')
    ax.scatter(Xtest,ttest,c='limegreen', marker='^', label='Univariate Test')
    ax.plot(Xtest,Xhyptest,'r--',label='Best Fit')

    # set xlabel and ylabel to AxisObject
    ax.set_xlabel('Floor Size (Square Feet)')
    ax.set_ylabel('House Price (Dollar)')
    ax.set_title('Univariate Regression')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('images/Univariate.png')
    plt.show()

##=======================================================================
## Main Program
##=======================================================================
def main():
    """Run main function."""
    parser = argparse.ArgumentParser('Univariate Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../data/univariate',
                        help='Directory for the univariate houses dataset.')
    FLAGS, unparsed = parser.parse_known_args()

    # Data file paths
    fh_train = FLAGS.input_data_dir + "/train.txt"
    fh_test  = FLAGS.input_data_dir + "/test.txt"

    # Print weight vector
    Xtrain, ttrain = read_data(fh_train)
    w = train(Xtrain, ttrain)
    print('Params Univariate: ', w, '\n')

    # Print RMSE and Cost
    E_rms_train_uni, J_train_uni, E_rms_test_uni, J_test_uni = univariate_reg(fh_train, fh_test)

    print("#"*50)
    print("Univariate Regression")

    # Print cost and RMSE on training data.
    print('E_rms_train Univariate: %0.2e' % E_rms_train_uni)
    print('J_train Univariate: %0.2e' % J_train_uni)

    # Print cost and RMSE on test data.
    print("\n")
    print('E_rms_test Univariate: %0.2e' % E_rms_test_uni)
    print('J_test Univariate: %0.2e' % J_test_uni)


    # Plotting
    myplot(fh_train, fh_test,w)




if __name__ == "__main__":
   import time

   # Beginning time
   program_begin_time = time.time()
   begin_ctime        = time.ctime()

   #  Run the main program
   main()


   # Print the time taken
   program_end_time = time.time()
   end_ctime        = time.ctime()
   seconds          = program_end_time - program_begin_time
   m, s             = divmod(seconds, 60)
   h, m             = divmod(m, 60)
   d, h             = divmod(h, 24)
   print("\n\nBegin time: ", begin_ctime)
   print("End   time: ", end_ctime, "\n")
   print("Time taken: {0: .0f} days, {1: .0f} hours, \
     {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
