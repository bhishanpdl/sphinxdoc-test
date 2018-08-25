#!python
# -*- coding: utf-8 -*-#
"""
:Title: Polynomial Regresssion with Ridge Regression.

@author: Bhishan Poudel

@date: Sep 22, 2017

@email: bhishanpdl@gmail.com

The cost function for the Ridge Regression is given by

.. math::

  J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2 + \
  \\frac{\lambda}{2} ||w||^2


Here, the first term is the half mean of the SSE.
And the second term is the shrinkage penalty.
The parameter :math:`\\lambda` is called shrinkage hyperparamter.
Since it is the hyperparamter we chose it from the validation set,
not from the train set.


The term :math:`||w||^2` is the L-2 regularizaton on the SSE term.
The square form is called Ridge Regression and the modulus form
:math:`|w|` is called Lasso Regresssion.


If we have both Lasso and Ridge regression it is called Elastic
Net Regression. Elastic Net Regression have the parameters:
:math:`\\lambda_1 ||w|| + \\lambda_2 ||w||^2`


If a group of predictors are highly correlated among themselves, LASSO
tends to pick only one of them and shrink the other to exact zero (or, very near to zero). Lasso can not do grouped selection and tends to choose only one variable.
It is good for eliminating trivial features but not good for grouped selection.
Lasso gives the sparse model and is computationally less expensive.


On the other hand, Ridge Regression penalize the term on the squares of the
magnitude. The weight are drawn near to zero but not exactly zero. This method
is computationally inefficient.

"""
# Imports
import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm
from numpy import sum, sqrt, array, log, exp
from numpy.core.umath_tests import inner1d
# from sklearn.metrics import mean_squared_error


# Read data matrix X and labels t from text file.
def read_data(infile):
    data = np.genfromtxt(infile, delimiter=None, dtype=np.double)
    X = data[:, :-1 ]
    t = data[:, [-1] ]

    #debug
    # print("X.shape = {}".format(X.shape))
    # print("t.shape = {}".format(t.shape))


    return X, t


def read_data_vander(infile, M):
    """Read the dataset and return vandermonde matrix Xvan for given degree M.

    This function returns vandermonde matrix of 1d array X.

    The vandermonde matrix will be of size len(X) * M.

    But here final Xvan will have shape sample * (degree+1)

    The first column of vandermonde matrix is all 1.

    The last column will be M-1 nth power of second column, NOT Mth power.

    The target t is of the size len(X)*1 i.e. N * 1 (N is sample size)

    Args:
      infile (str): input dataset text file, whitespace separated
      M (int): Degree of polynomial to fit

    .. note::

        Numpy vander function (Vandermonde Matrix).
        Refer `Numpy vander <https://docs.scipy.org/doc/numpy/reference/generated/numpy.vander.html>`_.

        Example::

            x = np.arange(1,6) # x must be 1d array
            x = np.array([1,2,3,4,5])
            xvan3 = np.vander(x, N=3,increasing=True)
            # shape of xvn is len(x) * degree
            # first column is all 1 and last power is excluded
            [[ 1  1  1]
            [ 1  2  4]
            [ 1  3  9]
            [ 1  4 16]
            [ 1  5 25]]

    .. note::

       Numpy array slicing::

        data     = np.arange(20).reshape((5,4))
        col0     = data[:, [0] ]
        col0_1   = data[:, [0,1]]
        col0_1a  = data[:, :2]
        not_col0 = data[:, 1:]
        not_last = data[:, :-1]

      """
    data = np.genfromtxt(infile, delimiter=None, dtype=np.double)
    X = data[:, :-1] # Design matrix X without t values of last column

    # Make the Vandermonde matrix from X
    # To use vandermonde X must be 1d array.
    # X[:, 0] is first column of input data X.
    Xvan = np.vander(X[:, 0], M + 1, increasing =True)
    t = data[:, [-1]]

    # debug
    # print("X.shape = ", X.shape)       # sample, 1
    # print("Xvan.shape = ", Xvan.shape) # sample, degree+1
    # print("t.shape = ", t.shape)       # sample, 1

    return Xvan, t



def train(X, t):
    """Train the data and return the weights w.

    This model uses OLS method to train the data without the penalty term.

    .. math::

      J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2

    Args:

      X (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X.

      t (column): target column vector

    .. note::

       Here the design matrix X should have one extra bias term.

    .. warning::

       The operator @ requires python >= 3.5

    .. note::

       Matrix properties.
       `Wikipedia <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.

       .. math::

         AB \\neq  BA \\\\
         (AB)^T =  B^T A^T \\\\
         (AB)^{-1} =  B^{-1} A^{-1} \\\\
         tr(AB) =  tr(BA) \\\\
         det(AB) = det(A) det(B) = det(B) det(A) = det(BA)

    """
    # w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
    w = (inv(X.T @ X))  @ (X.T @ t)

    # debug
    # print("X.shape = {}".format(X.shape))
    # print("t.shape = {}".format(t.shape))

    return w


def train_regularized(Xm1, t, lam, M):
    """Ridge Regularization (L2 normalization) with square penalty term.

    The cost function for ridge regularization is

    .. math::

      J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2 + \\frac{\lambda}{2} ||w||^2

    Minimizing cost function gives the weight vector w.
    Here :math:`\\lambda` is the hyperparameter chosen from validation set
    with lowest rmse for given values of degrees of polynomial. Different may
    give the same minimum rmse and we choose one of them.

    .. math::

      w = (\lambda N I) (X^T t)

    Args:

      Xm1 (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X.

      t (column): Target column vector. :math:`\\alpha no space before last`

      lam (float): The hyperparameter :math:`\\alpha > \\beta` for the regularization.

      M (int): Degree of the polynomial to fit.

    .. note::

       Here the design matrix X should have one extra bias term.
       The function read_data_vander returns X with one extra

    .. warning::

       The operator @ requires python >= 3.5

    """
    # debug
    # Example M = 9, Xm1 has shape 10,10 and t has shape 10,1
    # print("Xm1.shape = {}".format(Xm1.shape))
    # print("t.shape = {}".format(t.shape))


    # First get the identity matrix of size deg+1 by deg+1
    N = len(t)
    I = np.eye(M + 1)

    # weight for ridge regression
    w_ridge = inv(lam * N * I + Xm1.T @ Xm1 )   @ (Xm1.T @ t)

    return w_ridge

# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
    """Compute the RMSE.

    RMSE is the root mean square error.

    .. math:: RMSE = \sqrt{\sum_{i=1}^{n}  \\frac{(h_i - t_i)^2}{n} }

    Here the hypothesis h is the matrix product of X and w.
    Hypothesis h should have the same dimension as target vector t.


    The norm of 1d vector can be calculated as given in `Wikipedia Norm <https://en.wikipedia.org/wiki/Norm_(mathematics)>`_.

    :math:`||x|| = \sqrt{x_1^2 + x_2^2 + ... + x_n^2}`

    There are several methods to calculate hypothesis and norms.

    `Refer to stackoverflow <https://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy>`_.


    Python codes to calculate norm of a 1d vector::

        import numpy as np
        from numpy.core.umath_tests import inner1d

        V = np.random.random_sample((10**6,3,)) # 1 million vectors
        A = np.sqrt(np.einsum('...i,...i', V, V))
        B = np.linalg.norm(V,axis=1)
        C = np.sqrt((V ** 2).sum(-1))
        D = np.sqrt((V*V).sum(axis=1))
        E = np.sqrt(inner1d(V,V))

        print [np.allclose(E,x) for x in [A,B,C,D]] # [True, True, True, True]

        import cProfile
        cProfile.run("np.sqrt(np.einsum('...i,...i', V, V))") # 3 function calls in 0.013 seconds
        cProfile.run('np.linalg.norm(V,axis=1)')              # 9 function calls in 0.029 seconds
        cProfile.run('np.sqrt((V ** 2).sum(-1))')             # 5 function calls in 0.028 seconds
        cProfile.run('np.sqrt((V*V).sum(axis=1))')            # 5 function calls in 0.027 seconds
        cProfile.run('np.sqrt(inner1d(V,V))')                 # 2 function calls in 0.009 seconds.
        # np.eisensum can also be written as
        # np.sqrt(np.einsum('ij,ij->i',a,a))
        # NOTE:
        # inner1d is ~3x faster than linalg.norm and a hair faster than einsum
        # For small data set ~1000 or less numpy is faster
        # a_norm = np.sqrt(a.dot(a)) is faster than np.sqrt(np.einsum('i,i', a, a))



    We can calculate hypothesis as:
    :math:`h = X @ w`


    Or, we may use:
    :math:`h = X .dot(w)`

    One of the fastest methods to calculate the hypothesis is the
    np.einsum method. The explanation of `einsum` is given below:


    For example::

      w     X      t
      2,1   10,2   10,1
      i,j   k, i   k,j

      h = np.einsum('ij,ki->kj', w, X) = X @ w

    To find the norm of the residual matrix h-t we may use
    the code::

      # Using np.linalg.norm
      ht_norm = np.linalg.norm(h - t)

      # inner1d is the faster than np.linalg.norm subroutine.
      from numpy.core.umath_tests import inner1d
      ht_norm = np.sqrt(inner1d(h-t,h-t))

    To calculate RMSE we can also use sklearn library::

      from sklearn.metrics import mean_squared_error
      rmse = mean_squared_error(h, t)**0.5

    """

    # # print("w.shape = {}, X.shape = {} t.shape = {}".format(w.shape,X.shape,t.shape))
    # h = X.dot(w)
    # h = X @ w
    h = np.einsum('ij,ki->kj', w, X)
    sse = (h - t) ** 2
    mse = np.mean(sse)
    rmse = np.sqrt(mse)

    # Method from sklearn
    # rmse = mean_squared_error(X@w, t)**0.5 # 7.10437e-04

    return np.double(rmse)

def myplot(X, t,label,style):
    # matplotlib customization
    plt.style.use('ggplot')
    fig, ax = plt.subplots()


    # plot with label, title
    ax.plot(X,t,style,label=label)

    # set xlabel and ylabel to AxisObject
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Polynomial ' + label + ' data')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('images/hw01qn3_'+ label+'.png')
    plt.show()
    plt.close()

def plot_alldata():
    labels = ['dataset','devel','train','test']
    styles = ['ro','g^','bo','k>']
    for i, label in enumerate(labels):
        X, t = read_data('../data/polyfit/{}.txt'.format(label))
        myplot(X,t,label,styles[i])

def fit_unreg_poly(fh_train,fh_test,fh_valid,M):
    """Unregularized polynomial regression for degree 0 to 9.

    Here, the degree of the polynomial varies from 0-9.

    Args:
      fh_train (str): File path for train data
      fh_test (str): File path for test data
      fh_valid (str): File path for validation data

    Return: None

    """
    # Get Vandermonde matrix X and target t
    # First column is all 1 and shape of X is sample * deg+1
    # M = 9 X has 10 columns, with first column all ones.
    Xtrain, ttrain = read_data_vander(fh_train,M)
    Xtest, ttest   = read_data_vander(fh_test, M)
    Xvalid, tvalid = read_data_vander(fh_valid,M)

    # Look how they are
    # print("Xtrain = {}".format(Xtrain))
    # print("Xtrain.shape = {}".format(Xtrain.shape))
    # print("Xtrain[0] = {}".format(Xtrain[0]))


    # Values of degree of polynomials
    Mvals = np.arange(10)
    E_train, E_test = [], []

    for m in Mvals:

        # X values to use
        # XXX: Here inside for loop Xtrainm1 can not be written Xtrain
        # :, means all rows
        # 0:m+1 means columns 0 to m
        #
        # for loop m = 0 ... 9 we choose vandermonde matrix from above vander
        # matrix of degree 9.
        # Xtrain and Xtest both have 10 columns above for loop.
        Xtrainm1 = Xtrain[:, 0:m+1]
        Xtestm1  = Xtest[:, 0:m+1]

        # get weight w = (inv(X.T @ X))  @ (X.T @ t)
        #  w is a column vector of shape m+1, 1 (e.g. 10,1 for m=9 )
        w = train(Xtrainm1, ttrain)

        # get rmse = mean_squared_error(X@w, t)**0.5
        # NOTE:  h= X @ w
        # E = RMSE is scalar float number
        E1 = compute_rmse(Xtrainm1, ttrain, w)
        E2 = compute_rmse(Xtestm1, ttrain, w)

        # Append values to rmse list
        E_train = np.append(E_train, E1)
        E_test  = np.append(E_test, E2)

        # debug
        # print("\n")
        # print("#"*50)
        # print("degree = {}".format(m))
        # print("w.shape = {}".format(w.shape))
        # print("ttrain.shape = ", ttrain.shape)
        # print('Train RMSE = {:.5e}'.format(E1))

    # Elegant way of computing rmse for train and test
    # compute_rmse(X,t,w)
    # E_train = [compute_rmse( Xtrain[:, 0:m+1], ttrain, train(
    #                                                   Xtrain[:, 0:m+1], ttrain))
    #            for m in Mvals]
    #
    # E_test  = [compute_rmse(Xtest[:, 0:m+1],  ttrain, train(
    #                                                  Xtrain[:, 0:m+1], ttrain))
    #            for m in Mvals]

    # Plot unregularized polynomial regression
    plt.plot(Mvals, E_train, 'r-', label = "train")
    plt.plot(Mvals, E_test ,  'b--' , label = "test" )
    plt.xlabel("degree (M)")
    plt.ylabel("$E_{rms}$")
    plt.title("Unregularized Univariate Polynomial Regression")
    plt.legend()
    plt.savefig("images/unreg_poly_reg.png")
    plt.show()
    plt.close()

def fit_reg_poly(fh_train,fh_test,fh_valid):
    """Regularized polynomial with fixed degree M = 9.

    Here, ln lambda varies from -50 to 0 with step size 5.
    I.e. lamdda varies from exp(-50) to 1.

    We have to calculate weight vector w for each lambda.
    For degree M = 9, weight vector w has 10 elements.

    We also find RMSE for train and validation set for each lambda.
    Then we choose the hyperparameter lambda that gives the lowest
    RMSE on the validation set.

    Args:
      fh_train (str): File path for train data
      fh_test (str): File path for test data
      fh_valid (str): File path for validation data

    Return:
      lam_min_rmse_valid (float): The value of hyper parameter lambda
      that minimizes RMSE for the validation set.


    """
    # Degree of polynomial
    M = 9

    # Values of shrinkage hyperparameter lambda
    log_lambda_ridge = np.arange(-50, 0+5, 5)
    lambda_ridge = np.exp(log_lambda_ridge)

    # X,t for train,test and validation
    # vander gives bias term itself
    # Here, X matrix has M+1 columns. First column is all ones.
    Xtrain, ttrain = read_data_vander(fh_train,M)
    Xtest, ttest   = read_data_vander(fh_test, M)
    Xvalid, tvalid = read_data_vander(fh_valid,M)


    # Initiliaze rmse_train and rmse_validation
    E_train_ridge = []
    E_valid_ridge  = []
    for lam in lambda_ridge:

        # print("lam = {:.2e} log(lam) = {:.0f}".format(lam, np.log(lam)))
        # get w from training (note that we get lambda from validation)
        # w_ridge = inv(lam * N * I + Xm1.T @ Xm1 )   @ (Xm1.T @ t)
        w_ridge = train_regularized(Xtrain,ttrain,float(lam), M)

        # rmse for train and valid
        E1 = compute_rmse(Xtrain, ttrain, w_ridge)
        E2 = compute_rmse(Xvalid, tvalid, w_ridge)

        # Append rmse to the list
        E_train_ridge = np.append(E_train_ridge, E1)
        E_valid_ridge = np.append(E_valid_ridge, E2)

    print("\n")
    print("#"*60)
    print("Ridge Regression:")
    print("Degree of polynomial M = ", M)
    print('log(lam)   lam          E_train             E_valid')
    for i, lam in enumerate(lambda_ridge) :
        print(' {:.2f}   {:.5e} {:.14f}   {:.14f} '.format(
            np.log(lam), lam, E_train_ridge[i], E_valid_ridge[i] ))

    lam_min_rmse_valid_idx = np.where(E_valid_ridge == min(E_valid_ridge))[0]
    lam_min_rmse_valid_idx_last = lam_min_rmse_valid_idx[-1]
    idx = lam_min_rmse_valid_idx_last
    lam_min_rmse_valid = lambda_ridge[idx]
    print('-'*60)
    print("{}    {:.5e}                    {:.14f}".format(
        log(lambda_ridge[idx]), lam_min_rmse_valid, E_valid_ridge[idx]))

    # Plot
    plt.plot(log_lambda_ridge, E_train_ridge,
             color='r', marker='o', ls='-', label = "train")
    plt.plot(log_lambda_ridge, E_valid_ridge,
             color='b', marker= 'o', ls='--', label = "validation")
    plt.xlabel("log $\lambda$")
    plt.ylabel("$E_{rms}$")
    plt.title("Polynomial Regression Cross Validation")
    plt.legend()
    plt.savefig("images/reg_poly_reg.png")
    plt.show()
    plt.close()

    # Plot table
    fig, ax =plt.subplots()
    clust_data = np.array([log_lambda_ridge,lambda_ridge, E_train_ridge, E_valid_ridge]).T
    collabel=("log($\lambda$)", "$\lambda$", "$E_{train}$","$E_{valid}$")
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=clust_data,colLabels=collabel,loc='center')
    ax.plot(clust_data[:,0],clust_data[:,1])
    plt.title('Choosing hyperparameter $\lambda$ ')
    plt.savefig('images/table_reg_poly_fitting.png')
    plt.show()
    plt.close()

    return lam_min_rmse_valid

def comparison(fh_train,fh_test,fh_valid, lam_min_rmse_valid,M):
    """Compare the unregularized and regularized polynomial regression.

    Here, we compare test RMSE with and without ridge regularization for
    9th degree univariate polynomial regression.

    While fitting test data with ridge regression, we use the hyper parameter
    lambda that gives the minimum rmse in the cross-validation set.

    Args:
      fh_train (str): File path for train data
      fh_test (str): File path for test data
      fh_valid (str): File path for validation data
      lam_min_rmse_valid (float): The hyperparameter lambda that gives minimum
      rmse on cross validation set.

    Return: None

    """
    print("\n")
    print('#'*50)
    print("Comparison of regularized and unregularized cases:")
    # print("lam_min_rmse_valid = {}".format(lam_min_rmse_valid))

    # Get X and t from dataset
    Xtrain, ttrain = read_data_vander(fh_train,M)
    Xtest, ttest   = read_data_vander(fh_test, M)
    Xvalid, tvalid = read_data_vander(fh_valid,M)

    # Unregularized
    w = train(Xtrain, ttrain)
    E_rms_test = compute_rmse(Xtest, ttest, w)
    print('Test RMSE without regularization for M = 9: %0.4f.' % E_rms_test)

    w_ridge = train_regularized(Xtrain,ttrain, float(lam_min_rmse_valid), M)
    E_rms_test = compute_rmse(Xtest, ttest, w_ridge)
    print('Test RMSE with    regularization for M = 9: %0.4f.' % E_rms_test)



##=======================================================================
## Main Program
##=======================================================================
def main():
    """Run main function."""
    parser = argparse.ArgumentParser('Univariate Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../data/polyfit',
                        help='Directory for the polyfit dataset.')
    FLAGS, unparsed = parser.parse_known_args()

    ##=======================================================================
    ## Part 3b: Plotting dataset
    ##=======================================================================
    # Plot dataset
    plot_alldata()

    ##=======================================================================
    ## Part 3d: Polynomial Univariate Ridge Regularization
    ##=======================================================================
    fh_train = FLAGS.input_data_dir + "/train.txt"
    fh_test = FLAGS.input_data_dir + "/test.txt"
    fh_valid = FLAGS.input_data_dir + "/devel.txt"

    # unregularized
    fit_unreg_poly(fh_train,fh_test,fh_valid,M=9)

    # regularized
    lam_min_rmse_valid = fit_reg_poly(fh_train,fh_test,fh_valid)

    # compare them
    comparison(fh_train,fh_test,fh_valid, lam_min_rmse_valid,M=9)

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
    print("\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
