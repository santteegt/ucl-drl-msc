from fastFM.datasets import make_user_item_regression
from fastFM import mcmc
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
import scipy.sparse as sp
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
import time

parser = argparse.ArgumentParser(description='Transform Movielens data set to lightsvm format.')
parser.add_argument('base_dir', type=str, help='Base directory')
parser.add_argument('file', type=str, help='Dataset file to transform')
parser.add_argument('--kfold', type=bool, default=False, help='Process k-fold cross validation')
parser.add_argument('--plot', type=bool, default=False, help='Plot RMSE and parameters')
parser.add_argument('--iter', type=int, default=50, help='Number of samples for MCMC init')
parser.add_argument('--rank', type=int, default=8, help='Dimensionality of FM latent vectors')
parser.add_argument('--std-dev', type=float, default=0.1, help='Initial Standard Deviation')

# parameters
n_iter = 50
# rank = 4
rank = 8
# seed = 333
seed = 123
step_size = 1
# std_dev = 0.1
std_dev = 0.2


def runFM(X_train, y_train, X_test, y_test):
    """
    X_train = sp.csc_matrix(np.array([[6, 1],
                                [2, 3],
                                [3, 0],
                                [6, 1],
                                [4, 5]]), dtype=np.float64)
    y_train = np.array([298, 266, 29, 298, 848], dtype=np.float64)
    X_test = X_train
    y_test = y_train
    """

    """
    X, y, coef = make_user_item_regression(label_stdev=.4, random_state=seed)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    X_test = X_train
    y_test = y_train
    """

    start_time = time.time()

    fm = mcmc.FMRegression(n_iter=0, rank=rank, random_state=seed, init_stdev=std_dev)
    # initalize coefs
    fm.fit_predict(X_train, y_train, X_test)

    rmse_test = []
    rmse_new = []
    hyper_param = np.zeros((n_iter -1, 3 + 2 * rank), dtype=np.float64)
    for nr, i in enumerate(range(1, n_iter)):
        fm.random_state = i * seed
        y_pred = fm.fit_predict(X_train, y_train, X_test, n_more_iter=step_size)
        rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param[nr, :] = fm.hyper_param_

    print '------- restart ----------'
    values = np.arange(1, n_iter)
    rmse_test_re = []
    hyper_param_re = np.zeros((len(values), 3 + 2 * rank), dtype=np.float64)
    for nr, i in enumerate(values):
        fm = mcmc.FMRegression(n_iter=i, rank=rank, random_state=seed)
        y_pred = fm.fit_predict(X_train, y_train, X_test)
        rmse_test_re.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param_re[nr, :] = fm.hyper_param_

    print "Process finished in {} seconds".format(time.time() - start_time)
    print "Min RMSE on warmup model: {}".format(rmse_test[-1])
    print "Min RMSE on retrained model: {}".format(rmse_test_re[-1])

    return rmse_test, hyper_param, rmse_test_re, hyper_param_re


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = args.base_dir
    filename = args.file
    k_fold = args.kfold

    #parameters
    n_iter = args.iter
    rank = args.rank
    std_dev = args.std_dev

    # offset = '../../fastFM-notes/benchmarks/'
    offset = '../../'
    # train_path = offset + "data/ml-100k/u1.base.libfm"
    train_path = offset + os.path.join("data", base_dir, filename) # /ml-100k/u.sorted.data.libfm"
    # test_path = offset + "data/ml-100k/u1.test.libfm"
    # train_path = offset + "data/ml-100k/u1.base.sorted.libfm"
    # test_path = offset + "data/ml-100k/u1.test.sorted.libfm"
    # test_path = train_path

    from sklearn.datasets import load_svmlight_file
    # X_train, y_train = load_svmlight_file(train_path)
    # X_test,  y_test= load_svmlight_file(test_path)


    rmse_test = []
    rmse_test_re = []
    hyper_param = []
    hyper_param_re = []

    if not k_fold:
        X, y = load_svmlight_file(train_path)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
        X_train = sp.csc_matrix(X_train)
        X_test = sp.csc_matrix(X_test)
        # add padding for features not in test
        X_test = sp.hstack([X_test, sp.csc_matrix((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])

        rmse_test, hyper_param, rmse_test_re, hyper_param_re = runFM(X_train, y_train, X_test, y_test)
    else:
        #perform 5-fold cross validation
        rmse_test_fold = []
        rmse_test_re_fold = []
        hyper_param_fold = []
        hyper_param_re_fold = []

        k = 5

        for i in range(0, k):
            train_path = offset + os.path.join("data", base_dir, "u{}.base.libfm".format(i+1))
            test_path = offset + os.path.join("data", base_dir, "u{}.test.libfm".format(i+1))
            X_train, y_train = load_svmlight_file(train_path)
            X_test,  y_test= load_svmlight_file(test_path)

            X_train = sp.csc_matrix(X_train)
            X_test = sp.csc_matrix(X_test)
            # add padding for features not in test
            if X_train.shape[1] > X_test.shape[1]:
                X_test = sp.hstack([X_test, sp.csc_matrix((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])
            else:
                print "dia opuesto"
                X_train = sp.hstack([X_train, sp.csc_matrix((X_train.shape[0], X_test.shape[1] - X_train.shape[1]))])

            print "Executing {}-fold".format(i+1)
            rmse_test, hyper_param, rmse_test_re, hyper_param_re = runFM(X_train, y_train, X_test, y_test)
            rmse_test_fold.append(rmse_test)
            hyper_param_fold.append(hyper_param)
            rmse_test_re_fold.append(rmse_test_re)
            hyper_param_re_fold.append(hyper_param_re)

        rmse_test = np.sum(rmse_test_fold, axis=0) / k
        hyper_param = np.sum(hyper_param_fold, axis=0) / k
        rmse_test_re = np.sum(rmse_test_re_fold, axis=0) / k
        hyper_param_re = np.sum(hyper_param_re_fold, axis=0) / k

    if args.plot:

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))

        values = np.arange(1, n_iter)
        x = values * step_size
        burn_in = 5
        x = x[burn_in:]

        #with plt.style.context('ggplot'):
        axes[0, 0].plot(x, rmse_test[burn_in:], label='test rmse', color="r")
        axes[0, 0].plot(values[burn_in:], rmse_test_re[burn_in:], ls="--", color="r")
        axes[0, 0].legend()

        axes[0, 1].plot(x, hyper_param[burn_in:,0], label='alpha', color="b")
        axes[0, 1].plot(values[burn_in:], hyper_param_re[burn_in:,0], ls="--", color="b")
        axes[0, 1].legend()

        axes[1, 0].plot(x, hyper_param[burn_in:,1], label='lambda_w', color="g")
        #axes[2].plot(x, hyper_param[:,2], label='lambda_V', color="r")
        axes[1, 0].plot(values[burn_in:], hyper_param_re[burn_in:,1], ls="--", color="g")
        #axes[2].plot(values, hyper_param_re[:,2], label='lambda_V', ls="--", color="r")
        axes[1, 0].legend()

        axes[1, 1].plot(x, hyper_param[burn_in:,3], label='mu_w', color="g")
        #axes[3].plot(x, hyper_param[:,4], label='mu_V', color="r")
        axes[1, 1].plot(values[burn_in:], hyper_param_re[burn_in:,3], ls="--", color="g")
        #axes[3].plot(values, hyper_param_re[:,4], label='mu_V', ls="--", color="r")
        axes[1, 1].legend()

        plt.show()
        print "plotting done"

    #plt.savefig("../../fastFM-notes/jmlr/figs/mcmc_trace.pdf", bbox_inches='tight')
