import time
import sys      # NOQA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regressor import train_ridge_linear_model, train_lasso_model, \
    train_EN_model
from sklearn.metrics import mean_squared_error


def load_data():
    t1 = time.time()
    train_x = pd.read_hdf("../../yajun2/feature_train.h5")
    train_y = pd.read_hdf("../../yajun2/outcome_train.h5")
    test_x = pd.read_hdf("../../yajun2/feature_test.h5")
    test_y = pd.read_hdf("../../yajun2/outcome_test.h5")
    t2 = time.time()
    print("Time used in reading: %.2f sec" % (t2 - t1))
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x,
            "test_y": test_y}


def filter_max_amp(df):
    # comp_list = ["BHZ", "BHE", "BHN"]:
    comp_list = ["BHZ"]
    for comp in comp_list:
        shape0 = df.shape
        df = df[df["%s.disp.max_amp" % comp] < 1.0]
        shape1 = df.shape
        print("Shape change after filter max amp on {0}: {1} --> {2}".format(
            comp, shape0, shape1))

    return df


def split_data(df, train_percentage=0.8):
    msk = np.random.rand(len(df)) < train_percentage
    train = df[msk]
    test = df[~msk]
    return train, test


def log_scale_features(df):
    # drop un-used columns out
    if "channel" in df.columns:
        df.drop("channel", axis=1, inplace=True)

    if "source" in df.columns:
        df.drop("source", axis=1, inplace=True)

    for col in df.columns:
        if "kurt" in col:
            continue
        if "skew" in col:
            continue
        if "max_amp_loc" in col:
            continue
        df.loc[:, col] = np.log(df[col])

    return df


def plot_y(train_y, train_y_pred, test_y, test_y_pred):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(train_y, train_y_pred, alpha=0.5)
    plt.plot([2, 8], [2, 8])
    #plt.xlim([2.5, 7.5])
    #plt.ylim([2.5, 7.5])
    plt.title("Train")
    plt.subplot(1, 2, 2)
    plt.scatter(test_y, test_y_pred, alpha=0.5)
    plt.plot([2, 8], [2, 8])
    plt.title("Test")
    #plt.xlim([2.5, 7.5])
    #plt.ylim([2.5, 7.5])
    plt.show()


def over_sample_train_data(train_x, train_y, threshold=4.5, over_sample=10):
    if over_sample == 0:
        return

    row_counts = 0
    x_os = None
    y_os = []

    for irow in range(len(train_y)):
        if train_y[irow] > threshold:
            row_counts += 1

            _expand_x = np.tile(train_x[irow, :], (over_sample, 1))
            if x_os is None:
                x_os = _expand_x
            else:
                x_os = np.concatenate((x_os, _expand_x), axis=0)

            _expand_y = [train_y[irow], ] * over_sample
            y_os.extend(_expand_y)

    if len(x_os) != len(y_os):
        raise ValueError("Dimension mismatch between x_os and y_os: %s, %d"
                         % (x_os.shape, len(y_os)))

    print("Number of rows expanded initially: %d" % row_counts)
    print("Number of rows added(over_sample=%d): %d" % (over_sample, len(y_os)))

    train_x_os = np.concatenate((train_x, x_os), axis=0)
    train_y_os = np.concatenate((train_y, y_os), axis=0)

    print("Shape change of x after expanding: {0} --> {1}".format(
        train_x.shape, train_x_os.shape))
    print("Shape change of y after expanding: {0} --> {1}".format(
        train_y.shape, train_y_os.shape))

    return train_x_os, train_y_os


def train_linear_model(df_train_x, df_train_y, df_test_x, df_test_y,
                       model="ridge"):

    train_x = df_train_x.as_matrix()
    train_y = df_train_y.as_matrix()[:, 0]
    test_x = df_test_x.as_matrix()
    test_y = df_test_y.as_matrix()[:, 0]
    print("train x and y shape: ", train_x.shape, train_y.shape)
    print("test x and y shape: ", test_x.shape, test_y.shape)

    train_x, train_y = over_sample_train_data(
        train_x, train_y, threshold=4.5, over_sample=50)

    if model.lower() == "ridge":
        info = train_ridge_linear_model(
            train_x, train_y, test_x, sample_weight=None) 
    elif model.lower() == "lasso":
        info = train_lasso_model(train_x, train_y, test_x) 
    elif model.lower() == "en":
        info = train_EN_model(train_x, train_y, test_x) 
    else:
        raise ValueError("Error in model name: %s" % model)

    print("test_y and test_y_pred: ", test_y.shape, info["y"].shape)
    _mse = mean_squared_error(test_y, info["y"])
    _std = np.std(test_y - info["y"])
    print("MSE on test data: %f" % _mse)
    print("std of error on test data: %f" % _std)
    print("np mse: %f" % (((test_y - info["y"]) ** 2).mean()))

    plot_y(train_y, info["train_y"], test_y, info["y"])


def main():
    data = load_data()
    train_x = log_scale_features(data["train_x"])
    test_x = log_scale_features(data["test_x"])
    train_linear_model(train_x, data["train_y"], test_x, data["test_y"],
                       model="ridge")


if __name__ == "__main__":
    main()
