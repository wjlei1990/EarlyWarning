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
    df = pd.read_hdf("../../measurements.h5")
    t2 = time.time()
    print("Time used in reading: %.2f sec" % (t2 - t1))
    return df


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
    df.drop(["channel", "source"], axis=1, inplace=True)
    mags = df["magnitude"].copy()
    df.drop(["magnitude"], axis=1, inplace=True)

    for col in df.columns:
        if "kurt" in col:
            continue
        if "skew" in col:
            continue
        if "max_amp_loc" in col:
            continue
        df.loc[:, col] = np.log(df[col])

    df.loc[:, "magnitude"] = mags
    return df


def process_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    print("1: ", df.columns[df.isnull().any()].tolist())
    df = df.dropna(axis=1, how='any')
    # print("2: ", df.columns[df.isnull().any()].tolist())
    df = filter_max_amp(df)
    # df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    df = log_scale_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    print("2: ", df.columns[df.isnull().any()].tolist())
    df = df.dropna(axis=1, how='any')

    df_train, df_test = split_data(df, train_percentage=0.8)
    return df_train, df_test


def extract_feature_and_y(df):
    # print(df.isnull().any().any())
    y = np.array(df["magnitude"].as_matrix())
    df.drop("magnitude", axis=1, inplace=True)
    x = np.array(df.as_matrix())
    # print("x nan:", np.isnan(x).any())
    # print("y nan:", np.isnan(y).any())
    return x, y


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


def train(df_train, df_test):
    train_x, train_y = extract_feature_and_y(df_train)
    print("train x and y shape: {0} and {1}".format(
        train_x.shape, train_y.shape))
    test_x, test_y = extract_feature_and_y(df_test)
    print("test x and y shape: {0} and {1}".format(
        test_x.shape, test_y.shape))

    # print("train x nan:", np.isfinite(train_x).any())
    # print("train y nan:", np.isfinite(train_y).any())
    # print("test x nan:", np.isfinite(test_x).any())

    info = train_ridge_linear_model(train_x, train_y, test_x) 
    #info = train_lasso_model(train_x, train_y, test_x) 
    #info = train_EN_model(train_x, train_y, test_x) 

    _mse = mean_squared_error(test_y, info["y"])
    _std = np.std(test_y - info["y"])
    print("MSE on test data: %f" % _mse)
    print("std of error on test data: %f" % _std)

    plot_y(train_y, info["train_y"], test_y, info["y"])


def main():
    df = load_data()
    df_train, df_test = process_data(df)
    train(df_train, df_test)


if __name__ == "__main__":
    main()
