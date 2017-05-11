from __future__ import print_function, division
import pandas as pd
import numpy as np
import json
import math
import operator
from sklearn import preprocessing


INT_COL_DTYPE = np.int16
FLOAT_COL_DTYPE = np.float
CODE_RESERVE = set([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -14])
SEP = "|+|"


def load_json(fn):
    with open(fn) as fh:
        return json.load(fn)


def dump_json(content, fn):
    with open(fn, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)


def load_csv(fn):
    """ Load csv file into memory """
    df = pd.read_csv(fn, low_memory=False)
    return df


def load_hdf5(fn):
    df = pd.read_hdf(fn)
    return df


def check_dtypes(data):
    """
    Check the data types of each column in dataframe. Sometimes
    columns are "objects", which is not numerical.
    """
    dtypes = data.dtypes
    results = {}
    for idx, dt in enumerate(dtypes):
        if dt not in results:
            results[dt] = []
        results[dt].append(idx)
    return results


def get_columns_dtype(df):
    """ Get the dtypes of each column in the dataframe """
    results = {}
    for col in df:
        dt = df[col].dtype
        if dt not in results:
            results[dt] = []
        results[dt].append(col)
    return results


def non_reserved_vset(vset):
    new_vset = []
    for v in vset:
        if v in CODE_RESERVE:
            continue
        else:
            new_vset.append(v)

    return set(new_vset)


def add_neg_feature_columns(df, columns=None, verbose=True):
    """
    In the questionair, -9 -> -1 are reserved code for un-normal
    answers. So we create a new feature if the answer is in the
    range of [-1, -9]
    """
    if columns is None:
        columns = df.columns

    shape1 = df.shape
    ncols_add = 0
    for col in columns:
        new_col = "%s%s_neg_feature_" % (col, SEP)
        new_col_data = \
            ((df[col] < 0.0) & (df[col] >= -9.1)).astype(INT_COL_DTYPE)
        if new_col_data.sum() > 0:
            df[new_col] = new_col_data
            ncols_add += 1
    shape2 = df.shape

    if verbose:
        print("Columns added negative feature: %d/%d" %
              (ncols_add, len(columns)))
        print("Shape change after adding negative feature: {0} --> "
              "{1}".format(shape1, shape2))


def add_nan_feature_columns(df, columns=None, verbose=True):
    """
    add one new feature for each original column to keep the
    information of NaN values BEFORE impute.

    :param df: input dataframe
    :type df: pandas.DataFrame
    :return: No. change inplace
    """
    if columns is None:
        columns = df.columns

    shape1 = df.shape
    ncols_add = 0
    for col in columns:
        new_col = "{0}._nan_feature_".format(col)
        new_col_data = pd.isnull(df[col]).astype(INT_COL_DTYPE)
        if new_col_data.sum() > 0:
            df[new_col] = new_col_data
            ncols_add += 1
    shape2 = df.shape

    if verbose:
        print("Columns added nan feature: %d/%d" %
              (ncols_add, len(columns)))
        print("Shape change after adding the nan feature: {0} --> "
              "{1}".format(shape1, shape2))


def _isnan(v):
    if (isinstance(v, float) or isinstance(v, int)) and math.isnan(v):
        return True
    else:
        return False


def extract_column_vset(col_data, dropna=True):
    """ Extract the set of values and mode in one column data """
    vcounts = pd.Series(col_data).value_counts(dropna=dropna)

    index = vcounts.index
    maxv = 0
    maxk = None
    for k, v in vcounts.iteritems():
        if v > maxv:
            maxv = v
            maxk = k
    assert maxk == index[0]

    return index, index[0], vcounts


def extract_col_dtypes(df):
    col_dtypes = {}
    for col in df:
        dtype = df[col].dtype
        if dtype not in col_dtypes:
            col_dtypes[dtype] = []
        col_dtypes[dtype].append(col)

    return col_dtypes


def create_expand_columns(col_name, col_data, expand_vset=None,
                          dtype=INT_COL_DTYPE, skip_nan=True):
    if expand_vset is None:
        expand_vset, _, _ = extract_column_vset(col_data, skip_nan=skip_nan)

    new_data = {}
    for v in expand_vset:
        new_data[v] = (col_data == v).astype(dtype)
    # for v in expand_vset:
    #    new_data[v] = pd.Series(
    #        np.zeros(nsamples, dtype=dtype), index=col_data.index)
    # for idx, v in col_data.iteritems():
    #    if skip_nan and _isnan(v):
    #        continue
    #    new_data[v][idx] = 1

    expand_col_data = {}
    for v in expand_vset:
        expand_col_name = "%s.%s" % (col_name, v)
        expand_col_data[expand_col_name] = new_data[v]

    return expand_col_data


def check_y_null(data_y):
    """ Check the number of NaN values in each column of y data """
    results = {}
    for col in data_y.columns:
        data = data_y[col]
        null_idx = [idx for (idx, v) in enumerate(data.isnull()) if v]
        results[col] = null_idx
        print("Number of null values in [%s]: %d/%d" %
              (col, len(null_idx), len(data)))

    return results


def drop_unique_value_columns(df):
    """
    Drop columns with unique values.
    """
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[(nunique == 1) | (nunique == 0)].index
    df_new = df.drop(cols_to_drop, axis=1)
    print("Shape after drop unique value columns: {0} -> {1}".format(
        df.shape, df_new.shape))
    return df_new


def split_data(data_bg, train_y):
    """
    Split background data into train and test, based on the
    "challengeID" in train_y. In our dataset, there are
    in total of 4242 samples but only 2121 train samples.
    """
    train_y_ids = train_y["challengeID"]

    train_ids = set(train_y_ids).intersection(
        set(data_bg["challengeID"]))
    predict_ids = set(data_bg["challengeID"]) - set(train_y_ids)
    print("Number of elements in train dataset: %d" % len(train_ids))
    print("Number of elements in predict dataset: %d" % len(predict_ids))
    if len(train_ids.intersection(predict_ids)) != 0:
        raise ValueError("Split dataset error(overlap): %s"
                         % (train_ids.intersection(predict_ids)))
    if len(train_ids.union(predict_ids)) != len(data_bg["challengeID"]):
        raise ValueError("Split dataset error: missing")

    id_loc_info = dict(
        (_cidx, _idx) for (_idx, _cidx) in enumerate(data_bg["challengeID"]))
    train_x_locs = [id_loc_info[i] for i in train_ids]
    predict_x_locs = [id_loc_info[i] for i in predict_ids]
    if len(set(train_x_locs).intersection(set(predict_x_locs))) != 0:
        raise ValueError("Split dataset loc error(overlap)")

    train_x = data_bg.iloc[train_x_locs].copy()
    _validate_train_data(train_x, train_y)

    #predict_x = data_bg.iloc[predict_x_locs]
    # MODIFY HERE. USE ALL
    predict_x = data_bg
    return train_x, predict_x


def _validate_train_data(train_x, train_y):
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("Dimension error: ", train_x.shape,
                         ", ", train_y.shape)

    nsamples = train_x.shape[0]
    xids = list(train_x["challengeID"])
    yids = list(train_y["challengeID"])
    for i in range(nsamples):
        # check challengeID are the same
        id1 = xids[i]
        id2 = yids[i]
        if id1 != id2:
            raise ValueError("challengeID mismatch at %d: %d != %d"
                             % (i, id1, id2))

    print("Validation passed")


def extract_good_train_data(train_x, data_y, col):
    """
    clean the data since data_y contains NaN values in certain challengeID.
    The current plan is drop those row with NaN in y.
    """
    # extract good(not NaN) train_y
    good_ids = []
    good_locs = []
    bad_locs = []
    for iloc, v in enumerate(data_y[col].isnull()):
        if not v:
            good_ids.append(data_y["challengeID"][iloc])
            good_locs.append(iloc)
        else:
            bad_locs.append(iloc)

    print("Null y length: %d" % len(bad_locs))
    print("Not null y length: %d" % len(good_locs))
    train_y_clean = data_y.iloc[good_locs]
    print("train_y shape: ", train_y_clean.shape)

    train_x_clean = train_x.iloc[good_locs]
    print("train_x shape: ", train_x_clean.shape)
    train_x_bad = train_x.iloc[bad_locs]

    _validate_train_data(train_x_clean, train_y_clean)
    if train_x_clean.isnull().values.any():
        raise ValueError("NaN values in train_x_clean")
    if train_y_clean[col].isnull().values.any():
        raise ValueError("NaN values in train_y_clean")

    _good_ids = train_x_clean["challengeID"]
    for id1, id2 in zip(good_ids, _good_ids):
        if id1 != id2:
            raise ValueError("Error in good_ids")

    return train_x_clean.as_matrix(), train_y_clean[col].as_matrix(), \
        train_x_bad, _good_ids


def print_title(title, symbol="*", slen=20, simple_mode=False):
    total_len = len(title) + 2 + 2 * slen
    if not simple_mode:
        print(symbol * total_len)
    print(symbol * slen + " %s " % title + symbol * slen)
    if not simple_mode:
        print(symbol * total_len)


def validate_numerical_dtype(df):
    col_dtypes = get_columns_dtype(df)
    if np.dtype(object) in col_dtypes:
        cols_object = col_dtypes[np.dtype(object)]
        print("bad %s[%d]: %s" % (np.dtype(object), len(cols_object),
                                  cols_object))
        raise ValueError("Object still in columns")

    nan_flag = df.isnull().values.any()
    if nan_flag:
        for col in df:
            n_nan = df[col].isnull().sum()
            if n_nan > 0:
                print("column %s has nan values: %d" % (col, n_nan))
        raise ValueError("Still NaN values in current dataframe")

    print("-" * 10 + " Validation passed " + "-" * 10)


def standarize_feature(x, predict_x):
    # print("x and predict_x isfinite before standard: ", np.isinf(x).any(),
    #      np.isinf(predict_x).any())
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    predict_x = scaler.transform(predict_x)
    # print("x and predict_x isfinite after standard: ", np.isfinite(x).any(),
    #      np.isfinite(predict_x).any())
    return x, predict_x


def remean(data):
    mean = np.mean(data)
    return data - mean, mean


def safe_concat(df1, df2):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    common_sets = cols1.intersection(cols2)
    if len(common_sets) != 0:
        raise ValueError("Overlap in columns[%d]: %s"
                         % (len(common_sets), common_sets))

    return pd.concat([df1, df2], axis=1)
