from __future__ import print_function
import time
import numpy as np
from sklearn import linear_model
from utils import standarize_feature, print_title


def train_ridge_linear_model(_train_x, train_y, _predict_x,
                             sample_weight=None):
    print_title("Ridge Regressor")
    train_x, predict_x = \
        standarize_feature(_train_x, _predict_x)

    # using the default CV
    alphas = [0.1, 1, 10, 100, 1e3, 1e4, 2e4, 5e4, 8e4, 1e5, 1e6, 1e7, 1e8]
    reg = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)
    #reg.fit(train_x, train_y, sample_weight=sample_weight)
    reg.fit(train_x, train_y)
    cv_mse = np.mean(reg.cv_values_, axis=0)
    print("alphas: %s" % alphas)
    print("CV MSE: %s" % cv_mse)
    print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)

    # generate the prediction using the best model
    alpha = reg.alpha_
    reg = linear_model.Ridge(alpha=alpha)
    #reg.fit(train_x, train_y, sample_weight=sample_weight)
    reg.fit(train_x, train_y)
    predict_y = reg.predict(predict_x)
    train_y_pred = reg.predict(train_x)

    return {"y": predict_y, "train_y": train_y_pred, "coef": reg.coef_}


def train_lasso_model(_train_x, train_y, _predict_x):
    print_title("Lasso Regressor")

    train_x, predict_x = \
        standarize_feature(_train_x, _predict_x)

    reg = linear_model.LassoCV(
        precompute=True, cv=5, verbose=1, n_jobs=4)
    reg.fit(train_x, train_y)
    print("alphas: %s" % reg.alphas_)
    print("mse path: %s" % np.mean(reg.mse_path_, axis=1))

    itemindex = np.where(reg.alphas_ == reg.alpha_)
    print("itemindex: %s" % itemindex)
    _mse = np.mean(reg.mse_path_[itemindex[0], :])
    print("Best alpha using bulit-in LassoCV: %f(mse: %f)" %
          (reg.alpha_, _mse))

    alpha = reg.alpha_
    reg = linear_model.Lasso(alpha=alpha)
    reg.fit(train_x, train_y)
    n_nonzeros = (reg.coef_ != 0).sum()
    print("Non-zeros coef: %d" % n_nonzeros)
    predict_y = reg.predict(predict_x)
    train_y_pred = reg.predict(train_x)

    return {"y": predict_y, "train_y": train_y_pred, "coef": reg.coef_}


def train_lassolars_model(train_x, train_y, predict_x):
    print_title("LassoLars Regressor")
    reg = linear_model.LassoLarsCV(
        cv=10, n_jobs=3, max_iter=2000, normalize=False)
    reg.fit(train_x, train_y)
    print("alphas and cv_alphas: {0} and {1}".format(
        reg.alphas_.shape, reg.cv_alphas_.shape))
    print("alphas[%d]: %s" % (len(reg.cv_alphas_), reg.cv_alphas_))
    print("mse shape: {0}".format(reg.cv_mse_path_.shape))
    # print("mse: %s" % np.mean(_mse, axis=0))
    # print("mse: %s" % np.mean(_mse, axis=1))
    # index = np.where(reg.alphas_ == reg.alpha_)
    # print("itemindex: %s" % index)
    index = np.where(reg.cv_alphas_ == reg.alpha_)
    _mse_v = np.mean(reg.cv_mse_path_[index, :])
    print("mse value: %f" % _mse_v)

    print("best alpha: %f" % reg.alpha_)
    best_alpha = reg.alpha_
    reg = linear_model.LassoLars(alpha=best_alpha)
    reg.fit(train_x, train_y)
    n_nonzeros = (reg.coef_ != 0).sum()
    print("Non-zeros coef: %d" % n_nonzeros)
    predict_y = reg.predict(predict_x)
    return {'y': predict_y, "coef": reg.coef_}


def train_EN_model(_train_x, train_y, _predict_x):
    print_title("ElasticNet")
    train_x, predict_x = \
        standarize_feature(_train_x, _predict_x)

    #l1_ratios = [1e-4, 1e-3, 1e-2, 1e-1]
    #l1_ratios = [1e-5, 1e-4, 1e-3]
    l1_ratios = [0.9, 0.92, 0.95, 0.97, 0.99]
    #l1_ratios = [0.5]
    min_mse = 1
    for r in l1_ratios:
        t1 = time.time()
        reg_en = linear_model.ElasticNetCV(
            l1_ratio=r, cv=5, n_jobs=4, verbose=1, precompute=True)
        reg_en.fit(train_x, train_y)
        n_nonzeros = (reg_en.coef_ != 0).sum()
        _mse = np.mean(reg_en.mse_path_, axis=1)[
            np.where(reg_en.alphas_ == reg_en.alpha_)[0][0]]
        if _mse < min_mse:
            min_mse = _mse
            best_l1_ratio = r
            best_alpha = reg_en.alpha_
        t2 = time.time()
        print("ratio(%e) -- n: %d -- alpha: %f -- mse: %f -- "
              "time: %.2f sec" %
              (r, n_nonzeros, reg_en.alpha_, _mse, t2 - t1))

    print("Best l1_ratio and alpha: %f, %f" % (best_l1_ratio, best_alpha))
    # predict_model
    reg = linear_model.ElasticNet(l1_ratio=best_l1_ratio, alpha=best_alpha)
    reg.fit(train_x, train_y)
    predict_y = reg.predict(predict_x)
    train_y_pred = reg.predict(train_x)
    return {"y": predict_y, "train_y": train_y_pred, "coef": reg.coef_}
