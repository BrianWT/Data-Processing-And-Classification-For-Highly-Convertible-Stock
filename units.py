import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
from sklearn import svm, datasets
from sklearn import model_selection
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import _flatten
# from structure import Net


def load_dataset(method="randomover"):
    dataset = pd.read_csv("Pretreatment Data/Data.csv", encoding="GBK")

    Y_orig = np.array(dataset["Y"])
    X_orig = dataset.drop(["Y"], axis=1)
    X_orig = np.array(X_orig)
    X_train, X_test, Y_train, Y_test = train_test_split(X_orig, Y_orig, test_size=0.25, random_state=0)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # 过采样
    if method == "randomover":
        ros = RandomOverSampler(random_state=0)
        X_resampled, Y_resampled = ros.fit_sample(X_train, Y_train)
    # 使用SMOTE模型生成新的样本
    if method == "smote":
        smo = SMOTE(random_state=0)
        X_resampled, Y_resampled = smo.fit_resample(X_train, Y_train)
    # 使用ADASYN模型
    if method == "adasyn":
        ada = ADASYN(random_state=0)
        X_resampled, Y_resampled = ada.fit_resample(X_train, Y_train)
    # 欠采样
    if method == "randomunder":
        rus = RandomUnderSampler(random_state=0)
        X_resampled, Y_resampled = rus.fit_sample(X_train, Y_train)

    return X_resampled, Y_resampled, X_test, Y_test


def Logistical_parameter_sreeen(X_train, Y_train, X_test, Y_test):
    l1 = []
    l2 = []
    l1test = []
    l2test = []

    for i in np.linspace(0.05, 1, 19):
        lrl1 = LR(penalty="l1", solver="liblinear", C=i, class_weight='balanced', n_jobs=-1)
        lrl2 = LR(penalty="l2", solver="liblinear", C=i, class_weight='balanced', n_jobs=-1)

        lrl1 = lrl1.fit(X_train, Y_train)
        l1_dict = classification_report(Y_train, lrl1.predict(X_train), output_dict=True)
        l1.append(l1_dict["1"]["f1-score"])
        l1test_dict = classification_report(Y_test, lrl1.predict(X_test), output_dict=True)
        l1test.append(l1test_dict["1"]["f1-score"])

        lrl2 = lrl2.fit(X_train, Y_train)
        l2_dict = classification_report(Y_train, lrl2.predict(X_train), output_dict=True)
        l2.append(l2_dict["1"]["f1-score"])
        l2test_dict = classification_report(Y_test, lrl2.predict(X_test), output_dict=True)
        l2test.append(l2test_dict["1"]["f1-score"])

    graph = [l1, l2, l1test, l2test]
    color = ["green", "black", "lightgreen", "gray"]
    label = ["L1", "L2", "L1test", "L2test"]

    plt.figure(figsize=(6, 6))
    for i in range(len(graph)):
        plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
    plt.legend(loc=4)
    plt.show()

    return l1, l2, l1test, l2test


def SVM_parameter_sreeen(X_train, Y_train, X_test, Y_test):
    model1_1 = []
    model2_1 = []
    model2_2 = []
    model1_1_test = []
    model2_1_test = []
    model2_2_test = []

    for i in np.logspace(-2, 2, 25):
        t1 = time.time()
        lrl1_1 = svm.LinearSVC(penalty="l1", random_state=0, C=i, class_weight='balanced', loss="squared_hinge",
                               dual=False)
        lrl2_1 = svm.LinearSVC(penalty="l2", random_state=0, C=i, class_weight='balanced', loss="hinge")
        lrl2_2 = svm.LinearSVC(penalty="l2", random_state=0, C=i, class_weight='balanced', loss="squared_hinge",
                               dual=False)

        lrl1_1 = lrl1_1.fit(X_train, Y_train)
        lrl1_1_dict = classification_report(Y_train, lrl1_1.predict(X_train), output_dict=True)
        model1_1.append(lrl1_1_dict["1"]["f1-score"])
        lrl1_1_test_dict = classification_report(Y_test, lrl1_1.predict(X_test), output_dict=True)
        model1_1_test.append(lrl1_1_test_dict["1"]["f1-score"])

        lrl2_1 = lrl2_1.fit(X_train, Y_train)
        lrl2_1_dict = classification_report(Y_train, lrl2_1.predict(X_train), output_dict=True)
        model2_1.append(lrl2_1_dict["1"]["f1-score"])
        lrl2_1_test_dict = classification_report(Y_test, lrl2_1.predict(X_test), output_dict=True)
        model2_1_test.append(lrl2_1_test_dict["1"]["f1-score"])

        lrl2_2 = lrl2_2.fit(X_train, Y_train)
        lrl2_2_dict = classification_report(Y_train, lrl2_2.predict(X_train), output_dict=True)
        model2_2.append(lrl2_2_dict["1"]["f1-score"])
        lrl2_2_test_dict = classification_report(Y_test, lrl2_2.predict(X_test), output_dict=True)
        model2_2_test.append(lrl2_2_test_dict["1"]["f1-score"])
        t2 = time.time()
        print("C：", i, round((t2 - t1) / 60, 2), "M")

    graph = [model1_1, model2_1, model2_2, model1_1_test, model2_1_test, model2_2_test]
    color = ["black", "red", "blue", "black", "red", "blue"]
    label = ["model1_1", "model2_1", "model2_2", "model1_1_test", "model2_1_test", "model2_2_test"]

    plt.figure(figsize=(8, 8))
    for i in range(len(graph)):
        plt.plot(np.logspace(-2, 2, 25), graph[i], color[i], label=label[i])
    plt.legend(loc=4)
    plt.show()

    return model1_1, model2_1, model2_2, model1_1_test, model2_1_test, model2_2_test


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, 256)
        self.hidden2 = torch.nn.Linear(256, 128)
        self.hidden3 = torch.nn.Linear(128, 32)
        self.hidden4 = torch.nn.Linear(32, 10)
        self.out = torch.nn.Linear(10, n_output)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        # self.dropout3 = torch.nn.Dropout(p=0.2)
        self.do_bn1 = torch.nn.BatchNorm1d(256)
        self.do_bn2 = torch.nn.BatchNorm1d(128)
        self.do_bn3 = torch.nn.BatchNorm1d(32)
        self.do_bn4 = torch.nn.BatchNorm1d(10)

    def forward(self, x):
        z1 = self.hidden1(x)
        z1 = self.do_bn1(z1)
        z1 = self.dropout1(z1)
        a1 = F.relu(z1)
        z2 = self.hidden2(a1)
        z2 = self.do_bn2(z2)
        z2 = self.dropout2(z2)
        a2 = F.relu(z2)
        z3 = self.hidden3(a2)
        z3 = self.do_bn3(z3)
        # z3 = self.dropout3(z3)
        a3 = F.relu(z3)
        z4 = self.hidden4(a3)
        z4 = self.do_bn4(z4)
        a4 = F.relu(z4)
        y_predict = self.out(a4)

        return y_predict


def LightGBM_params_search(X_train, Y_train, params_test1, params_test2, params_test3, params_test4, params_test5,
                           params_test6):
    params_list = []

    ### 1.best_n_estimators
    data_train = lgb.Dataset(X_train, Y_train)
    cv_results = lgb.cv(params_test1, data_train, num_boost_round=10000, nfold=6, stratified=False, shuffle=True,
                        metrics='auc', early_stopping_rounds=50, seed=0)

    best_n_estimators = len(cv_results['auc-mean'])
    params_list.append(best_n_estimators)
    print('best n_estimators:', best_n_estimators)
    print('best cv score:', pd.Series(cv_results['auc-mean']).max())

    ### 2.max_depth和num_leaves
    gsearch1 = GridSearchCV(
        estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                     n_estimators=best_n_estimators, max_depth=6, bagging_fraction=0.8,
                                     feature_fraction=0.8), param_grid=params_test2, scoring='roc_auc',
        cv=5, n_jobs=-1, verbose=1)
    gsearch1.fit(X_train, Y_train)

    max_depth = gsearch1.best_params_["max_depth"]
    params_list.append(max_depth)
    num_leaves = gsearch1.best_params_["num_leaves"]
    params_list.append(num_leaves)
    print(gsearch1.best_params_)

    ### 3.max_bin和min_data_in_leaf
    gsearch2 = GridSearchCV(estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc',
                                                         learning_rate=0.1, n_estimators=best_n_estimators,
                                                         max_depth=max_depth,
                                                         num_leaves=num_leaves, bagging_fraction=0.8,
                                                         feature_fraction=0.8),
                            param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
    gsearch2.fit(X_train, Y_train)

    max_bins = gsearch2.best_params_["max_bin"]
    params_list.append(max_bins)
    min_data_in_leafs = gsearch2.best_params_["min_data_in_leaf"]
    params_list.append(min_data_in_leafs)
    print(gsearch2.best_params_)

    ### 4.feature_fraction、bagging_fraction以及bagging_freq
    gsearch3 = GridSearchCV(
        estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                     n_estimators=best_n_estimators, max_depth=max_depth, num_leaves=num_leaves,
                                     max_bin=max_bins, min_data_in_leaf=min_data_in_leafs),
        param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
    gsearch3.fit(X_train, Y_train)

    feature_fractions = gsearch3.best_params_["feature_fraction"]
    params_list.append(feature_fractions)
    bagging_fractions = gsearch3.best_params_["bagging_fraction"]
    params_list.append(bagging_fractions)
    bagging_freqs = gsearch3.best_params_["bagging_freq"]
    params_list.append(bagging_freqs)
    print(gsearch3.best_params_)

    ### 5.lambda_l1和lambda_l2
    gsearch4 = GridSearchCV(
        estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                     n_estimators=best_n_estimators, max_depth=max_depth, num_leaves=num_leaves,
                                     max_bin=max_bins, min_data_in_leaf=min_data_in_leafs,
                                     bagging_fraction=bagging_fractions, bagging_freq=bagging_freqs,
                                     feature_fraction=feature_fractions),
        param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
    gsearch4.fit(X_train, Y_train)

    lambda_L1 = gsearch4.best_params_["lambda_l1"]
    params_list.append(lambda_L1)
    lambda_L2 = gsearch4.best_params_["lambda_l2"]
    params_list.append(lambda_L2)
    print(gsearch4.best_params_)

    ### 6.min_split_gain
    gsearch5 = GridSearchCV(
        estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                     n_estimators=best_n_estimators, max_depth=max_depth, num_leaves=num_leaves,
                                     max_bin=max_bins, min_data_in_leaf=min_data_in_leafs,
                                     bagging_fraction=bagging_fractions, bagging_freq=bagging_freqs,
                                     feature_fraction=feature_fractions, lambda_l1=lambda_L1, lambda_l2=lambda_L2),
        param_grid=params_test6, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
    gsearch5.fit(X_train, Y_train)

    min_split_gain = gsearch5.best_params_["min_split_gain"]
    params_list.append(min_split_gain)
    print(gsearch5.best_params_)

    return params_list


def XGBoost__params_search(X_train, Y_train, param_test1, param_test2, param_test3, param_test4):
    params_list = []

    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=20, max_depth=5, min_child_weight=1,
                                                    gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', scale_pos_weight=1, seed=0),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch1.fit(X_train, Y_train)
    max_depthx = gsearch1.best_params_["max_depth"]
    params_list.append(max_depthx)
    min_child_weightx = gsearch1.best_params_["min_child_weight"]
    params_list.append(min_child_weightx)
    print(gsearch1.best_params_)

    gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=20, max_depth=max_depthx,
                                                    min_child_weight=min_child_weightx, gamma=0, subsample=0.8,
                                                    colsample_bytree=0.8,
                                                    objective='binary:logistic', scale_pos_weight=1, seed=0),
                            param_grid=param_test2, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch2.fit(X_train, Y_train)
    gammax = gsearch2.best_params_["gamma"]
    params_list.append(gammax)
    print(gsearch2.best_params_)

    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=20, max_depth=max_depthx,
                                                    min_child_weight=min_child_weightx, gamma=gammax, subsample=0.8,
                                                    colsample_bytree=0.8, objective='binary:logistic',
                                                    scale_pos_weight=1, seed=0),
                            param_grid=param_test3, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch3.fit(X_train, Y_train)
    subsamplex = gsearch3.best_params_["subsample"]
    params_list.append(subsamplex)
    colsample_bytreex = gsearch3.best_params_["colsample_bytree"]
    params_list.append(colsample_bytreex)
    print(gsearch3.best_params_)

    gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=20, max_depth=max_depthx,
                                                    min_child_weight=min_child_weightx, gamma=gammax,
                                                    subsample=subsamplex,
                                                    colsample_bytree=colsample_bytreex, objective='binary:logistic',
                                                    scale_pos_weight=1, seed=0),
                            param_grid=param_test4, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch4.fit(X_train, Y_train)
    reg_alphax = gsearch4.best_params_["reg_alpha"]
    params_list.append(reg_alphax)
    print(gsearch4.best_params_)

    return params_list


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.size_average = size_average
#
#     def forward(self, pred, target):
#         # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
#         # print(pred.shape)
#         # pred = nn.Sigmoid()(pred)
#         # print(pred.shape)
#         # print(pred.view(-1, 1).long().shape)
#         # print(target.view(-1, 1).long().shape)
#         # 此时 pred.size = target.size = (BatchSize,1)
#         pred = pred.view(-1, 1)
#         target = target.view(-1, 1)
#
#         # 预测样本为正负的概率, pred.size = (BatchSize,2)
#         pred = torch.cat((1 - pred, pred), dim=1)
#
#         # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
#         # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
#         class_mask = torch.zeros(pred.shape[0], pred.shape[1])
#         class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
#
#         # 所需概率值
#         probs = (pred * class_mask).sum(dim=1).view(-1, 1)
#         probs = probs.clamp(min=0.0001, max=1.0)
#
#         # 计算概率的 log 值
#         log_p = probs.log()
#
#         # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
#         alpha = torch.ones(pred.shape[0], pred.shape[1])
#         alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
#         alpha[:, 1] = alpha[:, 1] * self.alpha
#         alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
#
#         # 根据 Focal Loss 的公式计算 Loss
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#
#         return loss


# def load_model(path):
#
#     model = torch.load(path)
#
#     return model
#
#
# def load_1():
#     model = torch.load("C:/Users/jz/Desktop/A/模型及相关参数/Cross Loss神经网络筛选因子版本.pkl")
#     return model


# if __name__ == '__main__':
#     X_train, Y_train, X_test, Y_test = load_dataset(method="smote")
#     print("X_train shape: " + str(X_train.shape))
#     print("Y_train shape: " + str(Y_train.shape))
#     print("X_test shape: " + str(X_test.shape))
#     print("Y_test shape: " + str(Y_test.shape))


