import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


def set_seed(seed=42):
    """
    set all random seed to 42
    """
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def threshold_list(alist, threshold= 0.1):
    """
    get two list
    """
    big_thres = [i for i in alist if i >= threshold]
    small_thres = [i for i in alist if i < threshold]
    return big_thres, small_thres


def clean_all_feature_table(df, norm=True, norm_type='minmax', inf_times=2):
    """
    replace inf data in our DF, and can choose whether to normalize or not
    PS: if choose to normalize feature tables, do it before split data

    Args:
    -----
            df (dataframe): feature tabels, without column label and column label_index
            norm (bool): normalize features or not
            norm_type(str): minmax or standard
            inf_times: replace INF and give them values like max_column*inf_times

    Return:
            df_new: 2-d np array
    """
    df_new = df.copy()
    df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_new = df_new.fillna(df_new.max() * inf_times)

    if not norm:
        return df_new
    else:
        if norm_type == 'minmax':
            mm = MinMaxScaler()
            df_new = mm.fit_transform(np.array(df_new))
        elif norm_type == 'standard':
            sd = StandardScaler()
            df_new = sd.fit_transform(np.array(df_new))
        else:
            raise ValueError

    return df_new


def plot_pca(x):
    """
    Plot Principal Component Analysis cumulative of the explained variance

    Args:
    -----
            X (DataFrame): feature DataFrame

    """
    pca_sk = PCA()
    pca_sk.fit(x)

    # plotting the cumulative summation of the explained variance
    plt.figure()
    plt.plot(np.cumsum(pca_sk.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')  # for each component
    plt.title('Segmentation Dataset Explained Variance')
    plt.show(block=True)


def print_performance(y_true, y_pred):
    """
    Display performance of classifier by comparing classes. Used parameters to describe performance are
    accuracy, f1 score, recall and precision.

    Args:
    -----
            y_true (list): classes for every row of X
            y_pred (list): classes predicted by classifier

    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='macro')
    recall_w = recall_score(y_true, y_pred, average='macro')
    precision_w = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    print('accuracy: %.3f' % accuracy)
    print('f1 score macro av: %.3f' % f1_w)
    print('recall score macro av: %.3f' % recall_w)
    print('precision score macro av: %.3f' % precision_w)
    print('f1 score for every class: ', f1)
    print('recall: ', recall)
    print('precision: ', precision)
    print(' ')
    return accuracy, f1_w, recall_w, precision_w, f1, recall, precision


def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Perform 10fold crossvalidation, fit model with train data and evaluate its performance
    with print_performance() function and confusion matrix.

    Args:
    -----

            model (object): classifier object
            X_train (pd.DataFrame or np.array): feature matrix for train set
            y_train (list): classes for train set
            X_test (pd.DataFrame or np.array):feature matrix for test set
            y_test (list): classes for test set

    """
    print(model.__class__.__name__)
    print('Cross-validation')
    scores = cross_val_score(model, x_train, y_train, cv=10)
    print('Accuracy: ', scores)
    print('Mean accuracy:  %0.3f (+/- %0.3f)' % (scores.mean(), scores.std()))
    print('')

    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)

    print('Test')
    print_performance(y_test, y_pred_test)

    print('Confusion matrix')

    plot_confusion_matrix(model, x_test, y_test, normalize='pred', display_labels=['EO', 'EC'], values_format='.3f')
    plt.show()


def model_evaluation_dict(x, y, model, model_name, params):
    """
    Perform 10 fold crossvalidation, fit model with train data and evaluate its performance
    return performance dict

    """

    clf = GridSearchCV(model, params, cv=10)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf.fit(X_train, y_train)
    params = clf.best_params_

    Training_score = clf.score(X_train, y_train)
    Score = clf.score(X_test, y_test)
    Whole_score = clf.score(x, y)
    cvres = clf.cv_results_
    cvscore = cvres['mean_test_score'][clf.best_index_]
    macro_precision, macro_recall, macro_f1_score, macro_support = \
        precision_recall_fscore_support(y_test, clf.predict(X_test), average='macro')
    micro_precision, micro_recall, micro_f1_score, micro_support = \
        precision_recall_fscore_support(y_test, clf.predict(X_test), average='micro')
    if not params:
        # empty params dict
        params = 'default'
    # return a dictionary
    d_info = {'Classifier': model_name, 'param': params, 'Traing score': Training_score, ' Test Score': Score,
              'Whole score': Whole_score, 'CV Score': cvscore,
              'Precision(Macro)': macro_precision, 'Precision(Micro)': micro_precision,
              'Recall(Macro)': macro_recall, 'Recall(Micro)': micro_recall,
              'F1 Score(Macro)': macro_f1_score, 'F1 Score(Micro)': micro_f1_score}

    return d_info


def init_classifiers():
    """
    Initialize our machine learning classifier ---
    where catboost and NN (neural network classification) are not initialized,
    and most hyperparameters will take default values

    """

    model_names = ['SVM', 'LR', 'KNN', 'GBDT', 'DT', 'AdaB', 'RF', 'XGB', 'LGB', 'Catboost', 'NN']

    # the training parameters of each model
    param_grid_svc = [{}]
    param_grid_logistic = [{'C': [0.1], 'penalty': ['l1', 'l2']}]
    param_grid_knn = [{}, {'n_neighbors': list(range(3, 8))}]
    param_grid_gbdt = [{}]
    param_grid_tree = [{}]
    param_grid_boost = [{}]
    param_grid_rf = [{}]
    param_grid_xgb = [{}]
    param_grid_lgb = [{}]

    return ([(SVC(), model_names[0], param_grid_svc),
             (LogisticRegression(), model_names[1], param_grid_logistic),
             (KNeighborsClassifier(), model_names[2], param_grid_knn),
             (GradientBoostingClassifier(), model_names[3], param_grid_gbdt),
             (DecisionTreeClassifier(), model_names[4], param_grid_tree),
             (AdaBoostClassifier(), model_names[5], param_grid_boost),
             (RandomForestClassifier(), model_names[6], param_grid_rf),
             (xgb.XGBClassifier(), model_names[7], param_grid_xgb),
             (lgb.sklearn.LGBMClassifier(), model_names[8], param_grid_lgb)])


def prepare_signals(df_data, keep_min_length=True, padding=False, from_top=True, min_length=346, max_length=420):
    """

    Extract the DF signal data into a three-dim array -(batch_size, seq_length, embedding_dim),
    and truncate signals into a fixed length or padding them with zeros for a fixed length.

    Args:
    -----

            df_data (DataFrame): DF read by my function
            keep_min_length (bool): truncate signals into a fixed length
            padding (bool):  padding signals with zeros for a fixed length.
            from_top (bool): truncate or padding , from start or from end point
            min_length (int): the min signal length(got from 04 notebook)
            max_length (int): the possible max signal length(got from 04 notebook),

    """

    ch_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10',
                'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16']
    results = list()
    for i in range(df_data.shape[0]):
        current = list()
        for name in ch_names:
            current.append(np.array(eval(df_data[name][i])))
        current = np.transpose(np.array(current), axes=(1, 0))
        # print(current.shape)

        # keep all signal length 346
        if keep_min_length and not padding:
            if from_top:
                current = current[:min_length, :]
            else:
                current = current[current.shape[0] - min_length:, :]
        # padding signals with 0

        elif padding and not keep_min_length:
            # if bigger than max length
            if max_length < current.shape[0]:
                if from_top:
                    current = current[:max_length, :]
                else:
                    current = current[current.shape[0] - max_length:, :]
            else:

                padding_zeros = np.zeros((max_length - current.shape[0], 16))
                if from_top:
                    current = np.concatenate((padding_zeros, current))
                else:
                    current = np.concatenate((current, padding_zeros))
        else:
            raise ValueError

        results.append(current)

    return np.array(results)


from sklearn.metrics import ConfusionMatrixDisplay
from prettytable import PrettyTable
import joblib

class ConfusionMatrix(object):
    
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p-1, t-1] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity","F1score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1score = round(2*Precision*Recall/(Precision+Recall)) if Precision+Recall != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity, F1score])
            
        print(table)
        return str(acc)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')


        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def drop_ch_features(df,drop_list):
    
    results = list()
    fe_name = list(df.columns)
    
    for item in fe_name:
        for name in drop_list:
            if item.startswith(name):
                continue
            else:
                results.append(item)
    return df[results]


def threshold_list(alist, threshold= 0.1):
    big_thres = [i for i in alist if i >= threshold]
    small_thres = [i for i in alist if i < threshold]
    return big_thres, small_thres
