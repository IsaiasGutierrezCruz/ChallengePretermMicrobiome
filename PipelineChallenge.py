# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def feature_dropper(data, list_feat):
    return data.copy().drop(list_feat, axis=1)


def split_data_stratified(data, output_name, test_size=0.2) -> tuple:
    """
    Function to create a pair of stratified trains and test sets.

    Parameters
    ----------
    data: DataFrame
        Original dataframe to be divided
    output_name:str
        Name of the column with the output categories
    test_size:float
        Size of the test set

    Returns
    -------
        A tuple with a pair of stratified train and test dataframes, a pair of X_train and y_trian, and x_test and test sets
    """
    strati_train_set = None
    strati_test_set = None
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_indices, test_indices in split.split(data, data[output_name]):
        strati_train_set = data.loc[train_indices]
        strati_test_set = data.loc[test_indices]

    X_train, y_train = (
        strati_train_set.drop(output_name, axis=1),
        strati_train_set[output_name],
    )
    X_test, y_test = (
        strati_test_set.drop(output_name, axis=1),
        strati_test_set[output_name],
    )

    return strati_train_set, strati_test_set, X_train, y_train, X_test, y_test


def standard_data(x_train, x_test) -> tuple:
    """
    Function to standardize the x_train and x_test sets.

    Parameters
    ----------
    x_train: DataFrame
        X train dataset
    x_test : DataFrame
        X test dataset

    Returns
    -------
        Tuple with the standardise train and test sets
    """

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    # y_train = y_train.to_numpy()

    x_test = scaler.transform(x_test)
    # y_test = y_test.to_numpy()

    return x_train, x_test


def evaluate_model(
    data, model, output_name, n_splits=10, split_method="StratifiedShuffle"
):
    """
    Function to get the predicted targets using train and tests sets generated with a specific split method

    Parameters
    ----------
    data: DataFrame
        Original dataframe to be divided
    model: Sklearn Model
        Variable with the model of sklearn to evaluate
    output_name: str
        Name of the column with the output categories
    n_splits: int
        Number of splits to construct
    split_method: str
        Specify the method to split the data

    Returns
    -------
    tuple: ndarray
        Tuple containing the predicted and original targets of each pair of train and test sets.
    """
    predicted_targets = list()
    actual_targets = list()

    if split_method == "StratifiedShuffle":
        split = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2)
    else:
        split = None

    assert split is not None, "Be sure to have selected a correct split_method"

    for train_indices, test_indices in split.split(data, data[output_name]):
        strati_train_set = data.loc[train_indices]
        strati_test_set = data.loc[test_indices]

        x_train, y_train, x_test, y_test = standard_data(
            strati_train_set, strati_test_set, output_name=output_name
        )

        model.fit(x_train, y_train)
        predicted_labels = model.predict(x_test)

        predicted_targets.append(predicted_labels)
        actual_targets.append(y_test)

    return predicted_targets, actual_targets


def plot_confusion_matrix(
    y_test_list,
    predicted_label_list,
    classes,
    normalize=True,
    title="Normalized confusion matrix",
):
    """
    Construction of a heatmap with the confusion matrix generated with the original and predicted target lists

    Parameters
    ----------
    y_test_list: list
        List of arrays containing the original targets of each test set created
    predicted_label_list: list
        List of arrays containing the predicted targets of each test set created
    classes: tuple
        Names of the targets
    normalize:bool, default=True
        Variable to choose the normalization fo the confusion matrix
    title: str, default="Normalized confusion matrix"
        Name of use in the confusion matrix plot

    Returns
    -------
    figure
        Plot of the confusion matrix
    """

    cnf_matrix = np.empty(shape=[2, 2])

    for true_target, predict_target in zip(y_test_list, predicted_label_list):
        cnf_matrix = cnf_matrix + confusion_matrix(true_target, predict_target)

    np.set_printoptions(precision=2)

    if normalize:
        cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    figure = plt.figure()
    plt.imshow(cnf_matrix, interpolation="nearest", cmap=plt.get_cmap("YlGnBu"))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cnf_matrix.max() / 2.0

    for i, j in itertools.product(
        range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            format(cnf_matrix[i, j], fmt),
            horizontalalignment="center",
            color="black" if cnf_matrix[i, j] > thresh else "blue",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # return cnf_matrix
    return figure


def plot_roc_auc(actual_target, predicted_target):
    """
    Creation of the roc auc curve given a list of different true and predicted outputs

    Parameters
    ----------
    actual_target

    actual_target: list
        List of arrays containing the original targets of each test set created
    predicted_target: list
        List of arrays containing the predicted targets of each test set created

    Returns
    -------
    figure
        Plot of the roc auc curve


    """
    fig1 = plt.figure(figsize=[12, 12])
    ax1 = fig1.add_subplot(111, aspect="equal")

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1
    for true_target, predict_target in zip(actual_target, predicted_target):
        fpr, tpr, t = roc_curve(true_target, predict_target)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(
            fpr, tpr, lw=2, alpha=0.3, label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc)
        )
        i = i + 1

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        label=r"Mean ROC (AUC = %0.2f )" % mean_auc,
        lw=2,
        alpha=1,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    return fig1
