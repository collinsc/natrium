#! /usr/bin/env python3
""" Standalone script for evaluating a dataset.
Calculates measures of label quality and tries to spot outliers.

Usage:
    python3 investigate.py <filename>"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

def partition(data, ratio):
    """returns (training_data, training_labels, testing_data, testing_labels)"""

    data = data.loc[np.random.permutation(data.index)]

    partition_idx = int(data.shape[0] * ratio)
    train, test = np.split(data, [partition_idx])

    def splitDataLabels(data):
        labels = data["genre"].to_frame()
        data = data.drop(columns = ["genre"])
        return data , labels
    train_data, train_label = splitDataLabels(train)
    test_data, test_label = splitDataLabels(test)
    return train_data, train_label, test_data, test_label
 
def pca_partition(data, ratio, pc_count):
    """returns (training_data, training_labels, testing_data, testing_labels)"""

    data = data.loc[np.random.permutation(data.index)]

    partition_idx = int(data.shape[0] * ratio)
    train, test = np.split(data, [partition_idx])
    pca = PCA(n_components=pc_count)
    def splitDataLabels(data):
        labels = data["genre"].to_frame()
        data = data.drop(columns = ["genre"])
        return data , labels
    train_data, train_label = splitDataLabels(train)
    pca.fit(train_data)
    print("   pca variance:", pca.explained_variance_ratio_)
    test_data, test_label = splitDataLabels(test)
    return pca.transform(train_data), train_label, pca.transform(test_data), test_label

 
def classify(name, clf, train_data, train_labels, test_data, test_labels):

    print("   Fitting:", name)
    clf.fit(train_data, train_labels.values.ravel())
    print("   Predicting Probabilities:", name)
    scores = clf.predict_proba(test_data)
    print("   Predicting Classes:", name)
    predictions = clf.predict(test_data)
    return predictions, scores, test_labels.values.ravel()

def evaluate(name, classes, predicted, scores, actual, pca_predicted, pca_scores, pca_actual, graph = True):
    if graph:
        print("   Plotting:", name)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
        f.suptitle('%s ROC' % name)
    else:
        ax1 = None
        ax2 = None

    def ROC(ax, title, classes, predicted, scores, actual, graph):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        lw = 2
        acc =  np.sum(predicted ==  actual)/actual.shape[0]
        for i, cl in enumerate(classes):
            fpr[i], tpr[i], _ = roc_curve(actual, scores[:,i], pos_label=cl)
            roc_auc[i] = auc(fpr[i], tpr[i])
            if graph:
                ax.plot(fpr[i], tpr[i],lw=lw, label='%s, (auc = %0.2f)' % (cl, roc_auc[i], ))
        if graph:
            ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('%s \n(acc = %0.2f, sum_auc = %0.2f)' % (title, acc, sum(roc_auc.values())))
            ax.legend(loc="lower right")
        return sum(roc_auc.values())

    auc_sum = ROC(ax1, "Normal", classes, predicted, scores, actual, graph)
    pca_auc_sum = ROC(ax2, "PCA", classes, pca_predicted, pca_scores, pca_actual, graph)
 
    if graph:
        plt.ion()
        plt.show()
        plt.pause(0.1)
    return auc_sum, pca_auc_sum


    
def main():
    """The executable to read in the specified data file and perform the
    investigatory operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the data file to read in.")

    args = parser.parse_args()

    #read the dataframe in memory friendly chunks
    data_frame = pd.read_pickle(args.input_file.name).infer_objects()
    genre_list = sorted(["Punk", "Electronic","RnB", "Rap", "Country", "Metal", "Pop", "Rock"])
    data_frame = data_frame[data_frame["genre"].isin(genre_list)]
    print(data_frame.columns)
    data_frame = data_frame.drop(columns=["release_year"])

    names = [
            "Nearest Neighbors", 
            #"Linear SVM", 
            #"RBF SVM", 
            "Decision Tree", 
            "Neural Net", 
            "Naive Bayes" 
            ]

    classifiers = [
            KNeighborsClassifier(12),
            #SVC(kernel="linear", C=0.025),
            #SVC(gamma=2, C=1),
            DecisionTreeClassifier(max_depth=8),
            MLPClassifier(),
            GaussianNB(),
            ]
    partition_ratio = 0.8 
    print("PARTITION_RATIO = ", partition_ratio)
    pc_count = 12 
    print("PC_COUNT = ", pc_count)
    optimization_plots = False 
    print("OPTIMIZATION PLOTS", optimization_plots)

    if optimization_plots:
        knn_auc_sum = [] 
        knn_pca_auc_sum = [] 
        tree_auc_sum =[] 
        tree_pca_auc_sum =[] 
        for i in range(1,15):
            data = partition(data_frame, partition_ratio)
            pca_data = pca_partition(data_frame, partition_ratio, pc_count)
    
            results = classify(names[0],KNeighborsClassifier(i) , *data)
            pca_results = classify(names[0],KNeighborsClassifier(i) , *pca_data)
            auc, pca_auc = evaluate(names[0] + str(i), genre_list, *results, *pca_results, graph=False)
            knn_auc_sum.append(auc)
            knn_pca_auc_sum.append(pca_auc)
    
            results = classify(names[1],DecisionTreeClassifier(max_depth=i) , *data)
            pca_results = classify(names[1],DecisionTreeClassifier(max_depth=i) , *pca_data)
            auc, pca_auc = evaluate(names[1] + str(i), genre_list, *results, *pca_results, graph = False)
            tree_auc_sum.append(auc)
            tree_pca_auc_sum.append(pca_auc)
    
        plt.figure()
        plt.plot(range(1,15), knn_auc_sum, label="normal")
        plt.plot(range(1,15), knn_pca_auc_sum, label="pca")
        plt.title("KNN Optimization")
        plt.xlabel("Neighbor Count")
        plt.ylabel("Sum of AUC")
        plt.figure()
        plt.plot(range(1,15), tree_auc_sum, label="normal")
        plt.plot(range(1,15), tree_pca_auc_sum, label="pca")
        plt.title("Decision Tree Optimization")
        plt.xlabel("Tree Depth")
        plt.ylabel("Sum of AUC")
        plt.ion()
        plt.show()

    for name, clf in zip(names, classifiers):
        print("Starting:", name)
        print("   Partitioning:", name)
        data = partition(data_frame, partition_ratio)
        pca_data = pca_partition(data_frame, partition_ratio, pc_count)
        results = classify(name, clf, *data)
        pca_results = classify(name, clf, *pca_data)
        evaluate(name, genre_list, *results, *pca_results)
    input("Finished, press enter to close")


if __name__ == "__main__":
    main()
