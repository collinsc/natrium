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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

def partition(data, label_name, ratio):
    """ Partitions data set according to a provided ratio.
    params:
        data - The data set in a pandas data frame
        label_name - the name of the collumn in the data set that contains the labels
        ratio - the training/total data ratio
    
    returns: 
        training_data - The data set to train on
        training_labels - Indexed labels for training set
        testing_data - The data set to test on
        testing_labels - The data set to test on """

    data = data.loc[np.random.permutation(data.index)]

    partition_idx = int(data.shape[0] * ratio)
    train, test = np.split(data, [partition_idx])

    def splitDataLabels(data):
        """Separates labels from data."""
        labels = data[label_name].to_frame()
        data = data.drop(columns = [label_name])
        return data , labels

    train_data, train_label = splitDataLabels(train)
    test_data, test_label = splitDataLabels(test)
    return train_data, train_label, test_data, test_label
 
def pca_transform(pc_count, train_data, train_label, test_data, test_label):
    """Performs dimensionality reduction up to a given number of features according to pca. 
    params:
        training_data - The data set to train on
        training_labels - Indexed labels for training set
        testing_data - The data set to test on
        testing_labels - The data set to test on 
    
    returns: 
        training_data - The data set to train on (Transformed)
        training_labels - Indexed labels for training set
        testing_data - The data set to test on (Transformed)
        testing_labels - The data set to test on """
    pca = PCA(n_components=pc_count)
    pca.fit(pd.concat([train_data, test_data]))
    print("   pca variance:", pca.explained_variance_ratio_)
    return pca.transform(train_data), train_label, pca.transform(test_data), test_label

def center_and_scale(frame):
    """Centers and scales a pandas dataframe.
    params:
        frame - the frame to center and scale
    returns:
        frame - the centered and scaledframe
        """
    def center_scale_col(col):
        if (pd.api.types.is_numeric_dtype(col)):
            mean = col.mean()
            dev = col.std()
            col = (mean - col)/dev
        return col
    return frame.apply(center_scale_col)

 
def classify(name, clf, train_data, train_labels, test_data, test_labels):
    """Trains a classifier on a set of data, and then tests it on a a testing set.
    params:
        name - the name of the classifier (diagnostics)
        clf - the classifier object (must provide fit, and predict methods from scikit-learn)
    returns:
        predictions - the label of the most likely class for test set
        scores - the probabilites of each different class
        labels - the true labels of the classes"""
    print("   Fitting:", name)
    clf.fit(train_data, train_labels.values.ravel())
    print("   Predicting Probabilities:", name)
    scores = clf.predict_proba(test_data)
    print("   Predicting Classes:", name)
    predictions = clf.predict(test_data)
    return predictions, scores, test_labels.values.ravel()

def evaluate(name, classes, predicted, scores, actual, pca_predicted, pca_scores, pca_actual, graph = True):
    """Plots and runs statistics on a set of classifier data. Compares performance with,
    and without pca processing.
    params:
        name - the name of the classifier to compare
        classes - the sorted list of class labels
        predicted - the predicted class labels
        scores - the calculated probabilites for each cless
        actual - the actual class labels
        pca_predicted - the predicted class labels (pca processed)
        pca_scores - the calculated probabilites for each cless (pca processed)
        pca_actual - the actual class labels (pca processed) 
    returns:
        auc_sum - the auc sum for the classifier
        pca_auc_sum - the auc sum for the pca processed data for the classifier"""
    if graph:
        print("   Plotting:", name)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
        f.suptitle('%s ROC' % name)
    else:
        ax1 = None
        ax2 = None

    def ROC(ax, title, classes, predicted, scores, actual, graph):
        """Generate a subplot with an ROC curve for a set of classifier output"""
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        lw = 2
        for i, cl in enumerate(classes):
            if i <scores.shape[1]:
                fpr[i], tpr[i], _ = roc_curve(actual, scores[:,i], pos_label=cl)
                roc_auc[i] = auc(fpr[i], tpr[i])
                if graph:
                    ax.plot(fpr[i], tpr[i],lw=lw, label='%s, (auc = %0.2f)' % (cl, roc_auc[i] ))
        if graph:
            #generate dashed diagonal for ROC
            ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            acc =  np.sum(predicted == actual)/actual.shape[0]
            sum_auc = sum(roc_auc.values())
            mean_auc = sum_auc/len(roc_auc.values())
            ax.set_title('%s \n(acc = %0.2f sum_auc = %0.2f, mean_auc = %0.2f)' % (title, acc, sum_auc, mean_auc))
            ax.legend(loc="lower right")
        return sum(roc_auc.values())

    #calculate statistics and generate subplots
    auc_sum = ROC(ax1, "Normal", classes, predicted, scores, actual, graph)
    pca_auc_sum = ROC(ax2, "PCA", classes, pca_predicted, pca_scores, pca_actual, graph)
 
    if graph:
        #render the plot
        plt.ion()
        plt.show()
        plt.pause(0.1)

    return auc_sum, pca_auc_sum

def optimize_tree_depth(data_frame, genre_list, pc_count, partition_ratio):
    """Plot a graph of total auc v, tree depth with varying tree depth.
    params:
        pc_count - the number of features to keep for pca
        partition_ratio - the ratio of train/total for partitioning the data set"""
    print("OPTIMIZING DECISION TREES:")
    tree_auc_sum =[] 
    tree_pca_auc_sum =[] 
    name = "Decision Tree Optimization"
    for i in range(1,15):
        print("calculating depth:", i)
        data = partition(data_frame, "genre", partition_ratio)
        pca_data = partition(data_frame, "genre", partition_ratio)
        cleaned_data = pca_transform(pc_count, *pca_data)
        results = classify(name, DecisionTreeClassifier(max_depth=i) , *data)
        pca_results = classify(name, DecisionTreeClassifier(max_depth=i) , *cleaned_data)
        auc, pca_auc = evaluate(name +" Depth:"+ str(i), genre_list, *results, *pca_results, graph = False)
        tree_auc_sum.append(auc)
        tree_pca_auc_sum.append(pca_auc) 
    plt.figure()
    plt.plot(range(1,15), tree_auc_sum, label="Normal")
    plt.plot(range(1,15), tree_pca_auc_sum, label="PCA")
    plt.title(name)
    plt.xlabel("Tree Depth")
    plt.ylabel("Sum of AUC")
    plt.legend()
    plt.ion()
    plt.show()
    plt.pause(0.1)

def optimize_hidden_size(data_frame, genre_list, pc_count, partition_ratio):
    """Plot a graph of total auc v, tree depth with varying tree depth.
    params:
        pc_count - the number of features to keep for pca
        partition_ratio - the ratio of train/total for partitioning the data set"""
    print("OPTIMIZING NEURAL NET:")
    tree_auc_sum =[] 
    tree_pca_auc_sum =[] 
    name = "Neural Net Optimization"
    seed = [8]
    for i in range(1, 6, 1):
        print("    layer:", seed)
        data = partition(data_frame, "genre", partition_ratio)
        pca_data = partition(data_frame, "genre", partition_ratio)
        cleaned_data = pca_transform(pc_count, *pca_data)
        results = classify(name, MLPClassifier(hidden_layer_sizes=tuple(seed)) , *data)
        pca_results = classify(name, MLPClassifier(hidden_layer_sizes=tuple(seed)) , *cleaned_data)
        auc, pca_auc = evaluate(name +" Neurons:"+ str(i), genre_list, *results, *pca_results, graph = False)
        tree_auc_sum.append(auc)
        tree_pca_auc_sum.append(pca_auc) 
        seed.insert(0, seed[0]*2)
    plt.figure()
    plt.plot(range(1,6,1), tree_auc_sum, label="Normal")
    plt.plot(range(1,6,1), tree_pca_auc_sum, label="PCA")
    plt.legend()
    plt.title(name)
    plt.xlabel("Hidden Layer Depth")
    plt.ylabel("Sum of AUC")
    plt.ion()
    plt.show()
    plt.pause(0.1)

def optimize_pc_bayes(data_frame, genre_list, partition_ratio):
    """Plot a graph of total auc v, tree depth with varying tree depth.
    params:
        pc_count - the number of features to keep for pca
        partition_ratio - the ratio of train/total for partitioning the data set"""
    print("OPTIMIZING NEURAL NET:")
    tree_pca_auc_sum =[] 
    name = "Naive Bayes Optimization"
    for i in range(1, 22, 1):
        print("    features:", i)
        data = partition(data_frame, "genre", partition_ratio)
        pca_data = partition(data_frame, "genre", partition_ratio)
        cleaned_data = pca_transform(i, *pca_data)
        results = classify(name, GaussianNB() , *data)
        pca_results = classify(name, GaussianNB() , *cleaned_data)
        auc, pca_auc = evaluate(name +" features:"+ str(i), genre_list, *results, *pca_results, graph = False)
        tree_pca_auc_sum.append(pca_auc) 
    plt.figure()
    plt.plot(range(1,22,1), tree_pca_auc_sum)
    plt.title(name)
    plt.xlabel("Count of Features")
    plt.ylabel("Sum of AUC")
    plt.ion()
    plt.show()
    plt.pause(0.1)

    
def main():
    """The executable to read in the specified data file and perform the
    investigatory operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The path of the data file to read in.")

    args = parser.parse_args()

    #preprocess the data 
    data_frame = pd.read_pickle(args.input_file.name).infer_objects()
    genre_list = sorted(["Punk", "Electronic","RnB", "Rap", "Country", "Metal", "Pop", "Rock"])
    data_frame = data_frame[data_frame["genre"].isin(genre_list)]
    print(data_frame.columns)
    data_frame = data_frame.drop(columns=["release_year"])
    data_frame = center_and_scale(data_frame)

    #hyperparameters
    partition_ratio = 0.8 
    print("PARTITION_RATIO = ", partition_ratio)
    pc_count = 12 
    print("PC_COUNT = ", pc_count)
    optimization_plots = False 
    print("OPTIMIZATION PLOTS", optimization_plots)

    #plot the optimal decision tree depth
    if optimization_plots:
        optimize_tree_depth(data_frame, genre_list, pc_count, partition_ratio)
        optimize_hidden_size(data_frame, genre_list, pc_count, partition_ratio)
        optimize_pc_bayes(data_frame, genre_list, partition_ratio)

    names = [
            "Decision Tree", 
            "Neural Net", 
            "Naive Bayes" 
            ]

    classifiers = [
            DecisionTreeClassifier(max_depth=8),
            MLPClassifier(hidden_layer_sizes=(32,16,8)),
            GaussianNB()
            ]

    for name, clf in zip(names, classifiers):
        print("STARTING:", name)
        print("   Partitioning:", name)
        data = partition(data_frame, "genre", partition_ratio)
        pca_data = partition(data_frame, "genre", partition_ratio)
        cleaned_data = pca_transform(pc_count, *pca_data)
        results = classify(name, clf, *data)
        pca_results = classify(name, clf, *cleaned_data)
        evaluate(name, genre_list, *results, *pca_results)
    input("Finished, press enter to close")

if __name__ == "__main__":
    main()
