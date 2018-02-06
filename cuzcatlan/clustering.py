"""
Copied and modified from the dev branch of:
https://github.com/genepattern/HierarchicalClustering
on 2018-01-31
"""
import sys
import numpy as np
from statistics import mode
from sklearn.metrics import pairwise
from sklearn import metrics

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from sklearn.cluster import AgglomerativeClustering
import scipy
import itertools
from .elemental import *
from .information import *

# check if these are repeated:
import os
import sys
tasklib_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# sys.path.append(tasklib_path + "/ccalnoir")
import matplotlib as mpl
mpl.use('Agg')
# import pandas as pd
# import numpy as np
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import AgglomerativeClustering
# from time import time
# import cuzcatlan as cusca
sns.set_style("white")

SIGNIFICANT_DIGITS = 7

input_col_distance_dict = {
    # These are the values I expect
    "No column clustering": "No_column_clustering",
    "Uncentered correlation": "uncentered_pearson",
    "Pearson correlation": "pearson",
    "Uncentered correlation, absolute value": "absolute_uncentered_pearson",
    "Pearson correlation, absolute value": "absolute_pearson",
    "Spearman's rank correlation": "spearman",
    "Kendall's tau": "kendall",
    "Euclidean distance": "euclidean",
    "City-block distance": "manhattan",
    "No_column_clustering": "No_column_clustering",
    # These are the values the GpUnit tests give
    "0": "No_column_clustering",
    "1": "uncentered_pearson",
    "2": "pearson",
    "3": "absolute_uncentered_pearson",
    "4": "absolute_pearson",
    "5": "spearman",
    "6": "kendall",
    "7": "euclidean",
    "8": "manhattan",
    "9": "information_coefficient",
    # These are the values I expect from the comand line
    "no_col": "No_column_clustering",
    "uncentered_pearson": "uncentered_pearson",
    "pearson": "pearson",
    "absolute_uncentered_pearson": "absolute_uncentered_pearson",
    "absolute_pearson": "absolute_pearson",
    "spearman": "spearman",
    "kendall": "kendall",
    "euclidean": "euclidean",
    "manhattan": "manhattan",
    "Cosine": "cosine",
    "cosine": "cosine",
    "ic": "information_coefficient",
    "information_coefficient": "information_coefficient",
    "Information Coefficient": "information_coefficient",
}

input_row_distance_dict = {
    # These are the values I expect
    "No row clustering": "No_row_clustering",
    "Uncentered correlation": "uncentered_pearson",
    "Pearson correlation": "pearson",
    "Uncentered correlation, absolute value": "absolute_uncentered_pearson",
    "Pearson correlation, absolute value": "absolute_pearson",
    "Spearman's rank correlation": "spearman",
    "Kendall's tau": "kendall",
    "Euclidean distance": "euclidean",
    "City-block distance": "manhattan",
    "No_row_clustering": "No_row_clustering",
    # These are the values the GpUnit tests give
    "0": "No_row_clustering",
    "1": "uncentered_pearson",
    "2": "pearson",
    "3": "absolute_uncentered_pearson",
    "4": "absolute_pearson",
    "5": "spearman",
    "6": "kendall",
    "7": "euclidean",
    "8": "manhattan",
    "9": "information_coefficient",
    # These are the values I expect from the comand line
    "no_row": "No_row_clustering",
    "uncentered_pearson": "uncentered_pearson",
    "pearson": "pearson",
    "absolute_uncentered_pearson": "absolute_uncentered_pearson",
    "absolute_pearson": "absolute_pearson",
    "spearman": "spearman",
    "kendall": "kendall",
    "euclidean": "euclidean",
    "manhattan": "manhattan",
    "Cosine": "cosine",
    "cosine": "cosine",
    "ic": "information_coefficient",
    "information_coefficient": "information_coefficient",
    "Information Coefficient": "information_coefficient",
}

input_clustering_method = {
    # These are the values I expect
    'Pairwise complete-linkage': 'complete',
    'Pairwise average-linkage': 'average',
    'Pairwise ward-linkage': 'ward',
    # These are the values the GpUnit test give
    'm': 'complete',
    'a': 'average',  # I think this is the default
}


input_row_centering = {
    # These are the values I expect
    'No': None,
    'Subtract the mean from each row': 'Mean',
    'Subtract the median from each row': 'Median',
    # These are the values the GpUnit test give
    'None': None,
    'Median': 'Median',
    'Mean': 'Mean',
}

input_row_normalize = {
    # These are the values I expect
    'No': False,
    'Yes': True,
    # These are the values the GpUnit test give
    'False': False,
    'True': True,
}

input_col_centering = {
    # These are the values I expect
    'No': None,
    'Subtract the mean from each column': 'Mean',
    'Subtract the median from each column': 'Median',
    # These are the values the GpUnit test give
    'None': None,
    'Median': 'Median',
    'Mean': 'Mean',
}

input_col_normalize = {
    # These are the values I expect
    'No': False,
    'Yes': True,
    # These are the values the GpUnit test give
    'False': False,
    'True': True,
}


def parse_inputs(args=sys.argv):
    # inp = []
    # inp = args
    # Error handling:
    arg_n = len(args)
    if arg_n == 1:
        sys.exit("Not enough parameters files were provided. This module needs a GCT file to work.")
    elif arg_n == 2:
        gct_name = args[1]
        col_distance_metric = 'euclidean'
        output_distances = False
        row_distance_metric = 'No_row_clustering'
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC_out'
        row_normalization = False
        col_normalization = False
        row_centering = None
        col_centering = None
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric = euclidean (default value)")
        print("\toutput_distances =", output_distances, "(default: not computing it and creating a file)")
        print("\trow_distance_metric =", row_distance_metric, "(default: No row clustering)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC_out)")
        print("\trow_normalization =", row_normalization, "(default: False)")
        print("\tcol_normalization =", col_normalization, "(default: False)")
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 3:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = False
        row_distance_metric = 'No_row_clustering'
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC_out'
        row_normalization = False
        col_normalization = False
        row_centering = None
        col_centering = None
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", input_col_distance_dict[col_distance_metric])
        print("\toutput_distances =", output_distances, "(default: not computing it and creating a file)")
        print("\trow_distance_metric =", row_distance_metric, "(default: No row clustering)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC_out)")
        print("\trow_normalization =", row_normalization, "(default: False)")
        print("\tcol_normalization =", col_normalization, "(default: False)")
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 4:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = 'No_row_clustering'
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC_out'
        row_normalization = False
        col_normalization = False
        row_centering = None
        col_centering = None

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        if (output_distances == 'False') or (output_distances == 'F')\
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric, "(default: No row clustering)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC_out)")
        print("\trow_normalization =", row_normalization, "(default: False)")
        print("\tcol_normalization =", col_normalization, "(default: False)")
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 5:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = 'Pairwise average-linkage'
        # clustering_method = 'Pairwise complete-linkage'
        output_base_name = 'HC_out'
        row_normalization = False
        col_normalization = False
        row_centering = None
        col_centering = None

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        row_distance_metric = input_row_distance_dict[row_distance_metric]
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC_out)")
        print("\trow_normalization =", row_normalization, "(default: False)")
        print("\tcol_normalization =", col_normalization, "(default: False)")
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 6:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        row_distance_metric = input_row_distance_dict[row_distance_metric]
        clustering_method = input_clustering_method[clustering_method]
        if clustering_method not in linkage_dic.keys():
            exit("Clustering method chosen not supported. This should not have happened.")

        if (linkage_dic[clustering_method] == 'ward') and (col_distance_metric != 'average'):
            exit("When choosing 'Pairwise ward-linkage' the distance metric *must* be 'average' ")

        output_base_name = 'HC_out'
        row_normalization = False
        col_normalization = False
        row_centering = None
        col_centering = None
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name, "(default: HC_out)")
        print("\trow_normalization =", row_normalization, "(default: False)")
        print("\tcol_normalization =", col_normalization, "(default: False)")
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 7:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]
        output_base_name = args[6]
        row_normalization = False
        col_normalization = False
        row_centering = None
        col_centering = None

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        row_distance_metric = input_row_distance_dict[row_distance_metric]
        clustering_method = input_clustering_method[clustering_method]
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name)
        print("\trow_normalization =", row_normalization, "(default: False)")
        print("\tcol_normalization =", col_normalization, "(default: False)")
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 8:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]
        output_base_name = args[6]
        row_normalization = args[7]
        col_normalization = False
        row_centering = None
        col_centering = None

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        row_distance_metric = input_row_distance_dict[row_distance_metric]
        clustering_method = input_clustering_method[clustering_method]
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True

        row_normalization = input_row_normalize[row_normalization]
        # if (row_normalization == 'False') or (row_normalization == 'F') \
        #         or (row_normalization == 'false') or (row_normalization == 'f'):
        #     row_normalization = False
        # else:
        #     row_normalization = True

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name)
        print("\trow_normalization =", row_normalization)
        print("\tcol_normalization =", col_normalization, "(default: False)")
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 9:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]
        output_base_name = args[6]
        row_normalization = args[7]
        col_normalization = args[8]
        row_centering = None
        col_centering = None

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        row_distance_metric = input_row_distance_dict[row_distance_metric]
        clustering_method = input_clustering_method[clustering_method]
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True

        # Row normalization
        row_normalization = input_row_normalize[row_normalization]
        # if (row_normalization == 'False') or (row_normalization == 'F') \
        #         or (row_normalization == 'false') or (row_normalization == 'f'):
        #     row_normalization = False
        # else:
        #     row_normalization = True

        # Column normalization
        col_normalization = input_col_normalize[col_normalization]
        # if (col_normalization == 'False') or (col_normalization == 'F') \
        #         or (col_normalization == 'false') or (col_normalization == 'f'):
        #     col_normalization = False
        # else:
        #     col_normalization = True

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name)
        print("\trow_normalization =", row_normalization)
        print("\tcol_normalization =", col_normalization)
        print("\trow_centering =", row_centering, "(default: None)")
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 10:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]
        output_base_name = args[6]
        row_normalization = args[7]
        col_normalization = args[8]
        row_centering = args[9]
        col_centering = None

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        row_distance_metric = input_row_distance_dict[row_distance_metric]
        clustering_method = input_clustering_method[clustering_method]

        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True

        # Row normalization
        row_normalization = input_row_normalize[row_normalization]
        # if (row_normalization == 'False') or (row_normalization == 'F') \
        #         or (row_normalization == 'false') or (row_normalization == 'f'):
        #     row_normalization = False
        # else:
        #     row_normalization = True

        # Column normalization
        col_normalization = input_col_normalize[col_normalization]
        # if (col_normalization == 'False') or (col_normalization == 'F') \
        #         or (col_normalization == 'false') or (col_normalization == 'f'):
        #     col_normalization = False
        # else:
        #     col_normalization = True

        # row_centering
        row_centering = input_row_centering[row_centering]
        if (row_centering == 'None') or (col_normalization == 'N') \
                or (row_centering == 'none') or (col_normalization == 'n'):
            col_normalization = None

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name)
        print("\trow_normalization =", row_normalization)
        print("\tcol_normalization =", col_normalization)
        print("\trow_centering =", row_centering)
        print("\tcol_centering =", col_centering, "(default: None)")
    elif arg_n == 11:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]
        output_base_name = args[6]
        row_normalization = args[7]
        col_normalization = args[8]
        row_centering = args[9]
        col_centering = args[10]

        col_distance_metric = input_col_distance_dict[col_distance_metric]
        row_distance_metric = input_row_distance_dict[row_distance_metric]
        clustering_method = input_clustering_method[clustering_method]

        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True

        # Row normalization
        row_normalization = input_row_normalize[row_normalization]
        # if (row_normalization == 'False') or (row_normalization == 'F') \
        #         or (row_normalization == 'false') or (row_normalization == 'f'):
        #     row_normalization = False
        # else:
        #     row_normalization = True

        # Column normalization
        col_normalization = input_col_normalize[col_normalization]
        # if (col_normalization == 'False') or (col_normalization == 'F') \
        #         or (col_normalization == 'false') or (col_normalization == 'f'):
        #     col_normalization = False
        # else:
        #     col_normalization = True

        # row_centering
        row_centering = input_row_centering[row_centering]
        if (row_centering == 'None') or (col_normalization == 'N') \
                or (row_centering == 'none') or (col_normalization == 'n'):
            col_normalization = None

        # col_centering
        col_centering = input_col_centering[col_centering]
        if (col_centering == 'None') or (col_centering == 'N') \
                or (col_centering == 'none') or (col_centering == 'n'):
            col_centering = None

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name)
        print("\trow_normalization =", row_normalization)
        print("\tcol_normalization =", col_normalization)
        print("\trow_centering =", row_centering)
        print("\tcol_centering =", col_centering)
    else:
        sys.exit("Too many inputs. This module needs only a GCT file to work, "
                 "plus an optional input choosing between Pearson Correlation or Information Coefficient.")

    print(args)
    return gct_name, col_distance_metric, output_distances, row_distance_metric, clustering_method, output_base_name, \
        row_normalization, col_normalization, row_centering, col_centering


def plot_dendrogram(model, data, tree, axis, dist=mydist, title='no_title.png', col_thresh=0, **kwargs):
    #     plt.clf()

    # modified from https://github.com/scikit-learn/scikit-learn/pull/3464/files
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # TODO: Fix this mydist
    # distance = dendodist(children, euclidian_similarity)
    # distance = dendodist(children, dist)

    og_distances = better_dendodist(children, dist, tree, data, axis=axis)
    #     print(og_distances)
    #     og_distances = [abs(temp) for temp in og_distances]
    #     og_distances = [abs(temp) for temp in og_distances]
    #     print(og_distances)
    distance = np.cumsum(og_distances)
    #     distance = og_distances
    #     distance = better_dendodist(children, dist, tree, data, axis=axis)

    # norm_distances = []
    # for value in distance:
    #     norm_distances.append(1/value)
    # norm_distances = distance

    list_of_children = list(get_children(tree, leaves_are_self_children=False).values())
    no_of_observations = [len(i) for i in list_of_children if i]
    no_of_observations.append(len(no_of_observations) + 1)
    # print(len(no_of_observations))

    # print(children)

    # print(list(tree.values()))

    # print(norm_distances)

    # print(distance)
    if all(value == 0 for value in distance):
        # If all distances are zero, then use uniform distance
        distance = np.arange(len(distance))

    # print(distance)
    # print(np.cumsum(distance))

    # The number of observations contained in each cluster level
    # no_of_observations = np.arange(2, children.shape[0]+2)
    # print(no_of_observations)


    # Create linkage matrix and then plot the dendrogram
    # linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # linkage_matrix = np.column_stack([children, np.cumsum(distance), no_of_observations]).astype(float)
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # linkage_matrix = np.column_stack([children, norm_distances, no_of_observations]).astype(float)
    # print(linkage_matrix)
    # Plot the corresponding dendrogram

    #     print(linkage_matrix)


    R = dendrogram(linkage_matrix, color_threshold=col_thresh, **kwargs)
    #     R = dendrogram(linkage_matrix, **kwargs)
    #     [label.set_rotation(90) for label in plt.gca().get_xticklabels()]
    order_of_columns = R['ivl']
    # # print(order_of_columns)
    #     plt.gca().get_yaxis().set_visible(False)
    #     plt.savefig(title, dpi=300)
    #     plt.show()

    # n = len(linkage_matrix) + 1
    # cache = dict()
    # for k in range(len(linkage_matrix)):
    #     c1, c2 = int(linkage_matrix[k][0]), int(linkage_matrix[k][1])
    #     c1 = [c1] if c1 < n else cache.pop(c1)
    #     c2 = [c2] if c2 < n else cache.pop(c2)
    #     cache[n + k] = c1 + c2
    # order_of_columns = cache[2 * len(linkage_matrix)]
    return order_of_columns, linkage_matrix


def order_leaves(model, data, tree, labels, axis=0, dist=mydist, reverse = False):
    # Adapted from here: https://stackoverflow.com/questions/12572436/calculate-ordering-of-dendrogram-leaves

    children = model.children_
    # distance = better_dendodist(children, dist, tree, data, axis=axis)
    # if all(value == 0 for value in distance):
    #     distance = np.arange(len(distance))

    # list_of_children = list(get_children(tree, leaves_are_self_children=False).values())
    # no_of_observations = [len(i) for i in list_of_children if i]
    # no_of_observations.append(len(no_of_observations)+1)

    # Create linkage matrix and then plot the dendrogram
    # linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    pseudo_linkage_matrix = np.column_stack([children]).astype(float)

    n = len(pseudo_linkage_matrix) + 1

    # This orders leaves by number of clusters
    cache = dict()
    for k in range(len(pseudo_linkage_matrix)):
        c1, c2 = int(pseudo_linkage_matrix[k][0]), int(pseudo_linkage_matrix[k][1])
        c1 = [c1] if c1 < n else cache.pop(c1)
        c2 = [c2] if c2 < n else cache.pop(c2)
        cache[n + k] = c1 + c2
    numeric_order_of_leaves = cache[2 * len(pseudo_linkage_matrix)]

    if reverse:
        numeric_order_of_leaves = list(reversed(numeric_order_of_leaves))

    return [labels[i] for i in numeric_order_of_leaves]


def two_plot_two_dendrogram(model, dist=mydist, **kwargs):
    #modified from https://github.com/scikit-learn/scikit-learn/pull/3464/files
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    distance = dendodist(children, dist)
    if all(value == 0 for value in distance):
        # If all distances are zero, then use uniform distance
        distance = np.arange(len(distance))

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, color_threshold=0, orientation='left', **kwargs)
    # [label.set_rotation(90) for label in plt.gca().get_xticklabels()]
    order_of_rows = R['ivl']
    # print(order_of_columns)
    plt.gca().get_xaxis().set_visible(False)

    return list(reversed(order_of_rows))


def my_affinity_generic(M, metric):
    return np.array([np.array([metric(a, b) for a in M])for b in M])


def my_affinity_i(M):
    return np.array([[information_coefficient_dist(a, b) for a in M]for b in M])


def my_affinity_ai(M):
    return np.array([[absolute_information_coefficient_dist(a, b) for a in M]for b in M])


def my_affinity_p(M):
    return np.array([[custom_pearson_dist(a, b) for a in M]for b in M])


def my_affinity_s(M):
    return np.array([[custom_spearman_dist(a, b) for a in M]for b in M])


def my_affinity_k(M):
    return np.array([[custom_kendall_tau_dist(a, b) for a in M]for b in M])


def my_affinity_ap(M):
    return np.array([[absolute_pearson_dist(a, b) for a in M]for b in M])


def my_affinity_u(M):
    return np.array([[uncentered_pearson_dist(a, b) for a in M]for b in M])


def my_affinity_au(M):
    return np.array([[absolute_uncentered_pearson_dist(a, b) for a in M]for b in M])


def my_affinity_l1(M):
    return np.array([[custom_manhattan_dist(a, b) for a in M]for b in M])


def my_affinity_l2(M):
    return np.array([[custom_euclidean_dist(a, b) for a in M]for b in M])


def my_affinity_m(M):
    return np.array([[custom_manhattan_dist(a, b) for a in M]for b in M])


def my_affinity_c(M):
    return np.array([[custom_cosine_dist(a, b) for a in M]for b in M])


def my_affinity_e(M):
    # global dist_matrix
    # dist_matrix = np.array([[mydist(a, b) for a in M]for b in M])
    # return dist_matrix
    return np.array([[custom_euclidean_dist(a, b) for a in M] for b in M])


def count_diff(x):
    count = 0
    compare = x[0]
    for i in x:
        if i != compare:
            count += 1
    return count


def count_mislabels(labels, true_labels):
    # 2017-08-17: I will make the assumption that clusters have only 2 values.
    # clusters = np.unique(true_labels)
    # mislabels = 0
    # for curr_clust in clusters:
    #     print("for label", curr_clust)
    #     print("\t", labels[(true_labels == curr_clust)])
    #     compare_to = mode(labels[(true_labels == curr_clust)])
    #     print("\tcompare to:", compare_to, "mislables: ", np.count_nonzero(labels[(true_labels == curr_clust)] != compare_to))
    #     mislabels += np.count_nonzero(labels[(true_labels == curr_clust)] != compare_to)

    set_a = labels[true_labels == 0]
    set_b = labels[true_labels == 1]

    if len(set_a) <= len(set_b):
        shorter = set_a
        longer = set_b
    else:
        shorter = set_b
        longer = set_a

    long_mode = mode(longer)  # this what the label of the longer cluster should be.
    short_mode = 1 if long_mode == 0 else 0  # Choose the other value for the label of the shorter cluster

    # start with the longer vector:
    # print("The long set is", longer, "it has", np.count_nonzero(longer != long_mode), 'mislabels.')
    # print("The short set is", shorter, "it has", np.count_nonzero(shorter != short_mode), 'mislabels.')

    # np.count_nonzero(longer != long_mode) + np.count_nonzero(shorter != short_mode)

    return np.count_nonzero(longer != long_mode) + np.count_nonzero(shorter != short_mode)


def plot_heatmap(df, col_order, row_order, top=5, title_text='differentially expressed genes per phenotype'):
    if not(len(col_order), len(list(df))):
        exit("Number of columns in dataframe do not match the columns provided for ordering.")
    if not(len(row_order), len(df)):
        exit("Number of rows in dataframe do not match the columns provided for ordering.")
    # print(list(df), col_order)
    df = df[col_order]
    df = df.reindex(row_order)

    plt.clf()
    sns.heatmap(df.iloc[np.r_[0:top, -top:0], :], cmap='viridis')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Top {} {}'.format(top, title_text))
    plt.ylabel('Genes')
    plt.xlabel('Sample')
    plt.savefig('heatmap.png', dpi=300, bbox_inches="tight")


def parse_data(gct_name, row_normalization=False, col_normalization=False, row_centering=None, col_centering=None):
    f = open(gct_name)
    f.readline()
    size = f.readline().strip('\n').split('\t')

    data_df = pd.read_csv(gct_name, sep='\t', skiprows=2)
    # print(size)
    # print(list(data_df))
    # exit(data_df.shape)

    if 'Name' not in list(data_df):
        data_df['Name'] = data_df.iloc[:, 0]
        data_df.drop(data_df.columns[0], axis=1, inplace=True)
    if 'Description' not in list(data_df):
        data_df['Description'] = data_df['Name']

    data_df.set_index(data_df['Name'], inplace=True)
    og_full_gct = data_df.copy()
    og_full_gct.drop(['Name'], axis=1, inplace=True)
    data_df.drop(['Name', 'Description'], axis=1, inplace=True)
    plot_labels = list(og_full_gct.drop(['Description'], axis=1, inplace=False))
    data = data_df.as_matrix()
    row_labels = data_df.index.values

    og_data = data.copy()

    if row_centering is not None:
        if row_centering == 'Mean':
            row_means = np.mean(data, axis=1)
            row_means_col_vec = row_means.reshape((data.shape[0], 1))
            data = data - row_means_col_vec
        if row_centering == 'Median':
            row_medians = np.median(data, axis=1)
            row_medians_col_vec = row_medians.reshape((data.shape[0], 1))
            data = data - row_medians_col_vec

    if row_normalization:
        row_norm = np.sum(data * data, axis=1)
        row_norm_col_vec = row_norm.reshape((data.shape[0], 1))
        data = data / np.sqrt(row_norm_col_vec)

    if col_centering is not None:
        if col_centering == 'Mean':
            col_means = np.mean(data, axis=0)
            data = data - col_means
        if col_centering == 'Median':
            col_medians = np.median(data, axis=0)
            data = data - col_medians

    if col_normalization:
        col_norm = np.sum(data*data, axis=0)
        data = data/np.sqrt(col_norm)

    # print(data_df)
    # print(data)
    new_data_df = pd.DataFrame(data=data, index=data_df.index, columns=list(data_df))
    # print(new_data_df)
    # print(og_full_gct)
    new_full_gct = new_data_df.copy()
    new_full_gct.insert(0, column='Description', value=og_full_gct['Description'])
    # print(new_full_gct)
    # exit()

    return og_data, data_df, data, new_data_df, plot_labels, row_labels, og_full_gct, new_full_gct


str2func = {
    'custom_euclidean': my_affinity_e,
    'uncentered_pearson': my_affinity_u,
    'absolute_uncentered_pearson': my_affinity_au,
    'information_coefficient': my_affinity_i,
    'pearson': my_affinity_p,
    'spearman': my_affinity_s,
    'kendall': my_affinity_k,
    'absolute_pearson': my_affinity_ap,
    'l1': 'l1',
    'l2': 'l2',
    'manhattan': 'manhattan',
    'cosine': 'cosine',
    'euclidean': 'euclidean',
}


str2affinity_func = {
    'custom_euclidean': my_affinity_e,
    'uncentered_pearson': my_affinity_u,
    'absolute_uncentered_pearson': my_affinity_au,
    'information_coefficient': my_affinity_i,
    'pearson': my_affinity_p,
    'spearman': my_affinity_s,
    'kendall': my_affinity_k,
    'absolute_pearson': my_affinity_ap,
    'l1': my_affinity_l1,
    'l2': my_affinity_l2,
    'manhattan': my_affinity_m,
    'cosine': my_affinity_c,
    'euclidean': my_affinity_e,
}

str2dist = {
    'custom_euclidean': custom_euclidean_dist,
    'uncentered_pearson': uncentered_pearson_dist,
    'absolute_uncentered_pearson': absolute_uncentered_pearson_dist,
    'information_coefficient': information_coefficient_dist,
    'pearson': custom_pearson_dist,
    'spearman': custom_spearman_dist,
    'kendall': custom_kendall_tau_dist,
    'absolute_pearson': absolute_pearson_dist,
    'l1': custom_manhattan_dist,
    'l2': custom_euclidean_dist,
    'manhattan': custom_manhattan_dist,
    'cosine': custom_cosine_dist,
    'euclidean': custom_euclidean_dist,
}

str2similarity = {
    'custom_euclidean': custom_euclidean_sim,
    'uncentered_pearson': uncentered_pearson_corr,
    'absolute_uncentered_pearson': absolute_uncentered_pearson_corr,
    'information_coefficient': information_coefficient,
    'pearson': custom_pearson_corr,
    'spearman': custom_spearman_corr,
    'kendall': custom_kendall_tau_corr,
    'absolute_pearson': absolute_pearson_corr,
    'l1': custom_manhattan_sim,
    'l2': custom_euclidean_sim,
    'manhattan': custom_manhattan_sim,
    'cosine': custom_cosine_sim,
    # 'euclidean': pairwise.paired_euclidean_distances,
    'euclidean': custom_euclidean_sim,
    # 'euclidean': custom_euclidean_dist,
}

linkage_dic = {
    'Pairwise average-linkage': 'average',
    'Pairwise complete-linkage': 'complete',
    'Pairwise ward-linkage': 'ward',
    'average': 'average',
    'complete': 'complete',
    'ward': 'ward',
}


def make_tree(model, data=None):
    """
    Modified from:
    https://stackoverflow.com/questions/27386641/how-to-traverse-a-tree-from-sklearn-agglomerativeclustering
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    import itertools

    X = np.concatenate([np.random.randn(3, 10), np.random.randn(2, 10) + 100])
    model = AgglomerativeClustering(linkage="average", affinity="cosine")
    model.fit(X)

    ii = itertools.count(X.shape[0])
    [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]

    ---

    You can also do dict(enumerate(model.children_, model.n_leaves_))
    which will give you a dictionary where the each key is the ID of a node
    and the value is the pair of IDs of its children. – user76284

    :param model:
    :return: a dictionary where the each key is the ID of a node and the value is the pair of IDs of its children.
    """
    # ii = itertools.count(data.shape[0])  # Setting the counter at the number of leaves.
    # tree = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]
    # print(tree)
    # return tree

    return dict(enumerate(model.children_, model.n_leaves_))
    # return dict(enumerate(model.children_, 1))


def make_cdt(data, order_of_columns, order_of_rows, name='test.cdt', atr_companion=True, gtr_companion=False):
    # TODO: if order_of_columns == None, then do arange(len(list(data)))
    # TODO: if order_of_rows == None, then do arange(len(list(data)))
    # exit(data.to_csv())
    data.index.name = "ID"
    data.rename(columns={'Description': 'Name'}, inplace=True)

    temp = np.ones(len(data))
    data.insert(loc=1, column='GWEIGHT', value=temp)  # adding an extra column

    # These three lines add a row
    data.loc['EWEIGHT'] = list(np.ones(len(list(data))))
    newIndex = ['EWEIGHT'] + [ind for ind in data.index if ind != 'EWEIGHT']
    data = data.reindex(index=newIndex)

    if atr_companion:
        new_AID = ['', '']
        for element in range(len(order_of_columns)):
            temp = 'ARRY'+str(element)+'X'
            new_AID.append(temp)

        data.loc['AID'] = new_AID
        newIndex = ['AID'] + [ind for ind in data.index if ind != 'AID']
        data = data.reindex(index=newIndex)
        data = data[['Name', 'GWEIGHT']+order_of_columns]
    if gtr_companion:
        new_GID = ['']
        if atr_companion:
            new_GID = ['AID', 'EWEIGHT']  # This is to make sure we fit the CDT format
        # for element in np.sort(np.unique(GID)):
            # if 'NODE' in element:
            #     # print(element, 'GTR delete')
            #     pass
            # else:
            #     new_GID.append(element)
        for element in range(len(order_of_rows)):
            temp = 'GENE' + str(element) + 'X'
            new_GID.append(temp)

        data.insert(loc=0, column='GID', value=new_GID)  # adding an extra column
        data.insert(loc=0, column=data.index.name, value=data.index)  # Making the index a column

        # reorder to match dendogram
        temp = ['AID', 'EWEIGHT'] + order_of_rows
        data = data.loc[temp]
        # print(list(data.index))
        # print(data['GID'])
        # print(data['Name'])

        # Making the 'GID' the index -- for printing purposes
        data.index = data['GID']
        data.index.name = 'GID'
        data.drop(['GID'], axis=1, inplace=True)
        # print(list(data.index))

    # The first three lines need to be written separately due to a quirk in the CDT file format:

    # print(data.to_csv(sep='\t', index=True, header=True))
    f = open(name, 'w')
    f.write(data.to_csv(sep='\t', index=True, header=True))
    # f.write(data.to_csv(sep='\t', index=True, header=True))
    f.close()
    # pd.options.display.float_format = '{:3.3f}'.format
    data = data.round(2)
    # print(data.to_csv())
    # exit()
    # exit(data.to_csv(sep=' ', index=True, header=True, float_format='2',))
    return


def make_atr(col_tree_dic, data, dist, clustering_method='average', file_name='test.atr'):
    max_val = len(col_tree_dic)
    # AID = []

    # compute distances
    distance_dic = {}
    for node, children in col_tree_dic.items():
        val = centroid_distances(children[0], children[1], tree=col_tree_dic, data=data, axis=1,
                                 distance=dist, clustering_method=clustering_method)
        # print(dist, children, val)
        # print("Value is", val)
        distance_dic[node] = val

    # if dist == custom_euclidean_sim:
    #     print("Euclidean distance is especial, normalizing using this scheme:")
    #     low_norm = min(distance_dic.values())
    #     high_norm = max(distance_dic.values())
    #     for key in distance_dic.keys():
    #         # distance -= norm
    #         # distance_dic[key] = distance_dic[key]/high_norm
    #         # distance_dic[key] = (distance_dic[key]-low_norm)/high_norm
    #         # distance_dic[key] = distance_dic[key]/high_norm
    #         # distance_dic[key] = ((1/distance_dic[key])-high_norm)/low_norm
    #         print(distance_dic[key])

    f = open(file_name, 'w')
    for node, children in col_tree_dic.items():
        elements = [translate_tree(node, max_val, 'atr'), translate_tree(children[0], max_val, 'atr'),
                    translate_tree(children[1], max_val, 'atr'),
                    "{num:.{width}f}".format(num=distance_dic[node], width=SIGNIFICANT_DIGITS)]
        # print('\t', '\t'.join(elements))
        # AID.append(translate_tree(children[0], max_val, 'atr'))
        # AID.append(translate_tree(children[1], max_val, 'atr'))
        f.write('\t'.join(elements) + '\n')
        # print('\t'.join(elements) + '\n')
    f.close()

    return


def make_gtr(row_tree_dic, data, dist, clustering_method='average',  file_name='test.gtr'):
    max_val = len(row_tree_dic)
    # GID = []

    # compute distances
    distance_dic = {}
    for node, children in row_tree_dic.items():
        val = centroid_distances(children[0], children[1], tree=row_tree_dic, data=data, axis=0,
                                 distance=dist, clustering_method=clustering_method)
        distance_dic[node] = val

    f = open(file_name, 'w')
    for node, children in row_tree_dic.items():

        elements = [translate_tree(node, max_val, 'gtr'), translate_tree(children[0], max_val, 'gtr'),
                    translate_tree(children[1], max_val, 'gtr'),
                    "{num:.{width}f}".format(num=distance_dic[node], width=SIGNIFICANT_DIGITS)]
        # GID.append(translate_tree(children[0], max_val, 'gtr'))
        # GID.append(translate_tree(children[1], max_val, 'gtr'))
        f.write('\t'.join(elements)+'\n')
        # val -= 1
    f.close()

    return


def translate_tree(what, length, g_or_a):
    if 'a' in g_or_a:
        if what <= length:
            translation = 'ARRY'+str(what)+'X'
        else:
            translation = 'NODE' + str(what-length) + 'X'
    elif 'g' in g_or_a:
        if what <= length:
            translation = 'GENE'+str(what)+'X'
        else:
            translation = 'NODE' + str(what-length) + 'X'
    else:
        translation = []
        print('This function does not support g_or_a=', g_or_a)
    return translation


# def get_children_recursively(k, model, node_dict, leaf_count, n_samples, data, verbose=False, left=None, right=None):
#     # print(k)
#     i, j = model.children_[k]
#
#     if k in node_dict:
#         return node_dict[k]['children']
#
#     if i < leaf_count:
#         # print("i if")
#         left = [i]
#     else:
#         # print("i else")
#         # read the AgglomerativeClustering doc. to see why I select i-n_samples
#         left, node_dict = get_children_recursively(i - n_samples, model, node_dict,
#                                                    leaf_count, n_samples, data, verbose, left, right)
#
#     if j < leaf_count:
#         # print("j if")
#         right = [j]
#     else:
#         # print("j else")
#         right, node_dict = get_children_recursively(j - n_samples, model, node_dict,
#                                                     leaf_count, n_samples, data, verbose, left, right)
#
#     if verbose:
#         print(k, i, j, left, right)
#     temp = map(lambda ii: data[ii], left)
#     left_pos = np.mean(list(temp), axis=0)
#     temp = map(lambda ii: data[ii], right)
#     right_pos = np.mean(list(temp), axis=0)
#
#     # this assumes that agg_cluster used euclidean distances
#     dist = metrics.pairwise_distances([left_pos, right_pos], metric='euclidean')[0, 1]
#
#     all_children = [x for y in [left, right] for x in y]
#     pos = np.mean(list(map(lambda ii: data[ii], all_children)), axis=0)
#
#     # store the results to speed up any additional or recursive evaluations
#     node_dict[k] = {'top_child': [i, j], 'children': all_children, 'pos': pos, 'dist': dist,
#                     'node_i': k + n_samples}
#     return all_children, node_dict

# def recursive_atr


def get_children(tree, leaves_are_self_children=False):
    # this is a recursive function
    expanded_tree = {}
    for node in range(max(tree.keys())):
        if node <= len(tree):
            if leaves_are_self_children:
                expanded_tree[node] = [node]
            else:
                expanded_tree[node] = []

        else:
            # expanded_tree[node] = list_children_single_node(node, tree)
            expanded_tree[node] = list_children_single_node(node, tree, leaves_are_self_children)

    return expanded_tree


def list_children_single_node(node, tree, leaves_are_self_children=False, only_leaves_are_children=True):
    # children = []
    if node <= len(tree):
        if leaves_are_self_children:
            children = [node]
        else:
            children = []

    else:
        children = list(tree[node])

        # Check each child, and add their children to the list
        for child in children:
            if child <= len(tree):
                pass
            else:
                children += list_children_single_node(child, tree, only_leaves_are_children=True)
    if only_leaves_are_children:
        # print(sorted(np.unique(i for i in children if i <= len(tree))))
        # print()
        return [i for i in sorted(np.unique(children))if i <= len(tree)]
    else:
        return sorted(np.unique(children))


def centroid_distances(node_a, node_b, tree, data, axis=0, distance=mydist, clustering_method='average'):
    if axis == 0:
        pass
    elif axis == 1:
        data = np.transpose(data)
    else:
        exit("Variable 'data' does not have that many axises (╯°□°)╯︵ ┻━┻")

    children_of_a = list_children_single_node(node_a, tree=tree, leaves_are_self_children=True)
    children_of_b = list_children_single_node(node_b, tree=tree, leaves_are_self_children=True)

    # if distance == custom_euclidean_sim:
    #     print("Euclidean distance is especial, normalizing using this scheme:")
    #     distance = custom_euclidean_dist

    distances_list = []
    if clustering_method == 'average':
        for pair in itertools.product(data[children_of_a], data[children_of_b]):
            distances_list.append(distance(pair[0], pair[1]))
        return np.average(distances_list)
    elif clustering_method == 'complete':
        for pair in itertools.product(data[children_of_a], data[children_of_b]):
            distances_list.append(distance(pair[0], pair[1]))
        return np.min(distances_list)
    else:
        exit("Ony 'average' and 'complete' clustering methods are accepted at the moment (>_<)")


def euclidian_similarity(x, y):
    dist = mydist(x, y)
    # return 1/(1+dist)
    return 1/(np.exp(dist))


def better_dendodist(children, distance, tree, data, axis):
    distances_list = []
    for pair in children:
        distances_list.append(centroid_distances(pair[0], pair[1], tree, data, axis, distance=distance))
        # print(distance, pair, distances_list[-1])
    return distances_list


def HierarchicalClustering(pwd, gct_name, col_distance_metric, output_distances, row_distance_metric,
                           clustering_method, output_base_name, row_normalization, col_normalization,
                           row_centering, col_centering, custom_plot=False):
    # gct_name, col_distance_metric, output_distances, row_distance_metric, clustering_method, output_base_name, \
    # row_normalization, col_normalization, row_centering, col_centering = parse_inputs(sys.argv)

    og_data, og_data_df, data, data_df, col_labels, row_labels, og_full_gct, new_full_gct = \
        parse_data(gct_name, row_normalization, col_normalization, row_centering, col_centering)
    order_of_columns = list(data_df)
    order_of_rows = list(data_df.index)

    data_transpose = np.transpose(data)

    # print(data)
    # print(data_df)

    atr_companion = False
    col_model = None
    col_tree = None

    gtr_companion = False
    row_model = None
    row_tree = None

    AID = None
    GID = None

    if col_distance_metric != 'No_column_clustering':
        atr_companion = True
        col_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                            affinity=str2func[col_distance_metric])

        col_model.fit(data_transpose)
        col_tree = make_tree(col_model)
        order_of_columns = order_leaves(col_model, tree=col_tree, data=data_transpose,
                                        dist=str2similarity[col_distance_metric], labels=col_labels, reverse=True)

        make_atr(col_tree, file_name=output_base_name + '.atr', data=data,
                 dist=str2similarity[col_distance_metric], clustering_method=linkage_dic[clustering_method])

    if row_distance_metric != 'No_row_clustering':
        gtr_companion = True
        row_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                            affinity=str2func[row_distance_metric])
        # y_col = row_model.fit_predict(np.transpose(data))
        # print(y_col)
        row_model.fit(data)
        row_tree = make_tree(row_model)
        order_of_rows = order_leaves(row_model, tree=row_tree, data=data,
                                     dist=str2similarity[row_distance_metric], labels=row_labels)
        make_gtr(row_tree, data=data, file_name=output_base_name + '.gtr', dist=str2similarity[row_distance_metric])

    if output_distances:
        # TODO: check which col or row was selected, or both
        row_distance_matrix = str2affinity_func[row_distance_metric](data)
        # col_distance_matrix = str2affinity_func[col_distance_metric](np.transpose(data))
        dist_file = open(output_base_name + '_pairwise_distances.csv', 'w')
        dist_file.write('labels,')
        dist_file.write(",".join(col_model.labels_.astype(str)) + "\n")
        dist_file.write('samples,')
        dist_file.write(",".join(list(data_df)) + "\n")
        i = 0
        for row in row_distance_matrix:
            dist_file.write('distances row=' + str(i) + "," + ",".join(row.astype(str)) + "\n")
            i += 1

    make_cdt(data=new_full_gct, name=output_base_name + '.cdt', atr_companion=atr_companion,
             gtr_companion=gtr_companion,
             order_of_columns=order_of_columns, order_of_rows=order_of_rows)

    if custom_plot:
        # Plotting the heatmap with dendrogram

        plt.clf()
        # fig = plt.figure(figsize=(16, 9), dpi=300)
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
        gs.update(wspace=0.0, hspace=0.0)
        ax0 = plt.subplot(gs[0])  # Doing dendrogram first
        ax0.axis('off')

        col_order, link = plot_dendrogram(col_model, data, col_tree, axis=1, dist=custom_pearson_corr, col_thresh=15,
                                          title='no_title.png')
        col_order = [int(i) for i in col_order]

        # print(col_order)
        named_col_order = [col_labels[i] for i in col_order]
        # print(named_col_order)

        ax1 = plt.subplot(gs[1])

        #Row-normalizing for display purposes only:
        data_df = data_df.subtract(data_df.min(axis=1), axis=0)
        data_df = data_df.div(data_df.max(axis=1), axis=0)

        sns.heatmap(data_df[named_col_order], ax=ax1, cbar=False, cmap='bwr')
        # ax1.xaxis.tick_top()
        [label.set_rotation(90) for label in ax1.get_xticklabels()]
        plt.show()

    return col_model, row_model
