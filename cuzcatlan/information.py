"""Parts of this code are borrowed from the CCAL library: https://github.com/UCSD-CCAL"""
import sys
from numpy import asarray, exp, finfo, isnan, log, sign, sqrt, sum, sort, array, concatenate, where, nan_to_num, \
    apply_along_axis, array_split, empty, unique
from numpy.random import random_sample, seed, randint, choice, get_state, set_state, shuffle
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy.stats import norm, rankdata
import numpy as np
from pandas import DataFrame, Series, read_table
from os import mkdir, remove
from os.path import abspath, exists, isdir, islink, split, isfile
from shutil import copy, copytree, rmtree
from math import ceil, floor
sys.stdout.flush()
import matplotlib.pyplot as plt
sys.stdout.flush()
# from matplotlib.pyplot import
from matplotlib.pyplot import figure, subplot, savefig, gca, sca, suptitle, show
from matplotlib.gridspec import GridSpec
from matplotlib.cm import Paired, Set3, bwr, tab20, tab20b, tab20c
from scipy.cluster.hierarchy import dendrogram, linkage
from multiprocessing.pool import Pool
from statsmodels.sandbox.stats.multicomp import multipletests
from seaborn import heatmap, despine, set_style
from .ccal_style import *
import pandas as pd
import urllib.request
import validators

EPS = finfo(float).eps
RANDOM_SEED = 20121020


def information_coefficient_dist(x, y, n_grids=25, jitter=1E-10, random_seed=20170821):
    return 1 - information_coefficient(x, y, n_grids, jitter, random_seed)


def compute_information_coefficient(x, y, **kwargs):
    return information_coefficient(x, y, **kwargs)


def information_coefficient(x, y, n_grids=25,
                            jitter=1E-10, random_seed=20170821):
    """
    Compute the information coefficient between x and y, which are
        continuous, categorical, or binary vectors. This function uses only python libraries -- No R is needed.
    :param x: numpy array;
    :param y: numpy array;
    :param n_grids: int; number of grids for computing bandwidths
    :param jitter: number;
    :param random_seed: int or array-like;
    :return: float; Information coefficient
    """

    # Can't work with missing any value
    # not_nan_filter = ~isnan(x)
    # not_nan_filter &= ~isnan(y)
    # x = x[not_nan_filter]
    # y = y[not_nan_filter]

    x, y = drop_nan_columns([x, y])

    if (x == y).all():
        return 1
    else:
        try:
            # Need at least 3 values to compute bandwidth
            if len(x) < 3 or len(y) < 3:
                return 0
        except TypeError:
            # If x and y are numbers, we cannot continue and IC is zero.
            return 0

        x = asarray(x, dtype=float)
        y = asarray(y, dtype=float)

        # Add jitter
        seed(random_seed)
        x += random_sample(x.size) * jitter
        y += random_sample(y.size) * jitter

        # Compute bandwidths
        cor, p = pearsonr(x, y)

        # bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
        # bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

        # Compute P(x, y), P(x), P(y)
        # fxy = asarray(
        #     kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[
        #         2]) + EPS

        # Estimate fxy using scipy.stats.gaussian_kde
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        X, Y = np.mgrid[xmin:xmax:complex(0, n_grids), ymin:ymax:complex(0, n_grids)]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        # print(values)
        kernel = gaussian_kde(values)
        fxy = np.reshape(kernel(positions).T, X.shape) + EPS

        dx = (x.max() - x.min()) / (n_grids - 1)
        dy = (y.max() - y.min()) / (n_grids - 1)
        pxy = fxy / (fxy.sum() * dx * dy)
        px = pxy.sum(axis=1) * dy
        py = pxy.sum(axis=0) * dx

        # Compute mutual information;
        mi = (pxy * log(pxy / (asarray([px] * n_grids).T *
                               asarray([py] * n_grids)))).sum() * dx * dy

        # # Get H(x, y), H(x), and H(y)
        # hxy = - (pxy * log(pxy)).sum() * dx * dy
        # hx = -(px * log(px)).sum() * dx
        # hy = -(py * log(py)).sum() * dy
        # mi = hx + hy - hxy

        # Compute information coefficient
        ic = sign(cor) * sqrt(1 - exp(-2 * mi))

        # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
        if isnan(ic):
            ic = 0

        return ic


def absolute_information_coefficient_dist(x, y, n_grids=25, jitter=1E-10, random_seed=20170821):
    return 1 - absolute_information_coefficient(x, y, n_grids, jitter, random_seed)


def absolute_information_coefficient(x, y, n_grids=25,
                            jitter=1E-10, random_seed=20170821):
    """
    Compute the information coefficient between x and y, which are
        continuous, categorical, or binary vectors. This function uses only python libraries -- No R is needed.
    :param x: numpy array;
    :param y: numpy array;
    :param n_grids: int; number of grids for computing bandwidths
    :param jitter: number;
    :param random_seed: int or array-like;
    :return: float; Information coefficient
    """

    # Can't work with missing any value
    # not_nan_filter = ~isnan(x)
    # not_nan_filter &= ~isnan(y)
    # x = x[not_nan_filter]
    # y = y[not_nan_filter]

    x, y = drop_nan_columns([x, y])

    try:
        # Need at least 3 values to compute bandwidth
        if len(x) < 3 or len(y) < 3:
            return 0
    except TypeError:
        # If x and y are numbers, we cannot continue and IC is zero.
        return 0

    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)

    # Add jitter
    seed(random_seed)
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    # Compute bandwidths
    # cor, p = pearsonr(x, y)

    # bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    # bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    # Compute P(x, y), P(x), P(y)
    # fxy = asarray(
    #     kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[
    #         2]) + EPS

    # Estimate fxy using scipy.stats.gaussian_kde
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    X, Y = np.mgrid[xmin:xmax:complex(0, n_grids), ymin:ymax:complex(0, n_grids)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    fxy = np.reshape(kernel(positions).T, X.shape) + EPS

    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Compute mutual information;
    mi = (pxy * log(pxy / (asarray([px] * n_grids).T *
                           asarray([py] * n_grids)))).sum() * dx * dy

    # # Get H(x, y), H(x), and H(y)
    # hxy = - (pxy * log(pxy)).sum() * dx * dy
    # hx = -(px * log(px)).sum() * dx
    # hy = -(py * log(py)).sum() * dy
    # mi = hx + hy - hxy

    # Compute information coefficient
    ic = sqrt(1 - exp(-2 * mi))

    # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
    if isnan(ic):
        ic = 0

    return ic


def drop_nan_columns(arrays):
    """
    Keep only not-NaN column positions in all arrays.
    :param arrays: iterable of numpy arrays; must have the same length
    :return: list of numpy arrays; none of the arrays contains NaN
    """

    try:
        not_nan_filter = np.ones(len(arrays[0]), dtype=bool)
        # Keep column indices without missing value in all arrays
        for a in arrays:
            not_nan_filter &= ~np.isnan(a)

        return [a[not_nan_filter] for a in arrays]

    except TypeError:  # this means this is a number comparison, not a vector.
        # Keep "all" one column indices
        return arrays


# The following lines have been added to reproduce CCAL but without using R


def differential_gene_expression(
        gene_expression: "GCT filename; data matrix with input gene expression profiles",
        phenotype_file: "CLS filename; input binary phenotype/class distinction",
        output_filename: "Output files will have this name plus extensions .txt and .pdf",
        ranking_method: "The function to use to compute similarity between phenotypes and gene_expression",
        max_number_of_genes_to_show: "Maximum number of genes to show in the heatmap"=20,
        number_of_permutations: "Number of random permutations to estimate statistical significance "
                                "(p-values and FDRs)"=10,
        title: "The title of the heatmap"=None,
        random_seed: "Random number generator seed (can be set to a user supplied integer "
                     "for reproducibility)"=RANDOM_SEED):
    """
    Perform differential analysis on gene expression data of two phenotypes.
    :param phenotype_file: Series; input binary phenotype/class distinction
    :param gene_expression: DataFrame; data matrix with input gene expression profiles
    :param output_filename: str; output files will have this name plus extensions .txt and .pdf
    :param ranking_method:callable; the function to use to compute similarity between phenotypes and gene_expression.
    :param phenotypes_row_label: str; Name of phenotype row when input_phenotype is an array
    :param max_number_of_genes_to_show: int; maximum number of genes to show in the heatmap
    :param number_of_permutations: int; number of random permutations to estimate statistical significance (p-values and FDRs)
    :param title: str;
    :param random_seed: int | array; random number generator seed (can be set to a user supplied integer for reproducibility)
    :return: DataFrame; table of genes ranked by Information Coeff vs. phenotype
    """
    try:
        data_df = pd.read_table(gene_expression, header=2, index_col=0)
    except ValueError:
        data_df = gene_expression

    try:
        data_df.drop('Description', axis=1, inplace=True)
    except KeyError:
        pass

    try:
        temp = open(phenotype_file)
    except FileNotFoundError:
        if validators.url(phenotype_file):
            urlfile, __ = urllib.request.urlretrieve(phenotype_file)
        else:
            urlfile = phenotype_file

        temp = open(urlfile)
    temp.readline()
    temp.readline()
    classes = [int(i) for i in temp.readline().strip('\n').split(' ')]
    classes = pd.Series(classes, index=data_df.columns)

    gene_scores = make_match_panel(
        features=data_df,
        target=classes,
        function=ranking_method,
        target_ascending=False,
        n_top_features=0.99,
        max_n_features=max_number_of_genes_to_show,
        n_samplings=30,
        n_permutations=number_of_permutations,
        random_seed=random_seed,
        target_type='binary',
        title=title,
        file_path_prefix=output_filename)

    return gene_scores


def match_to_profile(
        gene_expression: "GCT filename; data matrix with input gene expression profiles",
        phenotype_input_method: "Select from the dropdown [CLS, Name, or Index] the type of input you have provided",
        phenotype_file: "Type the file name of the CLS file where the phenotypes are listed"=None,
        phenotype_column: "The column name in the GCT file where the gene name is present"='Index',
        name_of_phenotype_to_match: "The row/gene names the phenotype to match"=None,
        output_filename: "Output files will have this name plus extensions .txt and .pdf"= None,
        ranking_method: "The function to use to compute similarity between phenotypes and gene_expression"
                        ="compute_information_coefficient",
        max_number_of_genes_to_show: "Maximum number of genes to show in the heatmap"=20,
        number_of_permutations: "Number of random permutations to estimate statistical significance "
                                "(p-values and FDRs)"=10,
        title: "The title of the heatmap"='Differential Expression Results',
        random_seed: "Random number generator seed (can be set to a user supplied integer "
                     "for reproducibility)"=RANDOM_SEED):
    """
    Sort genes according to their association with a continuous phenotype or class vector.
    :param phenotype_file: Series; input binary phenotype/class distinction
    :param phenotype_column:
    :param name_of_phenotype_to_match:
    :param phenotype_input_method:
    :param gene_expression: DataFrame; data matrix with input gene expression profiles
    :param output_filename: str; output files will have this name plus extensions .txt and .pdf
    :param ranking_method:callable; the function to use to compute similarity between phenotypes and gene_expression.
    :param phenotypes_row_label: str; Name of phenotype row when input_phenotype is an array
    :param max_number_of_genes_to_show: int; maximum number of genes to show in the heatmap
    :param number_of_permutations: int; number of random permutations to estimate statistical significance (p-values and FDRs)
    :param title: str;
    :param random_seed: int | array; random number generator seed (can be set to a user supplied integer for reproducibility)
    :return: DataFrame; table of genes ranked by Information Coeff vs. phenotype
    """

    # TODO: add "if phenotypes_row_label is not None.
    # In this case it would check that Phenotypes is None. Only one can be not None.
    # Use phenotypes if both are provided.
    # Check that phenotypes_row_label actually exist in gene_expression DataFrame

    data_df = pd.read_table(gene_expression, header=2, index_col=0)
    try:
        data_df.drop('Description', axis=1, inplace=True)
    except KeyError:
        pass

    if phenotype_input_method == 'CLS':

        if validators.url(phenotypes_file):
            urlfile, __ = urllib.request.urlretrieve(phenotypes_file)
        else:
            urlfile = phenotypes_file

        temp = open(urlfile)
        temp.readline()
        temp.readline()
        classes = [float(i) for i in temp.readline().strip('\n').split(' ')]
        classes = pd.Series(classes, index=data_df.columns)
    elif phenotype_input_method == 'Name':
        classes = data_df[phenotype_column][name_of_phenotype_to_match]
    elif phenotype_input_method == 'Index':
        classes = data_df.loc[name_of_phenotype_to_match]

    # Turn a string into a callable function, if necessary
    if isinstance(ranking_method, str):
        ranking_method = eval(ranking_method)

    gene_scores = make_match_panel(
        target=classes,
        features=data_df,
        # max_n_unique_objects_for_drop_slices=1,
        function=ranking_method,
        target_ascending=False,
        n_top_features=0.99,
        max_n_features=max_number_of_genes_to_show,
        n_samplings=30,
        n_permutations=number_of_permutations,
        random_seed=random_seed,
        target_type='continuous',
        title=title,
        file_path_prefix=output_filename)

    return gene_scores


def make_match_panel(target,
                     features,
                     target_ascending=False,
                     scores=None,
                     function=compute_information_coefficient,
                     n_jobs=1,
                     scores_ascending=False,
                     n_top_features=25,
                     max_n_features=100,
                     n_samplings=3,
                     n_permutations=3,
                     random_seed=RANDOM_SEED,
                     indices=None,
                     figure_size=None,
                     title='Match Panel',
                     target_type='continuous',
                     features_type='continuous',
                     max_std=3,
                     target_annotation_kwargs={'fontsize': 12},
                     plot_column_names=False,
                     max_ytick_size=26,
                     file_path_prefix=None,
                     dpi=100):
    """
    Make match panel.
    Arguments:
        target (Series): (n_samples); must be 3 <= 0.632 * n_samples to compute
            MoE
        features (DataFrame): (n_features, n_samples)
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        function (callable): function for computing match scores between the
            target and each feature
        scores (DataFrame): (n_features, 4 ['Score', '<confidence> MoE',
            'p-value', 'FDR'])
        n_jobs (int): number of multiprocess jobs
        scores_ascending (bool): True (scores increase from top to bottom) |
            False
        n_top_features (number): number of features to compute MoE, p-value, and
            FDR; number threshold if 1 <= n_top_features, percentile threshold
            if n_top_features < 1, and don't compute if None
        max_n_features (int):
        n_samplings (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_samplings
        n_permutations (int): number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
        indices (iterable):
        figure_size (iterable):
        title (str): plot title
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        max_std (number):
        target_annotation_kwargs (dict):
        plot_column_names (bool): whether to plot column names
        max_ytick_size (int):
        file_path_prefix (str): file_path_prefix.match.txt and
            file_path_prefix.match.pdf will be saved
        dpi (int):
    Returns:
        DataFrame: (n_features, 4 ['Score', '<confidence> MoE', 'p-value',
            'FDR'])
    """

    # Sort target and features.columns (based on target)
    target = target.loc[target.index & features.columns].sort_values(
        ascending=target_ascending or target.dtype == 'O')

    # Drop constant rows
    features = drop_df_slices(
        features[target.index], 1, max_n_unique_objects=1)

    target_o_to_int = {}
    target_int_to_o = {}
    if target.dtype == 'O':
        # Make target numerical
        for i, o in enumerate(target.unique()):
            target_o_to_int[o] = i
            target_int_to_o[i] = o
        target = target.map(target_o_to_int)

    if target_type in ('binary', 'categorical'):

        # Cluster by group
        columns = cluster_2d_array_slices_by_group(
            nan_to_num(features.values), nan_to_num(target.values))
        features = features.iloc[:, columns]

    if scores is None:
        # Match
        scores = match(
            target.values,
            features.values,
            function,
            n_jobs=n_jobs,
            n_top_features=n_top_features,
            max_n_features=max_n_features,
            n_samplings=n_samplings,
            n_permutations=n_permutations,
            random_seed=random_seed)
        scores.index = features.index

        # Sort scores
        scores.sort_values('Score', ascending=scores_ascending, inplace=True)

        if file_path_prefix:
            # Save scores
            file_path_txt = file_path_prefix + '.match.txt'
            establish_path(file_path_txt)
            scores.to_csv(file_path_txt, sep='\t')

    # Select indices to plot
    if indices is None:
        indices = get_top_and_bottom_series_indices(scores['Score'],
                                                    n_top_features)
        if max_n_features < indices.size:
            indices = indices[:max_n_features // 2].append(
                indices[-max_n_features // 2:])
    else:
        indices = sorted(
            indices,
            key=lambda i: scores.loc[i, 'Score'],
            reverse=not scores_ascending)
    scores_to_plot = scores.loc[indices]
    features_to_plot = features.loc[scores_to_plot.index]

    # Make annotations
    annotations = DataFrame(index=scores_to_plot.index)
    # Make IC(MoE)s
    annotations['IC(\u0394)'] = scores_to_plot[['Score', '0.95 MoE']].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
    # Make p-value
    annotations['p-value'] = scores_to_plot['p-value'].apply('{:.2e}'.format)
    # Make FDRs
    annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    # Plot match panel
    if file_path_prefix:
        file_path_plot = file_path_prefix + '.match.pdf'
    else:
        file_path_plot = None
    plot_match_panel(target, target_int_to_o, features_to_plot, max_std,
                     annotations, figure_size, None, None, target_type,
                     features_type, title, target_annotation_kwargs,
                     plot_column_names, max_ytick_size, file_path_plot, dpi)

    return scores

def drop_df_slices(df,
                   axis,
                   only_obj=None,
                   max_n_unique_objects=None,
                   print_dropped_slices=False):
    """
    Drop df slices.
    Arguments:
        df (DataFrame):
        axis (int): 0 | 1
        only_obj (object):
        max_n_unique_objects (int): 0 < max_n_unique_objects
    Returns:
        DataFrame:
    """

    if only_obj is None and max_n_unique_objects is None:
        raise ValueError('Provide either only_obj or max_n_unique_objects.')

    # Select slices to be dropped
    dropped = array([False] * df.shape[[1, 0][axis]])

    if only_obj is not None:
        dropped |= (df == only_obj).all(axis=axis)

    if 0 < max_n_unique_objects:
        dropped |= df.apply(
            lambda s: s.unique().size <= max_n_unique_objects, axis=axis)

    # Drop
    print('Dropping {} axis-{} slices ...'.format(dropped.sum(), axis))

    if print_dropped_slices:
        print('* ======= Dropped slices ======= *')
        print(df.index[dropped].tolist())
        print('* ============================== *')

    if axis == 0:
        return df.loc[:, ~dropped]

    elif axis == 1:
        return df.loc[~dropped, :]


def cluster_2d_array_slices_by_group(array_2d, groups):
    """
    Cluster array_2d slices by group.
    Arguments:
        array_2d (array): (n_rows, n_columns)
        groups (array): (n_rows | n_columns); sorted group labels
    Returns:
        array: (n_rows | n_columns)
    """

    indices = []

    for i in get_1d_array_unique_objects_in_order(groups):

        group_indices = where(groups == i)[0]

        if groups.size == array_2d.shape[0]:

            clustered_indices = cluster_2d_array_rows(
                array_2d[group_indices, :])

        elif groups.size == array_2d.shape[1]:

            clustered_indices = cluster_2d_array_rows(
                array_2d[:, group_indices].T)

        else:
            raise ValueError(
                'groups length does not match the length of array_2d rows or columns.'
            )

        indices.append(group_indices[clustered_indices])

    return concatenate(indices)


def match(target,
          features,
          function,
          n_jobs=1,
          n_top_features=0.99,
          max_n_features=100,
          n_samplings=30,
          confidence=0.95,
          n_permutations=30,
          random_seed=RANDOM_SEED):
    """
    Compute: scores[i] = function(target, features[i]); compute margin of error
        (MoE), p-value, and FDR for n_top_features features.
    Arguments:
        target (array): (n_samples); must be 3 <= 0.632 * n_samples to compute
            MoE
        features (array): (n_features, n_samples)
        function (callable):
        n_jobs (int): number of multiprocess jobs
        n_top_features (number): number of features to compute MoE, p-value, and
            FDR; number threshold if 1 <= n_top_features, percentile threshold
            if n_top_features < 1, and don't compute if None
        max_n_features (int):
        n_samplings (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_samplings
        confidence (float):
        n_permutations (int): number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
    Returns:
        DataFrame: (n_features, 4 ['Score', '<confidence> MoE', 'p-value',
            'FDR'])
    """

    results = DataFrame(
        columns=['Score', '{} MoE'.format(confidence), 'p-value', 'FDR'])

    # Match
    print('Computing match score with {} ({} process) ...'.format(
        function, n_jobs))

    results['Score'] = concatenate(
        multiprocess(match_target_and_features,
                     [(target, fs, function)
                      for fs in array_split(features, n_jobs)], n_jobs))

    # Get top and bottom indices
    indices = get_top_and_bottom_series_indices(results['Score'],
                                                n_top_features)
    if max_n_features < indices.size:
        indices = indices[:max_n_features // 2].append(
            indices[-max_n_features // 2:])

    # Compute MoE
    if 3 <= n_samplings and 3 <= ceil(0.632 * target.size):

        results.loc[indices, '{} MoE'.format(
            confidence
        )] = match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
            target,
            features[indices],
            function,
            n_samplings=n_samplings,
            confidence=confidence,
            random_seed=random_seed)

    # Compute p-value and FDR
    if 1 <= n_permutations:

        permutation_scores = concatenate(
            multiprocess(permute_target_and_match_target_and_features,
                         [(target, fs, function, n_permutations, random_seed)
                          for fs in array_split(features, n_jobs)], n_jobs))

        p_values, fdrs = compute_empirical_p_values_and_fdrs(
            results['Score'], permutation_scores.flatten())

        results['p-value'] = p_values
        results['FDR'] = fdrs

    return results


def match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
        target,
        features,
        function,
        n_samplings=30,
        confidence=0.95,
        random_seed=RANDOM_SEED):
    """
    Match randomly sampled target and features to compute margin of errors.
    Arguments
        target (array): (n_samples); must be 3 <= 0.632 * n_samples to compute
            MoE
        features (array): (n_features, n_samples)
        function (callable):
        n_samplings (int): 3 <= n_samplings
        cofidence (float):
        random_seed (int | array):
    Returns:
        array: (n)
    """

    if n_samplings < 3:
        raise ValueError('Cannot compute MoEs because n_samplings < 3.')

    if ceil(0.632 * target.size) < 3:
        raise ValueError('Cannot compute MoEs because 0.632 * n_samples < 3.')

    print('Computing MoEs with {} samplings ...'.format(n_samplings))

    feature_x_sampling = empty((features.shape[0], n_samplings))

    seed(random_seed)
    for i in range(n_samplings):

        # Sample randomly
        random_indices = choice(target.size, ceil(0.632 * target.size))
        sampled_target = target[random_indices]
        sampled_features = features[:, random_indices]

        random_state = get_state()

        # Score
        feature_x_sampling[:, i] = match_target_and_features(
            sampled_target, sampled_features, function)

        set_state(random_state)

    # Compute MoE using bootstrapped score distributions
    return apply_along_axis(compute_margin_of_error, 1, feature_x_sampling)


def permute_target_and_match_target_and_features(target,
                                                 features,
                                                 function,
                                                 n_permutations=30,
                                                 random_seed=RANDOM_SEED):
    """
    Permute target and match target and features.
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
        n_permutations (int): 1 <= n_permutations
        random_seed (int | array):
    Returns:
        array: (n_features, n_permutations)
    """

    if n_permutations < 1:
        raise ValueError(
            'Not computing p-value and FDR because n_permutations < 1.')

    print('Computing p-values and FDRs with {} permutations ...'.format(
        n_permutations))

    feature_x_permutation = empty((features.shape[0], n_permutations))

    # Copy for inplace shuffling
    permuted_target = array(target)

    seed(random_seed)
    for i in range(n_permutations):
        if i % ceil(n_permutations/10) == 0:
            print('\t{}/{} ...'.format(i + 1, n_permutations))

        # Permute
        shuffle(permuted_target)

        random_state = get_state()

        # Match
        feature_x_permutation[:, i] = match_target_and_features(
            permuted_target, features, function)

        set_state(random_state)
    print('\t{}/{} - done.'.format(i + 1, n_permutations))

    return feature_x_permutation


def match_target_and_features(target, features, function):
    """
    Drop nan from target and features[i] and compute: scores[i] = function(
        target, features[i]).
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
    Returns:
        array: (n_features)
    """

    return apply_along_axis(drop_nan_and_apply_function_on_2_1d_arrays, 1,
                            features, target, function)


def drop_nan_and_apply_function_on_2_1d_arrays(array_1d_0, array_1d_1,
                                               function):
    """
    Drop nan and apply function(array_1d_0_without_nan, array_1d_1_without_nan).
    Arguments:
        array_1d_0 (array): (n)
        array_1d_1 (array): (n)
        function (callable):
    Returns:
        object:
    """

    nans = isnan(array_1d_0) | isnan(array_1d_1)

    array_1d_0_without_nan = array_1d_0[~nans]
    array_1d_1_without_nan = array_1d_1[~nans]

    return function(array_1d_0_without_nan, array_1d_1_without_nan)


def establish_path(path, path_type='file'):
    """
    Make directory paths up to the deepest directory in path.
    Arguments:
        path (str):
        path_type (str): 'file' | 'directory'
    Returns:
        None
    """

    # Work with absolute path
    path = abspath(path)

    # Check path ending
    if path_type == 'file':
        if path.endswith('/'):
            raise ValueError('File path should not end with \'/\'.')
    else:
        if not path.endswith('/'):
            path += '/'

    directory_path, file_name = split(path)

    # List missing directory paths
    missing_directory_paths = []

    while not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        # Check directory_path's directory_path
        directory_path, file_name = split(directory_path)

    # Make missing directories
    for directory_path in reversed(missing_directory_paths):
        mkdir(directory_path)
        print('Created directory {}.'.format(directory_path))


def get_top_and_bottom_series_indices(series, threshold):
    """
    Get top and bottom series indices.
    Arguments:
        series (Series):
        threshold (number): quantile if 0.5 < threshold < 1; ranking number if
            1 <= threshold
    Returns:
        Index: indices for the top and bottom objects
    """

    if 0.5 <= threshold < 1:

        top_and_bottom = (series <= series.quantile(1 - threshold)) | (
            series.quantile(threshold) <= series)

    elif 1 <= threshold:

        threshold = min(threshold, ceil(series.size / 2))

        rank = series.rank(method='dense')

        top_and_bottom = (rank <= threshold) | (
            (rank.max() - threshold) < rank)

    else:
        raise ValueError('threshold must be 0.5 <=.')

    return series.index[top_and_bottom]


def plot_match_panel(target, target_int_to_o, features, max_std, annotations,
                     figure_size, target_ax, features_ax, target_type,
                     features_type, title, target_annotation_kwargs,
                     plot_column_names, max_ytick_size, file_path, dpi):
    """
    Plot matches.
    Arguments:
        target (Series): (n_samples)
        target_int_to_o (dict):
        features (DataFrame): (n_features, n_samples)
        max_std (number):
        annotations (DataFrame): (n_features, 3)
        figure_size (iterable):
        target_ax (matplotlib ax):
        features_ax (matplotlib ax):
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): plot title
        target_annotation_kwargs (dict):
        plot_column_names (bool): whether to plot column names
        max_ytick_size (int):
        file_path (str):
        dpi (int):
    Returns:
        None
    """

    # Set target min, max, and colormap
    if target_type == 'continuous':
        # Normalize target for plotting
        target = Series(
            normalize_1d_array(target.values, method='-0-'),
            name=target.name,
            index=target.index)
        target_min, target_max, target_cmap = -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif target_type == 'categorical':
        n = unique(target).size
        if CMAP_CATEGORICAL_TAB20.N < n:
            # Make and use a Colormap with random colors
            cmap = make_random_categorical_colormap(n_colors=n)
        else:
            cmap = CMAP_CATEGORICAL_TAB20
        target_min, target_max, target_cmap = 0, n, cmap

    elif target_type == 'binary':
        target_min, target_max, target_cmap = 0, 1, CMAP_BINARY_WB

    else:
        raise ValueError('Unknown target_type: {}.'.format(target_type))

    # Set features min, max, and colormap
    if features_type == 'continuous':
        # Normalize featuers for plotting
        features = DataFrame(
            normalize_2d_array(features.values, method='-0-', axis=1),
            index=features.index,
            columns=features.columns)
        features_min, features_max, features_cmap = -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif features_type == 'categorical':
        n = unique(features).size
        if CMAP_CATEGORICAL_TAB20.N < n:
            # Make and use a Colormap with random colors
            cmap = make_random_categorical_colormap(n_colors=n)
        else:
            cmap = CMAP_CATEGORICAL_TAB20
        features_min, features_max, features_cmap = 0, n, cmap

    elif features_type == 'binary':
        features_min, features_max, features_cmap = 0, 1, CMAP_BINARY_WB

    else:
        raise ValueError('Unknown features_type: {}.'.format(features_type))

    # Set up figure
    if not figure_size:
        figure_size = (min(pow(features.shape[1], 1.8), FIGURE_SIZE[1]),
                       features.shape[0])

    # Set up grids and axes if target_ax or features_ax is not specified
    if target_ax is None or features_ax is None:
        figure(figsize=figure_size)
        gridspec = GridSpec(features.shape[0] + 1, 1)
        target_ax = subplot(gridspec[:1, 0])
        features_ax = subplot(gridspec[1:, 0])

    # Plot target heatmap
    heatmap(
        DataFrame(target).T,
        ax=target_ax,
        vmin=target_min,
        vmax=target_max,
        cmap=target_cmap,
        xticklabels=False,
        yticklabels=[target.name],
        cbar=False)

    # Decorate target heatmap
    decorate(
        ax=target_ax,
        despine_kwargs={'left': True,
                        'bottom': True},
        xlabel='',
        ylabel='',
        max_ytick_size=max_ytick_size)

    # Plot title
    if title:

        target_ax.text(
            target_ax.get_xlim()[1] / 2,
            -1,
            title,
            horizontalalignment='center',
            **FONT_LARGEST)

    # Plot target label
    if target_type in ('binary', 'categorical'):

        # Get boundary index
        boundary_indices = [0]
        prev_v = target[0]
        for i, v in enumerate(target[1:]):
            if prev_v != v:
                boundary_indices.append(i + 1)
            prev_v = v
        boundary_indices.append(features.shape[1])

        # Get label position
        label_positions = []
        prev_i = 0
        for i in boundary_indices[1:]:
            label_positions.append(i - (i - prev_i) / 2)
            prev_i = i

        # Plot target label
        unique_target_labels = get_unique_objects_in_order(target.values)
        for i, x in enumerate(label_positions):

            if target_int_to_o:
                t = target_int_to_o[unique_target_labels[i]]
            else:
                t = unique_target_labels[i]

            target_ax.text(
                x,
                -0.18,
                t,
                horizontalalignment='center',
                verticalalignment='bottom',
                rotation=90,
                **merge_dicts_with_function(
                    FONT_STANDARD, target_annotation_kwargs, lambda a, b: b))

    # Plot annotation header
    target_ax.text(
        target_ax.get_xlim()[1] * 1.018,
        0.5,
        ' ' * 5 + 'IC(\u0394)' + ' ' * 13 + 'p-value' + ' ' * 12 + 'FDR',
        verticalalignment='center',
        **FONT_STANDARD)

    # Plot annotation header separator line
    target_ax.plot(
        [target_ax.get_xlim()[1] * 1.02,
         target_ax.get_xlim()[1] * 1.4], [1, 1],
        '-',
        linewidth=1,
        color='#20D9BA',
        clip_on=False,
        aa=True)

    # Plot features heatmap
    heatmap(
        features,
        ax=features_ax,
        vmin=features_min,
        vmax=features_max,
        cmap=features_cmap,
        xticklabels=plot_column_names,
        cbar=False)

    # Decorate features heatmap
    decorate(
        ax=features_ax,
        despine_kwargs={
            'left': True,
            'bottom': True,
        },
        xlabel='',
        ylabel='',
        max_ytick_size=max_ytick_size)

    # Plot annotations
    for i, (a_i, a) in enumerate(annotations.iterrows()):
        features_ax.text(
            target_ax.axis()[1] * 1.018,
            i + 0.5,
            '\t'.join(a.tolist()).expandtabs(),
            verticalalignment='center',
            **FONT_STANDARD)

    # Save
    if file_path:
        save_plot(file_path, dpi=dpi)
    show(target_ax)


def save_plot(file_path, overwrite=True, dpi=100):
    """
    Establish file path and save plot.
    Arguments:
        file_path (str):
        overwrite (bool):
        dpi (int):
    Returns:
        None
    """

    # If the figure doesn't exist or overwriting
    if not isfile(file_path) or overwrite:

        establish_path(file_path)

        savefig(file_path, dpi=dpi, bbox_inches='tight')


def decorate(ax=None,
             style='ticks',
             despine_kwargs={},
             title=None,
             title_kwargs={},
             yaxis_position='left',
             xlabel=None,
             ylabel=None,
             xlabel_kwargs={},
             ylabel_kwargs={},
             xticks=None,
             yticks=None,
             max_n_xticks=None,
             max_n_yticks=None,
             max_xtick_size=None,
             max_ytick_size=None,
             xticklabels_kwargs={},
             yticklabels_kwargs={},
             legend_loc='best'):
    """
    Decorate an ax.
    """

    if ax:
        sca(ax)
    else:
        ax = gca()

    if legend_loc:
        ax.legend(
            loc=legend_loc,
            prop={
                'size': FONT_STANDARD['fontsize'],
                'weight': FONT_STANDARD['weight'],
            })

    # Set plot style
    set_style(style)
    despine(**despine_kwargs)

    # Title
    if title:
        title_kwargs = merge_dicts_with_function(FONT_LARGEST, title_kwargs,
                                                 lambda a, b: b)
        suptitle(title, **title_kwargs)

    # Set y axis position
    if yaxis_position == 'right':
        ax.yaxis.tick_right()

    # Style x label
    if xlabel is None:
        xlabel = ax.get_xlabel()
    xlabel_kwargs = merge_dicts_with_function(FONT_LARGER, xlabel_kwargs,
                                              lambda a, b: b)
    ax.set_xlabel(xlabel, **xlabel_kwargs)

    # Style y label
    if ylabel is None:
        ylabel = ax.get_ylabel()
    ylabel_kwargs = merge_dicts_with_function(FONT_LARGER, ylabel_kwargs,
                                              lambda a, b: b)
    ax.set_ylabel(ylabel, **ylabel_kwargs)

    # Style x ticks
    if xticks is not None:
        ax.set_xticks(xticks)

    if max_n_xticks and max_n_xticks < len(ax.get_xticks()):
        ax.set_xticks([])

    # Style x tick labels
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    if len(xticklabels):

        if xticklabels[0]:
            # Limit tick label size
            if max_xtick_size:
                xticklabels = [t[:max_xtick_size] for t in xticklabels]
        else:
            xticklabels = ax.get_xticks()

        # Set tick label rotation
        if 'rotation' not in xticklabels_kwargs:
            xticklabels_kwargs['rotation'] = 90

        xticklabels_kwargs = merge_dicts_with_function(
            FONT_SMALLER, xticklabels_kwargs, lambda a, b: b)

        ax.set_xticklabels(xticklabels, **xticklabels_kwargs)

    # Style y ticks
    if yticks is not None:
        ax.set_yticks(yticks)

    if max_n_yticks and max_n_yticks < len(ax.get_yticks()):
        ax.set_yticks([])

    # Style y tick labels
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    if len(yticklabels):

        if yticklabels[0]:
            # Limit tick label size
            if max_ytick_size:
                yticklabels = [t[:max_ytick_size] for t in yticklabels]
        else:
            yticklabels = ax.get_yticks()

        # Set tick label rotation
        if 'rotation' not in yticklabels_kwargs:
            yticklabels_kwargs['rotation'] = 0

        yticklabels_kwargs = merge_dicts_with_function(
            FONT_SMALLER, yticklabels_kwargs, lambda a, b: b)

        ax.set_yticklabels(yticklabels, **yticklabels_kwargs)


def get_1d_array_unique_objects_in_order(array_1d):
    """
    Get unique objects in the order of their appearance in array_1d.
    Arguments:
        array_1d (array): (n)
    Returns:
        array: (n_unique_objects); unique objects ordered by their appearances
            in array_1d
    """

    unique_objects_in_order = []

    for o in array_1d:

        if o not in unique_objects_in_order:
            unique_objects_in_order.append(o)

    return array(unique_objects_in_order)


def cluster_2d_array_rows(array_2d,
                          linkage_method='average',
                          distance_function='euclidean'):
    """
    Cluster array_2d rows.
    Arguments:
        array_2d (array): (n_rows, n_columns)
        linkage_method (str): linkage method compatible for
            scipy.cluster.hierarchy.linkage
        distance_function (str | callable): distance function compatible for
            scipy.cluster.hierarchy.linkage
    Returns:
        array: (n_rows); clustered row indices
    """

    clustered_indices = dendrogram(
        linkage(array_2d, method=linkage_method, metric=distance_function),
        no_plot=True)['leaves']

    return array(clustered_indices)


def multiprocess(function, args, n_jobs, random_seed=None):
    """
    Call function with args across n_jobs processes (n_jobs doesn't have to be
        the length of list_of_args).
    Arguments:
        function (callable):
        arg (iterable): args
        n_jobs (int): 0 < n_jobs
        random_seed (int | array):
    Returns:
        list:
    """

    if random_seed is not None:
        # Each process initializes with the current jobs' randomness (random
        # state & random state index). Any changes to these processes'
        # randomnesses won't update the current process' randomness.
        seed(random_seed)

    with Pool(n_jobs) as p:
        return p.starmap(function, args)


def compute_empirical_p_values_and_fdrs(values, random_values):
    """
    Compute empirical p-values and FDRs.
    Arguments:
        values (array): (n)
        random_values (array): (n_random_values)
    Returns:
        array: (n); p-values
        array: (n); FDRs
    """

    p_values_l = array(
        [compute_empirical_p_value(v, random_values, 'less') for v in values])
    p_values_g = array(
        [compute_empirical_p_value(v, random_values, 'great') for v in values])

    fdrs_l = multipletests(p_values_l, method='fdr_bh')[1]
    fdrs_g = multipletests(p_values_g, method='fdr_bh')[1]

    # Take smaller p-values
    p_values = where(p_values_l < p_values_g, p_values_l, p_values_g)

    # Take smaller FDRs
    fdrs = where(fdrs_l < fdrs_g, fdrs_l, fdrs_g)

    return p_values, fdrs


def compute_empirical_p_value(value, random_values, direction):
    """
    Compute empirical p-value.
    Arguments:
        value (float):
        random_values (array):
        direction (str): 'less' | 'great'
    Returns:
        float: p-value
    """

    if direction == 'less':
        significant_random_values = random_values <= value

    elif direction == 'great':
        significant_random_values = value <= random_values

    else:
        raise ValueError('Unknown direction: {}.'.format(direction))

    p_value = significant_random_values.sum() / random_values.size

    if p_value == 0:
        p_value = 1 / random_values.size

    return p_value


def compute_margin_of_error(array_1d, confidence=0.95):
    """
    Compute margin of error.
    Arguments:
        array_1d (array):
        confidence (float): 0 <= confidence <= 1
    Returns:
        float:
    """

    return norm.ppf(q=confidence) * array_1d.std() / sqrt(array_1d.size)


def normalize_1d_array(array_1d, method, rank_method='average'):
    """
    Normalize array_1d.
    Arguments:
        array_1d (array): (n)
        method (str): '-0-' | '0-1' | 'rank'
        rank_method (str): 'average' | 'min' | 'max' | 'dense' | 'ordinal'
    Returns:
        array: (n)
    """

    values = array_1d[~isnan(array_1d)]
    size = values.size
    mean = values.mean()
    std = values.std()
    min_ = values.min()
    max_ = values.max()

    if method == '-0-':

        if std:
            return (array_1d - mean) / std
        else:
            print('std == 0: / size instead of 0-1 ...')
            return array_1d / size

    elif method == '0-1':

        if max_ - min_:
            return (array_1d - min_) / (max_ - min_)
        else:
            print('(max - min) ==  0: / size instead of 0-1 ...')
            return array_1d / size

    elif method == 'rank':

        # Assign mean to nans
        array_1d[isnan(array_1d)] = mean
        return rankdata(array_1d, method=rank_method) / size

    else:
        raise ValueError('Unknown method: {}.'.format(method))


def make_random_categorical_colormap(n_colors=None, bad_color=None):
    """
    Make random categorical colormap.
    Arguments:
        n_colors (int):
        bad_color (matplotlib color):
    Returns:
        matplotlib.Colormap:
    """

    color_map = ListedColormap([make_random_color() for i in range(n_colors)])

    if bad_color:
        color_map.set_bad(bad_color)

    return color_map


def make_random_color():
    """
    Make a random color.
    Arguments:
        None
    Returns:
        str: hexcolor
    """

    return '#' + ''.join([choice('0123456789ABCDEF') for x in range(6)])


def normalize_2d_array(array_2d, method, axis=None, rank_method='average'):
    """
    Normalize array_2d.
    Arguments:
        array_2d (array): (n, m)
        method (str): '-0-' | '0-1' | 'rank'
        axis (int | str): 'global' | 0 | 1 |
        rank_method (str): 'average' | 'min' | 'max' | 'dense' | 'ordinal'
    Returns:
        array: (n, m)
    """

    if axis is None:

        values = array_2d[~isnan(array_2d)]
        size = values.size
        mean = values.mean()
        std = values.std()
        min_ = values.min()
        max_ = values.max()

        if method == '-0-':

            if std:
                return (array_2d - mean) / std
            else:
                print('std == 0: / size instead of 0-1 ...')
                return array_2d / size

        elif method == '0-1':

            if max_ - min_:
                return (array_2d - min_) / (max_ - min_)
            else:
                print('(max - min) ==  0: / size instead of 0-1 ...')
                return array_2d / size

        elif method == 'rank':

            array_2d[isnan(array_2d)] = mean

            return (rankdata(array_2d, method=rank_method) /
                    size).reshape(array_2d.shape)

        else:
            raise ValueError('Unknown method: {}.'.format(method))

    elif axis == 0 or axis == 1:

        return apply_along_axis(
            normalize_1d_array,
            axis,
            array_2d,
            method=method,
            rank_method=rank_method)

    else:
        raise ValueError('Unknown axis: {}.'.format(axis))


def get_unique_objects_in_order(iterable):
    """
    Get unique objects in the order of their appearance in iterable.
    Arguments:
        iterable (iterable): objects
    Returns:
        list: (n_unique_objects); unique objects ordered by their appearances
            in iterable
    """

    unique_objects_in_order = []

    for o in iterable:

        if o not in unique_objects_in_order:
            unique_objects_in_order.append(o)

    return unique_objects_in_order


def merge_dicts_with_function(dict_0, dict_1, function):
    """
    Merge dict_0 and dict_1, apply function to values keyed by the same key.
    Arguments:
        dict_0 (dict);
        dict_1 (dict):
        function (callable):
    Returns:
        dict: merged dict
    """

    merged_dict = {}

    for k in dict_0.keys() | dict_1.keys():

        if k in dict_0 and k in dict_1:
            merged_dict[k] = function(dict_0.get(k), dict_1.get(k))

        elif k in dict_0:
            merged_dict[k] = dict_0.get(k)

        elif k in dict_1:
            merged_dict[k] = dict_1.get(k)

        else:
            raise ValueError('dict_0 or dict_1 changed during iteration.')

    return merged_dict
