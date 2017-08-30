"""Parts of this code are borrowed from the CCAL library: https://github.com/UCSD-CCAL"""


from numpy import asarray, exp, finfo, isnan, log, sign, sqrt, sum, sort
from numpy.random import random_sample, seed
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
import numpy as np

EPS = finfo(float).eps


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
