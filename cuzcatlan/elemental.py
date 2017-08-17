import scipy
import numpy as np


def hellopipilworld():
    return u'Niltze Cemanahuac! (Hello "World"! in Nahuatl.)'


def custom_pearson(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def uncentered_pearson(x, y):
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        # x_squared =
        # y_squared =
        return np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y))))
