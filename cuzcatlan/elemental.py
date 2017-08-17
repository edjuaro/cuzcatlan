import scipy
import numpy as np


def hellopipilworld():
    return u'Niltze Cemanahuac! (Hello "World"! in Nahuatl.)'


def list2cls(in_list, name_of_out='output.cls'):
    """This function creates a CLS file from a list-like object"""
    # print("~~~~~"+str(metadata_subset.shape)+'~~~~~')
    cls = open(name_of_out, 'w')
    cls.write("{}\t{}\t1\n".format(len(in_list), len(np.unique(in_list))))
    cls.write("#\t{}\n".format("\t".join(np.unique(in_list).astype(str))))
    cls.write('\t'.join(in_list.astype(str))+'\n')
    cls.close()


def custom_pearson(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def custom_spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def custom_kendall_tau(x, y):
    return scipy.stats.kendalltau(x, y)[0]


def absolute_pearson(x, y):
    return np.abs(scipy.stats.pearsonr(x, y)[0])


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


def absolute_uncentered_pearson(x, y):
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        return np.abs(np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))))
