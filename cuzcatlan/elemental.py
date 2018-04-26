import scipy
import numpy as np
import pandas as pd

def hellopipilworld():
    return u'Niltze Cemanahuac! (Hello "World"! in Nahuatl.)'


def list2intlist(in_list, custom_order=None):
    unique_labels = pd.unique(in_list)
    translate = dict(zip(unique_labels, range(len(unique_labels))))
    if custom_order is not None:
        # print(in_list)
        new_list = [in_list[j] for j in custom_order]
        # print(new_list)
        # print([translate[i] for i in new_list])
        return [translate[i] for i in new_list]
    else:
        return [translate[i] for i in in_list]


def list2cls(in_list, name_of_out='output.cls', sep='\t'):
    """This function creates a CLS file from a list-like object"""
    # print("~~~~~"+str(metadata_subset.shape)+'~~~~~')
    cls = open(name_of_out, 'w')
    cls.write("{}{}{}{}1\n".format(len(in_list), sep, len(np.unique(in_list)), sep))
    cls.write("#{}{}\n".format(sep, sep.join(np.unique(in_list).astype(str))))
    cls.write(sep.join(in_list.astype(str))+'\n')
    cls.close()


def df2gct(df, extra_columns=0, use_index=True, name='output.gct', add_dummy_descriptions=False):
    if add_dummy_descriptions:
        if add_dummy_descriptions not in list(df):
            df.insert(0, 'Description', df.index)
            extra_columns += 1
    f = open(name, 'w')
    f.write("#1.2\n")
    f.write("\t".join([str(df.shape[0]), str(df.shape[1] - extra_columns)]) + "\n")
    f.write(df.to_csv(sep='\t', index=use_index))
    f.close()
    return


def custom_pearson_corr(x, y):
    # [-1,1]
    return scipy.stats.pearsonr(x, y)[0]


def custom_pearson_dist(x, y):
    # [0,2]
    return 1 - scipy.stats.pearsonr(x, y)[0]


def custom_spearman_corr(x, y):
    # [-1,1]
    return scipy.stats.spearmanr(x, y)[0]


def custom_spearman_dist(x, y):
    # [0,2]
    return 1 - scipy.stats.spearmanr(x, y)[0]


def absolute_spearman(x, y):
    # [0,1]
    return abs(scipy.stats.spearmanr(x, y)[0])


def custom_kendall_tau_corr(x, y):
    # [-1,1]
    return scipy.stats.kendalltau(x, y)[0]


def custom_kendall_tau_dist(x, y):
    # [0,2]
    return 1 - scipy.stats.kendalltau(x, y)[0]


def absolute_pearson_corr(x, y):
    # [0,1]
    return np.abs(scipy.stats.pearsonr(x, y)[0])


def absolute_pearson_dist(x, y):
    # [-1,0]
    return 1 - np.abs(scipy.stats.pearsonr(x, y)[0])


def uncentered_pearson_corr(x, y):
    # [-1,1]
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        # x_squared =
        # y_squared =
        return np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y))))


def uncentered_pearson_dist(x, y):
    # [0,2]
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        # x_squared =
        # y_squared =
        return 1 - (np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))))


def absolute_uncentered_pearson_corr(x, y):
    # [0,1]
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        return np.abs(np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))))


def absolute_uncentered_pearson_dist(x, y):
    # [-1,0]
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        return 1- (np.abs(np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y))))))


def mydist(p1, p2):
    # [0,inf)
    # a custom function that just computes Euclidean distance
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5


def custom_euclidean_dist(x, y):
    # [0,inf)
    return scipy.spatial.distance.euclidean(x, y)


def custom_euclidean_sim(x, y):
    # (0,1]
    return 1/(1+scipy.spatial.distance.euclidean(x, y))


def dendodist(V, dist=mydist):
    dists = np.array([dist(a[0], a[1]) for a in V])
    return np.cumsum(dists)


def custom_manhattan_dist(x, y):
    # [0,inf)
    return scipy.spatial.distance.cityblock(x, y)


def custom_manhattan_sim(x, y):
    # (0,1]
    return 1/(1+scipy.spatial.distance.cityblock(x, y))


def custom_cosine_dist(x, y):
    # [0,2]
    return scipy.spatial.distance.cosine(x, y)


def custom_cosine_sim(x, y):
    # [-1,1]
    return 1-scipy.spatial.distance.cosine(x, y)


def makeODF(output_data, vals, file_name='noname.odf'):
    f = open(file_name, 'w')
    f.write("ODF 1.0\n")  # Hard-coding spance, not tab here.
    f.write("HeaderLines=18\n")  # hard-coding 19+2-3 lines here. Needs to change.
    # f.write("COLUMN_NAMES:Rank\tFeature\tDescription\tScore\tFeature P\tFeature P Low\tFeature P High\tFDR(BH)\tQ Value"
    #         "\tBonferroni\tmaxT\tFWER\tFold Change\tALL Mean\tALL Std\tAML Mean\tAML Std\tk\n")
    # f.write("COLUMN_TYPES:int\tString\tString\tdouble\tdouble\tdouble\tdouble\tdouble\tdouble\tdouble\tdouble\tdouble"
    #         "\tdouble\tdouble\tdouble\tdouble\tdouble\tint\n")

    f.write("COLUMN_NAMES:"+"\t".join(list(output_data))+"\n")
    # f.write("COLUMN_DESCRIPTIONS:"+"\t".join(list(output_data))+"\n")
    f.write("COLUMN_TYPES:"+"\t".join(['String', 'String', 'double', 'String'])+"\n")  # TODO: automate this.
    #
    # print(list(output_data))
    # f.write("COLUMN_DESCRIPTIONS:\n")
    # f.write("Model=DiffEx\n")
    f.write("Model=Comparative Marker Selection\n")
    f.write("Dataset File="+vals['gct']+"\n")
    f.write("Class File="+vals['cls']+"\n")
    f.write("Permutations="+str(vals['n_perm'])+"\n")
    # f.write("Balanced=false\n")  # TODO: remove this one?
    # f.write("Complete=false\n")  # TODO: remove this one?
    # f.write("Test Direction=2 Sided\n")  # TODO: remove this one?
    f.write("Class 0="+vals['class_0']+"\n")
    f.write("Class 1="+vals['class_1']+"\n")
    f.write("Test Statistic="+vals['func']+"\n")
    f.write("pi0=\n")
    f.write("lambda=\n")
    f.write("pi0(lambda)=\n")
    f.write("cubic spline(lambda)=\n")
    f.write("Random Seed="+str(vals['rand_seed'])+"\n")
    f.write("Smooth p-values=\n")
    f.write("DataLines="+str(vals['dat_lines'])+"\n")
    f.write("RowNamesColumn=0\n")
    f.write("RowDescriptionsColumn=1\n")
    f.write(output_data.to_csv(sep='\t', index=False, header=False))
    return
