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


def custom_pearson_corr(x, y):
    # [-1,1]
    return scipy.stats.pearsonr(x, y)[0]


def custom_pearson_dist(x, y):
    # [-1,1]
    return 1 - scipy.stats.pearsonr(x, y)[0]


def custom_spearman_corr(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def custom_spearman_dist(x, y):
    return 1 - scipy.stats.spearmanr(x, y)[0]


def absolute_spearman(x, y):
    return abs(scipy.stats.spearmanr(x, y)[0])


def custom_kendall_tau_corr(x, y):
    return scipy.stats.kendalltau(x, y)[0]


def custom_kendall_tau_dist(x, y):
    return 1 - scipy.stats.kendalltau(x, y)[0]


def absolute_pearson_corr(x, y):
    return np.abs(scipy.stats.pearsonr(x, y)[0])


def absolute_pearson_dist(x, y):
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
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        return np.abs(np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))))


def absolute_uncentered_pearson_dist(x, y):
    if len(x) != len(y):
        # Uncentered Pearson Correlation cannot be computed for vectors of different length.
        print('Uncentered Pearson Correlation cannot be computed for vectors of different length.')
        return np.nan

    else:
        # Using the definition from eq (4) in https://www.biomedcentral.com/content/supplementary/1477-5956-9-30-S4.PDF
        return 1- (np.abs(np.sum(np.multiply(x, y)) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y))))))


def mydist(p1, p2):
    # a custom function that just computes Euclidean distance
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5


def custom_euclidean_dist(x, y):
    return scipy.spatial.distance.euclidean(x, y)


def custom_euclidean_sim(x, y):
    return 1/(1+scipy.spatial.distance.euclidean(x, y))


def dendodist(V, dist=mydist):
    dists = np.array([dist(a[0], a[1]) for a in V])
    return np.cumsum(dists)


def custom_manhattan_dist(x, y):
    return scipy.spatial.distance.cityblock(x, y)


def custom_manhattan_sim(x, y):
    return 1/(1+scipy.spatial.distance.cityblock(x, y))


def custom_cosine_dist(x, y):
    return scipy.spatial.distance.cosine(x, y)


def custom_cosine_sim(x, y):
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
