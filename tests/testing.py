import cuzcatlan as cusca
import pandas as pd
import numpy as np
from cuzcatlan import differential_gene_expression
from cuzcatlan import HierarchicalClustering
from cuzcatlan import compute_information_coefficient
import pickle

# TOP = 10
#
# RUN = False
#
# data_df = pd.read_table("test_data/all_aml_test.gct", header=2, index_col=0)
# data_df.drop('Description', axis=1, inplace=True)
# temp = open("test_data/all_aml_test.cls")
# temp.readline()
# temp.readline()
# classes = [int(i) for i in temp.readline().strip('\n').split(' ')]
# classes = pd.Series(classes, index=data_df.columns)
#
# if RUN:
#     scores = differential_gene_expression(phenotypes=classes, gene_expression=data_df, output_filename='DE_test',
#                                           ranking_method=cusca.custom_pearson_corr, number_of_permutations=10000)
#
#     pickle.dump(scores, open('match_results.p', 'wb'))
# else:
#     scores = pickle.load(open('match_results.p', 'rb'))
#
# # print(scores.iloc[np.r_[0:TOP, -TOP:0], :])
#
# scores['abs_score'] = abs(scores['Score'])
# scores['Feature'] = scores.index
# scores.sort_values('abs_score', ascending=False, inplace=True)
# scores.reset_index(inplace=True)
# scores['Rank'] = scores.index + 1
#
# print(scores.iloc[0:2*TOP, :])

print("Testing HC now...")
pwd = '.'
gct_name = "./test_data/test_BRCA_minimal_60x19.gct"
col_distance_metric = "pearson"
output_distances = False
row_distance_metric = 'No_row_clustering'
clustering_method = 'average'
output_base_name = 'OC_HC'
row_normalization = False
col_normalization = False
row_centering = 'Mean'
col_centering = 'Mean'
custom_plot = True

cusca.HierarchicalClustering(pwd,
                             gct_name,
                             col_distance_metric,
                             output_distances,
                             row_distance_metric,
                             clustering_method,
                             output_base_name,
                             row_normalization,
                             col_normalization,
                             row_centering,
                             col_centering,
                             custom_plot)
