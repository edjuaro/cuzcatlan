import cuzcatlan as cusca
import pandas as pd
import numpy as np
from cuzcatlan import differential_gene_expression
from cuzcatlan import compute_information_coefficient

data_df = pd.read_table("test_data/BRCA_minimal.gct", header=2, index_col=0)
data_df.drop('Description', axis=1, inplace=True)
temp = open("test_data/BRCA_minimal.cls")
temp.readline()
temp.readline()
classes = [int(i) for i in temp.readline().strip('\n').split(' ')]
classes = pd.Series(classes, index=data_df.columns)

differential_gene_expression(phenotypes=classes, gene_expression=data_df, output_filename='DE_test',
                             ranking_method=cusca.custom_pearson_corr)

