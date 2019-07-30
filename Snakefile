################################################################################
#                  _       
#  ___ _ __   __ _| | _____  ___ 
# / __| '_ \ / _` | |/ / _ \/ __|
# \__ \ | | | (_| |   <  __/\__ \
# |___/_| |_|\__,_|_|\_\___||___/
#                 
# Data integration and machine learning pipelines built on Snakemake
#
# https://github.com/khughitt/snakes
#
#  p3-cca 1.0
#  ----------
#
#  Config : /mnt/data2/Dropbox/r/nih/src/snakes-cca/config/config.yml
#  Date   : 2019-07-30 12:22:09
#
#  datasets:
#  - rnaseq: /data/projects/nih/p3/rnaseq/rnaseq_grch38_featurecounts_raw.tsv
#  - ac50: /data/projects/nih/p3/drug_response/curves/1.2/bg_adj_curves.csv
#
#  Output dir: output/1.0 
#
################################################################################
import glob
import gzip
import operator
import os
import yaml
import numpy as np
import pandas as pd
import pathlib
import warnings
from snakes import clustering, filters, gene_sets 
from snakes.rules import ActionRule, GroupedActionRule

# create output directory, if needed
output_dir = 'output/1.0' 

################################################################################
#
# Default target
#
################################################################################
rule all:
    input: "output/1.0/finished"
      
################################################################################
#
# rnaseq workflow
#
################################################################################
rule load_rnaseq:
    input: '/data/projects/nih/p3/rnaseq/rnaseq_grch38_featurecounts_raw.tsv'
    output: 'output/1.0/data/rnaseq/raw.csv'
    run:
        dat = pd.read_csv(input[0], sep='	', index_col=0, encoding='utf-8')
        dat.to_csv(output[0], index_label='symbol')

rule rnaseq_filtered:
    input: 'output/1.0/data/rnaseq/raw.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_filtered.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        cols_to_drop = ['KMS21BM_JCRB', 'Karpas417_ECACC', 'OCIMY1_PLB', 'OPM2_DSMZ']
        dat = dat.drop(columns=cols_to_drop, errors="ignore")
        dat.to_csv(output[0])

rule exclude_zero_variance:
    input: 'output/1.0/data/rnaseq/rnaseq_filtered.csv'
    output: 'output/1.0/data/rnaseq/exclude_zero_variance.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        dat = filters.filter_rows_by_func(df=dat, func=np.var, op=operator.gt, value=0, quantile=None)
        dat.to_csv(output[0])

rule rnaseq_genes:
    input: 'output/1.0/data/rnaseq/exclude_zero_variance.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_genes.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        dat = (dat / dat.sum()) * 1E6
        dat.to_csv(output[0])

rule rnaseq_ccle_sum:
    input: 'output/1.0/data/rnaseq/rnaseq_genes.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_ccle_sum.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        # load gmt file
        gmt_file = 'gene_sets/Cancer_Cell_Line_Encyclopedia.gmt.gz'

        if gmt_file.endswith('.gz'):
            fp = gzip.open(gmt_file, 'rt')
        else:
            fp = open(gmt_file, 'r')

        # iterate of gmt entries and construct a dictionary mapping from gene set names to lists
        # the genes they contain
        gsets = {}

        # gmt file column indices
        GENE_SET_NAME  = 0
        GENE_SET_START = 2

        # minimum number of genes required for a gene set to be used
        MIN_GENES = 5

        # iterate over gene sets, and those that meet the minimum size requirements
        for line in fp:
            # split line and retrieve gene set name and a list of genes in the set
            fields = line.rstrip('\n').split('\t')

            num_genes = len(fields) - 2

            if num_genes > MIN_GENES:
                gsets[fields[GENE_SET_NAME]] = fields[GENE_SET_START:len(fields)]

        fp.close()

        # apply function along gene sets and save output
        dat = gene_sets.gene_set_apply(dat, gsets, 'sum')

        # update row names to include dataset, gene set, and function applied
        #dat.index = ["_".join([gset_id_prefix, gene_set, func]) for gene_set in dat.index]
        #dat.to_csv(output[0], index_label='gene_set_id')
        dat.to_csv(output[0])

rule rnaseq_nci_nature2016_median:
    input: 'output/1.0/data/rnaseq/rnaseq_genes.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_nci_nature2016_median.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        # load gmt file
        gmt_file = 'gene_sets/NCI-Nature_2016.gmt.gz'

        if gmt_file.endswith('.gz'):
            fp = gzip.open(gmt_file, 'rt')
        else:
            fp = open(gmt_file, 'r')

        # iterate of gmt entries and construct a dictionary mapping from gene set names to lists
        # the genes they contain
        gsets = {}

        # gmt file column indices
        GENE_SET_NAME  = 0
        GENE_SET_START = 2

        # minimum number of genes required for a gene set to be used
        MIN_GENES = 5

        # iterate over gene sets, and those that meet the minimum size requirements
        for line in fp:
            # split line and retrieve gene set name and a list of genes in the set
            fields = line.rstrip('\n').split('\t')

            num_genes = len(fields) - 2

            if num_genes > MIN_GENES:
                gsets[fields[GENE_SET_NAME]] = fields[GENE_SET_START:len(fields)]

        fp.close()

        # apply function along gene sets and save output
        dat = gene_sets.gene_set_apply(dat, gsets, 'median')

        # update row names to include dataset, gene set, and function applied
        #dat.index = ["_".join([gset_id_prefix, gene_set, func]) for gene_set in dat.index]
        #dat.to_csv(output[0], index_label='gene_set_id')
        dat.to_csv(output[0])

rule rnaseq_drugbank_var:
    input: 'output/1.0/data/rnaseq/rnaseq_genes.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_drugbank_var.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        # load gmt file
        gmt_file = 'gene_sets/Human_DrugBank_all_symbol.gmt.gz'

        if gmt_file.endswith('.gz'):
            fp = gzip.open(gmt_file, 'rt')
        else:
            fp = open(gmt_file, 'r')

        # iterate of gmt entries and construct a dictionary mapping from gene set names to lists
        # the genes they contain
        gsets = {}

        # gmt file column indices
        GENE_SET_NAME  = 0
        GENE_SET_START = 2

        # minimum number of genes required for a gene set to be used
        MIN_GENES = 5

        # iterate over gene sets, and those that meet the minimum size requirements
        for line in fp:
            # split line and retrieve gene set name and a list of genes in the set
            fields = line.rstrip('\n').split('\t')

            num_genes = len(fields) - 2

            if num_genes > MIN_GENES:
                gsets[fields[GENE_SET_NAME]] = fields[GENE_SET_START:len(fields)]

        fp.close()

        # apply function along gene sets and save output
        dat = gene_sets.gene_set_apply(dat, gsets, 'var')

        # update row names to include dataset, gene set, and function applied
        #dat.index = ["_".join([gset_id_prefix, gene_set, func]) for gene_set in dat.index]
        #dat.to_csv(output[0], index_label='gene_set_id')
        dat.to_csv(output[0])

rule rnaseq_drugbank_final:
    input: 'output/1.0/data/rnaseq/rnaseq_drugbank_var.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_drugbank_final.csv'
    params:
        args = {'cutoff': 0.8, 'method': 'wgcna', 'nthreads': 0, 'use': 'pairwise.complete.obs', 'verbose': True, 'cor_method': 'wgcna'}
    script:
        '/mnt/data2/Dropbox/r/software/snakes/snakes/src/filter/filter_rows_max_correlation.R' 

rule rnaseq_msigdb_c3_tft_v62_median:
    input: 'output/1.0/data/rnaseq/rnaseq_genes.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_msigdb_c3_tft_v62_median.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        # load gmt file
        gmt_file = 'gene_sets/c3.tft.v6.2.symbols.gmt.gz'

        if gmt_file.endswith('.gz'):
            fp = gzip.open(gmt_file, 'rt')
        else:
            fp = open(gmt_file, 'r')

        # iterate of gmt entries and construct a dictionary mapping from gene set names to lists
        # the genes they contain
        gsets = {}

        # gmt file column indices
        GENE_SET_NAME  = 0
        GENE_SET_START = 2

        # minimum number of genes required for a gene set to be used
        MIN_GENES = 5

        # iterate over gene sets, and those that meet the minimum size requirements
        for line in fp:
            # split line and retrieve gene set name and a list of genes in the set
            fields = line.rstrip('\n').split('\t')

            num_genes = len(fields) - 2

            if num_genes > MIN_GENES:
                gsets[fields[GENE_SET_NAME]] = fields[GENE_SET_START:len(fields)]

        fp.close()

        # apply function along gene sets and save output
        dat = gene_sets.gene_set_apply(dat, gsets, 'median')

        # update row names to include dataset, gene set, and function applied
        #dat.index = ["_".join([gset_id_prefix, gene_set, func]) for gene_set in dat.index]
        #dat.to_csv(output[0], index_label='gene_set_id')
        dat.to_csv(output[0])

rule rnaseq_dsigdb_median:
    input: 'output/1.0/data/rnaseq/rnaseq_genes.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_dsigdb_median.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        # load gmt file
        gmt_file = 'gene_sets/DSigDB.gmt.gz'

        if gmt_file.endswith('.gz'):
            fp = gzip.open(gmt_file, 'rt')
        else:
            fp = open(gmt_file, 'r')

        # iterate of gmt entries and construct a dictionary mapping from gene set names to lists
        # the genes they contain
        gsets = {}

        # gmt file column indices
        GENE_SET_NAME  = 0
        GENE_SET_START = 2

        # minimum number of genes required for a gene set to be used
        MIN_GENES = 5

        # iterate over gene sets, and those that meet the minimum size requirements
        for line in fp:
            # split line and retrieve gene set name and a list of genes in the set
            fields = line.rstrip('\n').split('\t')

            num_genes = len(fields) - 2

            if num_genes > MIN_GENES:
                gsets[fields[GENE_SET_NAME]] = fields[GENE_SET_START:len(fields)]

        fp.close()

        # apply function along gene sets and save output
        dat = gene_sets.gene_set_apply(dat, gsets, 'median')

        # update row names to include dataset, gene set, and function applied
        #dat.index = ["_".join([gset_id_prefix, gene_set, func]) for gene_set in dat.index]
        #dat.to_csv(output[0], index_label='gene_set_id')
        dat.to_csv(output[0])

rule rnaseq_geo_disease_perturbations_median:
    input: 'output/1.0/data/rnaseq/rnaseq_genes.csv'
    output: 'output/1.0/data/rnaseq/rnaseq_geo_disease_perturbations_median.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        # load gmt file
        gmt_file = 'gene_sets/Disease_Perturbations_from_GEO_up.gmt.gz'

        if gmt_file.endswith('.gz'):
            fp = gzip.open(gmt_file, 'rt')
        else:
            fp = open(gmt_file, 'r')

        # iterate of gmt entries and construct a dictionary mapping from gene set names to lists
        # the genes they contain
        gsets = {}

        # gmt file column indices
        GENE_SET_NAME  = 0
        GENE_SET_START = 2

        # minimum number of genes required for a gene set to be used
        MIN_GENES = 5

        # iterate over gene sets, and those that meet the minimum size requirements
        for line in fp:
            # split line and retrieve gene set name and a list of genes in the set
            fields = line.rstrip('\n').split('\t')

            num_genes = len(fields) - 2

            if num_genes > MIN_GENES:
                gsets[fields[GENE_SET_NAME]] = fields[GENE_SET_START:len(fields)]

        fp.close()

        # apply function along gene sets and save output
        dat = gene_sets.gene_set_apply(dat, gsets, 'median')

        # update row names to include dataset, gene set, and function applied
        #dat.index = ["_".join([gset_id_prefix, gene_set, func]) for gene_set in dat.index]
        #dat.to_csv(output[0], index_label='gene_set_id')
        dat.to_csv(output[0])

################################################################################
#
# ac50 workflow
#
################################################################################
rule load_ac50:
    input: '/data/projects/nih/p3/drug_response/curves/1.2/bg_adj_curves.csv'
    output: 'output/1.0/data/ac50/raw.csv'
    run:
        dat = pd.read_csv(input[0], sep=',', index_col='cell_line', encoding='utf-8')
        dat.to_csv(output[0], index_label='sample_id')

rule ac50_filter_cols_name_in:
    input: 'output/1.0/data/ac50/raw.csv'
    output: 'output/1.0/data/ac50/ac50_filter_cols_name_in.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        cols_to_keep = ['drug_id', 'ac50', 'curve_slope']

        # if index column specified, ignore, remove it
        cols_to_keep = [x for x in cols_to_keep if x != dat.index.name]

        dat = dat[cols_to_keep]
        dat.to_csv(output[0])

rule ac50_filter_rows_name_not_in:
    input: 'output/1.0/data/ac50/ac50_filter_cols_name_in.csv'
    output: 'output/1.0/data/ac50/ac50_filter_rows_name_not_in.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        rows_to_drop = ['KMS21BM_JCRB', 'Karpas417_ECACC', 'OCIMY1_PLB', 'OPM2_DSMZ']
        dat = dat.drop(rows_to_drop)
        dat.to_csv(output[0])

rule drug_response_filtered:
    input: 'output/1.0/data/ac50/ac50_filter_rows_name_not_in.csv'
    output: 'output/1.0/data/ac50/drug_response_filtered.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        dat = filters.filter_rows_by_col(dat, col='curve_slope',
                                         op=operator.gt, value=0, quantile=None)
        # apply function to each group and filter results
        dat = filters.filter_rows_by_group_func(dat, "drug_id", 
                                                "drug_id", 
                                                "len", op=operator.ge, 
                                                value=20)

        cols_to_drop = ['curve_slope']
        dat = dat.drop(columns=cols_to_drop, errors="ignore")
        # apply function to each group and filter results
        dat = filters.filter_rows_by_group_func(dat, 
                                                "drug_id", 
                                                "ac50", 
                                                "mad", 
                                                op=operator.ge, 
                                                value=None, 
                                                quantile=0.5)

        dat.to_csv(output[0])

rule ac50_pivot_wide:
    input: 'output/1.0/data/ac50/drug_response_filtered.csv'
    output: 'output/1.0/data/ac50/ac50_pivot_wide.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        dat = dat.pivot(index=None, columns="drug_id", values="ac50")
        dat.to_csv(output[0])

rule ac50_impute_knn:
    input: 'output/1.0/data/ac50/ac50_pivot_wide.csv'
    output: 'output/1.0/data/ac50/ac50_impute_knn.csv'
    params:
        args = {'k': 5}
    script: '/mnt/data2/Dropbox/r/software/snakes/snakes/src/impute/impute_knn.R' 

rule ac50_transpose_data:
    input: 'output/1.0/data/ac50/ac50_impute_knn.csv'
    output: 'output/1.0/data/ac50/ac50_transpose_data.csv'
    run:
        dat = pd.read_csv(input[0], index_col=0)
        dat = dat.T
        dat.to_csv(output[0])

rule drug_response_final:
    input: 'output/1.0/data/ac50/ac50_transpose_data.csv'
    output: 'output/1.0/data/ac50/drug_response_final.csv'
    params:
        args = {'cutoff': 0.8, 'method': 'wgcna', 'nthreads': 0, 'use': 'pairwise.complete.obs', 'verbose': True, 'cor_method': 'wgcna'}
    script:
        '/mnt/data2/Dropbox/r/software/snakes/snakes/src/filter/filter_rows_max_correlation.R' 


################################################################################
#
# Data integration
#
################################################################################
rule drug_response_final_rnaseq_drugbank_final_cca:
    input: ['output/1.0/data/ac50/drug_response_final.csv', 'output/1.0/data/rnaseq/rnaseq_drugbank_final.csv']
    output: 'output/1.0/data_integration/drug_response_final_rnaseq_drugbank_final_cca.csv'
    params: {'method': 'cca'}
    script:
        '/mnt/data2/Dropbox/r/software/snakes/snakes/src/integrate/integrate_cca.R' 

################################################################################
#
# Training set construction
#
################################################################################
checkpoint create_training_sets:
  input:
    features=['output/1.0/data/rnaseq/rnaseq_genes.csv', 'output/1.0/data/rnaseq/rnaseq_ccle_sum.csv', 'output/1.0/data/rnaseq/rnaseq_nci_nature2016_median.csv', 'output/1.0/data/rnaseq/rnaseq_msigdb_c3_tft_v62_median.csv', 'output/1.0/data/rnaseq/rnaseq_drugbank_var.csv'],
    response="output/1.0/data/ac50/drug_response_final.csv"
  output: directory("output/1.0/training_sets/raw")
  params:
    output_dir="output/1.0/training_sets/raw",
    allow_mismatched_indices=False,
    include_column_prefix=True
  run:
    # create output directory
    if not os.path.exists(params.output_dir):
        os.mkdir(params.output_dir, mode=0o755)

    # load feature data
    feature_dat = pd.read_csv(input.features[0], index_col=0).sort_index()

    # update column names (optional)
    if params.include_column_prefix:
        prefix = pathlib.Path(input.features[0]).stem + "_"
        feature_dat.columns = prefix + feature_dat.columns

    if len(input.features) > 1:
        for filepath in input.features[1:]:
            dat = pd.read_csv(filepath, index_col=0).sort_index()

            # update column names (optional)
            if params.include_column_prefix:
                prefix = pathlib.Path(filepath).stem + "_"
                dat.columns = prefix + dat.columns

            # check to make sure there are no overlapping columns
            shared_columns = set(feature_dat.columns).intersection(dat.columns)

            if len(shared_columns) > 0:
                msg = f"Column names in {filepath} overlap with others in feature data."
                raise ValueError(msg)

            # check for index mismatches
            if not params.allow_mismatched_indices and not dat.index.equals(feature_dat.index):
                msg = f"Row names for {filepath} do not match other feature data indices."
                raise ValueError(msg)

            # merge feature data
            feature_dat = feature_dat.join(dat)

            # check to make sure dataset is not empty
            if feature_dat.empty:
                from pandas.errors import EmptyDataError
                msg = (f"Training set empty after merging {filepath}! Check to make "
                       "sure datasets have row names in common")
                raise EmptyDataError(msg)

    # load response dataframe
    response_dat = pd.read_csv(input.response, index_col=0).sort_index()

    # check for index mismatches
    if not params.allow_mismatched_indices and not response_dat.index.equals(feature_dat.index):
        msg = f"Row names for {input.response} do not match feature data indices."
        raise ValueError(msg)

    # check to make sure at least some shared indices exist
    if len(set(feature_dat.index).intersection(response_dat.index)) == 0:
        from pandas.errors import EmptyDataError
        msg = (f"Feature and response data have no shared row names!")
        raise EmptyDataError(msg)

    # iterate over columns in response data and create training sets
    for col in response_dat.columns:
        # get response column as a Series and rename to "response"
        dat = response_dat[col]
        dat.name = 'response'

        # add response data column to end of feature data and save to disk
        outfile = os.path.join(params.output_dir, "{}.csv".format(col))
        feature_dat.join(dat).to_csv(outfile)

rule collect_results:
    input:
      expand("output/1.0/{datasets}", datasets=['data/rnaseq/rnaseq_geo_disease_perturbations_median.csv', 'data/ac50/drug_response_final.csv']),
    output: touch("output/1.0/finished")

################################################################################
#
# Other settings
#
################################################################################
localrules: load_rnaseq, load_ac50

