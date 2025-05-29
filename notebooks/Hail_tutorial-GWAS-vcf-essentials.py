#!/usr/bin/env python
import datetime
import hail as hl

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

log_file_name = f"logs/hail-{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}.log"

# run Spark and init Hail
spark_conf = SparkConf().setAppName("hail-gwas")
sc = SparkContext(conf=spark_conf)
hl.init(sc=sc, log=log_file_name)

## VCF and sample data file 
vcf_fn = 'data/1kg.vcf'
annotations_fn = 'data/1kg_annotations.txt'

## Read a vcf file, convert and write it as matrix table
mt = hl.import_vcf(vcf_fn) # assign this to a dummy variable to avoid errors\n')
## Read table with sample info and set the key as the sample IDs
annotation_table = (hl.import_table(annotations_fn, impute=True)
         .key_by('Sample'))

mt = mt.annotate_cols(pheno = annotation_table[mt.s])

# sample_qc is a hail genetic method to compute per-sample metrics useful for quality control.
mt = hl.sample_qc(mt)
mt = mt.filter_cols((mt.sample_qc.dp_stats.mean >= 4) & (mt.sample_qc.call_rate >= 0.97))

# Filter entries by using the allele depth info
ab = mt.AD[1] / hl.sum(mt.AD)

# Hail boolean expression to select data to retain
filter_condition_ab = ((mt.GT.is_hom_ref() & (ab <= 0.1)) |
                        (mt.GT.is_het() & (ab >= 0.25) & (ab <= 0.75)) |
                        (mt.GT.is_hom_var() & (ab >= 0.9)))

fraction_filtered = mt.aggregate_entries(hl.agg.fraction(~filter_condition_ab))
print(f'Filtering {fraction_filtered * 100:.2f}% entries out of downstream analysis.')
mt = mt.filter_entries(filter_condition_ab)

## Variant QC and filtering
mt = hl.variant_qc(mt)
mt = mt.filter_rows(mt.variant_qc.AF[1] > 0.01) # It takes variants for which the alternate allele has a frequency larger than 1%
mt = mt.filter_rows(mt.variant_qc.p_value_hwe > 1e-6) # Hardy-Weinberg equilibrium pvalue cut-off

## Hardy-Weinberg normalized PCA
eigenvalues, pcs, _ = hl.hwe_normalized_pca(mt.GT)
mt = mt.annotate_cols(scores = pcs[mt.s].scores)

## Linear Regression with covariates
gwas = hl.linear_regression_rows(
    y=mt.pheno.CaffeineConsumption,
    x=mt.GT.n_alt_alleles(),
    covariates=[1.0, mt.pheno.isFemale, mt.scores[0], mt.scores[1], mt.scores[2]])

print ("Writing the filtered matrix table annotated with PCA scores")
mt.write("mt_filtered_PCA.mt", overwrite=True)
print ("Writing GWAS result table")
gwas.write("gwas_result_table.ht", overwrite=True)

