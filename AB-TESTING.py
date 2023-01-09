#1. Establish Hypotheses
#2. Assumption Check
# - 1. Normality Assumption
# - 2. Variance Homogeneity
# 3. Implementation of the Hypothesis
# - 1. Independent two-sample t-test (parametric test) if assumptions are met
# - 2. Mannwhitneyu test if assumptions are not met (non-parametric test)
# 4. Interpret results based on p-value
# Note:
# - Number 2 directly if normality is not achieved. If variance homogeneity is not provided, an argument is entered for number 1.
# - It can be useful to perform outlier analysis and correction before normality analysis.


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control_group = pd.read_excel("Datasets/ab_testing.xlsx", sheet_name="Control Group")
test_group = pd.read_excel("Datasets/ab_testing.xlsx", sheet_name="Test Group")

control_group.describe()
test_group.describe()

control_group["group"] = " control group "
test_group["group"] = " test group "

control_group.head()
test_group.head()
whole_data = pd.concat([control_group, test_group], ignore_index=True)
whole_data.sort_values(by="group", ascending=False)




# Hypothesis Testing

############################
#1. Set up the Hypothesis
############################

# H0: M1 = M2
# H1: M1 != M2

############################
#2. Assumption Check
############################

# Assumption of Normality
# Variance Homogeneity

############################
# Assumption of Normality
############################

# H0: Assumption of normal distribution is provided.
# H1:..not provided.



# 1-)#H0 : M1 = M2  " There is not a difference between the mean of the two groups for Purchase in terms of Statistics"
#H1 : M1!= M2 "There is a difference between the mean of the two groups for Purchase in terms of Statistics"

whole_data.groupby("group").agg({"Purchase": "mean"})

#2. Assumption Check
############################

# A) Assumption of Normality
############################

# H0: Assumption of normal distribution is provided.
# H1:..not provided.

test_stat, pvalue = shapiro(whole_data.loc[whole_data["group"] == " control group ", "Purchase"])
print('test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# HO CAN BE REJECTED if p-value < 0.05
# H0 CANNOT BE REJECTED if p-value !< 0.05

# for control group we have a p-value above 0.05 so we cannot reject the null hypothesis

test_stat, pvalue = shapiro(whole_data.loc[whole_data["group"] == " test group ", "Purchase"])
print('test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# for test group we have a p-value above 0.05 so we cannot reject the null hypothesis.

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue = ttest_indtest_stat, pvalue = ttest_ind(whole_data.loc[whole_data["group"] == " control group ", "Purchase"],
                            whole_data.loc[whole_data["group"] == " test group ", "Purchase"],equal_var=True)


print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))





# B) Variance Homogeneity Assumption
############################

# H0: Variances are Homogeneous
# H1: Variances Are Not Homogeneous

test_stat, pvalue = levene(whole_data.loc[whole_data["group"] == " control group ", "Purchase"],
                            whole_data.loc[whole_data["group"] == " test group ", "Purchase"])

print('test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# HO CANNOT BE REJECTED if p-value < 0.05 and Variances are not homogeneous

# 3. Implementation of the Hypothesis

# -Mannwhitneyu test if assumptions are not met (non-parametric test)

test_stat, pvalue = mannwhitneyu(whole_data.loc[whole_data["group"] == " control group ", "Purchase"],
                            whole_data.loc[whole_data["group"] == " test group ", "Purchase"])

print('test stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# since p-value is less than 0.05, we can reject the null hypothesis. So there is a difference between the mean of the two groups for Purchase in terms of Statistics.

whole_data.groupby("group").agg({"Purchase": ["mean", "sum"], "Earning": ["mean", "sum"]})

# we can offer average bidding price for test group as test group has got a difference from control group in terms of Statistics.







