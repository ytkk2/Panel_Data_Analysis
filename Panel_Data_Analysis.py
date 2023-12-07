# These libraries are needed to execute Fixed Effects Model and Random Effects Model
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
from linearmodels.panel import PanelOLS
from linearmodels.panel import RandomEffects

# These libraries are needed to do Hausman Specification Test
import numpy as np
from scipy import stats

# read dataset from csv file
# The number of dataset's index is 189
data = pd.read_csv("/Users/katoyutaka/Desktop//soturon/csv_files/all_pref_DID_TFP.csv")

# Convert the 'prefecture' column to a categorical data type
# for more efficient memory usage and statistical modeling
data["prefecture"] = data["prefecture"].astype("category")

# Set 'prefecture' and 'year' as a multi-level index to format the data as panel data,
# facilitating time series and cross-sectional analyses
panel_data = data.set_index(["prefecture", "year"])

# Pooled OLS Model
pooled_ols_model = PooledOLS(
    panel_data["TFP"], sm.add_constant(panel_data["DID"])
).fit()

# print(pooled_ols_model)


# Fixed Effects Model
fe_model = PanelOLS(
    panel_data["TFP"], sm.add_constant(panel_data["DID"]), entity_effects=True
).fit()

print(fe_model)

# Random Effects Model
re_model = RandomEffects(panel_data["TFP"], sm.add_constant(panel_data["DID"])).fit()

# print(re_model)

# Hausman Speification Test
# Get parameters from fe_model and re_model
fe_coef = fe_model.params
re_coef = re_model.params

# Calculate the difference in covariance matrices
cov_diff = fe_model.cov - re_model.cov

# ハウスマン検定統計量を計算
hausman_stat = (fe_coef - re_coef).T @ np.linalg.inv(cov_diff) @ (fe_coef - re_coef)

# 自由度を計算
df = len(fe_coef)

# P値を計算
p_value = stats.chi2.sf(hausman_stat, df)

# print("Hausman Test Statistic:", hausman_stat)
# print("Degrees of Freedom:", df)
# print("P-value:", p_value)
