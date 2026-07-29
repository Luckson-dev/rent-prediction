import pandas as pd
import numpy as np
import scipy.stats as ss

"""
  Calcule le V de Cramer entre deux variables catégorielles x et y

"""

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape

    # La formule de v de crammer
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

for col in category_cols:
    corr = cramers_v(df[col], df["Satisfait"])
    print(col, "-->", corr)