# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   Author name: Anar Huseynov
#   UM email: ahuseyno@umich.edu
# ---

import pandas as pd
import numpy as np
import scipy.stats
import numpy as np
from scipy import stats
import pandas as pd
from timeit import default_timer as timer
import math
from scipy.stats import bernoulli
from statistics import mean

# # Question 0

# ## Windowing Operations
# - Pandas contains a compact set of APIs for performing windowing operations - an operation that performs an aggregation over a sliding partition of values.
# - For more information [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html#exponentially-weighted-window)
# - Codes that I am going to use in this part are very similar to the ones used in provided link

# - Let's assume we want to calculate the sum of consequtive triplets of the array.

s = pd.Series(range(100))
s.rolling(window=3).sum()

# - Let's assume we want to calculate the moving average of the series.

s = pd.Series(np.random.normal(6, 1, 10))
s.rolling(window=3).mean()

# - Keep in mind that for the first couple of windows, entries that cannot be filled by elements of series (because length of periods before the chosen index is not enough to fill all entries) will be filled by `NaN` values by default. That is why we get `NaN` values for the first $2$ rolling means above. 

# ### Let's look at the usage of rolling window operations in time series data

# ### Using `Rolling` Function for Time Series Data 

s = pd.Series(np.random.normal(6, 1, 10), index=pd.date_range('2021-10-20', periods=10, freq='4h'))
s.rolling(window='1D').mean()

# In example above, frequency of the generated data is $4$ hours and windows are calculated for $1$ day. Also note that, length of a window is decreased when periods are not enough to fill all window. This feature can be managed manually by the `min_periods` which we will talk next.

# ### `min_periods`
# `min_periods` parameter dictates the minimum amount of `non-np.nan` values a window must have. For time-based windows, this parameter is defaulted to $1$ and for others it is defaulted to `window` parameter.

# length of windows can get as small as 1 if not enough non-np.nan values found.
s = pd.Series(np.random.normal(6, 1, 10))
s.rolling(window=3, min_periods = 1).sum()

# ### Centering Windows

# In addition, we can also specify in which way, windows are formed. By default, windows are formed in way that $i^{th}$ element is the last entry of $i^{th}$ window. We can change it to be the middle element of $i^{th}$ window.

s = pd.Series(range(10))
s.rolling(window=3, center=True, min_periods = 1).sum()

# ### Rolling window endpoints
#
# In addition, with the `closed` parameter the inclusion of the interval endpoints in rolling window calculations can be specified. There are 4 possible values for `closed`
# - "right" (default) $=>$ right closed, left open
# - "left" $=>$ right open, left closed
# - "both" $=>$ both closed
# - "neighter" $=>$ both open
#
# Let's examine them using code example

df = pd.Series(range(10))
final = pd.DataFrame()
df.rolling(window = 2, closed="right").sum()  # default
final["right"] = df.rolling(window = 2, closed="right").sum()
final["left"] = df.rolling(window = 2, closed="left").sum()
final["both"] = df.rolling(window = 2, closed="both").sum()
final["neigter"] = df.rolling(window = 2, closed="neither", min_periods = 1).sum()
final

# ## End of Tutorial