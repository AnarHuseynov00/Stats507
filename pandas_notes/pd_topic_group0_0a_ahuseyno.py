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

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: py:light,ipynb
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#
# **Group 0**
#   

import pandas as pd
import numpy as np

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Pivot tables](#Pivot-tables)
# + [One row to many](#One-row-to-many)
# + [DataFrame.pct_change()](#DataFrame.pct_change()) 
# + [Working with missing data](#Working-with-missing-data)
# + [Cumulative sums](#Title:-pandas.DataFrame.cumsum)
# + [Stack and unstack](#Stack-and-unstack)
# + [Pandas Query](#Pandas-Query) 
# + [Time Series](#Time-Series) 
# + [Window Functions](#Window-Functions) 
# + [Processing Time Data](#Processing-Time-Data)
# + [Pandas Time Series Analysis](#Title:-Pandas-Time-Series-Analysis)
# + [Pivot Table in pandas](#Pivot-Table-in-pandas)
# + [Multi-indexing](#Multi-indexing)
# + [Missing Data in Pandas](#Missing-Data-in-pandas)

# ## Pivot tables
# Zeyuan Li
# zeyuanli@umich.edu
# 10/19/2021
#
#

# ## Pivot tables in pandas
#
# The pivot tables in Excel is very powerful and convienent in handling with numeric data. Pandas also provides ```pivot_table()``` for pivoting with aggregation of numeric data. There are 5 main arguments of ```pivot_table()```:
# * ***data***: a DataFrame object
# * ***values***: a column or a list of columns to aggregate.
# * ***index***: Keys to group by on the pivot table index. 
# * ***columns***:  Keys to group by on the pivot table column. 
# * ***aggfunc***: function to use for aggregation, defaulting to ```numpy.mean```.

# ### Example

df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 6,
        "B": ["A", "B", "C"] * 8,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
        "D": np.random.randn(24),
        "E": np.random.randn(24),
        "F": [datetime.datetime(2013, i, 1) for i in range(1, 13)]
        + [datetime.datetime(2013, i, 15) for i in range(1, 13)],
    }
)
df


# ### Do aggregation
#
# * Get the pivot table easily. 
# * Produce the table as the same result of doing ```groupby(['A','B','C'])``` and compute the ```mean``` of D, with different values of D shown in seperate columns.
# * Change to another ***aggfunc*** to finish the aggregation as you want.

pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])


pd.pivot_table(df, values="D", index=["B"], columns=["A", "C"], aggfunc=np.sum)


# ### Display all aggregation values
#
# * If the ***values*** column name is not given, the pivot table will include all of the data that can be aggregated in an additional level of hierarchy in the columns:

pd.pivot_table(df, index=["A", "B"], columns=["C"])


# ### Output
#
# * You can render a nice output of the table omitting the missing values by calling ```to_string```

table = pd.pivot_table(df, index=["A", "B"], columns=["C"])
print(table.to_string(na_rep=""))

# ## One row to many
#
# *Kunheng Li(kunhengl@umich.edu)*

# The reason I choose this function is because last homework. Before the hint from teachers, I found some ways to transfrom one row to many rows. Therefore, I will introduce a function to deal with this type of data.

# First, let's see an example.

data = {
    "first name":["kevin","betty","tony"],
    "last name":["li","jin","zhang"],
    "courses":["EECS484, STATS507","STATS507, STATS500","EECS402,EECS482,EECS491"]   
}
df = pd.DataFrame(data)
df = df.set_index(["first name", "last name"])["courses"].str.split(",", expand=True)\
    .stack().reset_index(drop=True, level=-1).reset_index().rename(columns={0: "courses"})
print(df)

# This is the first method I want to introduce, stack() or unstack(), both are similar. 
# Unstack() and stack() in DataFrame are to make itself to a Series which has secondary index.
# Unstack() is to transform its index to secondary index and its column to primary index, however, 
# stack() is to transform its index to primary index and its column to secondary index.

# However, in Pandas 0.25 version, there is a new method in DataFrame called explode(). They have the result, let's see the example.

df["courses"] = df["courses"].str.split(",")
df = df.explode("courses")
print(df)


# ## DataFrame.pct_change()
# *Dongming Yang*

# +
# This function always be used to calculate the percentage change between the current and a prior element, and always be used to a time series     
# The axis could choose the percentage change from row or columns
# Creating the time-series index 
ind = pd.date_range('01/01/2000', periods = 6, freq ='W') 
  
# Creating the dataframe  
df = pd.DataFrame({"A":[14, 4, 5, 4, 1, 55], 
                   "B":[5, 2, 54, 3, 2, 32],  
                   "C":[20, 20, 7, 21, 8, 5], 
                   "D":[14, 3, 6, 2, 6, 4]}, index = ind) 
  
# find the percentage change with the previous row 
df.pct_change()

# find the percentage change with precvious columns 
df.pct_change(axis=1)

# +
# periods means start to calculate the percentage change between the periods column or row and the beginning

# find the specific percentage change with first row
df.pct_change(periods=3)

# +
# fill_method means the way to handle NAs before computing percentage change by assigning a value to that NAs
# importing pandas as pd 
import pandas as pd 
  
# Creating the time-series index 
ind = pd.date_range('01/01/2000', periods = 6, freq ='W') 
  
# Creating the dataframe  
df = pd.DataFrame({"A":[14, 4, 5, 4, 1, 55], 
                   "B":[5, 2, None, 3, 2, 32],  
                   "C":[20, 20, 7, 21, 8, None], 
                   "D":[14, None, 6, 2, 6, 4]}, index = ind) 
  
# apply the pct_change() method 
# we use the forward fill method to 
# fill the missing values in the dataframe 
df.pct_change(fill_method ='ffill')

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# -

# ## Working with missing data
# *Kailan Xu*

# - Detecting missing data
# - Inserting missing data
# - Calculations with missing data
# - Cleaning / filling missing data
# - Dropping axis labels with missing data

# ### 1. Detecting missing data

# As data comes in many shapes and forms, pandas aims to be flexible with regard to handling missing data. While NaN is the default missing value marker for reasons of computational speed and convenience, we need to be able to easily detect this value with data of different types: floating point, integer, boolean, and general object. In many cases, however, the Python None will arise and we wish to also consider that “missing” or “not available” or “NA”.

# +
import pandas as pd 
import numpy as np 

df = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["one", "two", "three"],
)
df2 = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])
df2
# -

# To make detecting missing values easier (and across different array dtypes), pandas provides the `isna()` and `notna()` functions, which are also methods on Series and DataFrame objects:

df2.isna()

df2.notna()

# ###  2. Inserting missing data

# You can insert missing values by simply assigning to containers. The actual missing value used will be chosen based on the dtype.
# For example, numeric containers will always use NaN regardless of the missing value type chosen:

s = pd.Series([1, 2, 3])
s.loc[0] = None
s

# Likewise, datetime containers will always use NaT.
# For object containers, pandas will use the value given:

s = pd.Series(["a", "b", "c"])
s.loc[0] = None
s.loc[1] = np.nan
s

# ### 3. Calculations with missing data

# - When summing data, NA (missing) values will be treated as zero.
# - If the data are all NA, the result will be 0.
# - Cumulative methods like `cumsum()` and `cumprod()` ignore NA values by default, but preserve them in the resulting arrays. To override this behaviour and include NA values, use `skipna=False`.

df2

df2["one"].sum()

df2.mean(1)

df2.cumsum()

df2.cumsum(skipna=False)

# ### 4. Cleaning / filling missing data

# pandas objects are equipped with various data manipulation methods for dealing with missing data.
# - `fillna()` can “fill in” NA values with non-NA data in a couple of ways, which we illustrate:

df2.fillna(0)

df2["one"].fillna("missing")

# ### 5.Dropping axis labels with missing data

# You may wish to simply exclude labels from a data set which refer to missing data. To do this, use `dropna()`:

df2.dropna(axis=0)


# # Title: pandas.DataFrame.cumsum
# - Name: Yixuan Feng
# - Email: fengyx@umich.edu

# ## pandas.DataFrame.cumsum
# - Cumsum is the cumulative function of pandas, used to return the cumulative values of columns or rows.

# ## Example 1 - Without Setting Parameters
# - This function will automatically return the cumulative value of all columns.

values_1 = np.random.randint(10, size=10) 
values_2 = np.random.randint(10, size=10) 
group = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'] 
df = pd.DataFrame({'group':group, 'value_1':values_1, 'value_2':values_2}) 
df

df.cumsum()

# ## Example 2 - Setting Parameters
# - By setting the axis to 1, this function will return the cumulative value of all rows.
# - By combining with groupby() function, other columns (or rows) can be used as references for cumulative addition.

df['cumsum_2'] = df[['group', 'value_2']].groupby('group').cumsum() 
df

# [link](https://github.com/fyx1009/Stats507/blob/main/pandas_notes/pd_topic_fengyx.py)

# ## Stack and Unstack
# **Heather Johnston**
#
# **hajohns@umich.edu**
#
# *Stats 507, Pandas Topics, Fall 2021*
#
# ### About stack and unstack
# * Stack and Unstack are similar to "melt" and "pivot" methods for transforming data
# * R users may be familiar with "pivot_wider" and "pivot_longer" (formerly "spread" and "gather")
# * Stack transforms column names to new index and values to column
#
# ### Example: Stack
# * Consider the `example` DataFrame below to be measurements of some value taken on different days at different times.
# * It would be natural to want these to be "gathered" into long format, which we can do using `stack`

example = pd.DataFrame({"day":["Monday", "Wednesday", "Friday"],
                        "morning":[4, 5, 6],
                        "afternoon":[8, 9, 0]})
example.set_index("day", inplace=True)
print(example)
print(example.stack())

# ### Example: Unstack
# * Conversely, for displaying data, it's often handy to have it in a wider format
# * Unstack is especially convenient after using `groupby` on a dataframe

rng = np.random.default_rng(100)
long_data = pd.DataFrame({"group":["a", "a", "a", "a", "b", "b", "b", "b"],
                          "program":["x", "y", "x", "y", "x", "y", "x", "y"],
                         "score":rng.integers(0, 100, 8),
                         "value":rng.integers(0, 20, 8)
                         })
long_data.groupby(["group", "program"]).mean()
long_data.groupby(["group", "program"]).mean().unstack()


# ## Pandas Query ##
#
# ### pd. query ##
#
# ###### Name: Anandkumar Patel
# ###### Email: patelana@umich.edu
# ###### Unique ID: patelana
#
# ### Arguments and Output
#
# **Arguments** 
#
# * expression (expr) 
# * inplace (default = False) 
#     * Do you want to operate directly on the dataframe or create new one
# * kwargs (keyword arguments)
#
# **Returns** 
# * Dataframe from provided query
#
# ## Why
#
# * Similar to an SQL query 
# * Can help you filter data by querying
# * Returns a subset of the DataFrame
# * loc and iloc can be used to query either rows or columns
#
# ## Query Syntax
#
# * yourdataframe.query(expression, inplace = True/False
#
# ## Code Example

import pandas as pd 
import numpy as np


import pandas as pd
import numpy as np
### Q0 code example

# created from arrays or tuples

arrays = [["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
          ["one", "two", "one", "two", "one", "two", "one", "two"]]
tuples = list(zip(*arrays)) # if from arrays, this step is dropped
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"]) 
# if from arrays, use pd.MultiIndex.from_arrays()

df1 = pd.Series(np.random.randn(8), index=index)

# created from product

iterables = [["bar", "baz", "foo", "qux"], ["one", "two"]]
df2 = pd.MultiIndex.from_product(iterables, names=["first", "second"])

#created directly from dataframe
df3 = pd.DataFrame([["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],
                  columns=["first", "second"])
pd.MultiIndex.from_frame(df)

# Basic Operation and Reindex

df1 + df1[:2]
df1 + df1[::2]

df1.reindex(index[:3])
df1.reindex([("foo", "two"), ("bar", "one"), ("qux", "one"), ("baz", "one")])

#Advanced Indexing 
df1 = df1.T
df1.loc[("bar", "two")]


import pandas as pd
df = pd.DataFrame({'A': range(1, 6),
                   'B': range(10, 0, -2),
                   'C C': range(10, 5, -1)})
print(df)

print('Below is the results of the query')

print(df.query('A > B'))


# ## Time Series
# **Name: Lu Qin**
# UM email: qinlu@umich.edu
#
# ### Overview
#  - Data times
#  - Time Frequency
#  - Time zone
#
# ### Import

import datetime
import pandas as pd
import numpy as np


# ### Datetime
#  - Parsing time series information from various sources and formats

dti = pd.to_datetime(
    ["20/10/2021", 
     np.datetime64("2021-10-20"), 
     datetime.datetime(2021, 10, 20)]
)

dti


# ### Time frequency
# - Generate sequences of fixed-frequency dates and time spans
# - Resampling or converting a time series to a particular frequency

# #### Generate

dti = pd.date_range("2021-10-20", periods=2, freq="H")

dti


# #### convert

idx = pd.date_range("2021-10-20", periods=3, freq="H")
ts = pd.Series(range(len(idx)), index=idx)

ts


# #### resample

ts.resample("2H").mean()


# ### Timezone
#  - Manipulating and converting date times with timezone information
#  - `tz_localize()`
#  - `tz_convert()`

dti = dti.tz_localize("UTC")
dti

dti.tz_convert("US/Pacific")


# ## Window Functions ##
# **Name: Stephen Toner** \
# UM email: srtoner@umich.edu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web


# Of the many funcitons in Pandas, one which is particularly useful for time
# series analysis is the window function. It lets us apply some aggregation 
# function over a specified lookback period on a rolling basis throughout the
# time series. This is particularly useful for financial analsyis of equity
# returns, so we will compute some financial metrics for Amazon stock using
# this techinique.

# Our first step is to import our data for Amazon ("AMZN") 
# over a healthy time horizon:

amzn_data = web.DataReader("AMZN", 
                           data_source = 'yahoo', 
                           start = "2016-10-01", 
                           end = "2021-10-01")

amzn_data.head()


# While the column labels are largely self-explanatory, two important notes
# should be made:
# * The adjusted close represents the closing price after all is said and done
# after the trading session ends; this may represent changes due to accounts 
# being settled / netted against each other, or from adjustments to financial
# reporting statements.
# * One reason for our choice in AMZN stock rather than others is that AMZN
# has not had a stock split in the last 20 years; for this reason we do not
# need to concern ourselves with adjusting for the issuance of new shares like
# we would for TSLA, AAPL, or other companies with large
# market capitalization.

# Getting back to Pandas, we have three main functions that allow us to
# perform Window operations:
# * `df.shift()`: Not technically a window operation, but helpful for
# computing calculations with offsets in time series
# * `rolling`: For a given fixed lookback period, tells us the 
# aggregation metric (mean, avg, std dev)
# * `expanding`: Similar to `rolling`, but the lookback period is not fixed. 
# Helpful when we want to have a variable lookback period such as "month to 
# date" returns

# Two metrics that are often of interest to investors are the returns of an
# asset and the volume of shares traded. Returns are either calculated on
# a simple basis:
# $$ R_s = P_1/P_0 -1$$
# or a log basis:
# $$ R_l = \log (P_1 / P_2) $$
# Simple returns are more useful when aggregating returns across multiple 
# assets, while Log returns are more flexible when looking at returns across 
# time. As we are just looking at AMZN, we will calculate the log returns
# using the `shift` function:

amzn_data["l_returns"] = np.log(amzn_data["Adj Close"]/
                                amzn_data["Adj Close"].shift(1))


plt.title("Log Returns of AMZN")
plt.plot(amzn_data['l_returns'])


# For the latter, we see that the
# volume of AMZN stock traded is quite noisy:

plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['Volume'])


# If we want to get a better picture of the trends, we can always take a
# moving average of the last 5 days (last full set of trading days):

amzn_data["vol_5dma"] = amzn_data["Volume"].rolling(window = 5).mean()
plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['vol_5dma'])


# When we apply this to a price metric, we can identify some technical patterns
# such as when the 15 or 50 day moving average crosses the 100 or 200 day
# moving average (known as the golden cross, by those who believe in it).

amzn_data["ma_15"] = amzn_data["Adj Close"].rolling(window = 15).mean()
amzn_data["ma_100"] = amzn_data["Adj Close"].rolling(window = 100).mean()

fig1 = plt.figure()
plt.plot(amzn_data["ma_15"])
plt.plot(amzn_data["ma_100"])
plt.title("15 Day MA vs. 100 Day MA")

# We can then use the `shift()` method to identify which dates have 
# golden crosses

gc_days = (amzn_data.eval("ma_15 > ma_100") & 
               amzn_data.shift(1).eval("ma_15 <= ma_100"))

gc_prices = amzn_data["ma_15"][gc_days]


fig2 = plt.figure()
plt.plot(amzn_data["Adj Close"], color = "black")
plt.scatter( x= gc_prices.index, 
                y = gc_prices[:],
                marker = "+", 
                color = "gold" 
                )

plt.title("Golden Crosses & Adj Close")


# The last feature that Pandas offers is a the `expanding` window function, 
# which calculates a metric over a time frame that grows with each additional 
# period. This is particularly useful for backtesting financial metrics
# as indicators of changes in equity prices: because one must be careful not
# to apply information from the future when performing backtesting, the 
# `expanding` functionality helps ensure we only use information up until the 
# given point in time. Below, we use the expanding function to plot cumulative
# return of AMZN over the time horizon.

def calc_total_return(x):
    """    
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.log(x[-1] / x[0]) 


amzn_data["Total Returns"] = (amzn_data["Adj Close"]
                              .expanding()
                              .apply(calc_total_return))

fig3 = plt.figure()
ax5 = fig3.add_subplot(111)
ax5 = plt.plot(amzn_data["Total Returns"])
plt.title("Cumulative Log Returns for AMZN")


# * ###  Processing Time Data
#
# **Yurui Chang**
#
# #### Pandas.to_timedelta
#
# - To convert a recognized timedelta format / value into a Timedelta type
# - the unit of the arg
#   * 'W'
#   * 'D'/'days'/'day'
#   * ‘hours’ / ‘hour’ / ‘hr’ / ‘h’
#   * ‘m’ / ‘minute’ / ‘min’ / ‘minutes’ / ‘T’
#   * ‘S’ / ‘seconds’ / ‘sec’ / ‘second’
#   * ‘ms’ / ‘milliseconds’ / ‘millisecond’ / ‘milli’ / ‘millis’ / ‘L’
#   * ‘us’ / ‘microseconds’ / ‘microsecond’ / ‘micro’ / ‘micros’ / ‘U’
#   * ‘ns’ / ‘nanoseconds’ / ‘nano’ / ‘nanos’ / ‘nanosecond’ / ‘N’
#
# * Parsing a single string to a Timedelta
# * Parsing a list or array of strings
# * Converting numbers by specifying the unit keyword argument

time1 = pd.to_timedelta('1 days 06:05:01.00003')
time2 = pd.to_timedelta('15.5s')
print([time1, time2])
pd.to_timedelta(['1 days 06:05:01.00003', '15.5s', 'nan'])

pd.to_timedelta(np.arange(5), unit='d')


# #### pandas.to_datetime
#
# * To convert argument to datetime
# * Returns: datetime, return type dependending on input
#   * list-like: DatetimeIndex
#   * Series: Series of datetime64 dtype
#   * scalar: Timestamp
# * Assembling a datetime from multiple columns of a DataFrame
# * Converting Pandas Series to datetime w/ custom format
# * Converting Unix integer (days) to datetime
# * Convert integer (seconds) to datetime

s = pd.Series(['date is 01199002',
           'date is 02199015',
           'date is 03199020',
           'date is 09199204'])
pd.to_datetime(s, format="date is %m%Y%d")

time1 = pd.to_datetime(14554, unit='D', origin='unix')
print(time1)
time2 = pd.to_datetime(1600355888, unit='s', origin='unix')
print(time2)


# # Title: Pandas Time Series Analysis
# ## Name: Kenan Alkiek (kalkiek)


from matplotlib import pyplot as plt

# Read in the air quality dataset
air_quality = pd.read_csv(
    'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2_long.csv')
air_quality["datetime"] = pd.to_datetime(air_quality["date.utc"])

# One common method of dealing with time series data is to set the index equal to the data
air_quality = air_quality.set_index('datetime')
air_quality.head()

# Plot the NO2 Over time for Paris france
paris_air_quality = air_quality[(air_quality['city'] == 'Paris') & (air_quality['country'] == 'FR')]

paris_air_quality.plot()
plt.ylabel("$NO_2 (µg/m^3)$")

# Plot average NO2 by hour of the day
fig, axs = plt.subplots(figsize=(12, 4))
air_quality.groupby("date.utc")["value"].mean().plot(kind='bar', rot=0, ax=axs)
plt.xlabel("Hour of the day")
plt.ylabel("$NO_2 (µg/m^3)$")
plt.show()

# Limit the data between 2 dates
beg_of_june = paris_air_quality["2019-06-01":"2019-06-03"]
beg_of_june.plot()
plt.ylabel("$NO_2 (µg/m^3)$")

# Resample the Data With a Different Frequency (and Aggregration)
monthly_max = air_quality.resample("M").max()
print(monthly_max)

# Ignore weekends and certain times
rng = pd.date_range('20190501 09:00', '20190701 16:00', freq='30T')

# Grab only certain times
rng = rng.take(rng.indexer_between_time('09:30', '16:00'))

# Remove weekends
rng = rng[rng.weekday < 5]

rng.to_series()


# ## Pivot Table in pandas
#
#
# *Mingjia Chen* 
# mingjia@umich.edu
#
# - A pivot table is a table format that allows data to be dynamically arranged and summarized in categories.
# - Pivot tables are flexible, allowing you to customize your analytical calculations and making it easy for users to understand the data.
# - Use the following example to illustrate how a pivot table works.


import numpy as np

df = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                   "B": [0, 1, 0, 1, 0],
                   "C": [1, 2, 2, 3, 3],
                   "D": [2, 4, 5, 5, 6],
                   "E": [2, 2, 4, 4, 6]})
print(df)


# ## Index
#
# - The simplest pivot table must have a data frame and an index.
# - In addition, you can also have multiple indexes.
# - Try to swap the order of the two indexes, the data results are the same.

tab1 = pd.pivot_table(df,index=["A"])
tab2 = pd.pivot_table(df,index=["A", "B"])
tab3 = pd.pivot_table(df,index=["B", "A"])
print(tab1)
print(tab2)
print(tab3)


# ## Values 
# - Change the values parameter can filter the data for the desired calculation.


pd.pivot_table(df,index=["B", "A"], values=["C", "D"])


# ## Aggfunc
#
# - The aggfunc parameter sets the function that we perform when aggregating data.
# - When we do not set aggfunc, it defaults aggfunc='mean' to calculate the mean value.
#   - When we also want to get the sum of the data under indexes:


pd.pivot_table(df,index=["B", "A"], values=["C", "D"], aggfunc=[np.sum,np.mean])


# ## Columns
#
# - columns like index can set the column hierarchy field, it is not a required parameter, as an optional way to split the data.
#
# - fill_value fills empty values, margins=True for aggregation


pd.pivot_table(df,index=["B"],columns=["E"], values=["C", "D"],
               aggfunc=[np.sum], fill_value=0, margins=1)


#
# Ziyi Gao
#
# ziyigao@umich.edu
#
# ## Multi-indexing
#
# - Aiming at sophisticated data analysis and manipulation, especially for working with higher dimensional data
# - Enabling one to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures
#
# ## Creating a multi-indexing dataframe and Reconstructing
#
# - It can be created from:
#     - a list of arrays (using MultiIndex.from_arrays())
#     - an array of tuples (using MultiIndex.from_tuples())
#     - a crossed set of iterables (using MultiIndex.from_product())
#     - a DataFrame (using MultiIndex.from_frame())
# - The method get_level_values() will return a vector of the labels for each location at a particular level
#
# ## Basic Indexing
#
# - Advantages of hierarchical indexing
#     - hierarchical indexing can select data by a “partial” label identifying a subgroup in the data
# - Defined Levels
#     - keeps all the defined levels of an index, even if they are not actually used
#     
# ## Data Alignment and Using Reindex
#
# - Operations between differently-indexed objects having MultiIndex on the axes will work as you expect; data alignment will work the same as an Index of tuples
# - The reindex() method of Series/DataFrames can be called with another MultiIndex, or even a list or array of tuples:
#
# ## Some Advanced Indexing
#
# Syntactically integrating MultiIndex in advanced indexing with .loc is a bit challenging
#
# - In general, MultiIndex keys take the form of tuples

# ## Missing Data in pandas
#
# #### Anuraag Ramesh: anuraagr@umich.edu
#
# - About
# - Calculations with missing data
# - Filling missing values
# - Interpolation
# - Replacing generic values

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from IPython.core.display import HTML,display
import random
import scipy.stats as sci
#------------------------------------------------------------------------------

# ## About
#
# Missing values are always present in datasets that are developed from the 
# real world , and it is important to understand the functions and 
# methods that are present to deal with them properly.

df = pd.DataFrame({'Name' : ['A' , 'B', 'C', 'D', 'E'],
                   'Score 1' :[90, 85, 86, 67, 45],
                   'Score 2' :[None , 78, 89, 56, 99], 
                   'Score 3' :[80, None , None, 56, 82],
                   'Score 4' : [68, 79, None , 26, 57]})

df

# ### Defining missing values

# In the dataset defined above, we can see that 
# there are few "NaN" of missing values.  
# - The missing or not avialable value is defined using `np.nan`.
# - We can find the missing values in a dataset using `isna()`. 
# The values that show 'True' are missing in the dataset
# - On the other hand, to find if a value is not null we use `notna()`

print(df.isna())
print('\n')
print(df.notna())

# - We can also use `np.nan()` as a 
# parameter to compare various values  
# - Using `isna()` to find the missing values in each column

print(df['Score 1'].isna())
print(df['Score 2'].isna())

# ## Calculations with missing data

# There is missing values in our dataset. But 
# there are several different ways we can 
# handle this to perform calculations.
#
# Suppose, we want to calculate 
# the average of scores for each person. 
# We can use these three methods.
# - Skip the missing values
# - Drop the column with missing values
# - Fill in the missing values with some other value
#
# Note : "NA'" values are automatically excluded while using groupby

# Skipping missing values
print(df.mean(skipna = True, axis = 1))

# +
# Dropping columns or rows with missing values

print(df.dropna(axis = 0)) #Row
print("\n")
print(df.dropna(axis = 1)) #Column
# -

# ## Filling missing values
#
# We can fill the missing values using different methods:
#
# - Filling missing values with 0
# - Filling missing values with a string - eg. NA
# - Filling missing with values with values 
# appearing before or after
# - Filling values with mean of a column

# Filling values with 0
df.fillna(0)
# Filling values with a string
df.fillna("NA")


# Filling values with values appearing after the
# missing values
df.fillna(method = "pad")

# Filling values with mean of individual columns
print(df.fillna(df.mean()))

# ## Interpolation
#
# This is the process of performing linear interpolation 
# to give an expectation assumption of missing values.
#
# There are several different methods of interpolation
#
# - linear : default method
# - quadratic
# - pchip
# - akima
# - spline
# - polynomial

df.interpolate()

df.interpolate(method = "akima")

# Below, we can see that the missing values in 
# `Score 3` is replaced by 55 and 45 respectively

df.interpolate(method = "quadratic")

# ## Replacing generic values
#
# We can simply replace the NaN values from the outside,
# by using `.replace()`  
#
# Here, we can assume and replace the value with a random
# value with 75.

df.replace(np.nan, 75)

# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

# # Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
#
# + [Windows operations](#Windows-operations)
# + [Pandas Idiom: Splitting](#Pandas-Idiom:-Splitting)
# + [Time Series](#Time-Series)
# + [Pandas pipeline](#Pandas-pipeline)
# + [Missing Data in Pandas](#Missing-Data-in-Pandas)
# + [Hierarchical Indexing](#Hierarchical-Indexing)
# + [Introduction to pandas.DataFrame.fillna()](#Introduction-to-pandas.DataFrame.fillna())
# + [Missing Data Cleaning](#Missing-Data-Cleaning)
# + [pandas.DataFrame.Insert()](#pandas.DataFrame.Insert())
# + [Pandas sort_values() tutorial](#Pandas-sort_values()-tutorial)





# # Windows operations
# *Tiejin Chen*; **tiejin@umich.edu**
#

# - In the region of data science, sometimes we need to manipulate
#   one raw with two raws next to it for every raw.
# - This is one kind of windows operation.
# - We define windows operation as an operation that
#   performs an aggregation over a sliding partition of values (from pandas' userguide)
# - Using ```df.rolling``` function to use the normal windows operation

rng = np.random.default_rng(9 * 2021 * 20)
n=5
a = rng.binomial(n=1, p=0.5, size=n)
b = 1 - 0.5 * a + rng.normal(size=n)
c = 0.8 * a + rng.normal(size=n)
df = pd.DataFrame({'a': a, 'b': b, 'c': c})
print(df)
df['b'].rolling(window=2).sum()

# ## Rolling parameter
# In ```rolling``` method, we have some parameter to control the method, And we introduce two:
# - center: Type bool; if center is True, Then the result will move to the center in series.
# - window: decide the length of window or the customed window

df['b'].rolling(window=3).sum()

df['b'].rolling(window=3,center=True).sum()

df['b'].rolling(window=2).sum()

# example of customed window

window_custom = [True,False,True,False,True]
from pandas.api.indexers import BaseIndexer
class CustomIndexer(BaseIndexer):
    def get_window_bounds(self, num_values, min_periods, center, closed):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            if self.use_expanding[i]:
                start[i] = 0
                end[i] = i + 1
            else:
                start[i] = i
                end[i] = i + self.window_size
        return start, end
indexer1 = CustomIndexer(window_size=1, use_expanding=window_custom)
indexer2 = CustomIndexer(window_size=2, use_expanding=window_custom)

df['b'].rolling(window=indexer1).sum()

df['b'].rolling(window=indexer2).sum()

# ## Windows operation with groupby
# - ```pandas.groupby``` type also have windows operation method,
#   hence we can combine groupby and windows operation.
# - we can also use ```apply``` after we use ```rolling```

df.groupby('a').rolling(window=2).sum()


def test_mean(x):
    return x.mean()
df['b'].rolling(window=2).apply(test_mean)













# # Pandas Idiom: Splitting
# Sean Kelly, seankell@umich.edu
#
# + [Splitting to analyze data](#Splitting-to-analyze-data)
# + [Splitting to create new Series](#Splitting-to-create-new-Series)
# + [Takeaways](#Takeaways)
#

# ## Pandas Idiom: Splitting
#
# - A useful way to utilize data is by accessing individual rows or groups of 
# rows and operating only on those rows or groups.  
# - A common way to access rows is indexing using the `loc` or `iloc` methods 
# of the dataframe. This is useful when you know what row indices you'd like to
# access.  
# - However, it is often required to subset a given data set based on some 
# criteria that we want each row of the subset to meet.  
# - We will look at selecting subsets of rows by splitting data based on row 
# values and performing analysis or calculations after splitting.
#
# ## Splitting to analyze data
#
# - Using data splitting makes it simple to create new dataframes representing 
# subsets of the initial dataframes
# - Find the average of one column of a group defined by another column

t_df = pd.DataFrame(
    {"col0":np.random.normal(size=10),
     "col1":np.random.normal(loc=10,scale=100,size=10),
     "col2":np.random.uniform(size=10)}
    )
t_below_average_col1 = t_df[t_df["col1"] < 10]
t_above_average_col1 = t_df[t_df["col1"] >= 10]
print([np.round(t_above_average_col1["col0"].mean(),4),
      np.round(t_below_average_col1["col0"].mean(),4)])

# ## Splitting to create new Series
#
# - We can use this splitting method to convert columns to booleans based on 
# a criterion we want that column to meet, such as converting a continuous 
# random variable to a bernoulli outcome with some probability p.

p = 0.4
t_df["col0_below_p"] = t_df["col2"] < p
t_df

# ## Takeaways
#
# - Splitting is a powerful but simple idiom that allows easy grouping of data
# for analysis and further calculations.  
# - There are many ways to access specific rows of your data, but it is
# important to use the right tool for the job.  
# - More information on splitting can be found [here][splitting].  
#
# [splitting]: https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#splitting











# # Time Series
# Name: Yu Chi
# UM email: yuchi@umich.edu

# - The topic I picked is Time Series in pandas, specifically about time zone
#  representation.
# - Pandas has simple functionality for performing resampling operations during
#  frequency conversion (e.g., converting secondly data into 5-minutely data).
# - This can be quite helpful in financial applications.
#
# - First we construct the range and how frequent we want to stamp the time.
# - `rng = pd.date_range("10/17/2021 00:00", periods=5, freq="D")`
# - In this example, the starting time is 00:00 on 10/17/2021, the frequency
#  is one day, and the period is 5 days long.
# - Now we can consstruct the time representation.
# - `ts = pd.Series(np.random.randn(len(rng)), rng)`
# - If we try printing out ts, it should look like the following:

rng = pd.date_range("10/17/2021 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)

# - Then we set up the time zone. In this example, I'll set it to UTC
#  (Coordinated Universal Time).
# - `ts_utc = ts.tz_localize("UTC")`
# - If we try printing out ts_utc, it should look like the following:

ts_utc = ts.tz_localize("UTC")
print(ts_utc)

# - If we want to know what the time is in another time zone, it can easily
#  done as the following:
# - In this example, I want to convert the time to EDT (US Eastern time).
# - `ts_edt = ts_utc.tz_convert("US/Eastern")`
# - Let's try printing out ts_edt:

ts_edt = ts_utc.tz_convert("US/Eastern")
print(ts_edt)











# # Pandas pipeline
#
# ## Overview
#
# Name: Jiale Zha
#
# Email: jialezha@umich.edu
#  
# - [About Pipeline](#About-Pipeline)
#
# - [API](#API)
#
# - [Examples](#Examples)
#
# - [Takeaways](#Takeaways)

#

# ## About Pipeline
#
# A common situation in our data analyses is that we need the output of a function to be one of the input of another function. Pipiline is just the concept for that situation as it means we could regard those functions as pipes and connect them, let the data stream go through them to get the final result.

#

# ## API
#
# The pipeline function in pandas could be used for Series and DataFrame, the general API for it is,
#
# `pandas.Series.pipe(func, *args, **kwargs)`
#
# `pandas.DataFrame.pipe(func, *args, **kwargs)`
#
# where the input parameter `func` is the function to apply next, `args` are positional arguments of the function, and `kwargs` is a dictionary of keyword arguments.

#

# ## Examples
#
# A very common example for pipeline is the computation of composition function, say if we want to compute the result of the following function, 
#
# `f_3(f_2(f_1(df), arg1=a), arg2=b, arg3=c)`

# A more readable code for the above function will be 
#
# `(df.pipe(f_1)                 
#     .pipe(f_2, arg1=a)         
#     .pipe(f_3, arg2=b, arg3=c)`

#

# In practice, if we have the following data, and we want to normalize it, we could use the pipe function to process it step by step.

data = pd.DataFrame(
    {'math':[96, 95, 25, 34],
     'stats': [88, 46, 23, 100],
    'computer': [86, 93, 34, 34]})
data

# We normalize the data by subtracting its mean and dividing its standard deviation.

(data
 # Compute the mean
 .pipe(pd.DataFrame.mean)
 # Subtract the mean, which is the 'other' parameter in the subtraction function
 .pipe((pd.DataFrame.sub, 'other'), data) 
 # Divided by the standard deviation of the original data
 .pipe(pd.DataFrame.div, data.std()))

#

# ### Takeaways
#
# - Use pipe method to do the multi-step data processing
#
# - Combine the pipe method with the other basic method in pandas













# # Missing Data in Pandas
#
# Shihao Wu, PhD student in statistics
#
# Reference: [https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
#
# There are 4 "slides" for this topic.
#
#
# ## Missing data
# Missing data arises in various circumstances in statistical analysis. Consider the following example:

# generate a data frame with float, string and bool values
df = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["1", "2", "3"],
)
df['4'] = "bar"
df['5'] = df["1"] > 0

# reindex so that there will be missing values in the data frame
df2 = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])

df2


# The missing values come from unspecified rows of data.

# ## Detecting missing data
#
# To make detecting missing values easier (and across different array dtypes), pandas provides the <code>isna()</code> and <code>notna()</code> functions, which are also methods on Series and DataFrame objects:

df2["1"]


pd.isna(df2["1"])


df2["4"].notna()


df2.isna()


# ## Inserting missing data
#
# You can insert missing values by simply assigning to containers. The actual missing value used will be chosen based on the dtype.
#
# For example, numeric containers will always use <code>NaN</code> regardless of the missing value type chosen:

s = pd.Series([1, 2, 3])
s.loc[0] = None
s


# Because <code>NaN</code> is a float, a column of integers with even one missing values is cast to floating-point dtype. pandas provides a nullable integer array, which can be used by explicitly requesting the dtype:

pd.Series([1, 2, np.nan, 4], dtype=pd.Int64Dtype())


# Likewise, datetime containers will always use <code>NaT</code>.
#
# For object containers, pandas will use the value given:

s = pd.Series(["a", "b", "c"])
s.loc[0] = None
s.loc[1] = np.nan
s


# ## Calculations with missing data
#
# Missing values propagate naturally through arithmetic operations between pandas objects.

a = df2[['1','2']]
b = df2[['2','3']]
a + b


# Python deals with missing value for data structure in a smart way. For example:
#
# * When summing data, NA (missing) values will be treated as zero.
# * If the data are all <code>NA</code>, the result will be 0.
# * Cumulative methods like <code>cumsum()</code> and <code>cumprod()</code> ignore <code>NA</code> values by default, but preserve them in the resulting arrays. To override this behaviour and include <code>NA</code> values, use <code>skipna=False</code>.

df2


df2["1"].sum()


df2.mean(1)


df2[['1','2','3']].cumsum()


df2[['1','2','3']].cumsum(skipna=False)


# Missing data is ubiquitous. Dealing with missing is unavoidable in data analysis. This concludes my topic here.












# # Hierarchical Indexing
#
# - Shushu Zhang
# - shushuz@umich.edu
# - Reference is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) 
#
# Hierarchical / Multi-level indexing is very exciting as it opens the door to some quite sophisticated data analysis and manipulation, especially for working with higher dimensional data. In essence, it enables you to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures like Series (1d) and DataFrame (2d).

# ### Creating a MultiIndex (hierarchical index) object
# - The MultiIndex object is the hierarchical analogue of the standard Index object which typically stores the axis labels in pandas objects. You can think of MultiIndex as an array of tuples where each tuple is unique. 
# - A MultiIndex can be created from a list of arrays (using MultiIndex.from_arrays()), an array of tuples (using MultiIndex.from_tuples()), a crossed set of iterables (using MultiIndex.from_product()), or a DataFrame (using MultiIndex.from_frame()). 
# - The Index constructor will attempt to return a MultiIndex when it is passed a list of tuples.

# Constructing from an array of tuples
arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]
tuples = list(zip(*arrays))
tuples
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
index

# ### Manipulating the dataframe with MultiIndex
#
# - Basic indexing on axis with MultiIndex is illustrated as below.
# - The MultiIndex keeps all the defined levels of an index, even if they are not actually used.

# Use the MultiIndex object to construct a dataframe 
df = pd.DataFrame(np.random.randn(3, 8), index=["A", "B", "C"], columns=index)
print(df)
df['bar']

#These two indexing are the same
print(df['bar','one'])
print(df['bar']['one'])

print(df.columns.levels)  # original MultiIndex
print(df[["foo","qux"]].columns.levels)  # sliced

# ### Advanced indexing with hierarchical index
# - MultiIndex keys take the form of tuples. 
# - We can use also analogous methods, such as .T, .loc. 
# - “Partial” slicing also works quite nicely.

df = df.T
print(df)
print(df.loc[("bar", "two")])
print(df.loc[("bar", "two"), "A"])
print(df.loc["bar"])
print(df.loc["baz":"foo"])


# ### Using slicers
#
# - You can slice a MultiIndex by providing multiple indexers.
#
# - You can provide any of the selectors as if you are indexing by label, see Selection by Label, including slices, lists of labels, labels, and boolean indexers.
#
# - You can use slice(None) to select all the contents of that level. You do not need to specify all the deeper levels, they will be implied as slice(None).
#
# - As usual, both sides of the slicers are included as this is label indexing.

# +
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]


miindex = pd.MultiIndex.from_product(
    [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
)


micolumns = pd.MultiIndex.from_tuples(
    [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
)


dfmi = (
    pd.DataFrame(
        np.arange(len(miindex) * len(micolumns)).reshape(
            (len(miindex), len(micolumns))
        ),
        index=miindex,
        columns=micolumns,
    )
    .sort_index()
    .sort_index(axis=1)
)

print(dfmi)
print(dfmi.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :])
# + [What is fillna method?](#What-is-fillna-method?)


# + [What parameter does this method have?](#What-parameter-does-this-method-have?)

# -









# + [How to use & Examples](#How-to-use-&-Examples) [markdown]
#
# # Introduction to pandas.DataFrame.fillna()
#
# **Xinfeng Liu(xinfengl@umich.edu)**
#
# ## What is fillna method?
#
# * pandas.DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)
# * Used to Fill NA/NaN values using the specified method.
#
# ## What parameter does this method have?
#
# * value(=None by default)
#    * value to use to fill null values
#    * could be a scalar or a dict/series/dataframe of values specifying which value to use for each index
# * method(=None by default)
#    * method to use for filling null values
#    * 'bfill'/'backfill': fill the null from the next valid obersvation
#    * 'ffill'/'pad': fill the null from the previous valid obersvation
# * axis(=None by default)
#    * fill null values along index(=0) or along columns(=1)
# * inplace(=False by default)
#    * if = True, fill in-place, and will not create a copy slice for a column in a DataFrame
# * limit(=None by default)
#    * a integer used to specify the maximum number of consecutive NaN values to fill. If there's a gap with more than this number of consecutive NaNs, it will be partially filled
# * downcast(=None by default)
#    * a dictionary of item->dtype of what to downcast if possible
#
# ## How to use & Examples

# +
import pandas as pd
import numpy as np

#create a dataframe with NaN
# this data frame represents the apple's hourly sales vlue 
# from 9am to 4pm in a store and it has some null values
df = pd.DataFrame([[25, np.nan, 23, 25, np.nan, 22, 20],
                   [22, 24, 25, np.nan, 21.5, np.nan, 20],
                   [27, 24.5, 20, 21, 19.5, 25, 22],
                   [19.5, np.nan, 22, np.nan, 27, 26, 21],
                   [21, 25.5, 26, np.nan, 22, 22, np.nan],
                   [30, np.nan, np.nan, 26, 29, 27.5, 35],
                   [27, 28, 30, 35, 37, np.nan, np.nan]],
                  columns=['monday', 
                           'tuesday', 
                           'wednesday', 
                           'thursday', 
                           'friday', 
                           'saturday',
                           'sunday'])
df
# -

# Now will can fill the null value using fillna method
df.fillna(method='ffill', axis=1)

# In this example, we used the previous valid value to fill the null value along the column. This actually make sense becuase each day's sale's value during the same period should be similar to each other. Therefore, fill the null with the same value as the day before will not affact the mean or variance the whole data














# # Missing Data Cleaning
#
# #### Chen Liu
#
# *ichenliu@umich.edu*

#
#
# - No data value is stored for some variables in an observation.
# - Here is a small example dataset we'll use in these slides.

# +
example = pd.DataFrame({
    'Col_1' : [1, 2, 3, 4, np.nan],
    'Col_2' : [9, 8, 7, 6, 5]
})
print(example)

### Insert missing data
example.loc[0, 'Col_1'] = None
print(example)
# -

# ## Calculation with missing data
#
# - When summing data, NA (missing) values will be 
#   treated as $0$ defaultly.
# - When producting data, NA (missing) values will be 
#   treated as $1$ defaultly.
# - To override this behaviour and include NA (missing) 
#   values, use `skipna=False`.
# - Calculate by series, NA (missing) values will yield 
#   NA (missing) values in result.

# +
### Default
print(example.sum())
print(example.prod())

### Include NA values
print(example.sum(skipna=False))
print(example.prod(skipna=False))

### Calculate by series
print(example['Col_1'] + example['Col_2'])
# -

# ## Logical operations with missing data
# - Comparison operation will always yield `False`.
# - `==` can not be used to detect NA (missing) values.
# - `np.nan` is not able to do logical operations, while `pd.NA` can. 

# +
### Comparison operation
print(example['Col_1'] < 3)
print(example['Col_1'] >= 3)

### These codes will yield all-False series
print(example['Col_1'] == np.nan)
print(example['Col_1'] == pd.NA)

### Logical operations of pd.NA
print(True | pd.NA, True & pd.NA, False | pd.NA, False & pd.NA)
# -

# ## Detect and delete missing data
# - `isna()` will find NA (missing) values.
# - `dropna()` will drop rows having NA (missing) values.
# - Use `axis = 1` in `dropna()` to drop columns having NA (missing) values.

# +
### Detect NA values
print(example.isna())

### Drop rows / columns having NA values 
print(example.dropna())
print(example.dropna(axis=1))
# -

# ## Fill missing data
# - `fillna()` can fill in NA (missing) values with 
#   non-NA data in a couple of ways.
# - Use `method='pad / ffill'` in `fillna()` to fill gaps forward.
# - Use `method='bfill / backfill'` in `fillna()` to fill gaps backward.
# - Also fill with a PandasObject like `DataFrame.mean()`.

# +
### Fill with a single value
print(example.fillna(0))

### Fill forward
print(example.fillna(method='pad'))

### Fill backward
print(example.fillna(method='bfill'))

### Fill with mean
print(example.fillna(example.mean()))
# -












# # pandas.DataFrame.Insert()
#
# Micah Scholes
#
# mscholes@umich.edu

# - The insert command is used to insert a new column to an existing dataframe.
# - While the merge command can also add columns to a dataframe, it is better for organizing data. The insert command works for data that is already organized where a column just needs to be added.

# ## Args
#
# The arguments for Insert() are:
# - loc: an integer representing the index location where the new column should be inserted
# - column: the name of the column. This should be a unique column name unless duplicates are desired.
# - value: a list, array, series, int, etc. of values to populate the column.
# - allow_duplicates: default is False. If set to true, it will allow you to insert a column with a duplicate name to an existing column.

# ## Example

# + [Topic Title](#Topic-Title)
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8], 'b': [3, 4, 5, 6, 7, 8, 9,
                                                        10]})
df.insert(2,'c',["a", "b", "c", "d", "e", "f", "g", "h"])
df

# +
# An error is raised if a duplicate column name is attempted to be inserted without setting allow_duplicates to true

df.insert(0,'a', [5, 6, 7, 8, 9,10, 11, 12])

df.insert(0,'a', [5, 6, 7, 8, 9,10, 11, 12], True)
df

# +
# Additionally, the values have to be the same length as the other columns, otherwise we get an error.

df.insert(0,'d', [5, 6, 7, 8, 9,10, 11])

df.insert(0,'d', 1)
df
# However if only 1 value is entered, it will populate the entire column with that value.
# -













# # Pandas sort_values() tutorial
#
# Alan Hedrick, ajhedri@umich.edu
#
# ## General overview
# - Sometimes, you may need to sort your data by column </li>
# - This can be done through using the sort_values() function through pandas </li>
# - Below is a code cell creating a data frame of rows corresponding to an individuals
#   name, age, ID number, and location</li>
# - The data frame will show it's initial state, and the be sorted by name</li>
#

dataframe = pd.DataFrame({"Name": ["Alan", "Smore's", "Sparrow", "Tonks", "Marina"],                          "Age": [22, 2, 1, 5, 21],                          "ID Num": [69646200, 20000000, 86753090, 48456002, 16754598],                          "Location": ["Michigan", "Michigan", "Michigan", "Texas", "Michigan"]})

print("Original dataframe")
print(dataframe)
print("")

print("Dataframe sorted by Name")
#sort the dataframe in alphabetical order by name
dataframe.sort_values(by='Name', inplace=True)
print(dataframe)

# ## Function breakdown
#
# - In order to call the function, you only need to fill the "by" parameter </li>
# - This parameter is set to the name of the column you wish to sort by</li>
# - You may be wondering what the "inplace" parameter is doing</li>
#   - sort_values() by default returns the sorted dataframe; however, it does not update the dataframe unless "inplace" is specified to be True</li>
# - Below is an example showing this fact, notice that without "inplace" the sorted dataframe must be set equal to another

print(dataframe)
print("")
dataframe.sort_values(by="Age")
print(dataframe)
#notice how the age column has not been sorted at all
print("")

new_df = dataframe.sort_values(by="Age")
print(new_df)
#it's been sorted!
#let's check the original again

print("")
print(dataframe)
print("")
dataframe.sort_values(by="Age", inplace=True)
print(dataframe)
#both are valid ways to use the function!

# ## Sorting in descending order
# - By default, sort_values() will sort columns in ascending order, but this can be easily changed
# - To do this, set the parameter, "ascending," to False

print(dataframe)
print("")
dataframe.sort_values(by="Age", inplace = True, ascending = False)
print(dataframe)
#now it's sorted by age in descending order!

# ## Sort by multiple columns
# - To do this, merely specify more columns such as in the example below
# - This can be useful when generating plots and tables to view specific data

# +
print(dataframe)
print("")

dataframe.sort_values(by=["ID Num", "Location"], inplace = True)
print(dataframe)


# +
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]


miindex = pd.MultiIndex.from_product(
    [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
)


micolumns = pd.MultiIndex.from_tuples(
    [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
)


dfmi = (
    pd.DataFrame(
        np.arange(len(miindex) * len(micolumns)).reshape(
            (len(miindex), len(micolumns))
        ),
        index=miindex,
        columns=micolumns,
    )
    .sort_index()
    .sort_index(axis=1)
)

print(dfmi)
print(dfmi.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :])
# + [What is fillna method?](#What-is-fillna-method?)



# + [What parameter does this method have?](#What-parameter-does-this-method-have?)

# -
