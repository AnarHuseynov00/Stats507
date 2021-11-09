
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

demo2011_2012 = pd.read_sas("DEMO_G(2011-2012).XPT")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
demo2013_2014 = pd.read_sas("DEMO_H(2013-2014).XPT")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
demo2015_2016 = pd.read_sas("DEMO_I(2015-2016).XPT")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
demo2017_2018 = pd.read_sas("DEMO_J(2017-2018).XPT")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]

# +
demo2011_2012.rename(columns = {"SEQN":"id", "RIAGENDR":"gender", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2011_2012["cohort"] = ["2011-2012" for i in range(demo2011_2012.shape[0])]

demo2013_2014.rename(columns = {"SEQN":"id", "RIAGENDR":"gender", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2013_2014["cohort"] = ["2013-2014" for i in range(demo2013_2014.shape[0])]

demo2015_2016.rename(columns = {"SEQN":"id", "RIAGENDR":"gender", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2015_2016["cohort"] = ["2015-2016" for i in range(demo2015_2016.shape[0])]

demo2017_2018.rename(columns = {"SEQN":"id", "RIAGENDR":"gender", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2017_2018["cohort"] = ["2017-2018" for i in range(demo2017_2018.shape[0])]
# -

final_demo_DF = pd.DataFrame()
final_demo_DF = final_demo_DF.append(demo2011_2012, ignore_index=True)
final_demo_DF = final_demo_DF.append(demo2013_2014, ignore_index=True)
final_demo_DF = final_demo_DF.append(demo2015_2016, ignore_index=True)
final_demo_DF = final_demo_DF.append(demo2017_2018, ignore_index=True)
final_demo_DF = final_demo_DF.fillna(0)
final_demo_DF['id'] = pd.to_numeric(final_demo_DF['id'], downcast="integer")
final_demo_DF['age'] = pd.to_numeric(final_demo_DF['age'], downcast="integer")
final_demo_DF['gender'] = final_demo_DF['gender'].astype(str)
final_demo_DF['RIDSTATR'] = pd.to_numeric(final_demo_DF['RIDSTATR'], downcast="integer")
final_demo_DF['SDMVPSU'] = pd.to_numeric(final_demo_DF['SDMVPSU'], downcast="integer")
final_demo_DF['SDMVSTRA'] = pd.to_numeric(final_demo_DF['SDMVSTRA'], downcast="integer")
final_demo_DF['race and ethnicity'] = final_demo_DF['race and ethnicity'].astype(str)
final_demo_DF['education'] = pd.to_numeric(final_demo_DF['education'], downcast="integer")
final_demo_DF['martial status'] = final_demo_DF['martial status'].astype(str)

final_demo_DF.to_pickle("./final_Demo_modified_dataset.pkl")
