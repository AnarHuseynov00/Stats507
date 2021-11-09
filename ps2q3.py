import numpy as np
import time
import pandas as pd

demo2011_2012 = pd.read_sas("DEMO_G(2011-2012).XPT")[["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
demo2013_2014 = pd.read_sas("DEMO_H(2013-2014).XPT")[["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
demo2015_2016 = pd.read_sas("DEMO_I(2015-2016).XPT")[["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
demo2017_2018 = pd.read_sas("DEMO_J(2017-2018).XPT")[["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
                                                      "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]

# +
# Then, we rename the first 5 columns (not others since there are no short explanations for them. For now it is better to keep 
# them in shorter code version). Also we add "cohort" column to all data frames
demo2011_2012.rename(columns = {"SEQN":"id", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2011_2012["cohort"] = ["2011-2012" for i in range(demo2011_2012.shape[0])]

demo2013_2014.rename(columns = {"SEQN":"id", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2013_2014["cohort"] = ["2013-2014" for i in range(demo2013_2014.shape[0])]

demo2015_2016.rename(columns = {"SEQN":"id", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2015_2016["cohort"] = ["2015-2016" for i in range(demo2015_2016.shape[0])]

demo2017_2018.rename(columns = {"SEQN":"id", "RIDAGEYR":"age", "RIDRETH3":"race and ethnicity", 
                                "DMDEDUC2":"education", "DMDMARTL":"martial status"}, inplace = True)
demo2017_2018["cohort"] = ["2017-2018" for i in range(demo2017_2018.shape[0])]
# -

# We create a single dataframe to collect all dataframes
final_demo_DF = pd.DataFrame()

final_demo_DF = final_demo_DF.append(demo2011_2012, ignore_index=True)
final_demo_DF = final_demo_DF.append(demo2013_2014, ignore_index=True)
final_demo_DF = final_demo_DF.append(demo2015_2016, ignore_index=True)
final_demo_DF = final_demo_DF.append(demo2017_2018, ignore_index=True)

# we clear all nan values and replace them with 0's
final_demo_DF = final_demo_DF.fillna(0)

# we convert columns to appropriate types
final_demo_DF['id'] = pd.to_numeric(final_demo_DF['id'], downcast="integer")
final_demo_DF['age'] = pd.to_numeric(final_demo_DF['age'], downcast="integer")
final_demo_DF['RIDSTATR'] = pd.to_numeric(final_demo_DF['RIDSTATR'], downcast="integer")
final_demo_DF['SDMVPSU'] = pd.to_numeric(final_demo_DF['SDMVPSU'], downcast="integer")
final_demo_DF['SDMVSTRA'] = pd.to_numeric(final_demo_DF['SDMVSTRA'], downcast="integer")
final_demo_DF['race and ethnicity'] = final_demo_DF['race and ethnicity'].astype(str)
final_demo_DF['education'] = final_demo_DF['education'].astype(str)
final_demo_DF['martial status'] = final_demo_DF['martial status'].astype(str)

# Here is the types
print(final_demo_DF.dtypes)

# we save the new single dataframe
final_demo_DF.to_pickle("./final_Demo_dataset.pkl")

# we check it again
unpickled_df = pd.read_pickle("./final_Demo_dataset.pkl")

# we check the shape
print(final_demo_DF.shape)

# > So there are $39156$ data points in demographics dataset

# we do the same for oral and dental data. First we should get all needed keys
Target_keys = ["SEQN", "OHDDESTS"]
all_keys = pd.read_sas("OHXDEN_G(2011-2012).XPT").keys()
for i in range(len(all_keys)):
    if all_keys[i][0:3] == "OHX" and (all_keys[i][-2:] == "TC" or all_keys[i][-3:] == "CTC"):
        Target_keys.append(all_keys[i])

# We read the data
oral_dental_2011_2012 = pd.read_sas("OHXDEN_G(2011-2012).XPT")[Target_keys]
oral_dental_2013_2014 = pd.read_sas("OHXDEN_H(2013-2014).XPT")[Target_keys]
oral_dental_2015_2016 = pd.read_sas("OHXDEN_I(2015-2016).XPT")[Target_keys]
oral_dental_2017_2018 = pd.read_sas("OHXDEN_J(2017-2018).XPT")[Target_keys]

# we create a new dictionary to change column names
label_rep = {}
for i in range(len(Target_keys)):
    if Target_keys[i][-2:] == "TC":
        label_rep[Target_keys[i]] = "tooth counts " + Target_keys[i][-4:-2]
    if Target_keys[i][-3:] == "CTC":
        label_rep[Target_keys[i]] = "coronal cavities " + Target_keys[i][-5:-3]

# we change column names
oral_dental_2011_2012.rename(columns=label_rep, inplace = True)
oral_dental_2013_2014.rename(columns=label_rep, inplace = True)
oral_dental_2015_2016.rename(columns=label_rep, inplace = True)
oral_dental_2017_2018.rename(columns=label_rep, inplace = True)

# We create final oral and dent data frame
final_oral_dent_data = pd.DataFrame()

# we collect all dataframes into single dataframe
final_oral_dent_data = final_oral_dent_data.append(oral_dental_2011_2012, ignore_index=True)
final_oral_dent_data = final_oral_dent_data.append(oral_dental_2013_2014, ignore_index=True)
final_oral_dent_data = final_oral_dent_data.append(oral_dental_2015_2016, ignore_index=True)
final_oral_dent_data = final_oral_dent_data.append(oral_dental_2017_2018, ignore_index=True)

# we clear data frame from nan's and replace them with 0's
final_oral_dent_data = final_oral_dent_data.fillna(0)

# we convert all numeric types to int32 (applicable ones)
final_oral_dent_data = final_oral_dent_data.astype('int32', errors="ignore")

print(final_oral_dent_data.dtypes)

# we save data
final_oral_dent_data.to_pickle("./final_oral_dent_dataset.pkl")

# we check data
unpickled_df = pd.read_pickle("./final_oral_dent_dataset.pkl")

# we print the shape
print(final_oral_dent_data.shape)

# > So there are $35909$ data points in oral and dent dataset