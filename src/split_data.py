"""

The script obtains correlation heatmap, adds feature 
& executes train-test splitting 

"""

from utility import create_log, parse_config, read_data
from split_data_util import analyze_corr, split_data

create_log("split_data.log")  # Creating log file

##################################################
#-----------------Reading Dataset-----------------
##################################################

config_path = "../config/config.yaml"   
config = parse_config(config_path)   # read config file
data_path = config["split_data"]["data"]   # read dataset
df = read_data(data_path)


#######################################
#---------Analyzing Correlation--------
#######################################

analyze_corr(df, "Churn")


#%%
# Adding new feature
df["feature_2"] = df["MonthlyCharges"]**(0.2) + df["SeniorCitizen"]**(0.35) - df["OnlineSecurity"]**(0.1)

#%%
#######################################
#---------- Train-Test Split ----------
#######################################

split_data(df, "Churn")