"""

The script obtains various visualizations of the cleaned dataset 
& stores them in "visualization" directory

"""
import os
from utility import create_log, parse_config, read_data
from data_analysis_util import dataset_balance, box_plot, grouped_bar_plot

create_log("data_analysis.log")  # Creating log file

# setting config
config_path = "config/config.yaml"   
config = parse_config(config_path)   # read config file


#%%
##################################################
#-----------------Reading Dataset-----------------
##################################################

data_path = config["data_analysis"]["data"]   # read dataset
df_clean = read_data(data_path)

os.chdir('visualizations')  # directory to save visualization figures

#%%
#################################################
#---------------- Dataset Balance ---------------
#################################################

dataset_balance(df_clean, "Churn", "dataset_balance")

#%%
############################################
# --------- 1. Grouped Bar Plots -----------
############################################

grouped_bar_plot(df_clean, "gender", "Churn", "gender_vs_churn")

grouped_bar_plot(df_clean, "SeniorCitizen", "Churn", "senior_vs_churn")

grouped_bar_plot(df_clean, "Partner", "Churn", "partner_vs_churn")

grouped_bar_plot(df_clean, "Dependents", "Churn", "dependent_vs_churn")

grouped_bar_plot(df_clean, "PhoneService", "Churn", "phone_vs_churn")

grouped_bar_plot(df_clean, "MultipleLines", "Churn", "multiple_vs_churn")

grouped_bar_plot(df_clean, "InternetService", "Churn", "internet_vs_churn")

grouped_bar_plot(df_clean, "OnlineSecurity", "Churn", "security_vs_churn")

grouped_bar_plot(df_clean, "OnlineBackup", "Churn", "backup_vs_churn")

grouped_bar_plot(df_clean, "DeviceProtection", "Churn", "protection_vs_churn")

grouped_bar_plot(df_clean, "TechSupport", "Churn", "techsupport_vs_churn")

grouped_bar_plot(df_clean, "StreamingTV", "Churn", "streamtv_vs_churn")

grouped_bar_plot(df_clean, "StreamingMovies", "Churn", "streammov_vs_churn")

grouped_bar_plot(df_clean, "Contract", "Churn", "contract_vs_churn")

grouped_bar_plot(df_clean, "PaperlessBilling", "Churn", "paperless_vs_churn")

grouped_bar_plot(df_clean, "PaymentMethod", "Churn", "paymethod_vs_churn")


#%%
#################################################
#--------------- 2. Box Plots -------------------
#################################################

box_plot(df_clean, "tenure", "Churn", "tenure_vs_churn")

box_plot(df_clean, "MonthlyCharges", "Churn", "monthcharges_vs_churn")


#%%

os.chdir('..')  # resetting to project path 