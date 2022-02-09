"""

The script converts the columns to categorical features 
& removes the outliers

"""

from utility import create_log, parse_config, read_data
from prepare_data_util import balance_the_dataset, convert_cat, cols_with_ouliers

create_log("prepare_data.log")  # Creating log file

##################################################
#-----------------Reading Dataset-----------------
##################################################
config_path = "../config/config.yaml"   
config = parse_config(config_path)   # read config file
data_path = config["prepare_data"]["data"]   # read dataset
df = read_data(data_path)

#%% Balancing Dataset

df = balance_the_dataset(df, "Churn", 1000) 


#%%
#################################################
##----------- Categorical Coversion--------------
#################################################

col_list = ["gender", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
            "DeviceProtection", "TechSupport", "StreamingTV","StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges", "Churn"]
df = convert_cat(df, col_list)  # converting to categories


################################################
#----------------Outlier Removal----------------
################################################

cols_with_outliers = cols_with_ouliers(df)   # Finding columns with outliers


df.to_csv("../data/prepared_data.csv", index = False)   # Saving file