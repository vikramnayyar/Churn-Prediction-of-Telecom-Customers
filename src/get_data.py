"""

The script reads and analyzes the dataset. After cleaning, it is oberved that
dataset cleaning is not required.

Also, the **locations** and **restaurant type** dictionaries are saved. This is 
required for the application to accept user inputs. 

"""

from logzero import logger
from utility import create_log, parse_config, read_data
from get_data_util import analyze_data

create_log("get_data.log")  # Creating log file

# setting config
config_path = "config/config.yaml"   
config = parse_config(config_path)   # read config file

#%%
##################################################
#-------------------Reading Data------------------
##################################################
data_path = config["get_data"]["data"]   # read dataset
df = read_data(data_path)
        
analyze_data(df)   # Analyzing dataset

#%%
##################################################
#-----------------Cleaning Dataset----------------
##################################################

# cleaning TotalCharges Column

nan_indices = df.index[df['TotalCharges'] == ' '].tolist()   # finding non-float values
df = df.drop(nan_indices)    # dropping non-float values
df = df.reset_index()     # resetting indices
df = df.drop(['index'], axis = 1)   # removing index col

df.TotalCharges = df.TotalCharges.astype('float')   # coverting TotalCharges col to float type


# removing unnecessary columns
df = df.drop(["customerID"], axis = 1)


# %%
# Saving cleaned data
df.to_csv('data/clean_data.csv', index = False)    # Saving the file in the path

logger.info("Cleaned dataset was saved successfully.")

