"""

The script models the split data using several classification models (sklearn, cbc, xgb, lgbm, neural networks),
then; selects, optimizes and saves the best model 

"""

from logzero import logger
import pickle

from model_data_util import compare_models, normalize_data, build_nn, evaluate_nn, plot_feature_importance, optimize_best_model
from utility import create_log, parse_config, read_data

create_log("model_data.log")  # Creating log file

##################################################
#-------------Reading Dataset & Config------------
##################################################
config_path = "../config/config.yaml"   
config = parse_config(config_path)   # read config file

train_data = read_data(config["model_data"]["train_data"])
train_labels = read_data(config["model_data"]["train_labels"])
test_data = read_data(config["model_data"]["test_data"])
test_labels = read_data(config["model_data"]["test_labels"])

df = read_data(config["split_data"]["data"])


#%%
##################################################
#-------------sklearn Models Comparison-----------
##################################################
model_comparison = compare_models(train_data, train_labels, test_data, test_labels)


#%%
##################################################
#-------------Neural Network Modelling-----------
##################################################

#%%
# refine features
train_data_nn = train_data.drop(["StreamingTV", "gender", "PhoneService", "MultipleLines"], axis = 1)  
test_data_nn = test_data.drop(["StreamingTV", "gender", "PhoneService", "MultipleLines"], axis = 1)  


#%%
# feature normalization

train_data_nn, test_data_nn = normalize_data(df, train_data_nn, test_data_nn)

#%%
# building nn
classifier = build_nn(train_data_nn)

#%% Train nn
x = len(train_data.columns)

history = classifier.fit(train_data_nn, train_labels, 
                         validation_data=(test_data_nn, test_labels), 
                         batch_size = config["model_data"]["nn"]["batch_size"], # 32 
                         epochs = config["model_data"]["nn"]["epochs"])   # default epochs =  25,   batch_size = 10

#%% Evaluate nn performance

evaluate_nn(history)


#%%
##################################################
#-------------Overall Model Comparison-----------
##################################################

accuracy = max(history.history['accuracy'])   

model_comparison = model_comparison.append({'model_name': "Neural Network", 
                                                    'Accuracy': accuracy}, ignore_index = True)
model_comparison.sort_values(by = ['Accuracy'], ascending = False, inplace = True )     
model_comparison.reset_index(drop = True)

logger.info(model_comparison.head(10))


#%%
####################################################
###--------Best Model Performance Optimization------
####################################################

# select best model
best_model = model_comparison.iloc[0][0]
logger.info("Best Model is {}".format(best_model))

#%%
# refine feature

plot_feature_importance(train_data, train_labels, best_model)

train_data_bm = train_data.drop(["SeniorCitizen", "PhoneService"], axis = 1)   # "StreamingMovies" is required
test_data_bm = test_data.drop(["SeniorCitizen", "PhoneService"], axis = 1)

#%% optimize model

best_model, accuracy = optimize_best_model(train_data_bm, train_labels, test_data_bm, test_labels)

logger.info("Accuracy of the best model after optimization is {}".format(accuracy))
    
#%%
# Saving Model
file = open('../model/model.pkl', 'wb')   # Open a file to store model
pickle.dump(best_model, file)   # dumping information to the file
file.close()