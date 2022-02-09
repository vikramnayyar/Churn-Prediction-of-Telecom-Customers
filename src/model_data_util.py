"""

The script declares the functions used in 'model_data.py'

"""

import pandas as pd
import pathlib
import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier  
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

#import xgboost as xgb
from xgboost import XGBClassifier

#import lightgbm as lgb
from lightgbm import LGBMClassifier


# keras 
from keras.models import Sequential
from keras.layers import Dropout, Dense
from utility import parse_config


##################################################
#-----------------Reading Config------------------
##################################################

config_path = "../config/config.yaml"   
config = parse_config(config_path)   # read config file

##################################################
#-----------------Declaring Functions-------------
##################################################

def compare_models(train_data, train_labels, test_data, test_labels):
    
    model_comparison = pd.DataFrame()
    model_names = [ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, 
                   GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier,
                   XGBClassifier, CatBoostClassifier, LGBMClassifier]
    
    
    model_labels = ["ExtraTreesClassifier (ETC)", "AdaBoostClassifier (ABC)", "Bagging Classifier (BC)", 
                    "GradientBoostClassifier (GBC)", "RandomForestClassifier (RFC)", "DecisionTreeClassifier (DTC)", 
                    "XGBClassifier (XGB)", "CatBoostClassifier (CBC)", "LGBMClassifier (LGBM)"]
    i = 0
    
    for model_name in model_names:
        
        model_label = model_labels[i]
        i += 1   
        model = model_name()   # learning_rate does not work here
        model.fit(train_data, train_labels)
                
        accuracy = evaluate_model(model, test_data, test_labels, model_label)

        model_comparison = model_comparison.append({'model_name': model_name, 
                                                    'Accuracy': accuracy}, ignore_index = True)
    
    model_comparison.sort_values(by = ['Accuracy'], ascending = False, inplace = True ) 
    
    model_comparison.reset_index(drop = True)
    return model_comparison


def evaluate_model(model, test_data, test_labels, model_label):
    
    pred = model.predict(test_data)
    accuracy = accuracy_score(test_labels, pred)
    cm = confusion_matrix(test_labels, pred)
    
    if accuracy > 0.8:    
        plot_cm(model_label, cm, accuracy)
    return accuracy


def plot_cm(model, cm, accuracy):

    fig = plt.figure(figsize=(7, 5))
    plt.title(model, size = 15)
     
    # Declaring heatmap labels
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    # Plotting heatmap 
    cmap = custom_cmap()
    sns.heatmap(cm, annot=labels, annot_kws={"size": 15}, fmt = '', cmap=cmap)
    
    
    # Adding figure labels
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values \n \n Accuracy: {}'.format(round(accuracy, 4)))
    
    os.chdir('../visualizations')
    plt.savefig('cm_{}'.format(model))   # save figure
    os.chdir('../src')
    plt.close()

def custom_cmap():
    import matplotlib.colors
    
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "#e9fcdc"], 
              [norm(-0.6), "#d9f0c9"], 
              [norm( 0.6), "#4CBB17"],
              [norm( 1.0), "#0B6623"]]
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    return cmap
    


def normalize_data(df, train_data, test_data):
    
    from sklearn import preprocessing
    import pandas as pd
    min_max_scaler = preprocessing.MinMaxScaler()
    
    x_scaled = min_max_scaler.fit_transform(train_data)
    train_data_nn = pd.DataFrame(x_scaled)
    
    x_scaled = min_max_scaler.fit_transform(test_data)
    test_data_nn = pd.DataFrame(x_scaled)
    
    return train_data_nn, test_data_nn



def build_nn(train_data):
    classifier = Sequential()
     # Adding the input layer and the first hidden layer
    classifier.add(Dense(30, activation = 'relu', input_dim = len(train_data.columns)))  # tanh is better
   
    # Adding dropout
    classifier.add(Dropout(rate = 0.15))
     # Adding the second hidden layer
    classifier.add(Dense(30, activation = 'relu'))

    # Adding dropout
    classifier.add(Dropout(rate = 0.15))

    # Adding the output layer
    classifier.add(Dense(1, activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  # RMSProp is also equivalent
    
    return classifier



def evaluate_nn(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy']) 
    plt.plot(history.history['val_accuracy']) 
    plt.title('Neural Network Accuracy') 
    plt.ylabel('accuracy') 
    plt.xlabel('epoch') 
    plt.legend(['train', 'test'], loc='upper left') 
    
    tgt_path = pathlib.Path.cwd().parent.joinpath('visualizations/nn_performance.png')  # declaring file path
    plt.savefig(tgt_path)



def plot_feature_importance(train_data, train_labels, best_model):

    best_model = best_model()
    best_model.fit(train_data, train_labels) 
    
    feature_importance = pd.DataFrame(best_model.feature_importances_,
                                   index = train_data.columns,
                                   columns=['importance']).sort_values('importance',ascending=False)
    
    # Plotting feature importance
    plt.figure(figsize=(20,8))
    plt.plot(feature_importance)
    plt.scatter(y=feature_importance.importance,x=feature_importance.index)
    plt.title(config["model_data"]["feature_importance"]["title"], fontsize = 16)
    plt.ylabel(config["model_data"]["feature_importance"]["ylabel"], fontsize=14)
    plt.xlabel(config["model_data"]["feature_importance"]["xlabel"], fontsize = 14)
    plt.xticks(rotation = 45)
    plt.grid()
    
    os.chdir("../visualizations")
    plt.savefig("feature_importance.png")
    os.chdir("../src")
    
    
def optimize_best_model(train_data, train_labels, test_data, test_labels):

    # Assigning categorical features
    rfc_params = {"n_estimators": config["model_data"]["rfc_params"]["n_estimators"],   # 900
                  "min_samples_split": config["model_data"]["rfc_params"]["min_samples_split"], 
                  "min_samples_leaf" : config["model_data"]["rfc_params"]["min_samples_leaf"], 
                  "max_features" : config["model_data"]["rfc_params"]["max_features"]  # log2
                  }       

    # Declaring model
    model = RandomForestClassifier(**rfc_params)
    
    # Fitting model to train set 
    model.fit(train_data, train_labels)
    
    model_label = "optimized_rfc"         
    accuracy = evaluate_model(model, test_data, test_labels, model_label)
    return model, accuracy    