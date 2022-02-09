"""

The script declares the functions used in 'split_data.py'

"""

import pathlib
from logzero import logger

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split



def analyze_corr(df, col):
    # fig
    fig = plt.figure(figsize=(12, 12))
    
    # mask
    mask = np.triu(df.corr())
    
    # axes 
    axes = fig.add_axes([0, 0, 1, 1])
    sns.heatmap(df.dropna().corr(), annot=True, mask=mask, square=True, fmt='.2g',
                vmin=-1, vmax=1, center= 0, cmap='viridis', linecolor='white', 
                cbar_kws= {'orientation': 'vertical'}, ax=axes) 
    
    # title
    axes.text(-1, -1.5, 'Correlation', color='black', fontsize=24, fontweight='bold')
    
    plt.savefig('../visualizations/correlation_heatmap.png')
    
    # Printing correlations
    corr_matrix = df.corr()
    logger.info("The correlation of 'rating' with other columns is: {}".format(corr_matrix[col].sort_values()))



def split_data(df, col):
    train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 25)

    train_labels = train_set[col]
    train_data = train_set.drop(col, axis = 1)

    test_labels = test_set[col]
    test_data = test_set.drop(col, axis = 1)

    # Saving train and test sets 
    tgt_path = pathlib.Path.cwd().parent.joinpath('data/train_labels.csv')  # declaring file path
    train_labels.to_csv(tgt_path, index = False)   # saving file
    
    tgt_path = pathlib.Path.cwd().parent.joinpath('data/train_data.csv')  # declaring file path
    train_data.to_csv(tgt_path, index = False)   # saving file
    
    tgt_path = pathlib.Path.cwd().parent.joinpath('data/test_labels.csv')  # declaring file path
    test_labels.to_csv(tgt_path, index = False)   # saving file
    
    tgt_path = pathlib.Path.cwd().parent.joinpath('data/test_data.csv')  # declaring file path
    test_data.to_csv(tgt_path, index = False)   # saving file
    
    logger.info(f"\nRows in train data : {len(train_set)}\nRows in train labels: {len(train_labels)}\nRows in test data: {len(test_set)}\nRows in test labels: {len(test_labels)}\n")
