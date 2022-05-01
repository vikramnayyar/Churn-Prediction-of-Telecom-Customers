"""

The script declares functions used in 'data_analysis.py'

"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from utility import parse_config

config_path = "config/config.yaml"   
config = parse_config(config_path)   # read config file


def dataset_balance(df_clean, col, plot_name):
    fig, ax = plt.subplots()
    sns.countplot(x = col, data = df_clean, palette = 'viridis')
    
    plt.title('Churn Distribution of Telecom Customers', fontsize = 16)
    plt.xlabel('Deposit', fontsize = 14)
    plt.ylabel('Total Customers', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.savefig("plot_name")
  

def grouped_bar_plot(df_clean, col, label, plot_type):
    
    fig, ax = plt.subplots()

    sns.catplot(col, hue = label, data=df_clean, kind="count", palette={'No':"#80e880", 'Yes':"#2626ff"}, legend = False)
    
    color_patches = [
        Patch(facecolor="#80e880", label="Non-Churners"),
        Patch(facecolor="#2626ff", label="Churners")
    ]
    
    plt.title("{} vs Churn".format(col), size = 18, y=1.08) 
    plt.xlabel("{}".format(col), size = 14)
    plt.ylabel("Total Customers", size = 14)
    plt.xticks(size = 12, rotation = 'vertical')
    plt.legend(handles = color_patches, fontsize = 12,  bbox_to_anchor=(1.4,1.05))
    
    plt.savefig(plot_type)  # saving figure
    plt.close(1)



def box_plot(df_clean, col, label, plot_type):
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("{} vs Churn".format(col), size = 18, y=1.08)
    
    # Subplot 1
    ax[0].hist(df_clean[df_clean[label]=='No'][col], bins=30, alpha=0.5, color="green", label="Non-Churners")
    ax[0].hist(df_clean[df_clean[label]=='Yes'][col], bins=30, alpha=0.5, color="blue", label="Churners")
    
    ax[0].set_xlabel("{}".format(col), size = 14)
    ax[0].set_ylabel("Total Customers".format(col), size = 14)
    ax[0].legend(fontsize = 11);
    
    # Subplot 2
    sns.boxplot(x=col, y=label, data=df_clean, orient="h", palette={ 'No':"#80e880", 'Yes':"#2626ff"}, ax = ax[1])
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlabel("{}".format(col), size = 14)
    
    color_patches = [
        Patch(facecolor="#80e880", label="Non-Churners"),
        Patch(facecolor="#2626ff", label="Churners")
    ]
    ax[1].legend(handles=color_patches, fontsize=11);
  
    plt.savefig(plot_type)  # saving figure
