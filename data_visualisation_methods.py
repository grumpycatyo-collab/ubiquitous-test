import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.ensemble import IsolationForest

def plot_numerical_columns_distribution(df, n_rows, n_cols):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))  
    axes = axes.flatten()

    for i, col in enumerate(numerical_columns):
        sns.histplot(df[col], bins=30, kde=False, ax=axes[i])  
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    for i in range(len(numerical_columns), n_rows * n_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    
    
def plot_categorical_columns(df, n_cols, n_rows):
    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()

    if n_rows is None:
        n_rows = math.ceil(len(categorical_columns) / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), squeeze=False)

    row = 0
    for i, column in enumerate(categorical_columns):
        col = i % n_cols
        sns.countplot(x=column, data=df, ax=axs[row, col], width=0.4)
        axs[row, col].set_title(f'Bar Graph of {column}')
        axs[row, col].set_xlabel(column)
        axs[row, col].set_ylabel('Count')
        axs[row, col].grid(True)
        axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=45)

        if (i + 1) % n_cols == 0:
            row += 1

    if len(categorical_columns) % n_cols != 0:
        axs.flatten()[-1].axis('off')

    plt.tight_layout()
    plt.show()
    
def target_countplot(df,column,target):
    plt.figure(figsize=(12, 6))

    top_categories = df[column].value_counts().nlargest(40)
    sns.countplot(x=column, data=df, order=top_categories.index, hue=target)

    plt.xticks(rotation=90, ha='right')
    plt.xlabel('Data')
    plt.ylabel('Count')
    plt.title(f'Count Plot for {column} column')
    plt.show()


def plot_outliers_boxplot(df, n_cols=2):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    figsize=(16, 6)
    num_rows = math.ceil(len(numerical_columns) / n_cols)
    fig, axs = plt.subplots(num_rows, n_cols, figsize=(figsize[0], figsize[1]*num_rows))

    row = 0
    for i, column in enumerate(numerical_columns):
        col = i % n_cols

        sns.boxplot(y=column, data=df, ax=axs[row, col], width=0.4)
        axs[row, col].set_title(f'Boxplot of {column}')
        axs[row, col].set_ylabel('Values')

        if (i + 1) % n_cols == 0:
            row += 1

    if len(numerical_columns) % n_cols != 0:
        axs.flatten()[-1].axis('off')

    plt.tight_layout()
    plt.show()
    
