import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd

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