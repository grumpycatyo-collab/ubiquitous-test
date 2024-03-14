import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

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
    

def calculate_metrics(model, X_test, y_test, regression=True, verbose=True):
    if verbose:
        print(f"Calculating metrics for {type(model).__name__} model...")
        print("Model information:")
    
    y_pred = model.predict(X_test)
    if regression:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mse ** 0.5
        
        if verbose:
            print(f"Regression Metrics:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}\n")
        
        return mse, mae, rmse
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        if verbose:
            print(f"Classification Metrics:")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
        
        return accuracy, precision, recall, f1
    

def create_classification_table(models, X, y , cv=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    table_data = {'Estimator': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'Best Model': []}
    best_model = None
    best_accuracy = 0  
    for name, model in models.items():
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_weighted')
        f1_cv = scores.mean()
        model.fit(X_train, y_train)
        accuracy, precision, recall, f1 = calculate_metrics(model, X_test, y_test, regression=False, verbose=False)
        table_data['Estimator'].append(name)
        table_data['Accuracy'].append(accuracy)
        table_data['Precision'].append(precision)
        table_data['Recall'].append(recall)
        table_data['F1-Score'].append(f1)
        table_data['Best Model'].append(False)
        
        if f1 > best_accuracy:
            best_model = model
            best_accuracy = f1
    
    if best_model:
        best_model_index = table_data['Estimator'].index(name)
        table_data['Best Model'][best_model_index] = True
    
    return pd.DataFrame(table_data), best_model


def test_train_plot(model,x,y):
    kf = KFold(n_splits=4)
    f1_train = []
    f1_test = []
    for train_index, test_index in kf.split(x):
           x_train, x_test = x.iloc[train_index], x.iloc[test_index]
           y_train, y_test = y.iloc[train_index], y.iloc[test_index]
           model.fit(x_train, y_train)
           y_train_pred = model.predict(x_train)
           y_test_pred = model.predict(x_test)
           f1_train.append(f1_score(y_train, y_train_pred,average='weighted'))
           f1_test.append(f1_score(y_test, y_test_pred,average='weighted'))
    plt.figure(figsize=(7,7))
    folds = range(1, kf.get_n_splits()+1 )
    plt.plot(folds, f1_train, 'o-', color='green', label='train')
    plt.plot(folds, f1_test, 'o-', color='blue', label='test')
    plt.legend()
    plt.grid()
    plt.title(model)
    plt.xlabel('Number of fold')
    plt.ylabel('f1_score')
    plt.show()