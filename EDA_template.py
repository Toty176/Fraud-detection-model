#!/usr/bin/env python
# coding: utf-8

# # EDA Reusable Template

# ### Import important libraries

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler


# ### Loading the data set basic information

# In[7]:


def overview(df):
    print("Shape:", df.shape)
    print("\n Columns:", df.columns.tolist())
    print("\n Data Types:\n", df.dtypes)


# ### Exploring missing values

# In[8]:


def missing_values(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    summary = pd.DataFrame({'Missing': missing, 'Percent': missing_percent})
    summary = summary[summary['Missing'] > 0]
    print("\n Missing Values Summary:\n", summary)


# ### Describe the Given Data Set

# In[9]:


def describe_data(df):
    print("\n Statistical Summary:\n")
    display(df.describe())


# ### List columns that uses a certain type

# In[ ]:


def list_columns(df,*types):
    cat_cols = df.select_dtypes(include=types).columns
    print("\n columns with the following types", types, ":\n", cat_cols.tolist())
    return cat_cols.tolist()


# ### Convert To Categorical

# In[11]:


def convert_specific_to_category(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to category.")
        else:
            print(f"Column '{col}' not found in DataFrame.")


# ### Correlation Matrix

# In[ ]:


def correlation_matrix(df, figsize=(10, 6), annot=True):

    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        print(" Not enough numeric columns to compute correlation.")
        return
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


# ### Distribution plots

# In[13]:


def plot_distributions(df, bins=30):

    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) == 0:
        print("No numerical columns to plot.")
        return
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=bins, color='skyblue')
        plt.title(f'Distribution of "{col}"')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ### Box plots

# In[14]:


def plot_boxplots(df):
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) == 0:
        print("No numeric columns to plot.")
        return

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f'Boxplot of "{col}"')
        plt.xlabel(col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ### Change Columns Types

# In[15]:


def convert_columns_dtype(df, columns, dtype):
    for col in columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
                print(f"Converted '{col}' to {dtype}.")
            except Exception as e:
                print(f"Failed to convert '{col}' to {dtype}: {e}")
        else:
            print(f"Column '{col}' not found in DataFrame.")

    return df


# ### Normalization and Standardization

# #### min-max

# In[16]:


def min_max_normalize(df, columns):
    scaler = MinMaxScaler()
    for col in columns:
        if col in df.columns:
            df[col + '_normalized'] = scaler.fit_transform(df[[col]])
            print(f" Min-Max normalized '{col}' → '{col}_normalized'")
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return df


# #### Z-Score 

# In[17]:


from sklearn.preprocessing import StandardScaler

def z_score_standardize(df, columns):
    scaler = StandardScaler()
    for col in columns:
        if col in df.columns:
            df[col + '_zscore'] = scaler.fit_transform(df[[col]])
            print(f"Z-score standardized '{col}' → '{col}_zscore'")
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return df


# #### Log Transformation

# In[18]:


def log_transform(df, columns):

    for col in columns:
        if col in df.columns:
            if (df[col] < 0).any():
                print(f" Skipped '{col}' (contains negative values).")
                continue
            df[col + '_log'] = np.log1p(df[col])
            print(f"Log-transformed '{col}' → '{col}_log'")
        else:
            print(f" Column '{col}' not found in DataFrame.")
    return df


# ### Total of a certain Group

# In[ ]:


def total_column_per_group(dataset,Group,column,AVG=False):
    if AVG:
        total_column = dataset.groupby(Group)[column].mean().sort_values(ascending=True)
    else:
        total_column = dataset.groupby(Group)[column].sum().sort_values(ascending=False)
    highest = total_column.head(1)
    lowest = total_column.tail(1)
    return total_column, highest, lowest


# ### TO add at the begining of each code

# In[ ]:


import sys

# Add your folder path (note: use raw string r"..." to handle backslashes)
sys.path.append(r"C:\Users\Taha Sherif\OneDrive - New Giza University\education\Data analysis\EDA Reusable")
from EDA_template import *

