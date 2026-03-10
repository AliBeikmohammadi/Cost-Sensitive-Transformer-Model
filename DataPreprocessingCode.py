#!/usr/bin/env python
# coding: utf-8

# In[8]:


import random
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, BayesianRidge
from sklearn.metrics import confusion_matrix, precision_score, recall_score, make_scorer, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.backend as k
from tensorflow.keras import layers, models
from tensorflow.keras import losses
from tensorflow.keras.metrics import Recall
from tensorflow import keras
from keras import metrics 
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFAutoModel
import imblearn
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE
from imblearn.over_sampling import KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, TomekLinks
from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler
from imblearn.pipeline import Pipeline as Pipeline2
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from collections import Counter
import plotly.express as px
import scikitplot as skplt
from focal_loss import BinaryFocalLoss
import matplotlib.cm as cm

# Set a seed
SEED = 0

# Set a seed on the three different libraries
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# In[76]:


# Load data and replace 'na' strings with NaN
df_train_val = pd.read_csv('aps_failure_training_set.csv', na_values='na')
df_test = pd.read_csv('aps_failure_test_set.csv', na_values='na')

# Replace 'neg' with 0 and 'pos' with 1
df_train_val['class'].replace({'neg': 0, 'pos': 1}, inplace=True)
df_test['class'].replace({'neg': 0, 'pos': 1}, inplace=True)

# Split the data into feature and label sets
X_train_val_raw = df_train_val.drop('class', axis=1)
y_train_val_raw = df_train_val['class'].ravel()
X_test_raw = df_test.drop('class', axis=1)
y_test_raw = df_test['class'].ravel()

# Print data shape and class balance information
print(X_train_val_raw.shape, X_test_raw.shape)
print(df_train_val['class'].value_counts())
print(df_test['class'].value_counts())


# In[79]:


# Check the class imbalance in the train set
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

colors = ["blue", "red"]
class_names = ['Negative', 'Positive']


# plot countplot for train/validation set
sns.countplot(data=df_train_val, x='class', ax=axs[0], palette=colors)
axs[0].set_xlabel(' Class Labels', fontsize=12)
axs[0].set_ylabel('Number of Instances', fontsize=12)
axs[0].set_title('Class Distribution in Training/Validation Set', fontsize=12)
axs[0].set_xticklabels(class_names)
for p in axs[0].patches:
    axs[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)
# plot countplot for test set
sns.countplot(data=df_test, x='class', ax=axs[1], palette=colors)
axs[1].set_xlabel(' Class Labels', fontsize=12)
axs[1].set_ylabel('')
axs[1].set_title('Class Distribution in Test Set', fontsize=12)
axs[1].set_xticklabels(class_names)
for p in axs[1].patches:
    axs[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# In[81]:


# Find the percentage of missing values in the dataset per columns
missing_values = X_train_val_raw.isna().mean()

# Generate a list of tuples with the percentage of missing values and the number of columns with that percentage of missing values
list_missing_values = []
for x in range(5, 90, 5):
    list_missing_values.append((x/100, missing_values.loc[missing_values > x/100].count()))

# Plot the percentage of missing values
x_line = [tuple[0] for tuple in list_missing_values]
y_line = [tuple[1] for tuple in list_missing_values]

fig, ax = plt.subplots(figsize=(8, 4))

colors = cm.Greys(np.linspace(0.2, 1, len(x_line)))  # Adjust the range to control the spectrum

sns.barplot(x=x_line, y=y_line, ax=ax, palette=colors)

ax.set_xlabel('Percentage of Missing Values', fontsize=12)
ax.set_ylabel('Number of Attributes Affected', fontsize=12)

# Add percentage values to the bars
for i, v in enumerate(y_line):
    ax.annotate(str(v), xy=(i, v), ha='center', va='bottom', fontsize=10)

plt.show()


# In[82]:


# Drop columns with 10% or more of missing values
columns_to_drop = missing_values.loc[missing_values >= 0.1].index.to_list()

# Print shape before dropping columns
print(X_train_val_raw.shape, X_test_raw.shape)

# Remove the same columns both from the train and the test set
X_train_val_cut = X_train_val_raw.drop(columns=columns_to_drop, axis=1)
X_test_cut = X_test_raw.drop(columns=columns_to_drop, axis=1)

# Print shape after dropping columns
print(X_train_val_cut.shape, X_test_cut.shape)


# In[28]:


# Initialize the imputer object
imputer_model =  BayesianRidge()
imputer = IterativeImputer(estimator=imputer_model, sample_posterior=False, max_iter=100, 
                           tol=0.001, n_nearest_features=None, initial_strategy='mean', 
                           imputation_order='ascending', skip_complete=False, verbose=1, 
                           random_state=SEED, add_indicator=False, keep_empty_features=False)
# Fit the model
imputer.fit(X_train_val_cut)

# Fit and transform the imputer on the dataset
X_train_val_cut_imputed = imputer.transform(X_train_val_cut)
X_test_cut_imputed = imputer.transform(X_test_cut)


# In[29]:


# Iterative imputer and Anomaly Detection are computationally expensive, therefore, we store the result to avoid these steps in further iterations
pd.DataFrame(X_train_val_cut_imputed).to_csv('aps_failure_train_valid_set_cut_imputed.csv', index=False, header=X_train_val_cut.columns)
pd.DataFrame(y_train_val_raw).to_csv('aps_failure_train_valid_set_label.csv', index=False, header=['class'])

pd.DataFrame(X_test_cut_imputed).to_csv('aps_failure_test_set_cut_imputed.csv', index=False, header=X_train_val_cut.columns)
pd.DataFrame(y_test_raw).to_csv('aps_failure_test_set_label.csv', index=False, header=['class'])


# In[16]:


#SVMSMOTE+EditedNearestNeighbours
# define the oversampling method
over = SVMSMOTE(sampling_strategy=0.5, random_state=SEED, k_neighbors=5, 
                n_jobs=-1, m_neighbors=10, svm_estimator=None, out_step=0.5)
# define the undersampling method
under = RepeatedEditedNearestNeighbours(sampling_strategy='majority', n_neighbors=3, 
                                        max_iter=100, kind_sel='all', n_jobs=-1)
resample = Pipeline2(steps=[('o', over), ('u', under)])
# fit and apply the pipeline
# transform the dataset
X_train_val_cut_imputed_resampled, y_train_val_raw_resampled = resample.fit_resample(X_train_val_cut_imputed, y_train_val_raw)
# summarize the new class distribution
counter_resampled = Counter(y_train_val_raw_resampled)
counter = Counter(y_train_val_raw)
print(counter)
print(counter_resampled)


# In[24]:


pd.DataFrame(X_train_val_cut_imputed_resampled).to_csv('aps_failure_train_valid_set_cut_imputed_resampled.csv', index=False, header=X_train_val_cut.columns)
pd.DataFrame(y_train_val_raw_resampled).to_csv('aps_failure_train_valid_set_label_resampled.csv', index=False, header=['class'])


# In[87]:


# Create an instance of the MinMax class
scaler = MinMaxScaler()

# Fit the scaler to the dataset
scaler.fit(X_train_val_cut_imputed_resampled)

# Scaled the data between 0 and 1
X_train_val_cut_imputed_resampled_scaled = scaler.transform(X_train_val_cut_imputed_resampled)
X_test_cut_imputed_scaled  = scaler.transform(X_test_cut_imputed)


# In[89]:


X_train_val_cut_imputed_scaled = scaler.transform(X_train_val_cut_imputed)
pd.DataFrame(X_train_val_cut_imputed_scaled).to_csv('aps_failure_train_valid_set_cut_imputed_scaled.csv', index=False, header=X_train_val_cut_imputed.columns)
pd.DataFrame(X_train_val_cut_imputed_resampled_scaled).to_csv('aps_failure_train_valid_set_cut_imputed_resampled_scaled.csv', index=False, header=X_train_val_cut.columns)
pd.DataFrame(X_test_cut_imputed_scaled).to_csv('aps_failure_test_set_cut_imputed_scaled.csv', index=False, header=X_train_val_cut.columns)

