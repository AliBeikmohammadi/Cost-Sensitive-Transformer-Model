#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, make_scorer, f1_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.backend as k
from keras import metrics
from tensorflow.keras import layers, models
from tensorflow.keras import losses
from tensorflow import keras
from tensorflow.keras.metrics import Recall, Accuracy, Precision, FalsePositives, FalseNegatives
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from transformers import TFAutoModel
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE
from imblearn.over_sampling import KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, TomekLinks
from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler 
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px

import scikitplot as skplt

from focal_loss import BinaryFocalLoss

import argparse

# In[3]:


def cost_sensitive_metric_dl(y_true, y_pred):
    """
    Computes a cost-sensitive metric for classification models based on the number of false positives and false negatives.
    
    Args:
    y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
        
    Returns:
    cost: float
        The cost of misclassification, computed as 500 times the number of false negatives plus 10 times the number of false positives.
    """
    # Cast the predicted labels to float type
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)

    # Compute the number of false positives, and false negatives
    fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(y_true, 0), tf.math.equal(y_pred, 1)), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(y_true, 1), tf.math.equal(y_pred, 0)), tf.float32))

    # Compute the cost function
    cost = 500 * fn + 10 * fp
    return cost


def cost_sensitive_score(y_true, y_pred):
    """
    Calculate the cost-sensitive score of the model.
    
    This function calculates the total cost of classification by weighting the false positive and false negative costs.
    The cost for each false positive is 10, and the cost for each false negative is 500.
    
    Parameters:
    -----------
    y_true: array-like of shape (n_samples,)
        True binary labels.
    y_pred: array-like of shape (n_samples,)
        Predicted binary labels.

    Returns:
    --------
    cost: float
        The cost-sensitive score of the model.
    """
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate the total cost of classification
    false_positive_cost = 10
    false_negative_cost = 500
    cost = false_positive_cost * fp + false_negative_cost * fn
    
    return cost

# Create a scoring object using the custom cost-sensitive function
my_func = make_scorer(cost_sensitive_score, greater_is_better=False)


def test(fname):
    X_train_val_cut_imputed_scaled = pd.read_csv('aps_failure_train_valid_set_cut_imputed_scaled.csv')
    y_train_val_raw = pd.read_csv('aps_failure_train_valid_set_label.csv')['class'].ravel()
    X_test_cut_imputed_scaled = pd.read_csv('aps_failure_test_set_cut_imputed_scaled.csv')
    y_test_raw = pd.read_csv('aps_failure_test_set_label.csv')['class'].ravel()
    model = models.load_model("./Model/"+fname, custom_objects={'cost_sensitive_metric_dl':cost_sensitive_metric_dl}, compile=False)   
    # Compile the model
    model.compile(metrics=[cost_sensitive_metric_dl, Accuracy(),Precision(),Recall(),FalsePositives(),FalseNegatives()])

    # Make predictions
     # Get the probability of the binary classifier
    y_pred_proba_nn_Train = model.predict(X_train_val_cut_imputed_scaled)
    y_pred_proba_nn_Test = model.predict(X_test_cut_imputed_scaled)

    # Predict the class label
    y_pred_Train_2 = np.array([0 if x <= 0.5 else 1 for x in y_pred_proba_nn_Train])
    y_pred_Test_2 = np.array([0 if x <= 0.5 else 1 for x in y_pred_proba_nn_Test])

    # Print total cost
    print(f'Train set: Based on standard threshold: Total cost is: {cost_sensitive_score(y_train_val_raw, y_pred_Train_2)}, with a Precision {precision_score(y_train_val_raw, y_pred_Train_2)} and a Recall {recall_score(y_train_val_raw, y_pred_Train_2)}.')
    print(f'Test set: Based on standard threshold: Total cost is: {cost_sensitive_score(y_test_raw, y_pred_Test_2)}, with a Precision {precision_score(y_test_raw, y_pred_Test_2)} and a Recall {recall_score(y_test_raw, y_pred_Test_2)}.')

    # Print the confusion matrix
    skplt.metrics.plot_confusion_matrix(y_train_val_raw, y_pred_Train_2, normalize=False, title="(standard threshold: Train) Total Cost: " + str(cost_sensitive_score(y_train_val_raw, y_pred_Train_2)))
    skplt.metrics.plot_confusion_matrix(y_test_raw, y_pred_Test_2, normalize=False, title="(standard threshold: Test) Total Cost: " + str(cost_sensitive_score(y_test_raw, y_pred_Test_2)))
    plt.show()


# In[26]:


# Define the TransformerModel class
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

def maincode(args):
    # Define hyperparameters and create the model instance
    SEED = args.seed_number
    
    head_size = args.head_size 
    num_heads = args.num_heads 
    ff_dim = args.ff_dim 
    num_transformer_blocks = args.num_transformer_blocks 
    mlp_dropout = args.mlp_dropout 
    dropout = args.dropout 
    
    # Define the learning rate for the optimizer
    lr = args.learning_rate #learning rate
    
    # Define the pos_weight and gamma hyperparameters
    pos_weight = args.pos_weight
    gamma = args.gamma 
    
    # Define the batch size
    batch_size= args.batch_size
    
    # Define epochs
    epochs = args.num_epochs
    mlp_u = args.mlp_units
    mlp_units=[]
    temp = str.split(mlp_u, '_')
    for i in temp:
        mlp_units.append(int(i))
    
    fname='mlp'+mlp_u+'blocks'+str(num_transformer_blocks)+'heads'+str(num_heads)         +'size'+str(head_size)+'ffdim'+str(ff_dim)+'mlpdropout'+str(mlp_dropout)+'dropout'+str(dropout)+'alpha'+         str(pos_weight)+'gamma'+str(gamma)+'lr'+str(lr)+'batch'+str(batch_size)+'epochs'+str(epochs)+'run'
    
    # Set a seed on the three different libraries
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    # Set hyperparameters

    X_train_val_cut_imputed_resampled_scaled = pd.read_csv('aps_failure_train_valid_set_cut_imputed_resampled_scaled.csv')
    y_train_val_raw_resampled = pd.read_csv('aps_failure_train_valid_set_label_resampled.csv')['class'].ravel()
    X_test_cut_imputed_scaled = pd.read_csv('aps_failure_test_set_cut_imputed_scaled.csv')
    y_test_raw = pd.read_csv('aps_failure_test_set_label.csv')['class'].ravel()

    # Split data into train and test set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val_cut_imputed_resampled_scaled, y_train_val_raw_resampled, 
                                                          test_size=0.1, stratify=y_train_val_raw_resampled, random_state=SEED)
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    input_shape = X_train.shape[1:]  # Define the input shape according to your data

    model = build_model(
        input_shape=input_shape, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks, mlp_units=mlp_units, mlp_dropout=mlp_dropout,
        dropout=dropout)

    # Define the BinaryFocalLoss with the hyperparameters
    loss = BinaryFocalLoss(pos_weight=pos_weight, gamma=gamma)

    # Define the optimizer with the selected learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=opt, loss=loss, 
                  metrics=[cost_sensitive_metric_dl, 
                          ])

    #model.summary()

    TB= TensorBoard(
        log_dir="./TB/"+fname,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None)
    CK= tf.keras.callbacks.ModelCheckpoint(
        filepath = "./Model/"+fname+"epoch{epoch:01d}",
        monitor = "val_cost_sensitive_metric_dl",
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = "min",
        save_freq = "epoch",)

    if not os.path.exists("./CSV/"):
        os.makedirs("./CSV/")
    csv_logger = CSVLogger("./CSV/"+fname+'.csv', separator=",", append=False)
    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), 
              callbacks=[TB, csv_logger, CK]) 
    if not os.path.exists("./Model/"):
        os.makedirs("./Model/")
    model.save("./Model/"+fname)    
    # Evaluate the model
    test(fname)


# In[ ]:


if __name__ == '__main__':
 # Parse command line arguments
 parser = argparse.ArgumentParser(description='Train a Transformer on APS')
 parser.add_argument('--seed_number', type=int, default=0,
                     help='seed number')
 parser.add_argument('--learning_rate', type=float, default=0.0005,
                     help='learning rate for the optimizer')
 parser.add_argument('--pos_weight', type=float, default=0.95,
                     help='pos_weight for the focal loss')
 parser.add_argument('--gamma', type=float, default=1.5,
                     help='gamma for the focal loss')
 parser.add_argument('--num_epochs', type=int, default=8000,
                     help='number of epochs to train')
 parser.add_argument('--batch_size', type=int, default=72,
                     help='batch size')
 parser.add_argument('--head_size', type=int, default=256,
                     help='head size')
 parser.add_argument('--num_heads', type=int, default=4,
                     help='num heads')
 parser.add_argument('--ff_dim', type=int, default=4,
                     help='ff dim')
 parser.add_argument('--num_transformer_blocks', type=int, default=4,
                     help='num transformer blocks') 
 parser.add_argument('--mlp_dropout', type=float, default=0.4,
                     help='mlp_dropout') 
 parser.add_argument('--dropout', type=float, default=0.25,
                     help='dropout') 
 parser.add_argument('--mlp_units', type=str, default='128_64',
                     help='mlp units')    
args = parser.parse_args()
# Train the network
maincode(args)  

