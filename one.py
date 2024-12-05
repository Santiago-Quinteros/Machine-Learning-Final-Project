import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from utils import *


## Preprocess data
def preprocessing(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, header='infer')

    # Clean 
    df = replace_nonalphanumerical(df)
    
    # fill missing values
    df = fill_missing_values(df)
    
    #Convert text to numbers for training
    X_data, Y_data=mapping(df)

    # Center and normalize 
    X_data = center_normalize(X_data)

    return X_data, Y_data


# Prepare data for training
def dataset_loader(X_data, y_data, test_size,one_hot_encoding=True):

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    if one_hot_encoding:
        # One hot encoding
        n_classes = 2
        y_data_ = np.zeros((len(y_data), n_classes))
        for i in range(len(y_data)):
            y_data_[i,y_data[i]]=1

        y_data = y_data_


    # Random train/test split
    sort_indices = np.random.permutation(len(X_data))
    X_data = X_data[sort_indices]
    y_data = y_data[sort_indices]

    test_len = int(test_size*len(X_data))

    X_train = X_data[:-test_len]
    y_train = y_data[:-test_len]

    X_test = X_data[-test_len:]
    y_test = y_data[-test_len:]
    

    return X_train, y_train, X_test, y_test



class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Train model
def train_model(model, X_train, y_train, learning_rate, epochs, verbose):
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    batch_size = 4

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):

            inputs = X_train[i:i + batch_size]

            targets = y_train[i:i + batch_size]

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            # Print the loss for every epoch
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    return model 



# Display model performance
def model_performance(model , X_test, y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    correct_predictions = 0

    for i in range(0, len(X_test)):
        inputs = X_test[i]
        targets = y_test[i]

        outputs = model(inputs)
        predicted_labels = torch.argmax(outputs)
        target = torch.argmax(targets)

        correct_predictions += (predicted_labels == target).sum().item()


    print(f"Model Accuracy: {100*correct_predictions/len(X_test):.2f}%")

    return correct_predictions/len(X_test)


#Decision trees:
def Decision_tree(X_train, y_train, max_depth):
    dec_tree = DecisionTreeClassifier(max_depth=max_depth)
    dec_tree.fit(X_train, y_train)
    return dec_tree

#SVM:
def SVM_train(X_train, y_train):
    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)
    return svm_model

#KNN:
def KNN_train(X_train, y_train):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    return knn_model

#Logistic regression:
def Logistic_regression_train(X_train, y_train):
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    return log_reg_model
