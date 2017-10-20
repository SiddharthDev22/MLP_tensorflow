from pyexcel_xlsx import get_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def onehot_encode(Y):
    """ Change the label format from integer to one hot encode"""
    onehot_encoder = OneHotEncoder(sparse=False)
    Y = Y.reshape(len(Y), 1)
    Y = onehot_encoder.fit_transform(Y)
    return Y

def read_dataset(filename, datasetNo=1, normalize=True, type="classification"):
    """ Read dataset from xlsx file"""
    sheetNo = (datasetNo - 1) * 2
    # read the excel file
    data = get_data(filename, start_row=1, start_column=1)
    if type == "classification":
        label = 0
    elif type == "regression":
        label = 1
    X_train = np.array(data[list(data.keys())[sheetNo]])[:, 2:].astype(np.float32)
    Y_train = np.array(data[list(data.keys())[sheetNo]])[:, label].astype(np.float32)
    X_valid = np.array(data[list(data.keys())[sheetNo+1]])[:, 2:].astype(np.float32)
    Y_valid = np.array(data[list(data.keys())[sheetNo+1]])[:, label].astype(np.float32)

    if type == "regression":
        X_train = X_train[Y_train != 0, :]
        Y_train = Y_train[Y_train != 0].reshape((-1, 1))
        X_valid = X_valid[Y_valid != 0, :]
        Y_valid = Y_valid[Y_valid != 0].reshape((-1, 1))
    elif type == "classification":
        Y_train = onehot_encode(Y_train)
        Y_valid = onehot_encode(Y_valid)

    if normalize:
        X_train = normalize_feat(X_train)
        X_valid = normalize_feat(X_valid)


    return X_train, Y_train, X_valid, Y_valid


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

def normalize_feat(X):
    """ normalize the features of set to [0 1]"""
    return X / X.max(axis=0)
