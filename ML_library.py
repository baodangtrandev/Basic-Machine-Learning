import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class BasicML:
    def __init__(self):
        pass


class Metrics(BasicML):
    def __init__(self):
        super().__init__()
    
    def R2_Score(self, prediction, actual):
    # Nếu actual là pandas Series, chuyển sang numpy array
        if isinstance(actual, pd.Series):
            actual = actual.to_numpy()  # Chuyển sang numpy array
        
        # Nếu prediction là pandas Series, chuyển sang numpy array
        if isinstance(prediction, pd.Series):
            prediction = prediction.to_numpy()  # Chuyển sang numpy array
        
        # Tính toán R2 Score
        SSres = np.sum((prediction - actual) ** 2)
        SSmean = np.sum((actual - np.mean(actual)) ** 2)
        
        return 1 - SSres / SSmean



class BasicLinearModel(BasicML):
    def __init__(self):
        super().__init__()
        pass


class BasicLinearRegression(BasicLinearModel):
    def __init__(self):
        coeff = None
        super().__init__()
        
    def train(self, X, y):
        """Train the linear regression model."""
        try:
            X = X.to_numpy()
            ones = np.ones((X.shape[0],1))
            Xbar = np.concatenate((ones,X), axis=1)
            y = y.to_numpy()
            A = np.dot(Xbar.T,Xbar)
            b = np.dot(Xbar.T,y)
            w = np.dot(np.linalg.pinv(A),b)
            self.coeff = w
        except:
            raise ("Error size of dataset is ambigious")
        
        

    def test(self, X):
        """Make predictions using the trained model."""
        if isinstance(X, pd.Series):
            X = X.to_numpy() 
        X = X.to_numpy()
        ones = np.ones((X.shape[0],1))
        X_test = np.concatenate((ones,X), axis=1)
        return np.dot(X_test,self.coeff)


    
