import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist


class BasicML:
    def __init__(self):
        pass


class Metrics(BasicML):
    def __init__(self):
        super().__init__()
    
    def R2_Score(self, prediction, actual):
        if isinstance(actual, pd.Series):
            actual = actual.to_numpy() 
        if isinstance(prediction, pd.Series):
            prediction = prediction.to_numpy() 
        
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

class BasicClusterModel(BasicML):
    def __init__(self):
        super().__init__()
        pass


class BasicKmeanCluster:
    def __init__(self):
        self.labels = []
        self.centroids = []
        self.clusters = 1
    
    def kmeans_init_centroid(self, dataset: any, cluster: int):
        return dataset[np.random.choice(dataset.shape[0], cluster, replace=True)]

    def kmeans_assign_labels(self, dataset, centroids: list):
        D = cdist(dataset, centroids, metric='euclidean')
        return np.argmin(D, axis=1)
    
    def kmeans_update_centroid(self, dataset, labels: list, clusters: int):
        centroids = np.zeros((clusters, dataset.shape[1]))
        
        for cluster in range(clusters):  
            dataset_of_cluster = dataset[labels == cluster, :]
            centroids[cluster, :] = np.mean(dataset_of_cluster, axis=0)
        
        return centroids
    
    def has_converged(self, cur_centroids, new_centroids):
        return set([tuple(a) for a in cur_centroids]) == set([tuple(b) for b in new_centroids])
    
    def kmeans(self, dataset, clusters: int):
        it = 0
        self.clusters = clusters
        self.centroids.append(self.kmeans_init_centroid(dataset, clusters))  
        while True:
            self.labels.append(self.kmeans_assign_labels(dataset, self.centroids[-1]))
            new_centroid = self.kmeans_update_centroid(dataset, self.labels[-1], self.clusters)
            if self.has_converged(self.centroids[-1], new_centroid):
                break
            it += 1
            self.centroids.append(new_centroid)
        return (self.centroids[-1], self.labels[-1], it)
    
    def kmeans_display_with_centers(self,X, label,centers):
        K = np.amax(label) + 1
        X_label = []
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  
        shapes = ['^', 'o', 's', 'P', 'D', '*', 'X']  
        for k in range(K):
            X_label.append(X[label == k, :])
            color = colors[k % len(colors)]  
            shape = shapes[k % len(shapes)]  
            plt.plot(X_label[k][:, 0], X_label[k][:, 1], color + shape, markersize=4, alpha=0.8)

        for k in range(K):
            shape = shapes[k%len(shapes)]
            plt.plot(centers[k,0], centers[k,1], 'y'+shape , markersize = 10, alpha = 1)

        plt.axis('equal')
        plt.plot()
        plt.show()
        
    def kmeans_display(self,X, label):
        K = np.amax(label) + 1
        X_label = []
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  
        shapes = ['^', 'o', 's', 'P', 'D', '*', 'X']  

        for k in range(K):
            X_label.append(X[label == k, :])
            color = colors[k % len(colors)]  
            shape = shapes[k % len(shapes)]  
            plt.plot(X_label[k][:, 0], X_label[k][:, 1], color + shape, markersize=4, alpha=0.8)
        plt.axis('equal')
        plt.plot()
        plt.show()

class BasicKNN(BasicML):
    def __init__(self):
        super().__init__()
        

class BasicGradientDesent(BasicML):
    def __init__(self):
        super().__init__()
    

        
        
        
        
        
        
        
        
    
    
