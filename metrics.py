# functions for scoring models
import numpy as np

def r_squared(y_true, y_pred):
    """
    Model evaluation by the R^2 metric. Comparison against the mean

    Parameters
    ------
    y_true: nd array, shape(n,1)
            true ouputs
    y_pred: nd array, shape(n,1)
            models predictions

    Returns
    --------
    r_squared: float -> [0,1]
                the r squared value of the data

    """
    y_bar = np.mean(y_true) # scalar mean of target

    ss_res = np.sum((y_true - y_pred) **2)
    ss_tot = np.sum((y_true - y_bar) **2)

    return 1 - ( ss_res / ss_tot )
    
def adjusted_r_squared(y_true,y_pred,d):
    """
    For comparing models with different numbers of features

    Parameters
    ------
    y_true: nd array, shape(n,1)
            true ouputs
    y_pred: nd array, shape(n,1)
            model predictions
    d: int
       number of features
    

    Returns
    --------
    adjusted_r_squared: float -> [0,1]
                the adjusted r squared value of the data


    """
    n = y_true.shape[0]

    Rsquared = r_squared(y_true,y_pred)

    return 1 - (((1 - Rsquared ) * (n -1)) / (n - d -1))

def confusion_matrix(y_true,y_pred,num_classes):
    """
    The confusion matrix is used to evaluate the perfomance of the model. [i,j] is how many inputs with true label i where predicted j. Thus everything off the diagonal (i != j) is an error.

    Parameters
    ---------
    y_true: nd array, shape(n,1)
            vector of outputs.
    y_pred: nd array, shape(n,1)
            vector of predicted class labels for each input.

    Returns
    --------
    C: nd array, shape(k,k)
        confusion matrix
    """
    n = y_pred.shape[0]

    y_pred = y_pred.flatten().astype(int)


    C = np.zeros(shape=(num_classes,num_classes),dtype=int)
    np.add.at(C,(y_true,y_pred),1)

    return C

def accuracy(C):
    """
    Shows the accuracy of a classifier - The fraction of correct predictions

    Parameters
    ---------
    C: nd array, shape(k,k)
                      confusion matrix. k - nujmber of classes

    Returns
    --------
    accuracy: np.float
              accuracy score
    
    
    """
    return np.trace(C) / np.sum(C)

def precision(C):
    """
    Shows the per-class precision of a classifier - The fraction of positive predictions that where true positives

    Parameters
    ---------
    C: nd array, shape(k,k)
                      confusion matrix. k - nujmber of classes

    Returns
    --------
    precision: nd array, shape(k,)
               per-class precision
    
    
    """
    return np.diag(C) / np.sum(C,axis=0)

def recall(C):
    """
    Shows the per-class recall of a classifier - Of all the true positivies, how many did the model actually find?
    Parameters
    ---------
    C: nd array, shape(k,k)
                      confusion matrix. k - nujmber of classes

    Returns
    --------
    recall: nd array, shape(k,)
            per-class recall
    
    
    """
    return np.diag(C) / np.sum(C,axis=1)

def f1_score(C):
    """
    Harmonic mean of precision and recall. Gives a single number that balances both metrics.
    Parameters
    ---------
    C: nd array, shape(k,k)
                      confusion matrix. k - nujmber of classes

    Returns
    --------
    F1_Score: np.float
              F1 score    
    
    """
    P = precision(C)
    R = recall(C)




    return (2 *(P * R)) / (P + R)

def silhouette(X,y_pred,centroids):
    """
    Returns the silhouette score for a clustering model.

    Parameters
    ---------
    X: nd array, shape(n,d)
       input points. n examples, d features

    y_pred: nd array, shape(n,1)
            predicted cluster label. y[i] is in the range [0,k-1] for k clusters

    centroids: nd array, shape(k,d)
               k centroids with d features.

    Returns
    --------

    silhouette score: np.float
                    the silhoette score   
    
    """
    k = centroids.shape[0]

    pairwise_distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis,:,:], axis =2)

    mean_cluster_distances = []
    for i in range(k):
        mask = y_pred == i
        mean_cluster_distances.append(np.mean(pairwise_distances[:,mask],axis=1))

    mean_cluster_distances = np.stack(mean_cluster_distances,axis=1)

    silhouette_scores = []
    for i in range(X.shape[0]):
        distances = mean_cluster_distances[i]

        a =distances[y_pred[i]]

        distances[y_pred[i]] = np.inf

        b = np.min(distances)

        s = (b - a) / (max(a,b))
        silhouette_scores.append(s)

    return np.mean(silhouette_scores)


