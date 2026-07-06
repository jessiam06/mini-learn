import numpy as np
import textwrap

class LinearRegressor():
    
    def __init__(
            self, 
            mode = "analytic",
            alpha = 0.1,
            iterations = 100, 
            regulariser = "none", 
            lmbda = 1.0
            ):
        
        """
        Initialises the Linear Regression Class

        Parameters
        -----------
        mode: str. "analytic" or "iterative"
              Defines whether to use analytic solution or iterative solution
        alpha: float
               learning rate
        iterations: int
                    number of iterations for iterative solution
        regulariser: str. "none" or "ridge"
                     chooses which regulariser to use
        lmbda: float >=0
               regularisation strength
        """

        self.mode = mode
        self.alpha = alpha
        self.iterations = iterations
        self.regulariser = regulariser
        self.lmbda = lmbda

        # To ne calculated
        self.w_hat = None
        
        
    def __grad_mse(self,w):
        """
        Gradient for Mean Squared Error Loss
        """
        return (2 / self.n) * ( self.X.T @ ( ( self.X @ w )  - self.y) )


    def __grad_ridge(self,w):
        """
        Gradient for L2 Ridge Loss
        """
        mask = np.ones_like(w)
        mask[-1] = 0
        return self.__grad_mse(w) + 2 * self.lmbda * mask *  w
        
    def _analytic_solve(self,X,y):
        """
        Uses the normal equations to find the analytic solution

        Parameters
        --------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples. d - number of features
        y: nd array, shape(n,1)
           output vector

        Returns
        --------
        weights: nd array, shape((d+1),1)
                 augmented weights + bias array
        
        """
        # shapes
        n = X.shape[0]
        d = X.shape[1]

        match self.regulariser:
            case "none":
                self.w_hat = np.linalg.solve( X.T @ X, X.T @ y )
            case "ridge":
                self.w_hat = np.linalg.solve(X.T @ X + n * self.lmbda * np.eye(d),X.T @ y)
            case _:
                raise ValueError(f"Unknown regulariser '{self.regulariser}'. Choose 'none' or 'ridge'.")
        

        return self.w_hat
            
    def _iterative_solve(self,X,y):
        """
        Finds the approximate solution via gradient descent with MSE loss.

        Parameters
        ---------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples. d - number of features

        Returns
        -------
        weights: nd array, shape((d+1),1)
                 augmented weights + bias array
        
        """
        # shapes
        n = X.shape[0]
        d = X.shape[1]

        
        # gradient wrt weights of MSE loss. w is nd array, shape((d+1),1)
        match self.regulariser:
            case "none":
                grad = self.__grad_mse
            case "ridge":
                grad = self.__grad_ridge
            case _:
                raise ValueError(f"Unknown regulariser '{self.regulariser}'. Choose 'none' or 'ridge'.")
       
        # intialise w randomly
        self.w_hat = np.random.default_rng(42).standard_normal((d, 1))

        # update via gradient descent
        for _ in range(self.iterations):
            self.w_hat  = self.w_hat - self.alpha * grad(self.w_hat)

        return self.w_hat
        

    def fit(self, X, y):
        """
        Performs Linear Regression on the labelled data to calculate optimal model weights
        
        Parameters
        ----------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples. d - number of features
        y: nd array, shape(n,1)
            output vector

        Returns
        ------
        self: LinearRegressor()
               returns itself, useful for chaining operations
        """
        # augment the input with constant 1
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        # shapes
        n = X.shape[0]
        d = X.shape[1]

        match self.mode:
            case "analytic":
                self._analytic_solve(X,y)
            case "iterative":
                self._iterative_solve(X,y)
            case _:
                raise ValueError(f"Unknow mode '{self.mode}'. Should be 'analytic' or 'iterative'.")

        return self # this allows us to chain: model.fit(X,y).predict(X_test)

    def predict(self,X):
        """
        Predicts an ouptut from unlabelled data

        Parameters
        ------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples. d - number of features
        
        Returns
        -------
        y: nd array, shape(n,1)
           output vector
        """

        if self.w_hat is None:
            raise RuntimeError("Model has not yet been fitted. Call fit() first")

        # augment the input
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.w_hat
    

    def r_squared(self,X,y):
        """
        Model evaluation by the R^2 metric. Comparison against the mean

        Parameters
        ------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples. d - number of features
        y: nd array, shape(n,1)
            output vector

        Returns
        --------
        r_squared: float -> [0,1]
                   the r squared value of the data

        """
        y_hat = self.predict(X)
        y_bar = np.mean(y) # scalar mean of target

        ss_res = np.sum((y - y_hat) **2)
        ss_tot = np.sum((y - y_bar) **2)

        return 1 - ( ss_res / ss_tot )
    

    def adjusted_r_squared(self,X,y):
        """
        For comparing models with different numbers of features

        Parameters
        ------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples. d - number of features
        y: nd array, shape(n,1)
            output vector

        Returns
        --------
        adjusted_r_squared: float -> [0,1]
                   the adjusted r squared value of the data


        """
        
        # shapes
        n = X.shape[0]
        d = X.shape[1] - 1 # doesn't include bias

        Rsquared = self.r_squared(X,y)

        return 1 - (((1 - Rsquared ) * (n -1)) / (n - d -1))
     
class LogisticRegressor():
    def __init__(
            self,
            num_classes = 2,
            alpha = 0.1,
            iterations = 100,
            regulariser = "ridge",
            lmbda = 1.0
            ):
        
        """
        Initialies a Logistic Regression model

        Parameters
        --------
        num_classes: int
                     number of class labels. (0,1 ..., k)
        alpha: float
               learning rate
        iterations: int
                    number of iterations in the training loop
        
        """
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.iterations = iterations
        self.regulariser = regulariser
        self.lmbda = lmbda

        # randomly initialise weights
        self.weights = None

    def _softmax(self,Z):
        """
        Computes the softmax probabilities of a logit matrix Z

        Parameters
        ------
        Z: nd array, shape(n,k)
           matrix of logits. n - number of inputs, k - number of classes

        Returns
        -------
        P: nd array, shape(n,k)
           matrix of probabilities
        """
        Z = Z - np.max(Z, axis=1,keepdims=True)
        exp_Z = np.exp(Z)

        return exp_Z / np.sum(exp_Z, axis=1,keepdims=True)

    def _cel_grad(self,P,y,X):
        """
        Gradient for Cross Entropy Loss w.r.t weights
         
        Parameters
        --------
        P: nd array, shape(n,k)
           matrix of class probabilities. n - number of inputs, k number of classes
        Y: nd array, shape(n,k)
           matrix of one hot encoded labels. y_i,j = 1 if the ith input is of class j, 0 otherwise
        X: nd array, shape(n,d+1)
           augmented input matrix. n - number of inputs, d+1, number of features
        
        Returns
        --------
        grad_L: nd array, shape(k,d+1)
                gradient of Cross Entropy Loss w.r.t weights
        """
        n = X.shape[0]
        return  ((P - y).T @ X ) / n

    def fit(self,X,y):
        """
        Performs Logistic Regression on the input data

        Parameters
        -------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples, d - number of features

        y: nd array, shape(n,k)
           one hot encoding of class. y_i = 1 if x is in class i, 0 otherwise. k - number of classes
        
        Returns
        -------
        self: LogisticRegressor
              returns itself, allowing chaining. e.g model.fit(X,y).predict(X_test)
        
        """
        # augment the input with constant 1
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        # shapes
        n = X.shape[0]
        d = X.shape[1]

        
        # randomly initialise weights
        self.weights = np.random.default_rng(42).standard_normal((self.num_classes,d))

       

        for _ in range(self.iterations):
             # logits matrix. Row i contains all k logits for input x. 
            Z = X @ self.weights.T

            # softmax to get probabilities. [i,k] contains the probability that input i is in class k.
            P = self._softmax(Z)

            match self.regulariser:
                case "none":
                    loss = self._cel_grad(P,y,X)
                case "ridge":
                    mask = np.ones_like(self.weights)
                    mask[-1] = 0
                    loss = self._cel_grad(P,y,X) + 2 * self.lmbda * mask *  self.weights
                case _:
                    raise ValueError(f"unregocnized regulariser '{self.regulariser}'. Use 'none' or 'ridge'.")

            
            self.weights = self.weights - self.alpha * loss

        return self

    def predict(self,X):
        """
        Returns the most likely class for each input

        Parameters
        ----------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of observations, d - number of features

        Returns
        ------
        out: nd array, shape(n,1)
             the most likely class label
               
        """
        if self.weights is None:
            raise RuntimeError("Model has not yet been fitted. Call fit() first.")


        # augment the input with constant 1
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        
        if self.weights is None:
            raise RuntimeError("Model has not yet been fit. call model.fit()")
        
        # logits
        Z = X @ self.weights.T

        #probabilities
        P = self._softmax(Z)

        return np.argmax(P,axis=1).reshape(-1,1) # the -1 tells numpy to infer that dimension automatically
    
    def _confusion_matrix(self,y_true,y_pred):
        """
        The confusion matrix is used to evaluate the perfomance of the model. [i,j] is how many inputs with true label i where predicted j. Thus everything off the diagonal (i != j) is an error.

        Parameters
        ---------
        y_true: nd array, shape(n,k)
                matrix where each row is the one hot encoding of the corresponding input.
        y_pred: nd array, shape(n,1)
                vector of predicted class labels for each input.

        Returns
        --------
        C: nd array, shape(k,k)
           confusion matrix
        """
        n = y_pred.shape[0]

        y_true = np.argmax(y_true,axis=1)
        y_pred = y_pred.flatten().astype(int)


        C = np.zeros(shape=(self.num_classes,self.num_classes),dtype=int)
        np.add.at(C,(y_true,y_pred),1)

        return C

    def evaluate(self,y_true,y_pred,verbose = True):
        """
        Shows the accuracy, precision, recall and F1 score of the model

        Parameters
        ---------
        y_true: nd array, shape(n,k)
                matrix where each row is the one hot encoding of the corresponding input.
        y_pred: nd array, shape(n,1)
                vector of predicted class labels for each input.
        verbose: bool
                 Choice of whether to return a verbose string or a minimal dictionary

        Returns
        --------
        evaluated: str, dict
                   string or dictionary containing accuracy, precision, recall and F1 score
        """
        C = self._confusion_matrix(y_true,y_pred)
        
        accuracy = np.trace(C) / np.sum(C)

        precision = np.diag(C) / np.sum(C,axis=0)
        
        recall = np.diag(C) / np.sum(C,axis=1)

        F1_score = (2 *(precision * recall)) / (precision + recall)

        out = textwrap.dedent(f"""
        --------------
        Accuracy:    {accuracy}
        Fraction of correct predictions

        Precision:   {precision}
        Fraction of positive predictions that are true positive (per class)

        Recall:      {recall}
        Fraction of all true positives to all predictions (per class)

        F1 Score:    {F1_score}
        Harmonic mean of precision and recall
        --------------
        """).strip()

        return out if verbose else {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        F1_score
        }




class DecisionTreeNode():
    """
    A node of the decision tree, to be created and traversed recursively.
    """
    def __init__(self):
        self.feature     = None # which feaure to split on
        self.threshold   = None # threshold for that feature
        self.left        = None # Node or leaf
        self.right       = None # Node or leaf
        self.leaf_class  = None # only set if this is a leaf

    def is_leaf(self):
        """
        Returns
        ------
        is_leaf: bool
                 checks if this node is a leaf or not
        """
        return self.leaf_class is not None
    
class DecisionTree():
    def __init__(
            self,
            num_classes = 2,
            max_depth = 5,
            min_samples = 2):
        """
        Initialises a Classification Tree Model

        Parameters
        ---------
        num_classes: int
                     number of classes the model expects
        max_depth: int
                     maximum number of splits the tree can make, to prevent overfitting

        min_samples: int
                     minimum number of samples per node needed for a split
        """
        self.root = None
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples = min_samples

    def _entropy(self,Y):
        """
        Returns the Shannon Entropy of the given data. A measure of how "mixed" the data is.

        Parameters
        --------
        Y: ndarray, shape(n,1)
           class labels. n examples, numbered 0-k for k+1 classes
        
        Returns
        --------
        Entropy: float
                 Shannon Entropy
        
        """
        p = self._proportions(Y)
        p = p[p > 0]  # mask out zeros before taking log

        return -1 * np.sum(p * np.log2(p))

    def _proportions(self,Y):
        """
        Returns the proportion of the given data that each class accounts for

        Parameters
        --------
        Y: ndarray, shape(n,1)
           class labels. n examples, numbered 0-(k-1) for k classes
        
        Returns
        --------
        proportions: ndarray, size k
                    array of the proportions of each class. Used to calculation entropy
                    
        """
        counts = np.bincount(Y.flatten(),minlength=self.num_classes)
        proportions = counts / Y.size
        return proportions
    
    def _build_tree(self,X,Y, depth):
        """
        Recursively builds the decision tree by finding the split that maximises information gain at each step

        Parameters
        -------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples, d - number of features

        Y: nd array, shape(n,1)
           class labels for each input. Y_i takes a value from 0-(k-1), for k classes

        depth: int
               the depth of the tree at the current node. should be set to 0 upon starting the algorithm
        
        Returns
        -------
        self: DecisionTreeNode
              the root node of the full decision tree, that can be traversed recusrively
        
        """
  
        node = DecisionTreeNode()

        # ------ base cases, return leaves ------
        all_same_classes = len(np.unique(Y)) == 1
        too_deep = depth >= self.max_depth
        too_few_samples = X.shape[0] < self.min_samples

        if all_same_classes or too_deep or too_few_samples:
            node.leaf_class = np.bincount(Y.flatten(), minlength=self.num_classes).argmax()
            return node

        # compute thresholds
        thresholds = np.sort(X,axis=0)    # sorts all the columns (features)
        thresholds = (thresholds[:-1, :] + thresholds[1:, :]) / 2 # calculates the midpoints for each pair of consecutive entries

        # find best information gain
        best_feature = None
        best_threshold = None 
        best_information_gain = 0 

       

        # loop through each threshold
        parent_entropy = self._entropy(Y)

        for (_, col_index), threshold in np.ndenumerate(thresholds):
            # use a boolean mask to make the split
            row_mask = X[:, col_index] >= threshold

            child_A = X[row_mask]
            labels_A = Y[row_mask]

            child_B = X[~row_mask]
            labels_B = Y[~row_mask]

            # calculate information gain
            weighted_child_entropy = ((child_A.shape[0] / X.shape[0]) * self._entropy(labels_A) + (child_B.shape[0] / X.shape[0]) * self._entropy(labels_B))
            
            information_gain = parent_entropy - weighted_child_entropy
            
            if information_gain >= best_information_gain:
                best_information_gain = information_gain
                best_feature = col_index
                best_threshold = threshold


        # recurse on the best ones

        if best_feature is None:
            # return if all splits produce no information gain
            node.leaf_class = np.bincount(Y.flatten(), minlength=self.num_classes).argmax()
            return node


        node.feature = best_feature
        node.threshold = best_threshold

        best_mask = X[:, best_feature] >= best_threshold

        node.left  = self._build_tree(X[best_mask],  Y[best_mask],  depth + 1)
        node.right = self._build_tree(X[~best_mask], Y[~best_mask], depth + 1)

        return node



    def fit(self,X,Y):
        """
        Generates a Classification and Regression Decision Tree

        Parameters
        -------
        X: nd array, shape(n,d)
           matrix of inputs. n - number of examples, d - number of features

        Y: nd array, shape(n,1)
           class labels for each input. Y_i takes a value from 0-k
        
        Returns
        -------
        self: Decision Tree
              returns itself, allowing chaining. e.g model.fit(X,y).predict(X_test)
        
        """
        self.root = self._build_tree(X,Y,depth=0)
        return self
    
    def _predict_one(self,x,node):
        """
        predicts the class of a single training example

        Parameters
        -------
        x: ndarray, shape(1,d)
           a single example, with d features

        node: DecisionTreeNode
              the current node of the decision tree to check against

        
        Returns
        -------
        leaf_class: int
                    the class prediction for this example
        
        
        """
        if node.is_leaf():
            return node.leaf_class
        
        if x[node.feature] >= node.threshold:
            return self._predict_one(x,node.left)
        else:
            return self._predict_one(x,node.right)
        

    def predict(self,X):
        """
        predicts the class labels for a matrix of inputs

        Parameters
        -------
        X: ndarray, shape(n,d)
           a set of unlabelled data. n examples with d features

        Returns
        -------
        classes: ndarray, shape(n,1)
                    the class predictions for each example
        
        
        """
        if self.root is None:
            raise RuntimeError("Model has not yet been fitted. Call model.fit() first")
        
        return np.array([self._predict_one(x,self.root) for x in X]).reshape(-1,1)
    

class KMeans():
    def __init__(
            self,
            k=3,
            iterations=300):
        """
        Initialises a K++ means clustering model

        Parameters
        ---------
        k: int
           Number of clusters. This must  be known beforehand

        iterations: int
                    maximum number of iterations the model should run for, if centroids have not yet stabilised

        
        """
        
        self.k = k
        self.iterations = iterations

        self.centroids = None
        self.nearest_cluster = None
        


    def fit(self,X):
        """
        Runs the K++ Means algorithm to calculate cluster centroids, and assign all points to a single cluster

        Parameters
        ----------
        X: ndarray, shape(n,d)
           input data. n examples, d features

        Returns
        --------
        model: self
               fitted model
        """


        # intialise centroids (k++ means)
        centroids = np.zeros(shape=(self.k,X.shape[1]))

        # uniformly choose first centroid    
        centroids[0] = X[np.random.choice(X.shape[0])]


        for i in range(1,self.k):
            # shape (n, i ,d) pairwise difference vectors
            distances = X[:, np.newaxis, :] - centroids[:i, : ]
            
            # shape (n,i) squared distance from every point to every centroid
            squared_distances = np.sum(distances**2, axis=2)

            # shape (n,) each points distance to its nearest centroid
            min_squared_distance = np.min(squared_distances, axis=1)

            probabilities = min_squared_distance / np.sum(min_squared_distance)

            centroids[i] = X[np.random.choice(X.shape[0],p=probabilities)]

        nearest_cluster = None        
        for i in range(self.iterations):
            # assignment step
            # shape (n, i ,d) pairwise difference vectors
            distances = X[:, np.newaxis, :] - centroids
            
            # shape (n,i) squared distance from every point to every centroid
            squared_distances = np.sum(distances**2, axis=2)

            nearest_cluster = np.argmin(squared_distances, axis=1)



            old_centroids = centroids.copy()
            # update step
            for j in range(self.k):
                mask = nearest_cluster == j

                new_centroid = np.mean(X[mask],axis = 0)
                centroids[j] = new_centroid

            if np.array_equal(old_centroids, centroids):
                break
                    
                    

        self.centroids = centroids
        self.nearest_cluster = nearest_cluster

        return self
    
    def predict(self,X):
        """
        Returns the closest cluster for each point

        Parameters
        ----------
        X: ndarray, shape(n,d)
           input data. n examples, d features

        Returns
        --------
        nearest_cluster: ndarray, shape(n,1)
                         cluster associated with each datapoint
        """

        if self.centroids is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # shape (n, i ,d) pairwise difference vectors
        distances = X[:, np.newaxis, :] - self.centroids
        
        # shape (n,i) squared distance from every point to every centroid
        squared_distances = np.sum(distances**2, axis=2)

        nearest_cluster = np.argmin(squared_distances, axis=1)

        return nearest_cluster.reshape(-1, 1)
    

class GaussianMixture():
    def __init__(
            self,
            k = 2,
            iterations = 300):
        
        self.k = k
        self.iterations = iterations

        self.mus  = None # means vectors
        self.pis  = None # mixing weights
        self.covs = None # covariance matrices
    
    def _vectorised_multivariate_pdf(self,X,Mu,Cov,Cov_inv):

        diffs = X[:, np.newaxis, :] - Mu[np.newaxis, : , :] # shape(n,k,d)

        quad_form = np.einsum('ijl,jlm,ijm->ij', diffs, Cov_inv, diffs) # shape(n,k)

        d = Cov.shape[1]

        normaliser = 1 / np.sqrt(((2 * np.pi)** d)  * np.linalg.det(Cov))

        return normaliser * np.exp(-0.5 * quad_form)


    def fit(self,X):
        # run k means to get reasonable centroids
        k_means = KMeans(self.k, self.iterations)

        # starting parameters
        self.mus  = k_means.fit(X).centroids
        self.pis  = np.full(self.k, 1/self.k)
        self.covs = np.tile(np.eye(X.shape[1]),(self.k,1,1))



        for _ in range(self.iterations):
            # E-step
            
            Cov_inv = np.linalg.inv(self.covs)

            gaussian_density = self._vectorised_multivariate_pdf(X,self.mus,self.covs,Cov_inv) # shape(n,k)

            r_numerators = self.pis * gaussian_density

            responsibilities = r_numerators / (np.sum(r_numerators,axis=1,keepdims=True))

            old_log_likelihood = np.sum(np.log(np.sum(r_numerators, axis=1)))

            # M-step

            N_k = np.sum(responsibilities,axis=0)

            self.mus = responsibilities.T @ X / N_k[:, np.newaxis]

            self.pis = N_k / X.shape[0]

            diffs = X[:, np.newaxis, :] - self.mus[np.newaxis, : , :] # shape(n,k,d)
            self.covs = np.einsum('ij,ijl,ijm->jlm', responsibilities, diffs, diffs) / N_k[:, np.newaxis, np.newaxis]

            self.covs += 1e-6 * np.eye(X.shape[1]) # regularisation term. If a cluster falls onto a single point, covariance matrix will be singular.

            # recompute with updated parameters
            cov_inv_new = np.linalg.inv(self.covs)
            new_density = self._vectorised_multivariate_pdf(X, self.mus, self.covs, cov_inv_new)
            new_r_numerators = self.pis * new_density

            new_log_likelihood = np.sum(np.log(np.sum(new_r_numerators, axis=1)))

            if abs(new_log_likelihood - old_log_likelihood) < 1e-6:
                break


        return self
    
    def predict(self,X):
        if self.mus is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        Cov_inv = np.linalg.inv(self.covs)

        gaussian_density = self._vectorised_multivariate_pdf(X,self.mus,self.covs,Cov_inv) # shape(n,k)

        r_numerators = self.pis * gaussian_density

        responsibilities = r_numerators / (np.sum(r_numerators,axis=1,keepdims=True))

        return responsibilities



