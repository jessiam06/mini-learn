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

