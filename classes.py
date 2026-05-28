import numpy as np

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
            ):
        pass


    def fit(self,X,y):
        pass

    def predict(self,X,y):
        pass 