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
        X: nd array, shape(n,d)
           n input vectors with d features collected as the rows of a matrix

        y: nd array, shape(n,1)
           n outputs collected as a vector
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
        
    def analytic_solve(self,X,y):
        """
        Uses the normal equations to find the analytic solution

        Parameters:
        --------
        regulariser: str, "none", "ridge"
                     None uses normal eqations. Ridge uses ridge regression
        lmbda: float >= 0
                  Regularisation strength

        Returns:
        -------
        nd array, shape((d+1),1)
        
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
            
    def iterative_solve(self,X,y):
        """
        Finds the approximate solution via gradient descent with MSE loss.
        Parameters:
        ---------
        alpha: float
               learning rate

        iterations: int
                    number of iterations
        
        regulariser: str, "none", "ridge"
                     None uses normal eqations. Ridge uses ridge regression

        Returns:
        -------
        nd array, shape((d+1),1)
        
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
        # augment the input with constant 1
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        # shapes
        n = X.shape[0]
        d = X.shape[1]

        match self.mode:
            case "analytic":
                self.analytic_solve(X,y)
            case "iterative":
                self.iterative_solve(X,y)
            case _:
                raise ValueError(f"Unknow mode '{self.mode}'. Should be 'analytic' or 'iterative'.")

        return self # this allows us to chain: model.fit(X,y).predict(X_test)

    def predict(self,X):
        """
        Parameters:
        ------
        X: nd array, shape(n,d)
           input matrix
        
        Returns:
        -------
        y: nd array, shape(n,1)
           outputs
        """

        if self.w_hat is None:
            raise RuntimeError("Model has not yet been fitted. Use analytic_solve or iterative_solve to calculate weights")

        # augment the input
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.w_hat
    

    def r_squared(self,X,y):
        """
        Model evaluation by the R^2 metric. Comparison against the mean
        """
        y_hat = self.predict(X)
        y_bar = np.mean(self.y) # scalar mean of target

        ss_res = np.sum((y - y_hat) **2)
        ss_tot = np.sum((y - y_bar) **2)

        return 1 - ( ss_res / ss_tot )
    

    def adjusted_r_squared(self,X,y):
        """
        For comparing models with different numbers of features
        """
        
        # shapes
        n = X.shape[0]
        d = X.shape[1]

        Rsquared = self.r_squared(X,y)

        return 1 - (((1 - Rsquared ) * (n -1)) / (n - d -1))
     
