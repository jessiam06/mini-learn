import numpy as np

class LinearRegressor():
    
    def __init__(self, X, y):
        """
        Initialises the Linear Regression Class

        Parameters
        -----------
        X: nd array, shape(n,d)
           n input vectors with d features collected as the rows of a matrix

        y: nd array, shape(n,1)
           n outputs collected as a vector
        """

        # augemnt the input with constant 1

        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        # data
        self.X = X
        self.y = y

        # shapes
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]

        # weights, ouptut
        self.w_hat = None
        self.y_hat = None

        # loss gradients
        
        
        def __grad_mse(self,w):
            return (2 / self.n) * ( self.X.T @ ( ( self.X @ w )  - self.y) )


        def __grad_ridge(self,w,lmbda):
            mask = np.ones_like(w)
            mask[-1] = 0
            return self.__grad_mse(w) + 2 * lmbda * mask *  w
        

        mask = np.ones((self.d,1))
        mask[-1] = 0 # we don't want to regularise bias
        

    


    def analytic_solve(self,regulariser = "none",lmbda = 1):
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
        match regulariser:
            case "none":
                self.w_hat = np.linalg.solve( self.X.T @ self.X, self.X.T @ self.y )
            case "ridge":
                self.w_hat = np.linalg.solve(self.X.T @ self.X + self.n * lmbda * np.eye(self.d),self.X.T @ self.y)
            case _:
                self.w_hat = np.linalg.solve( self.X.T @ self.X, self.X.T @ self.y )
        

        return self.w_hat
            
    def iterative_solve(self, alpha, iterations,regulariser = "none"):
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
        
        # gradient wrt weights of MSE loss. w is nd array, shape((d+1),1)
        match regulariser:
            case "none":
                grad = self.__grad_mse
            case "ridge":
                grad = self.__grad_ridge
            case _:
                raise ValueError(f"Unknown regulariser '{regulariser}'. Choose 'none' or 'ridge'.")
       
        # intialise w randomly
        self.w_hat = np.random.default_rng(42).standard_normal((self.d, 1))

        # update via gradient descent
        for _ in range(iterations):
            self.w_hat  = self.w_hat - alpha * grad(self.w_hat)

        return self.w_hat
    
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
        # augment
        if self.w_hat is None:
            raise RuntimeError("Model has not yet been fitted. Use analytic_solve or iterative_solve to calculate weights")

        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))
        self.y_hat = X @ self.w_hat

        return self.y_hat
    

    def R_squared(self):
        """
        Model evaluation by the R^2 metric. Comparison against the mean
        """
        if self.y is None:
            raise RuntimeError("Model has not yet been fitted. Use analytic_solve or iterative_solve to calculate outputs")

        y_bar = np.mean(self.y) # scalar mean of target

        return 1 - ( (np.linalg.norm(self.y - self.y_hat)**2) / (np.linalg.norm(self.y - y_bar))**2 )
    

    def adjusted_R_squared(self):
        """
        For comparing models with different numbers of features
        """
        Rsquared = self.R_squared()

        return 1 - (((1 - Rsquared ) * (self.n -1)) / (self.n - self.d -1))
     
