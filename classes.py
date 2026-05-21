import numpy as np

class LinearRegressor():
    
    def __init__(self, X, y):
        """
        Initialises the Linear Regression Class

        Parameters
        -----------
        X: nd array, shape(n,d)
           n input vectors with d features collected as the rows of a matrix

        Y: nd array, shape(n,1)
           n outputs collected as a vector
        """

        # augemnt the input with constant 1

        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        # data
        self.X = X
        self.y = y

        self.X_t = np.transpose(self.X) # transpose for calculation

        # shapes
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]

        # weights
        self.w_hat = None

        # loss gradients
        
        self.mse = lambda w: (2 / self.n) * ( self.X_t @ ( ( self.X @ w )  - self.y) )
        self.ridge = lambda w,lmbda: (2 / self.n) * ( self.X_t @ ( ( self.X @ w )  - self.y) ) + 2 * lmbda * w

    


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
                self.w_hat = np.linalg.solve( self.X_t @ self.X, self.X_t @ self.y )
            case "ridge":
                self.w_hat = np.linalg.solve(self.X_t @ self.X + self.n * lmbda * np.identity(self.d),self.X_t @ self.y)
            case _:
                self.w_hat = np.linalg.solve( self.X_t @ self.X, self.X_t @ self.y )
        

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
                grad = self.mse
            case "ridge":
                grad = self.ridge
            case _:
                grad = self.mse
       
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
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.w_hat
    
