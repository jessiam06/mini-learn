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

    


    def analytic_solve(self):
        """
        Uses the normal equations to find the analytic solution

        Returns:
        -------
        nd array, shape((d+1),1)
        
        """
        self.w_hat = np.linalg.solve( self.X_t @ self.X, self.X_t @ self.y )
        return self.w_hat

        
    def iterative_solve(self, alpha, iterations):
        """
        Finds the approximate solution via gradient descent with MSE loss
        Parameters:
        ---------
        alpha: float
               learning rate

        iterations: int
                    number of iterations

        Returns:
        -------
        nd array, shape((d+1),1)
        
        """
        
        # gradient wrt weights of MSE loss. w is nd array, shape((d+1),1)
        grad = lambda w: (2 / self.n) * ( self.X_t @ ( ( self.X @ w )  - self.y) )

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
    
