import numpy as np
from kernels import BaseKernel

class KernelDensityEstimate:
    def __init__(self, X, kernel):
        """
        Initialise the KDE with a data array X and an instance of the BaseKernel class
        """ 
        assert X.ndim == 2, 'The data array X must have two dimensions'
        self.X = X # data array, observations as rows, covariates as columns
        
        assert isinstance(kernel, BaseKernel), 'The kernel must be an instance of BaseKernel'
        self.kernel = kernel
    
    def __call__(self, x):
        """Compute the weighted kernel density estimate at the point x, or an array of points x"""
        assert x.ndim == 2, 'The evaluation arra x must have two dimensions'
        assert self.h > 0, 'The bandwidth parameter should be positive' 
        
        # Compute ( x_i - X_j ) / h array to pass into the kernel
        eval_pts = ( 1 / self.h ) * ( x[np.newaxis, :] - self.X[:, np.newaxis])[:, :, 0]

        return ( 1 / self.h ) * ( self.kernel( eval_pts ) ).mean(axis = 0)
        
    def silverman(self):
        """
        Use the normal distribution approximation bandwidth, also called Silverman's rule of thumb.
        
        Generally only recommended for data that is close to Gaussian, and for Gaussian kernels.
        """
        std = self.X.std()
        q3, q1 = np.percentile(self.X, [75 ,25])
        iqr = q3 - q1
        
        h = 0.9 * np.min([std, iqr]) * ( self.X.shape[0] ** (-1/5) )
        print(f"Silverman's rule of thumb: h = {round(h, 4)}")
        self.h = h
                