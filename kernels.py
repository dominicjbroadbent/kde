import numpy as np

class BaseKernel:
    def __init__(self):
        """Initialise the kernel instance"""
        pass
    
    def __call__(self, x):
        """Evaluate the kernel function at x"""
        raise NotImplementedError()
        
class Gaussian(BaseKernel):
    def __init__(self):
        """Initialise the Gaussian kernel"""
        super().__init__()
        
    def __call__(self, x):
        """Evaluate the kernel function at x"""
        return ( 1 / np.sqrt( 2 * np.pi ) ) * np.exp( ( - 1 / 2 ) * x**2 )

class Epanechnikov(BaseKernel):
    def __init__(self):
        """Initialise the Epanechnikov kernel"""
        super().__init__()
    
    def __call__(self, x):
        """Evaluate the kernel function at x"""
        x = np.atleast_1d(np.array(x)) # Accept floats and arrays
        mask = ( np.abs(x) <= 1 )
        x[mask] = ( 3 / 4) * ( 1 - x[mask] **2 )
        x[np.logical_not(mask)] = 0
        return x      