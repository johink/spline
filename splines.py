#Implementation of spline regression
import numpy as np
from sklearn.linear_model import LinearRegression

class SplineRegression(object):
    """Class which implements linear regression using simple splines to fit non-linear patterns
    """
    def __init__(self):
        self.betas = None
        self.intercepts = None
        self.cutpoints = None
        
        #Contains betas corrected for prior beta values
        self._betas = None
    
    def fit(self, xs, ys, cutpoints = []):
        """ 
Parameters
----------
xs : Numpy Array
    An array containing the independent (x) variable values
ys : Numpy Array
    An array containing the dependent (y) variable values
cutpoints : Iterable of numeric values (default [])
    The values contained in cutpoints will be used to determine which xs and ys are relevant for each individual spline.  If an empty list is passed, fit will be equivalent to an ordinary simple regression
    
Returns
-------
betas : Numpy Array
    An array containing the slopes of each spline
intercepts : Numpy Array
    An array containing the intercepts of each spline
    
Description
-----------
Fits a series of connected linear regressions to describe the data in xs and ys.  A separate regression is conducted for each cutpoint "zone."  For example, with cutpoints [0, 10], three regressions will be conducted:  On all data below 0, all data between 0 and 10, and all data above 10.
"""
        if isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray) and len(xs) == len(ys):
            cutpoints = list(cutpoints)
            if cutpoints == [] or type(cutpoints[0]) in [int,float,long]:
                lr = LinearRegression()
                self.cutpoints = cutpoints

                #We add to the cutpoints two additional values, so that we may both upper- and lower-bound for all values
                cuts = [min(xs)] + cutpoints + [max(xs)]
                betas = []
                intercepts = []
                for i in range(1, len(cuts)):
                    sub_mask = (xs >= cuts[i - 1]) & (xs <= cuts[i])

                    #sklearn weirdness requires us to reshape these into proper (n, 1) arrays instead of (n,)
                    lr.fit(xs[sub_mask].reshape(-1, 1), ys[sub_mask].reshape(-1, 1))
                    betas.append(lr.coef_[0][0])
                    intercepts.append(lr.intercept_[0])
                    self.betas = np.array(betas)
                    self._betas = self.betas - np.hstack([0, self.betas[:-1]])
                    self.intercepts = np.array(intercepts)
                return self.betas, self.intercepts
            else:
                raise ValueError("The values in cutpoints must be numeric")
        else:
            raise ValueError("Make sure xs and ys are both Numpy Arrays of the same length")
    
    def predict(self, xs):
        """
        Parameters
        ----------
        xs : Numpy Array
            An array containing the x values for which to receive predicted y-values

        Returns
        -------
        y_preds : Numpy Array
            An array containing the predicted y values
        """
        if self.betas is None:
            raise BaseException("Cannot call predict before a model is fitted")

        if isinstance(xs, np.ndarray) and len(xs) > 0:
            zero = np.zeros(len(xs))
            result = self.intercepts[0] + xs * self._betas[0]
            for beta, cutpoint in zip(self._betas[1:], self.cutpoints):
                result += np.maximum(zero, xs - cutpoint) * beta
            return result
        elif not xs:
            raise ValueError("xs array cannot be empty!")
        else:
            raise TypeError("xs must be a Numpy Array")