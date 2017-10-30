import numpy as np
from scipy.stats import kstest, norm


class LinearRegression(object):
    '''Ordinary Least Squares...'''
    def __init__(self, X, y, fit_intercept=True):
        self.X = X
        self.y = y
        self.fit_intercept = fit_intercept
        self._coeff = None

    def fit(self, check_residuals=True, threshold=0.05):
        if self.fit_intercept:
            self.X = self._add_intercept(self.X)
        self._solve_ols()

        if check_residuals:
            print 'checking residuals...'
            if self._check_residuals(threshold):
                print '...residuals are gaussian distributed at %3.2f...' % threshold
            else:
                print '...residuals are Not gaussian distributed...'

    def _add_intercept(self, X):
        '''add a column of 1s in the X matrix...'''
        return np.insert(X, 0, np.ones_like(X[:, 0]), axis=1)

    def _solve_ols(self):
        '''matrix solution for OLS...'''
        XT = self.X.transpose()
        XTX = np.dot(XT, self.X)
        XTX_i = np.linalg.inv(XTX)
        self._coeff = np.dot(np.dot(XTX_i, XT), self.y)

    def _calculate_residuals(self):
        return self.y - np.dot(self.X, self._coeff)

    def _check_residuals(self, threshold=0.05):
        '''check residuals using ks_test for normality...'''
        residuals = self._calculate_residuals()
        mu, std = np.mean(residuals), np.std(residuals)

        def g_cdf(x):
            return norm.cdf(x, mu, std)

        # standard 2-sided ks test...
        t_stat, p_value = kstest(residuals, g_cdf)
        # returns True for gaussian noise
        return p_value > threshold

    def calc_r_squared(self, adjusted=True):
        '''returns the standard R2 value...'''
        n_obs, n_var = self.X.shape
        y_ = np.mean(self.y)
        p = np.dot(self.X, self._coeff)

        ss_t = np.sum(np.square(self.y - y_))
        ss_e = np.sum(np.square(self.y - p))

        r2 = 1.0 - ss_e/ss_t
        if adjusted:
            return 1.0 - (1.0 - r2) * ((n_obs - 1) / (n_obs - n_var))
        return r2
