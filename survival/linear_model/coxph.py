# This file is part of the submission of the Chair for Computer Aided
# Medical Procedures, Technische Universitaet Muenchen, Germany to the
# Prostate Cancer DREAM Challenge 2015.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import warnings

import numpy
from scipy.linalg import solve
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted

from ..base import SurvivalAnalysisMixin
from ..util import check_arrays_survival

__all__ = ['CoxPHSurvivalAnalysis']


class CoxPHOptimizer:
    """Negative partial log-likelihood of Cox proportional hazards model"""

    def __init__(self, X, event, time, alpha):
        self.x = X
        self.event = event
        self.time = time
        self.alpha = alpha
        self._n_samples = self.x.shape[0]

    def loss(self, w):
        """Compute negative partial log-likelihood

        Parameters
        ----------
        w : array, shape = [n_features]
            Estimate of coefficients

        Returns
        -------
        loss : float
            Average negative partial log-likelihood
        """
        xw = numpy.dot(self.x, w)

        at_risk = numpy.empty(self.x.shape[0])
        for i in range(self.x.shape[0]):
            idx = self.time >= self.time[i]
            at_risk[i] = logsumexp(xw[idx])

        loss = numpy.mean(self.event * (xw - at_risk))
        if self.alpha > 0:
            loss -= 0.5 * self.alpha * squared_norm(w)

        return -loss

    def gradient(self, w):
        """Compute gradient of negative partial log-likelihood

        Parameters
        ----------
        w : array, shape = [n_features]
            Estimate of coefficients

        Returns
        -------
        gradient : ndarray, shape = [n_features]
            Gradient with respect to model's coefficients
        """
        em = numpy.exp(numpy.dot(self.x, w))
        # multiply row x[i, :] by em[i]
        x_scaled = em[:, numpy.newaxis] * self.x

        gf = numpy.zeros(self.x.shape[1])

        for i in range(self.x.shape[0]):
            if not self.event[i]:
                continue

            idx = numpy.flatnonzero(self.time >= self.time[i])
            wi = numpy.sum(em.take(idx))

            # build column-wise sum, i.e. sum over all samples/rows
            gf += self.x[i, :] - numpy.sum(x_scaled.take(idx, axis=0), axis=0) / wi

        if self.alpha > 0:
            gf -= self.alpha * w

        gf /= self._n_samples

        assert numpy.isfinite(gf).all()
        return -gf

    def hessian(self, w):
        """Compute Hessian matrix

        Parameters
        ----------
        w : array, shape = [n_features]
            Estimate of coefficients

        s : array, shape = [n_features]
            Vector to be multiplied by Hessian

        Returns
        -------
        hessian : ndarray, shape = [n_features, n_features]
            Hessian matrix with respect to model's coefficients
        """
        em = numpy.exp(numpy.dot(self.x, w))

        n_samples, n_features = self.x.shape
        hess = numpy.zeros((n_features, n_features))

        for i in range(n_samples):
            if not self.event[i]:
                continue

            idx = numpy.flatnonzero(self.time >= self.time[i])
            emi = em.take(idx)
            wi = numpy.sum(emi)

            v = -numpy.outer(emi, emi) / (wi * wi)
            numpy.fill_diagonal(v, (emi + numpy.diagonal(v)) / wi)

            vv = numpy.take(self.x, idx, axis=0)

            hess += numpy.dot(numpy.dot(vv.T, v), vv)

        if self.alpha > 0:
            diag_idx = numpy.diag_indices(n_features)
            hess[diag_idx] = hess.diagonal() + self.alpha

        hess /= self._n_samples

        return hess


class CoxPHSurvivalAnalysis(BaseEstimator, SurvivalAnalysisMixin):
    """Cox proportional hazards model.

    Uses the Breslow method to handle ties and Newton-Raphson optimization.

    Parameters
    ----------
    alpha : float, optional, default = 0
        Regularization parameter for ridge regression penalty.

    n_iter : int, optional, default = 100
        Maximum number of iterations.

    tol : float, optional, default = 1e-8
        Convergence criteria. Convergence is based on the deviance::

        |deviance_old - deviance_new| / (|deviance_old| + 0.1) < tol

    verbose : bool, optional, default = False
        Whether to print statistics on the convergence
        of the optimizer.

    Attributes
    ----------
    `coef_` : ndarray, shape = [n_features]
        Coefficients of the model

    References
    ----------
    .. [1] Cox, D. R. Regression models and life tables (with discussion).
           Journal of the Royal Statistical Society. Series B, 34, 187-220, 1972.
    """

    def __init__(self, alpha=0, n_iter=100, tol=1e-8, verbose=False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        """Minimize negative partial log-likelihood for provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data matrix

        y : list of two arrays of same length
            The first element is a boolean array of event indicators and
            the second element the observed survival/censoring times.
        """
        X, event, time = check_arrays_survival(X, y)

        if self.alpha < 0:
            raise ValueError("alpha must be positive, but was %r" % self.alpha)

        optimizer = CoxPHOptimizer(X, event, time, self.alpha)

        w0 = numpy.zeros(X.shape[1])
        i = 0
        deviance_old = 2 * optimizer.loss(w0)
        while True:
            g = optimizer.gradient(w0)
            h = optimizer.hessian(w0)
            w0 -= solve(h, g, overwrite_a=True, overwrite_b=True, check_finite=False)

            deviance = 2 * optimizer.loss(w0)
            res = numpy.abs(deviance - deviance_old) / (numpy.abs(deviance_old) + 0.1)
            if res < self.tol:
                break
            elif i > self.n_iter:
                warnings.warn(('Optimization did not converge: Maximum number of iterations has been exceeded.'),
                              stacklevel=2)
                break

            deviance_old = deviance
            i += 1

        if self.verbose:
            print("Optimization stopped after %d iterations" % (i + 1))

        self.coef_ = w0

        return self

    def predict(self, X):
        """Predict risk scores.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data matrix.

        Returns
        -------
        risk_score : array, shape = [n_samples]
            Predicted risk scores.
        """
        check_is_fitted(self, "coef_")

        X = numpy.atleast_2d(X)

        return numpy.dot(X, self.coef_)
