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
import numpy
import pandas
from rpy2 import robjects
from pandas.rpy.common import convert_to_r_dataframe
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..base import SurvivalAnalysisMixin, ExternalREstimatorMixin
from ..util import check_pandas_survival

__all__ = ['RandomSurvivalForest']

_rf = robjects.packages.importr('randomForestSRC')
_stats = robjects.packages.importr("stats", robject_translations={'format_perc': '_format_perc'})


def _convert_to_r(data):
    if not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)

    return convert_to_r_dataframe(data, strings_as_factors=True)


class RandomSurvivalForest(BaseEstimator, SurvivalAnalysisMixin, ExternalREstimatorMixin):
    """Wrapper around rfsrc function of the randomForestSRC R package.

    Parameters
    ----------
    ntree : integer, optional, default=1000
        Number of trees in the forest.

    mtry : integer, optional, default=sqrt(n_features)
        Number of variables randomly selected as candidates for each
        node split. The default is `sqrt(n_features)`, except for regression
        families where `n_features/3` is used. Values are rounded up.

    nodesize : integer, optional
        Minimum number of unique cases (data points) in a terminal
        node.  The defaults are: survival (3), competing risk (6),
        regression (5), classification (1), mixed outcomes (3).

    nodedepth : integer, optional
        Maximum depth to which a tree should be grown. The default
        behaviour is that this parameter is ignored.

    splitrule : string, optional
        Splitting rule used to grow trees.  Available rules are as follows:

           * Regression analysis: The default rule is weighted mean-squared error splitting
           * Classification analysis: The default rule is Gini index splitting
           * Survival analysis: Two rules are available. (1) The default
             rule is `logrank` which implements log-rank splitting;
             (2) `logrankscore` implements log-rank score splitting

    nsplit : non-negative integer, optional
        If non-zero, the specified tree splitting rule is randomized
        which can significantly increase speed.

    importance: string, optional, default='none'
        Method for computing variable importance (VIMP).
        Calculating VIMP can be computationally expensive when the
        number of variables is high, thus if VIMP is not needed
        consider setting `importance="none"`.

    na.action : string, optional, default='na.omit'
        Action taken if the data contains NA's.  Possible values
        are `na.omit`, `na.impute` or `na.random`. The default
        `na.omit` removes the entire record if even one of its
        entries is ‘NA’ (for x-variables this applies only to those
        specifically listed in 'formula').  Selecting `na.impute` or
        `na.random` imputes the data.

    nimpute : non-negative integer, optional, default=1
        Number of iterations of the missing data algorithm.
        Performance measures such as out-of-bag (OOB) error rates
        tend to become optimistic if `nimpute` is greater than 1.

    proximity : string or boolean, optional
        Should the proximity between observations be calculated?
        Creates an n by n matrix, which can be large. Choices are
        `inbag`, `oob`, `all`, `True`, or `False`.  Note that
        setting `proximity = True` is equivalent to `proximity = "inbag"`.

    seed : negative integer, optional
        Negative integer specifying seed for the random number generator.

    Attributes
    ----------
    ``model_`` : object
        Underlying R object containing the model.

    ``family_`` : string
         The family used in the analysis.

    ``proximity_`` : array-like, shape=[n_samples, n_samples]
        Proximity matrix recording the frequency of pairs of data
        points occur within the same terminal node.

    References
    ----------
    .. [1] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S.,
           "Random survival forests", The Annals of Applied Statistics, 2(3), 841-860, 2008.
    """

    def __init__(self, ntree=1000, mtry=None, nodesize=None, nodedepth=None, splitrule=None,
                 nsplit=0, importance='none', na_action='na.omit', nimpute=1, proximity=False,
                 seed=None):
        self.ntree = ntree
        self.mtry = mtry
        self.nodesize = nodesize
        self.nodedepth = nodedepth
        self.splitrule = splitrule
        self.nsplit = nsplit
        self.na_action = na_action
        self.nimpute = nimpute
        self.proximity = proximity
        self.importance = importance
        self.seed = seed

        self.classes_ = None
        self.fit_features_ = None
        self.n_features_ = None

    @property
    def family_(self):
        check_is_fitted(self, "model_")
        return self.model_.rx2('family')[0]

    @property
    def proximity_(self):
        check_is_fitted(self, "model_")
        if self.proximity:
            return numpy.asarray(self.model_.rx2('proximity'))

    def _get_r_params(self, arguments=None):
        if self.seed is not None and self.seed >= 0:
            raise ValueError('seed must be negative integer')

        all_params = self.get_params(deep=False)
        if arguments is None:
            arguments = all_params.keys()

        params = {}
        for key in arguments:
            value = all_params[key]
            if value is not None:
                if isinstance(value, numpy.integer):
                    value = int(value)
                params[key] = value

        params['na.action'] = params.pop('na_action', 'na.omit')
        return params

    def _get_r_predict_params(self):
        return self._get_r_params(['na_action', 'importance', 'proximity', 'seed'])

    def _fit(self, formula, rdata, params):
        if self.seed is not None:
            robjects.r('set.seed(%d)' % self.seed)
        self.model_ = _rf.rfsrc(formula, data=rdata, forest=True, **params)

    def _set_fit_features(self, filter_func, data_frame):
        self.fit_features_ = pandas.Index(filter(filter_func, data_frame.columns))

    def _fit_survival(self, X, event, time):
        data = pandas.concat((X, time, event), axis=1)

        formula = robjects.Formula("Surv({0}, {1}) ~ .".format(time.name, event.name))
        rdata = _convert_to_r(data)
        params = self._get_r_params()
        self._set_fit_features(lambda v: v != time.name and v != event.name, X)
        self._fit(formula, rdata, params)

    def fit(self, X, y):
        """Build a random survival forest from training data.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape = [n_samples, n_features]
            Data matrix.

        y : structured array or pandas.DataFrame, shape = [n_samples]
            A structured array (or data frame) containing the binary event indicator
            as first field (column), and time of event or time of censoring as
            second field (column).

        Returns
        -------
        self
        """
        X, event, time = check_pandas_survival(X, y)
        event = event.astype(numpy.int32)

        self._fit_survival(X, event, time)

        self.n_features_ = X.shape[1]
        return self

    def _predict(self, X, params):
        rdata = _convert_to_r(X)
        ret = _stats.predict(self.model_, newdata=rdata, **params)
        return ret

    def predict(self, X):
        """Predict hazard.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape = [n_samples, n_features]
            The input samples. If a :class:`pandas.DataFrame` is provided,
            the columns must match the columns that were used during training.

        Returns
        -------
        y : array of shape = [n_samples]
            Predicted hazard.
        """
        check_is_fitted(self, "model_")

        if X.shape[1] != self.n_features_:
            raise ValueError('expected %d features, but got %d' % (self.n_features_, X.shape[1]))

        if isinstance(X, pandas.DataFrame) and not X.columns.equals(self.fit_features_):
            raise ValueError('columns in test data do not match original training data')

        list_vector = self._predict(X, self._get_r_predict_params())
        predictions = numpy.asarray(list_vector.rx2('predicted'))

        return predictions
