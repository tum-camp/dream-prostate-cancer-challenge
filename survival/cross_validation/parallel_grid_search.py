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
from collections import Mapping
import logging

from IPython.parallel import Client, interactive
import numpy
import pandas
from sklearn.base import BaseEstimator
from sklearn.cross_validation import _check_cv, check_scoring, is_classifier
from sklearn.grid_search import ParameterGrid, _check_param_grid
from sklearn.utils import check_X_y

from . import _safe_split_params, _safe_split
from ..util import check_arrays_survival

__all__ = ['NestedGridSearchCV']

LOG = logging.getLogger(__name__)


# see http://stackoverflow.com/questions/10857250/python-name-space-issues-with-ipython-parallel
@interactive
def _fit_and_score_estimator(data):
    fold_id, train_index, test_index, params = data
    _fit_params = fit_params if fit_params is not None else {}
    _predict_params = predict_params if predict_params is not None else {}

    result = params.copy()
    try:
        result['score'] = _fit_and_score(clone(estimator), x, y, scorer, train_index, test_index,
                                         params, _fit_params, _predict_params)
    except Exception as e:
        Application.instance().log.exception(e)
        result['score'] = float('nan')

    result['n_samples_test'] = test_index.shape[0]
    result['fold'] = fold_id

    return result


def _get_best_parameters(fold_results, param_names, param_grid):
    """Get best setting of parameters from grid search

    Parameters
    ----------
    fold_results : pandas.DataFrame
        Contains performance measures as well as hyper-parameters
        as columns. Must contain a column 'fold'.

    param_names : list
        Names of the hyper-parameters. Each name should be a column
        in ``fold_results``.

    param_grid : list of dict
        Parameter grid to evaluate. Dictionaries in list are evaluated
        independently of each other.

    Returns
    -------
    max_performance : pandas.Series
        Maximum performance and its hyper-parameters
    """
    if pandas.isnull(fold_results.loc[:, 'score']).all():
        raise ValueError("Results are all NaN")

    max_performance = pandas.Series(index=param_names, dtype=numpy.object)
    all_mean_scores = numpy.empty(len(param_grid))
    for i, p in enumerate(param_grid):
        pnames = list(p.keys())
        # average across inner folds
        grouped = fold_results.drop('fold', axis=1).groupby(pnames)
        # cast to object to retain individual dtypes when doing row indexing
        grid_mean = grouped['score'].mean().reset_index().astype(numpy.object)

        # highest average across folds
        max_idx = grid_mean.loc[:, 'score'].idxmax()

        # best parameters
        best_score = grid_mean.loc[max_idx, :]
        all_mean_scores[i] = best_score['score']
        max_performance.update(best_score.drop('score'))

    max_performance['score'] = all_mean_scores.mean()
    return max_performance


def _get_parameter_names(param_grid):
    names = []
    for p in param_grid:
        names.extend(list(p.keys()))
    return names


class NestedGridSearchCV(BaseEstimator):
    """Cross-validation with nested hyper-parameter search for each training fold.

    The data is first split into ``cv`` train and test sets. For each training set.
    a grid search over the specified set of parameters is performed (inner cross-validation).
    The set of parameters that achieved the highest average score across all inner folds
    is used to re-fit a model on the entire training set of the outer cross-validation loop.
    Finally, results on the test set of the outer loop are reported.

    If auxiliary functions are used in functions passed to ``scoring`` or ``inner_cv``,
    they must be imported into the `__main__` namespace via
    :func:`IPython.parallel.DirectView.sync_imports` or be defined in the main script.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings. See :class:`sklearn.grid_search.ParameterGrid`.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        See :func:`sklearn.metrics.get_scorer` for details.

    cv : integer or cross-validation generator, default=3
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    inner_cv : integer or callable, default=3
        If an integer is passed, it is the number of folds.
        If callable, the function must have the signature ``inner_cv_func(X, y)``
        and return a cross-validation object, see :mod:`sklearn.cross_validation`
        module for the list of possible objects.

    profile : str, optional
        The IPython profile to use.

    Attributes
    ----------
    best_params_ : pandas.DataFrame
        Contains selected parameter settings for each fold.
        The validation score refers to average score across all folds of the
        inner cross-validation, the test score to the score on the test set
        of the outer cross-validation loop.

    grid_scores_ : list of pandas.DataFrame
        Contains full results of grid search for each training set of the
        outer cross-validation loop.

    scorer_ : callable
        Scorer function used on the held out data to choose the best
        parameters for the model.
    """

    def __init__(self, estimator, param_grid, scoring=None, cv=None, inner_cv=None,
                 profile=None):
        self.scoring = scoring
        self.estimator = estimator

        if isinstance(param_grid, Mapping):
            self.param_grid = [param_grid]
        else:
            self.param_grid = param_grid

        self.scoring = scoring
        self.cv = cv
        self.inner_cv = inner_cv
        self.profile = profile

        _check_param_grid(param_grid)

    def _grid_search_params_iter(self, train_X, train_y):
        if callable(self.inner_cv):
            inner_cv = self.inner_cv(train_X, train_y)
        else:
            inner_cv = _check_cv(self.inner_cv, train_X, train_y, classifier=is_classifier(self.estimator))

        param_iter = ParameterGrid(self.param_grid)
        LOG.info("Performing grid search over %d configurations" % len(param_iter))

        for fold_id, (train_index, test_index) in enumerate(inner_cv):
            for parameters in param_iter:
                yield fold_id + 1, train_index, test_index, parameters

    def _grid_search(self, train_X, train_y, train_params, predict_params):
        # distribute data to engines
        self._dview.push({'x': train_X, 'y': train_y, 'estimator': self.estimator, 'scorer': self.scorer_,
                          'fit_params': train_params, 'predict_params': predict_params}, block=True)

        cv_iter = self._grid_search_params_iter(train_X, train_y)

        ar = self._lview.map(_fit_and_score_estimator, cv_iter, chunksize=1)
        N = len(ar)
        while not ar.ready():
            ar.wait(5)
            LOG.info('Done %4i out of %4i', ar.progress, N)

        return pandas.DataFrame(ar.result)

    def _cv_iter_with_params(self, cv, params):
        for i, (train_index, test_index) in enumerate(cv):
            fold_id = i + 1
            parameters = params.loc[fold_id, :].to_dict()

            yield fold_id, train_index, test_index, parameters

    def _fit_and_score_with_parameters(self, X, y, cv, params, fit_params, predict_params):
        LOG.info("Performing testing with best parameters per fold")

        # distribute data to engines
        self._dview.push({'x': X, 'y': y, 'fit_params': fit_params, 'predict_params': predict_params}, block=True)

        cv_iter = self._cv_iter_with_params(cv, params)
        results = self._lview.map(_fit_and_score_estimator, cv_iter, block=True)

        df = pandas.DataFrame(results)
        df.set_index('fold', inplace=True)
        return df

    def _fit_holdout(self, X, y, fit_params, predict_params, X_test, y_test):
        """Perform cross-validation on training data for hyper-parameter search and test on hold-out data"""
        grid_results = self._grid_search(X, y, fit_params, predict_params)

        param_names = _get_parameter_names(self.param_grid)
        max_performance = _get_best_parameters(grid_results, param_names, self.param_grid)
        LOG.info("Best performance for training data:\n%s", max_performance) 

        best_parameters = pandas.DataFrame([max_performance], copy=True)
        best_parameters['score (Test)'] = 0.0
        best_parameters.rename(columns={'score': 'score (Validation)'}, inplace=True)

        LOG.info("Performing testing on hold-out data with best parameters")

        x_train_test = numpy.row_stack((X, X_test))
        y_train_test = numpy.concatenate((y, y_test), axis=0)
        train_index = numpy.arange(X.shape[0])
        test_index = numpy.arange(X.shape[0], x_train_test.shape[0])

        # distribute data to engines
        self._dview.push({'x': x_train_test, 'y': y_train_test,
                          'fit_params': fit_params, 'predict_params': predict_params}, block=True)

        results = self._lview.map(_fit_and_score_estimator,
                                  [(0, train_index, test_index, best_parameters.loc[0, param_names].to_dict())],
                                  block=True)

        scores = results[0]
        best_parameters['score (Test)'] = scores['score']

        self.best_params_ = best_parameters
        self.grid_scores_ = grid_results

    def _fit(self, X, y, cv, fit_params, predict_params):
        """Perform nested cross-validation"""
        param_names = _get_parameter_names(self.param_grid)
        n_folds = len(cv)

        best_parameters = []
        grid_search_results = []
        for i, (train_index, test_index) in enumerate(cv):
            LOG.info("Training fold %d of %d", i + 1, n_folds)

            train_X, train_y = _safe_split(X, y, train_index)
            train_params = _safe_split_params(train_index, X.shape[0], fit_params)
            train_predict_params = _safe_split_params(train_index, X.shape[0], predict_params)

            grid_results = self._grid_search(train_X, train_y, train_params, train_predict_params)
            grid_search_results.append(grid_results)

            max_performance = _get_best_parameters(grid_results, param_names, self.param_grid)
            LOG.info("Best performance for fold %d:\n%s", i + 1, max_performance)
            max_performance['fold'] = i + 1
            best_parameters.append(max_performance)

        best_parameters = pandas.DataFrame(best_parameters, dtype=numpy.object)
        best_parameters.set_index('fold', inplace=True)
        best_parameters['score (Test)'] = 0.0
        best_parameters.rename(columns={'score': 'score (Validation)'}, inplace=True)

        scores = self._fit_and_score_with_parameters(X, y, cv, best_parameters.loc[:, param_names],
                                                     fit_params, predict_params)
        best_parameters['score (Test)'] = scores['score']

        self.best_params_ = best_parameters
        self.grid_scores_ = grid_search_results

    def _init_cluster(self):
        rc = Client(profile=self.profile)
        dview = rc[:]
        lview = rc.load_balanced_view()

        with dview.sync_imports():
            from IPython.config import Application
            from survival.cross_validation import _fit_and_score
            from sklearn.base import clone

        return dview, lview

    def fit(self, X, y, fit_params=None, predict_params=None, X_test=None, y_test=None):
        """Do nested cross-validation.

        If ``X_test`` and ``y_test`` are not provided, nested cross-validation using
        ``X`` and ``y`' is performed, i.e., data is first split into *K* folds, where
        *K-1* folds are used for training and hyper-parameter selection and the
        remaining fold for testing. The training portion is again split into *T* folds
        to perform a grid-search over hyper-parameters. The parameters that achieved the
        best average performance across the *T* inner cross-validation folds are selected.
        Using these parameters, a model is trained on the entire training data and applied
        to the *K*-th testing fold.

        If ``X_test`` and ``y_test`` are provided, a regular cross-validation is performed on
        ``X`` and ``y`` to determine hyper-parameters as for the inner cross-validation above.
        Using the best performing parameters, a model is trained on all of ``X`` and ``y`` and
        applied to ``X_test`` and ``y_test`` for testing.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Feature matrix.

        y : structured array, shape = [n_samples]
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        fit_params : dict
            Additional arguments passed to the fit method.

        predict_params : dict
            Additional arguments passed to the predict method.

        X_test : array-like, shape = [n_test_samples, n_features]
            Hold-out data to perform testing on.

        y_test : array-like or sequence, shape = [n_test_samples]
            Target values of hold-out test data.

        Returns
        -------
        self
        """
        if y.dtype.names is None:
            X, y = check_X_y(X, y)
        else:
            X, event, time = check_arrays_survival(X, y, force_all_finite=False)
            y = numpy.fromiter(zip(event, time), dtype=[('event', numpy.bool), ('time', numpy.float64)])

        if X_test is not None:
            X_test, event_test, time_test = check_arrays_survival(X_test, y_test, force_all_finite=False)
            y_test = numpy.fromiter(zip(event_test, time_test), dtype=[('event', numpy.bool), ('time', numpy.float64)])

        cv = _check_cv(self.cv, X, y, classifier=is_classifier(self.estimator))

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        self._dview, self._lview = self._init_cluster()
        if X_test is None:
            self._fit(X, y, cv, fit_params, predict_params)
        else:
            self._fit_holdout(X, y, fit_params, predict_params, X_test, y_test)

        del self._dview
        del self._lview

        return self
