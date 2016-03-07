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
import numbers

import numpy
from sklearn.base import clone, BaseEstimator
from sklearn.cross_validation import _check_cv
from sklearn.externals.joblib import Parallel, delayed

from .stacking import Stacking
from ..cross_validation import _fit_and_score

__all__ = ["EnsembleSelection", "EnsembleSelectionRegressor", "MeanEstimator"]


class EnsembleAverage(BaseEstimator):
    def __init__(self, base_estimators, name=None):
        self.base_estimators = base_estimators
        self.name = name
        if hasattr(self.base_estimators[0], "classes_"):
            self.classes_ = self.base_estimators[0].classes_

    def get_base_params(self):
        return self.base_estimators[0].get_params()

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        prediction = numpy.zeros(X.shape[0])
        for est in self.base_estimators:
            prediction += est.predict(X)

        return prediction / len(self.base_estimators)

    def predict_proba(self, X):
        prediction = None
        for est in self.base_estimators:
            p = est.predict_proba(X)
            if prediction is None:
                prediction = p
            else:
                prediction += p

            if numpy.any(est.classes_ != self.classes_):
                raise ValueError("class labels are different")

        return prediction / len(self.base_estimators)


class MeanEstimator(BaseEstimator):
    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        return X.mean(axis=X.ndim - 1)


def _fit_and_score_fold(est, x, y, scorer, train_index, test_index, fit_params, idx, fold):
    score = _fit_and_score(est, x, y, scorer, train_index, test_index,
                           est.get_params(), fit_params, {})
    return idx, fold, score, est


def _predict(estimator, X, idx):
    return idx, estimator.predict(X)


def _score_regressor(estimator, X, y, idx):
    name_time = y.dtype.names[1]
    error = (estimator.predict(X).ravel() - y[name_time]) ** 2
    return idx, error


class BaseEnsembleSelection(Stacking):

    def __init__(self, meta_estimator, base_estimators, scorer=None, n_estimators=0.2,
                 min_score=0.66, min_correlation=0.6,
                 cv=None, n_jobs=1, verbose=0):
        super().__init__(meta_estimator=meta_estimator,
                         base_estimators=base_estimators)

        self.scorer = scorer
        self.n_estimators = n_estimators
        self.min_score = min_score
        self.min_correlation = min_correlation
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._extra_params.extend(["scorer", "n_estimators", "min_score", "min_correlation",
                                   "cv", "n_jobs", "verbose"])

    def __len__(self):
        if hasattr(self, "fitted_models_"):
            return len(self.fitted_models_)
        else:
            return 0

    def _check_params(self):
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must not be zero or negative")

        if self.n_estimators >= 1.0:
            self.n_estimators = int(self.n_estimators)
            if self.n_estimators > len(self.base_estimators):
                raise ValueError(
                    "n_estimators (%d) must not exceed number of base learners (%d)" % (self.n_estimators,
                                                                                        len(self.base_estimators)))

        if isinstance(self.n_estimators, numbers.Integral):
            self.n_estimators_ = self.n_estimators
        else:
            self.n_estimators_ = max(int(self.n_estimators * len(self.base_estimators)), 1)

        if not callable(self.scorer):
            raise TypeError("scorer is not callable")

        if not (-1. <= self.min_correlation <= 1):
            raise ValueError("min_correlation must be in [-1; 1], but was %r" % self.min_correlation)

    def _create_base_ensemble(self, out, n_estimators, n_folds):
        """For each base estimator collect models trained on each fold"""
        ensemble_scores = numpy.empty((n_estimators, n_folds))
        base_ensemble = numpy.empty_like(ensemble_scores, dtype=numpy.object)
        for model, fold, score, est in out:
            ensemble_scores[model, fold] = score
            base_ensemble[model, fold] = est

        return ensemble_scores, base_ensemble

    def _create_cv_ensemble(self, base_ensemble, idx_models_included, model_names=None):
        """For each selected base estimator, average models trained on each fold"""
        fitted_models = numpy.empty(len(idx_models_included), dtype=numpy.object)
        for i, idx in enumerate(idx_models_included):
            model_name = self.base_estimators[idx][0] if model_names is None else model_names[idx]
            avg_model = EnsembleAverage(base_ensemble[idx, :], name=model_name)
            fitted_models[i] = avg_model

        return fitted_models

    def _fit_and_score_ensemble(self, X, y, cv, **fit_params):
        """Create a cross-validated model by training a model for each fold with the same model parameters"""
        fit_params_steps = self._split_fit_params(fit_params)

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose
        )(
            delayed(_fit_and_score_fold)(clone(estimator), X, y,
                                         self.scorer,
                                         train_index, test_index,
                                         fit_params_steps[name],
                                         i, fold)
            for i, (name, estimator) in enumerate(self.base_estimators)
            for fold, (train_index, test_index) in enumerate(cv))

        return self._create_base_ensemble(out, len(self.base_estimators), len(cv))

    def _add_diversity_score(self, scores, predictions):
        n_models = predictions.shape[1]

        cor = numpy.corrcoef(predictions, rowvar=0)
        numpy.fill_diagonal(cor, 0)

        final_scores = scores.copy()
        diversity = numpy.apply_along_axis(
            lambda x: (n_models - numpy.sum(x >= self.min_correlation)) / n_models,
            0, cor)

        final_scores += diversity
        return final_scores

    def _fit(self, X, y, cv, **fit_params):
        raise NotImplementedError()

    def fit(self, X, y=None, **fit_params):
        """Fit ensemble of models

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data.

        y : array-like, optional
            Target data if base estimators are supervised.

        Returns
        -------
        self
        """
        self._check_params()

        cv = _check_cv(self.cv, X, y)
        self._fit(X, y, cv, **fit_params)

        return self


class EnsembleSelection(BaseEnsembleSelection):
    """Ensemble selection for survival analysis that accounts for a score and correlations between predictions.

    The ensemble is pruned during training only according to the specified score (accuracy) and
    additionally for prediction according to the correlation between predictions (diversity).

    The hillclimbing is based on cross-validation to avoid having to create a separate validation set.

    Parameters
    ----------
    base_estimators : list
        List of (name, estimator) tuples (implementing fit/predict) that are
        part of the ensemble.

    scorer : callable
        Function with signature ``func(estimator, X_test, y_test, **test_predict_params)`` that evaluates the error
        of the prediction on the test data. The function should return a scalar value.
        *Larger* values of the score are assumed to be better.

    n_estimators : float or int
        If a float, the percentage of estimators in the ensemble to retain, if an int the
        absolute number of estimators to retain.

    min_score : float, optional, default = 0.66
        Threshold for pruning estimators based on scoring metric. After `fit`, only estimators
        with a score above `min_score` are retained.

    min_correlation : float, optional, default = 0.6
        Threshold for Pearson's correlation coefficient that determines when predictions of
        two estimators are significantly correlated.

    cv : int, a cv generator instance, or None
        The input specifying which cv generator to use. It can be an
        integer, in which case it is the number of folds in a KFold,
        None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator. The generator has to ensure
        that each sample is only used once for testing.

    n_jobs : int, default 1
        Number of jobs to run in parallel.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    `scores_` : ndarray, shape = [n_base_estimators,]
        Array of scores (relative to best performing estimator)

    `fitted_models_` : ndarray
        Selected models during training based on `scorer`.

    References
    ----------

    .. [1] Caruana, R., Munson, A., Niculescu-Mizil, A.
        "Getting the most out of ensemble selection". 6th IEEE International Conference on Data Mining, 828-833, 2006

    .. [2] Rooney, N., Patterson, D., Anand, S., Tsymbal, A.
        "Dynamic integration of regression models. International Workshop on Multiple Classifier Systems".
        Lecture Notes in Computer Science, vol. 3181, 164-173, 2004
    """

    def __init__(self, base_estimators, scorer=None, n_estimators=0.2,
                 min_score=0.2, min_correlation=0.6,
                 cv=None, n_jobs=1, verbose=0):
        super().__init__(meta_estimator=MeanEstimator(),
                         base_estimators=base_estimators,
                         scorer=scorer,
                         n_estimators=n_estimators,
                         min_score=min_score,
                         min_correlation=min_correlation,
                         cv=cv,
                         n_jobs=n_jobs,
                         verbose=verbose)

    def _fit(self, X, y, cv, **fit_params):
        scores, base_ensemble = self._fit_and_score_ensemble(X, y, cv, **fit_params)
        self.fitted_models_, self.scores_ = self._prune_by_cv_score(scores, base_ensemble)

    def _prune_by_cv_score(self, scores, base_ensemble, model_names=None):
        mean_scores = scores.mean(axis=1)
        idx_good_models = numpy.flatnonzero(mean_scores >= self.min_score)
        if len(idx_good_models) == 0:
            raise ValueError("no base estimator exceeds min_score, try decreasing it")

        total_score = mean_scores[idx_good_models]
        max_score = total_score.max()
        total_score /= max_score

        fitted_models = self._create_cv_ensemble(base_ensemble, idx_good_models, model_names)

        return fitted_models, total_score

    def _prune_by_correlation(self, X):
        n_models = len(self.fitted_models_)

        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_predict)(est, X, i)
            for i, est in enumerate(self.fitted_models_))

        predictions = numpy.empty((X.shape[0], n_models), order="F")
        for i, p in out:
            predictions[:, i] = p

        if n_models > self.n_estimators_:
            final_scores = self._add_diversity_score(self.scores_, predictions)
            sorted_idx = numpy.argsort(-final_scores, kind="mergesort")

            selected_models = sorted_idx[:self.n_estimators_]
            return predictions[:, selected_models]

        return predictions

    def _predict_estimators(self, X):
        predictions = self._prune_by_correlation(X)
        return predictions


class EnsembleSelectionRegressor(BaseEnsembleSelection):
    """Ensemble selection for regression that accounts for the accuracy and correlation of errors.

    The ensemble is pruned during training according to estimators' accuracy and the correlation
    between prediction errors per sample. The accuracy of the *i*-th estimator defined as
    :math:`\\frac{ \min_{i=1,\ldots, n}(error_i) }{ error_i }`.
    In addition to the accuracy, models are selected based on the correlation between residuals
    of different models (diversity). The diversity of the *i*-th estimator is defined as
    :math:`\\frac{n-count}{n}`, where *count* is the number of estimators for whom the correlation
    of residuals exceeds `min_correlation`.

    The hillclimbing is based on cross-validation to avoid having to create a separate validation set.

    Parameters
    ----------
    base_estimators : list
        List of (name, estimator) tuples (implementing fit/predict) that are
        part of the ensemble.

    scorer : callable
        Function with signature ``func(estimator, X_test, y_test, **test_predict_params)`` that evaluates the error
        of the prediction on the test data. The function should return a scalar value.
        *Smaller* values of the score are assumed to be better.

    n_estimators : float or int
        If a float, the percentage of estimators in the ensemble to retain, if an int the
        absolute number of estimators to retain.

    min_score : float, optional, default = 0.66
        Threshold for pruning estimators based on scoring metric. After `fit`, only estimators
        with a accuracy above `min_score` are retained.

    min_correlation : float, optional, default = 0.6
        Threshold for Pearson's correlation coefficient that determines when residuals of
        two estimators are significantly correlated.

    cv : int, a cv generator instance, or None
        The input specifying which cv generator to use. It can be an
        integer, in which case it is the number of folds in a KFold,
        None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator. The generator has to ensure
        that each sample is only used once for testing.

    n_jobs : int, default 1
        Number of jobs to run in parallel.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    `scores_` : ndarray, shape = [n_base_estimators,]
        Array of scores (relative to best performing estimator)

    `fitted_models_` : ndarray
        Selected models during training based on `scorer`.

    References
    ----------

    .. [1] Caruana, R., Munson, A., Niculescu-Mizil, A.
        "Getting the most out of ensemble selection". 6th IEEE International Conference on Data Mining, 828-833, 2006

    .. [2] Rooney, N., Patterson, D., Anand, S., Tsymbal, A.
        "Dynamic integration of regression models. International Workshop on Multiple Classifier Systems".
        Lecture Notes in Computer Science, vol. 3181, 164-173, 2004
    """
    def __init__(self, base_estimators, scorer=None, n_estimators=0.2,
                 min_score=0.66, min_correlation=0.6,
                 cv=None, n_jobs=1, verbose=0):
        super().__init__(meta_estimator=MeanEstimator(),
                         base_estimators=base_estimators,
                         scorer=scorer,
                         n_estimators=n_estimators,
                         min_score=min_score,
                         min_correlation=min_correlation,
                         cv=cv,
                         n_jobs=n_jobs,
                         verbose=verbose)

    def _fit(self, X, y, cv, **fit_params):
        scores, base_ensemble = self._fit_and_score_ensemble(X, y, cv, **fit_params)
        fitted_models, scores = self._prune_by_cv_score(scores, base_ensemble)

        if len(fitted_models) > self.n_estimators_:
            fitted_models, scores = self._prune_by_correlation(fitted_models, scores, X, y)

        self.fitted_models_ = fitted_models
        self.scores_ = scores

    def _prune_by_cv_score(self, scores, base_ensemble, model_names=None):
        mean_scores = scores.mean(axis=1)
        mean_scores = mean_scores.min() / mean_scores

        idx_good_models = numpy.flatnonzero(mean_scores >= self.min_score)
        if len(idx_good_models) == 0:
            raise ValueError("no base estimator exceeds min_score, try decreasing it")

        fitted_models = self._create_cv_ensemble(base_ensemble, idx_good_models, model_names)

        return fitted_models, mean_scores

    def _prune_by_correlation(self, fitted_models, scores, X, y):
        n_models = len(fitted_models)

        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_score_regressor)(est, X, y, i)
            for i, est in enumerate(fitted_models))

        error = numpy.empty((X.shape[0], n_models), order="F")
        for i, err in out:
            error[:, i] = err

        final_scores = self._add_diversity_score(scores, error)
        sorted_idx = numpy.argsort(-final_scores, kind="mergesort")

        selected_models = sorted_idx[:self.n_estimators_]

        return fitted_models[selected_models], final_scores

    def _predict_estimators(self, X):
        n_models = len(self.fitted_models_)

        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_predict)(est, X, i)
            for i, est in enumerate(self.fitted_models_))

        predictions = numpy.empty((X.shape[0], n_models), order="F")
        for i, p in out:
            predictions[:, i] = p

        return predictions
