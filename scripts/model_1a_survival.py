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
from os.path import basename

import numpy
import pandas
from pymongo import MongoClient
from sklearn.base import clone
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import KFold

from survival.column import categorical_to_numeric
from survival.datasets import load_arff_file
from survival.meta import EnsembleSelection


def _fit_and_score_estimator(data):
    estimator, train_index, test_index, fit_params, idx, fold, name = data

    score = _fit_and_score(estimator, X, y, scorer, train_index, test_index,
                           estimator.get_params(), fit_params, {})

    filtered_params = {k: v for k, v in estimator.get_params().items()
                       if not (isinstance(v, BaseEstimator) or callable(v))}

    client = MongoClient(mongodb_host)
    db = client.ensemble_selection_survival
    db.cv_scores.insert({"name": name, "id": idx, "fold": fold, "score": score,
                         "params": filtered_params})

    path = join(models_dir, name, "fold_%d.pickle.gz" % fold)
    with gzip.open(path, "wb", compresslevel=6) as fp:
        estimator.model_score = score
        estimator.model_fold = fold
        pickle.dump(estimator, fp)

    return idx, fold, score, None


def _load_and_predict(data):
    idx, base_estimator_name = data
    base_estimator_dir = join(models_dir, base_estimator_name)
    estimators = []
    for afile in os.listdir(base_estimator_dir):
        path = join(base_estimator_dir, afile)
        with gzip.open(path, "rb") as fp:
            estimator = pickle.load(fp)
            if isinstance(estimator, ExternalREstimatorMixin):
                estimator.model_ = ri2py(estimator.model_)
        estimators.append(estimator)

    avg_estimator = EnsembleAverage(estimators, name=base_estimator_name)
    pred = avg_estimator.predict(X)
    if not isinstance(estimator, ExternalREstimatorMixin):
        pred = numpy.exp(pred)

    return idx, pred


def _score_cindex(est, X_test, y_test, **kwargs):
    y_pred = est.predict(X_test)
    name_event, name_time = y_test.dtype.names

    result = concordance_index_censored(y_test[name_event], y_test[name_time], y_pred)
    return result[0]


def _create_directories(model_dir, base_estimators):
    from os.path import join
    from os import makedirs

    for name, _ in base_estimators:
        makedirs(join(model_dir, name), exist_ok=True)


class ParallelEnsembleSelection(EnsembleSelection):
    """Fits estimators in parallel using IPython.parallel and stores
    all results in MongoDB
    """
    def __init__(self, base_estimators, scorer=None, n_estimators=0.2,
                 min_score=0.66, min_correlation=0.6,
                 cv=None, verbose=0):
        super().__init__(base_estimators=base_estimators,
                         scorer=scorer,
                         n_estimators=n_estimators,
                         min_score=min_score,
                         min_correlation=min_correlation,
                         cv=cv,
                         verbose=verbose)

    def _fit_and_score_ensemble(self, X, y, cv, **fit_params):
        """Create a cross-validated model by training a model for each fold with the same model parameters"""
        fit_params_steps = self._split_fit_params(fit_params)

        dview.push({"X": X, "y": y, "scorer": self.scorer}, block=True)
        out = lview.map_sync(_fit_and_score_estimator,
                             ((estimator, train_index, test_index, fit_params_steps[name],
                               i, fold, name)
                              for i, (name, estimator) in enumerate(self.base_estimators)
                              for fold, (train_index, test_index) in enumerate(cv)))

        return self._create_base_ensemble(out, len(self.base_estimators), len(cv))

    def _prune_by_correlation(self, X):
        n_models = len(self.fitted_models_)

        dview.push({"X": X}, block=True)
        out = lview.map_sync(_load_and_predict, ((i, est_name)
                                                 for i, est_name in enumerate(self.fitted_models_)))

        predictions = numpy.empty((X.shape[0], n_models), order="F")
        for i, p in out:
            predictions[:, i] = p

        if n_models > self.n_estimators_:
            final_scores = self._add_diversity_score(self.scores_, predictions)
            sorted_idx = numpy.argsort(-final_scores, kind="mergesort")

            client = MongoClient(mongodb_host)
            db = client.ensemble_selection_survival
            for i in range(final_scores.shape[0]):
                db.corr_scores.insert({"name": self.fitted_models_[i],
                                       "score": final_scores[i]})

            selected_models = sorted_idx[:self.n_estimators_]
            return predictions[:, selected_models]

        return predictions

    def load_from_db(self):
        self._check_params()

        client = MongoClient(mongodb_host)
        db = client.ensemble_selection_survival
        # average cross-validation scores of base estimators across folds (sorted in ascending order)
        out = db.cv_scores.aggregate([{"$group": {"_id": "$name", "mean": {"$avg": "$score"}}},
                                      {"$sort": {"mean": 1}}])
        scores = pandas.DataFrame(out["result"])
        idx = int(scores["mean"].searchsorted(self.min_score, side="left"))
        if idx == scores.shape[0]:
            raise ValueError("no base estimator exceeds min_score, try decreasing it")

        selected = scores.iloc[idx:, :]
        self.fitted_models_ = selected["_id"].values
        self.scores_ = selected["mean"].values

        return self


def _init_cluster_and_database(profile=None):
    rc = Client(profile=profile)
    _dview = rc[:]
    _lview = rc.load_balanced_view()

    with _dview.sync_imports():
        import os
        from os.path import join
        import gzip
        import pickle
        import numpy
        from pymongo import MongoClient
        from rpy2.robjects.conversion import ri2py
        from sklearn.base import BaseEstimator
        from survival.base import ExternalREstimatorMixin
        from survival.cross_validation import _fit_and_score
        from survival.metrics import concordance_index_censored
        from survival.meta.ensemble_selection import EnsembleAverage

    _dview.push({"mongodb_host": mongodb_host, "models_dir": models_dir}, block=True)

    return _dview, _lview


def append_estimators(est, param_grid, base_estimators):
    grid = ParameterGrid(param_grid)

    for i, params in enumerate(grid):
        e = clone(est)
        est.set_params(**params)
        name = "%s_%d" % (e.__class__.__name__, i)
        base_estimators.append((name, e))
    return base_estimators


def create_estimator(data, seed):
    from survival.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis,\
        RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from survival.kernels import ClinicalKernelTransform
    from survival.svm import FastSurvivalSVM

    base_estimators = []
    param_grid = {"n_estimators": [100, 500, 1000, 1500],
                  "subsample": [1.0, 0.75, 0.5],
                  "learning_rate": [0.25, 0.125, 0.06]}
    est = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    param_grid = {"ntree": [1000],
                  "nodesize": [3, 5, 10, 25, 50, 100],
                  "splitrule": ["logrank"],
                  "nsplit": [0, 2, 5, 10]}
    est = RandomSurvivalForest(seed=-seed)
    append_estimators(est, param_grid, base_estimators)

    param_grid = {"n_estimators": [100, 500, 1000, 1500],
                  "subsample": [1.0, 0.75, 0.5],
                  "learning_rate": [0.25, 0.125, 0.06],
                  "max_leaf_nodes": [5, 10, 20],
                  "min_samples_split": [2, 5, 10, 20],
                  "max_features": [None, "sqrt", 0.5, 0.75]}
    est = GradientBoostingSurvivalAnalysis(random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    param_grid = {"alpha": 2. ** numpy.arange(-12, 13, 2)}
    transform = ClinicalKernelTransform(fit_once=True)
    transform.prepare(data)
    est = FastSurvivalSVM(kernel=transform.pairwise_kernel, random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    cv = KFold(data.shape[0], n_folds=5, shuffle=True, random_state=seed)
    return ParallelEnsembleSelection(base_estimators, n_estimators=0.05, scorer=_score_cindex, cv=cv)


class _DummyEstimator(object):
    def fit(self):
        return

    def predict(self):
        return


def fit_and_dump(_x, _y, args):
    data = _x.copy()
    _x = categorical_to_numeric(_x)

    model = create_estimator(data, args.seed)
    print("Number of base estimators: %d" % len(model.base_estimators))

    print("Purging MongoDB cv_scores database")
    client = MongoClient(mongodb_host)
    db = client.ensemble_selection_survival
    db.cv_scores.remove({})

    print("Purging MongoDB corr_scores database")
    client = MongoClient(mongodb_host)
    db = client.ensemble_selection_survival
    db.corr_scores.remove({})

    print("Fitting %r" % model)
    _create_directories(args.models_dir, model.base_estimators)
    return model.fit(_x.values, _y)


def main(args):
    _x, _y, _x_test, _y_test = load_arff_file(args.input, [args.event, args.time], args.outcome,
                                              args.test, to_numeric=False)

    model = fit_and_dump(_x, _y, args)
    model.load_from_db()
    print("Ensemble size: %d" % len(model))

    if _x_test is not None:
        _x_test = categorical_to_numeric(_x_test)
    p = model.predict(_x_test.values)

    result = pandas.DataFrame({"RISK": p, "RPT": _x_test.index.to_series()})
    result.set_index("RPT", inplace=True)

    _results_file = 'results-%s-%s.csv' % (basename(args.input), "ensemble_selection_survival")
    result.to_csv(_results_file)


if __name__ == '__main__':
    from IPython.parallel import Client
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to ARFF file to use for training', required=True)
    parser.add_argument('-t', '--test', help='Path to ARFF file to use for testing')
    parser.add_argument('--event', help='Attribute denoting binary event indicator', required=True)
    parser.add_argument('--time', help='Attribute denoting survival/censoring time', required=True)
    parser.add_argument('--outcome', default="1",
                        help="Value denoting the outcome of interest for the event indicator attribute")
    parser.add_argument('-p', '--profile', default='default', help='Name of IPython parallel profile')
    parser.add_argument('-s', '--seed', type=int, default=19, help='Random number seed')
    parser.add_argument('--metric', default='cindex', choices=['cindex', 'timeroc'],
                        help='Which metric to use for performance evaluation')
    parser.add_argument('--host', default="localhost",
                        help="Name of host running MongoDB server")
    parser.add_argument('--models-dir', default=".ensemble_models")

    _args = parser.parse_args()

    mongodb_host = _args.host
    models_dir = _args.models_dir
    dview, lview = _init_cluster_and_database(_args.profile)

    try:
        main(_args)
    finally:
        dview.clear()
