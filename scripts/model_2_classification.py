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
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import get_scorer

from survival.column import categorical_to_numeric
from survival.datasets import load_arff_file
from survival.meta.ensemble_selection import BaseEnsembleSelection, MeanEstimator


def _load_and_predict_proba(data):
    idx, base_estimator_name = data
    base_estimator_dir = join(models_dir, base_estimator_name)
    estimators = []
    for afile in os.listdir(base_estimator_dir):
        path = join(base_estimator_dir, afile)
        with gzip.open(path, "rb") as fp:
            estimator = pickle.load(fp)
        estimators.append(estimator)

    avg_estimator = EnsembleAverage(estimators, name=base_estimator_name)
    pred_prob = avg_estimator.predict_proba(X)
    assert pred_prob.ndim == 2

    return idx, pred_prob, avg_estimator.classes_


def _fit_and_score_estimator(data):
    estimator, train_index, test_index, fit_params, idx, fold, name = data

    score = _fit_and_score(estimator, X, y, scorer, train_index, test_index,
                           estimator.get_params(), fit_params, {})

    filtered_params = {k: v for k, v in estimator.get_params().items()
                       if not (isinstance(v, (BaseEstimator, numpy.ndarray)) or callable(v))}

    client = MongoClient(mongodb_host)
    db = client.ensemble_selection_classification
    db.cv_scores.insert({"name": name, "id": idx, "fold": fold, "score": score,
                         "params": filtered_params})

    path = join(models_dir, name, "fold_%d.pickle.gz" % fold)
    with gzip.open(path, "wb", compresslevel=6) as fp:
        estimator.model_score = score
        estimator.model_fold = fold
        pickle.dump(estimator, fp)

    return idx, fold, score, None


class ParallelEnsembleSelectionClassifier(BaseEnsembleSelection):

    def __init__(self, base_estimators, scorer=None, n_estimators=0.2,
                 min_score=0., cv=None, verbose=0):
        super().__init__(meta_estimator=MeanEstimator(),
                         base_estimators=base_estimators,
                         scorer=scorer,
                         n_estimators=n_estimators,
                         min_score=min_score,
                         cv=cv,
                         verbose=verbose)

    def _prune_by_cv_score(self, scores, base_ensemble):
        mean_scores = scores.mean(axis=1)
        idx_good_models = numpy.flatnonzero(mean_scores >= self.min_score)
        if len(idx_good_models) == 0:
            raise ValueError("no base estimator exceeds min_score, try decreasing it")

        sorted_idx = numpy.argsort(-mean_scores, kind="mergesort")
        selected_models = sorted_idx[:self.n_estimators_]

        return base_ensemble[selected_models], mean_scores[selected_models]

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

    def _fit(self, X, y, cv, **fit_params):
        scores, base_ensemble = self._fit_and_score_ensemble(X, y, cv, **fit_params)
        self.fitted_models_, self.scores_ = self._prune_by_cv_score(scores, base_ensemble)

    def _predict_estimators(self, X):
        n_models = len(self.fitted_models_)

        dview.push({"X": X}, block=True)
        out = lview.map_sync(_load_and_predict_proba, ((i, est_name)
                                                       for i, est_name in enumerate(self.fitted_models_)))

        n_classes = out[0][1].shape[1]
        self.classes_ = out[0][2]
        predictions = numpy.empty((X.shape[0], n_classes, n_models), order="F")
        for i, p, _ in out:
            predictions[:, :, i] = p

        return predictions

    def load_from_db(self):
        self._check_params()

        client = MongoClient(mongodb_host)
        db = client.ensemble_selection_classification
        # average cross-validation scores of base estimators across folds (sorted in ascending order)
        out = db.cv_scores.aggregate([{"$group": {"_id": "$name", "mean": {"$avg": "$score"}}},
                                      {"$sort": {"mean": 1}}])
        scores = pandas.DataFrame(out["result"])
        idx = int(scores["mean"].searchsorted(self.min_score, side="left"))
        if idx == scores.shape[0]:
            raise ValueError("no base estimator exceeds min_score, try decreasing it")

        selected = scores.iloc[idx:, :].copy()
        selected.sort("mean", ascending=False, kind="mergesort", inplace=True)
        selected = selected.iloc[:self.n_estimators_, :]

        self.fitted_models_ = selected["_id"].values
        self.scores_ = selected["mean"].values

        return self


def append_estimators(est, param_grid, base_estimators):
    grid = ParameterGrid(param_grid)

    for i, params in enumerate(grid):
        e = clone(est)
        est.set_params(**params)
        name = "%s_%d" % (e.__class__.__name__, i)
        base_estimators.append((name, e))
    return base_estimators


def create_estimator(x, y, seed):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from survival.kernels import ClinicalKernelTransform
    from survival.meta.balanced_classifier import BalancedRandomForestClassifier, BalancedGradientBoostingClassifier

    base_estimators = []
    param_grid = {"n_estimators": [1000],
                  "min_samples_split": [3, 5, 10, 25, 50, 100],
                  "criterion": ["gini", "entropy"],
                  "max_features": [None, "sqrt", "log2", 0.1]}
    est = RandomForestClassifier(n_estimators=1000, class_weight="auto", random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    est = BalancedRandomForestClassifier(n_estimators=1000, random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    param_grid = {"n_estimators": [100, 500, 1000, 1500],
                  "subsample": [1.0, 0.75, 0.5],
                  "learning_rate": [0.25, 0.125, 0.06],
                  "max_leaf_nodes": [5, 10, 20],
                  "min_samples_split": [2, 5, 10, 20],
                  "max_features": [None, "sqrt", 0.5, 0.75]}
    est = GradientBoostingClassifier(random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    est = BalancedGradientBoostingClassifier(random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    param_grid = {"C": 2. ** numpy.arange(-12, 13, 2)}
    transform = ClinicalKernelTransform(fit_once=True)
    transform.prepare(x)

    est = SVC(kernel=transform, probability=True, class_weight="auto", max_iter=200, random_state=seed)
    append_estimators(est, param_grid, base_estimators)

    cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)
    return ParallelEnsembleSelectionClassifier(base_estimators, n_estimators=0.05, cv=cv)


def _create_directories(model_dir, base_estimators):
    from os.path import join
    from os import makedirs

    for name, _ in base_estimators:
        makedirs(join(model_dir, name), exist_ok=True)


def fit_and_dump(_x, _y, args):
    data = _x.copy()
    _x = categorical_to_numeric(_x)
    _y = _y[args.event].cat.codes.values

    model = create_estimator(data, _y, args.seed)
    if args.metric == 'avgprec':
        scoring_func = get_scorer("average_precision")
    else:
        scoring_func = get_scorer("roc_auc")
    model.set_params(scorer=scoring_func)

    print("Number of base estimators: %d" % len(model.base_estimators))

    print("Purging MongoDB cv_scores database")
    client = MongoClient(mongodb_host)
    db = client.ensemble_selection_classification
    db.cv_scores.remove({})

    print("Fitting %r" % model)
    _create_directories(args.models_dir, model.base_estimators)
    return model.fit(_x.values, _y)


def main(args):
    _x, _y, _x_test, _y_test = load_arff_file(args.input, [args.event, 'ENDTRS_C', 'ENTRT_PC'],
                                              survival=False, path_testing=args.test, to_numeric=False)

    model = fit_and_dump(_x, _y, args)
    model.load_from_db()
    print("Ensemble size: %d" % len(model))

    if _x_test is not None:
        _x_test = categorical_to_numeric(_x_test)
    proba = model.predict(_x_test.values)
    pred_labels = model.classes_.take(numpy.argmax(proba, axis=1), axis=0)

    result = pandas.DataFrame({"RISK": proba[:, model.classes_ == 1].ravel(),
                               "RPT": _x_test.index.to_series(),
                               "DISCONT": pred_labels})
    result.set_index("RPT", inplace=True)

    _results_file = 'results-%s-%s.csv' % (basename(args.input), "ensemble_selection_classification")
    result.to_csv(_results_file)


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
        from sklearn.base import BaseEstimator
        from sklearn.metrics import mean_squared_error
        from survival.cross_validation import _fit_and_score
        from survival.meta.ensemble_selection import EnsembleAverage

    _dview.push({"mongodb_host": mongodb_host, "models_dir": models_dir}, block=True)

    return _dview, _lview


if __name__ == '__main__':
    from IPython.parallel import Client
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Path to ARFF file to load')
    parser.add_argument('-t', '--test', help='Path to ARFF file to use for testing')
    parser.add_argument('--event', required=True,
                        help='Attribute denoting class label')
    parser.add_argument('-p', '--profile', default='default', help='Name of IPython parallel profile')
    parser.add_argument('-s', '--seed', type=int, default=19, help='Random number seed')
    parser.add_argument('--metric', default='avgprec', choices=['avgprec', 'rocauc'],
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
