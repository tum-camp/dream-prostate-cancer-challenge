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
import datetime
import json
import logging
from os.path import basename

import pandas
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import get_scorer

from survival.cross_validation.parallel_grid_search import NestedGridSearchCV
from survival.datasets import load_arff_file
from survival.column import categorical_to_numeric

LOG = logging.getLogger('validate')


def get_estimator(method, seed, data):
    if "rf" == method:
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=1000, class_weight="auto", random_state=seed)
    elif "brf" == method:
        from survival.meta.balanced_classifier import BalancedRandomForestClassifier
        est = BalancedRandomForestClassifier(n_estimators=1000, random_state=seed)
    elif "bgbc" == method:
        from survival.meta.balanced_classifier import BalancedGradientBoostingClassifier
        est = BalancedGradientBoostingClassifier(random_state=seed)
    elif "gbc" == method:
        from sklearn.ensemble import GradientBoostingClassifier
        est = GradientBoostingClassifier(random_state=seed)
    elif method == "ck_svm":
        from sklearn.svm import SVC
        from survival.kernels import ClinicalKernelTransform

        transform = ClinicalKernelTransform(fit_once=True)
        transform.prepare(data)

        est = SVC(kernel=transform, probability=True, class_weight="auto", random_state=seed)
    else:
        raise ValueError("invalid method %r" % method)

    return est


def get_param_grid(grid_file):
    with open(grid_file) as fp:
        param_grid = json.load(fp)

    if not isinstance(param_grid, dict):
        raise ValueError("expected param_grid of type dict, but got %s" % type(param_grid))

    return param_grid


def print_settings(estimator, x, y):
    LOG.info("Training data shape = (%d, %d)", *x.shape)
    LOG.info("class labels: %s", pandas.Series(y).value_counts())
    LOG.info("%r", estimator)


def run_grid_search(estimator, param_grid, metric, X, y, X_test, y_test, seed, profile):
    _train_test_iter = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)
    inner_cv_func = lambda zx, zy: StratifiedShuffleSplit(zy, n_iter=10, test_size=0.2, random_state=seed)
    if metric == 'avgprec':
        scoring_func = get_scorer("average_precision")
    else:
        scoring_func = get_scorer("roc_auc")

    _grid_search = NestedGridSearchCV(estimator, param_grid, scoring_func, cv=_train_test_iter,
                                      inner_cv=inner_cv_func, profile=profile)
    _grid_search.fit(X, y, X_test=X_test, y_test=y_test)
    return _grid_search


def write_results(best_params, output):
    LOG.info('Writing output to {0}'.format(output))

    with open(output, 'w') as fp:
        dt = datetime.datetime.now()
        fp.write("# Created on {0}\n".format(dt.strftime("%Y-%m-%d %H:%M")))
        fp.write("# {0}\n".format(" ".join(sys.argv)))
        best_params.to_csv(fp)


def main(args):
    LOG.info("Using IPython profile %s", args.profile)
    rc = parallel.Client(profile=args.profile)

    _x, _y, _x_test, _y_test = load_arff_file(args.input, [args.event, 'ENDTRS_C', 'ENTRT_PC'],
                                              survival=False, path_testing=args.test, to_numeric=False)
    _data = _x.copy()
    _x = categorical_to_numeric(_x)
    if _x_test is not None:
        _x_test = categorical_to_numeric(_x_test)
    _y = _y[args.event].cat.codes.values

    _estimator = get_estimator(args.method, args.seed, _data)
    _param_grid = get_param_grid(args.params)
    print_settings(_estimator, _x, _y)

    _grid_search = run_grid_search(_estimator, _param_grid, args.metric,
                                   _x, _y, _x_test, _y_test,
                                   args.seed, args.profile)

    if args.test is None:
        _output = "results-%s-%s-%s.csv" % (args.metric, basename(args.input).rstrip(".arff"), args.method)
    else:
        _output = "results-%s-%s+%s-%s.csv" % (args.metric, basename(args.input).rstrip(".arff"),
                                               basename(args.test).rstrip(".arff"),
                                               args.method)
    write_results(_grid_search.best_params_, _output)

    rc[:].clear()


if __name__ == '__main__':
    from IPython import parallel
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method',
                        choices=["brf", "bgbc", "rf", "gbc", "ck_svm"],
                        help='Name of method to use', required=True)
    parser.add_argument('-i', '--input', required=True,
                        help='Path to ARFF file to load')
    parser.add_argument('-p', '--params', required=True,
                        help='Path to JSON file defining parameter grid')
    parser.add_argument('--event', required=True,
                        help='Attribute denoting class label')
    parser.add_argument('--test', help='Path to ARFF file to use for testing (optional)')
    parser.add_argument('--profile', default='default', help='Name of IPython parallel profile')
    parser.add_argument('-s', '--seed', type=int, default=19, help='Random number seed')
    parser.add_argument('--metric', default='avgprec', choices=['avgprec', 'rocauc'],
                        help='Which metric to use for performance evaluation')

    _args = parser.parse_args()

    main(_args)
