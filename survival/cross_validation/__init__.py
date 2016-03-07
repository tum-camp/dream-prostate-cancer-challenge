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
from sklearn.utils import safe_indexing


def _safe_split_params(indices, n_samples, fit_params):
    """Returns dict with values being numpy arrays restricted
    to the rows denoted by indices.

    Values that are not numpy arrays or which have number of rows
    different from `n_samples` are passed along unmodifed.

    Parameters
    ----------
    indices : array
        Indices according to which arrays will be subsampled.

    n_samples : int
        Number of rows/samples the data matrix has.

    fit_params : dict
        Arrays to subsample.
    """
    if fit_params is None:
        return

    new_params = {}
    for key, value in fit_params.items():
        if hasattr(value, "shape") and value.shape[0] == n_samples:
            new_params[key] = safe_indexing(value, indices)
        else:
            new_params[key] = value
    return new_params


def _safe_split(X, y, indices):
    """Create subset of dataset"""
    X_subset = safe_indexing(X, indices)
    if y is not None:
        y_subset = safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset


def _fit_and_score(est, x, y, scorer, train_index, test_index, parameters, fit_params, predict_params):
    """Train survival model on given data and return its score on test data"""
    X_train, y_train = _safe_split(x, y, train_index)
    train_params = _safe_split_params(train_index, x.shape[0], fit_params)

    # Training
    est.set_params(**parameters)
    est.fit(X_train, y_train, **train_params)

    # Testing
    test_predict_params = _safe_split_params(test_index, x.shape[0], predict_params)
    X_test, y_test = _safe_split(x, y, test_index)

    score = scorer(est, X_test, y_test, **test_predict_params)
    if not isinstance(score, numbers.Number):
        raise ValueError("scoring must return a number, got %s (%s) instead."
                         % (str(score), type(score)))

    return score
