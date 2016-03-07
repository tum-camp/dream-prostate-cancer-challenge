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
from sklearn.utils import check_consistent_length, check_array

__all__ = ['check_arrays_survival', 'check_pandas_survival', 'safe_concat']


def check_arrays_survival(X, y, force_all_finite=True):
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    X : array-like
        Data matrix containing feature vectors.

    y : structured array with two fields
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    X : array, shape=[n_samples, n_features]
        Feature vectors.

    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    if not isinstance(y, numpy.ndarray) or y.dtype.fields is None or len(y.dtype.fields) != 2:
        raise ValueError('y must be a structured array with the first field'
                         ' being a binary class event indicator and the second field'
                         ' the time of the event/censoring')

    event_field, time_field = y.dtype.names

    X = check_array(X, dtype=float, ensure_min_samples=2, force_all_finite=force_all_finite)
    event = check_array(y[event_field], ensure_2d=False)
    if not numpy.issubdtype(event.dtype, numpy.bool_):
        raise ValueError('elements of event indicator must be boolean, but found {0}'.format(event.dtype))

    if not numpy.any(event):
        raise ValueError('all samples are censored')

    if not numpy.issubdtype(y[time_field].dtype, numpy.number):
        raise ValueError('time must be numeric, but found {0}'.format(y[time_field].dtype))

    time = check_array(y[time_field], dtype=float, ensure_2d=False)
    check_consistent_length(X, event, time)
    return X, event, time


def check_frame(frame, ensure_2d=True, ensure_min_samples=1, ensure_min_features=1,
                copy=False):
    frame = pandas.DataFrame(frame, copy=copy)

    if ensure_2d and frame.ndim != 2:
        raise ValueError("Found DataFrame with dim %d. Expected 2" % frame.ndim)

    if ensure_min_samples > 0:
        n_samples = frame.shape[0]
        if n_samples < ensure_min_samples:
            raise ValueError("Found DataFrame with %d sample(s) while a"
                             " minimum of %d is required."
                             % (n_samples, ensure_min_samples))

    if ensure_min_features > 0 and frame.ndim == 2:
        n_features = frame.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found DataFrame with %d feature(s) while"
                             " a minimum of %d is required."
                             % (n_features, ensure_min_features))
    return frame


def check_pandas_survival(X, y):
    """Check that all data frames have consistent first dimensions.

    Parameters
    ----------
    X : pandas.DataFrame
        Data matrix containing feature vectors.

    y : pandas.DataFrame with two columns
        The first column contains the binary event indicator,
        and the second columns the time of event or time of censoring.

    Returns
    -------
    X : pandas.DataFrame, shape=[n_samples, n_features]
        Feature vectors.

    event : pandas.Series, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : pandas.Series, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    X = check_frame(X, ensure_min_samples=2)
    if not isinstance(y, pandas.DataFrame):
        y = pandas.DataFrame(y)

    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError('y must be a DataFrame with the first column'
                         ' being a binary class event indicator and the second column'
                         ' the time of the event/censoring')

    event = y.iloc[:, 0]
    time = y.iloc[:, 1]

    if not pandas.core.common.is_bool_dtype(event.dtype):
        raise ValueError('elements of event indicator must be boolean, but found {0}'.format(event.dtype))

    if not event.any():
        raise ValueError('all samples are censored')

    if not pandas.core.common.is_numeric_dtype(time.dtype):
        raise ValueError('time must be numeric, but found {0}'.format(time.dtype))

    check_consistent_length(X, event, time)
    return X, event, time


def safe_concat(objs, *args, **kwargs):
    """Alternative to :func:`pandas.concat` that preserves categorical variables.

    Parameters
    ----------
    objs : a sequence or mapping of Series, DataFrame, or Panel objects
        If a dict is passed, the sorted keys will be used as the `keys`
        argument, unless it is passed, in which case the values will be
        selected (see below). Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised
    axis : {0, 1, ...}, default 0
        The axis to concatenate along
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis(es)
    join_axes : list of Index objects
        Specific indexes to use for the other n - 1 axes instead of performing
        inner/outer set logic
    verify_integrity : boolean, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation
    keys : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level
    levels : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys
    names : list, default None
        Names for the levels in the resulting hierarchical index
    ignore_index : boolean, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the the index values on the other
        axes are still respected in the join.
    copy : boolean, default True
        If False, do not copy data unnecessarily

    Notes
    -----
    The keys, levels, and names arguments are all optional

    Returns
    -------
    concatenated : type of objects
    """
    axis = kwargs.pop("axis", 0)
    categories = {}
    for df in objs:
        if isinstance(df, pandas.Series):
            if pandas.core.common.is_categorical_dtype(df.dtype):
                categories[df.name] = {"categories": df.cat.categories, "ordered": df.cat.ordered}
        else:
            dfc = df.select_dtypes(include=["category"])
            for col in range(dfc.shape[1]):
                s = dfc.iloc[:, col]
                if s.name in categories:
                    if axis == 1:
                        raise ValueError("duplicate columns %s" % s.name)
                    categories[s.name]["categories"].union(s.cat.categories)
                else:
                    categories[s.name] = {"categories": s.cat.categories, "ordered": s.cat.ordered}
                df[s.name] = df[s.name].astype(object)

    concatenated = pandas.concat(objs, *args, axis=axis, **kwargs)

    for name, params in categories.items():
        concatenated[name] = pandas.Categorical(concatenated[name], **params)

    return concatenated
