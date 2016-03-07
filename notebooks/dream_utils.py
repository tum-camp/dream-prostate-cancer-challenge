import numpy
import pandas
from scipy.stats import skew


def is_useless_column(series):
    """Check if series is useless

    A series is useless, if

        1. It contains only missing values
        2. It categorical with only a single category
        3. It is continuous and has a variance smaller 0.001

    Parameters
    ----------
    series : pandas.Series
        Series to check

    Returns
    -------
    is_useless : bool
        Whether the given series is useless
    """
    complete_series = series[pandas.notnull(series)]
    if len(complete_series) == 0:
        return True

    if numpy.issubdtype(complete_series.dtype, numpy.floating) or\
            numpy.issubdtype(complete_series.dtype, numpy.integer):
        return complete_series.var() < 0.001
    else:
        return complete_series.nunique() <= 1


def get_useless_columns(data):
    """Get list of column names that are useless

    Parameters
    ----------
    data : pandas.DataFrame
        Data to check for useless columns

    Returns
    -------
    idx : pandas.Series
        Boolean series indicating useless columns
    """
    idx = data.apply(is_useless_column, axis=0)
    return idx


def drop_useless(table, suffix=None, sep=" ", inplace=False, useless_cols=None):
    useless = get_useless_columns(table)
    if useless_cols is None:
        useless_cols = useless[useless].index
    else:
        useless_cols.union(useless[useless].index)

    if suffix is not None:
        cols = set()
        for suf in suffix:
            cols.update([c + sep + suf for c in useless_cols])

        dep = table.columns.isin(cols)
        more_useless = table.columns[dep]

    all_useless = useless_cols.union(more_useless)
    new_table = table.drop(all_useless, axis=1, inplace=inplace)
    if inplace:
        new_table = table

    dn = new_table.select_dtypes(include=[numpy.number]).apply(lambda x: x.nunique())
    dc = dn < 10
    if dc.any():
        for col in dn[dc].index:
            print("%s has only %d unique values" % (col, dn[col]))

    return new_table


def transfer_categories(ref, other):
    """Ensures that categorical attributes of ref are also present in other.

    If a categorical attribute is present in `ref` and `other`, categories that
    are present in `ref` but not in `other` are added to `other`. The order of
    attributes will be the same as in `ref`.

    `other` must not have categories that are missing in `ref.

    Parameters
    ----------
    ref : pandas.DataFrame
        Data frame to use as reference.

    other : pandas.DataFrame
        Data frame to modify.

    Returns
    -------
    updated_other : pandas.DataFrame
        New data frame with same categories as `ref` for each categorical column.

    updated : dict of pandas.Index
        Categories that have been added for the respective column.
    """
    ref_cols = ref.select_dtypes(include=["category"]).columns

    new_other = other.copy()

    updates = {}
    for col in ref_cols:
        if col not in other.columns:
            continue

        if not pandas.core.common.is_categorical_dtype(other[col].dtype):
            if pandas.isnull(other[col]).all():
                codes = numpy.repeat(-1, other[col].shape[0])
                rc = ref[col].cat
                new_other[col] = pandas.Categorical.from_codes(codes, categories=rc.categories,
                                                               ordered=rc.ordered, name=col)
                continue
            else:
                raise TypeError(col + " of other is not categorical: %s" % other[col].dtype)

        cat_ref = ref[col].cat.categories
        cat_other = other[col].cat.categories

        if not cat_ref.equals(cat_other):
            d = cat_ref.difference(cat_other)
            updates[col] = d
            rc = ref[col]
            new_other[col] = pandas.Series(pandas.Categorical(new_other[col].astype("object").values,
                                                              ordered=rc.cat.ordered, categories=rc.cat.categories),
                                           name=col)

    return new_other, updates


def safe_log(x):
    if x.min() <= -1:
        raise ValueError("{0} min is {1}".format(x.name, x.min()))
    return numpy.log(x + 1)


def safe_anscombe(_x):
    """Apply Anscombe transform to each column"""
    def _anscombe(x):
        if (x[pandas.notnull(x)] <= 0).all():
            x = -x
        if x.min() < 0:
            raise ValueError("{0} has negative values".format(x.name))
        return 2 * numpy.sqrt(x + 3./.8)

    if _x.ndim == 1:
        return _anscombe(_x)
    return _x.apply(_anscombe, axis=0, reduce=False, raw=True)


def log_transform(data, columns, inplace=False):
    """Apply log transform to each column and add 'log' prefix to columns
    that were transformed"""
    new_columns = {s: "log {0}".format(s) for s in columns}
    data.loc[:, columns] = data.loc[:, columns].apply(safe_log)
    data_new = data.rename(columns=new_columns, inplace=inplace)
    return data_new, new_columns


def detect_and_correct_skewness(data, threshold=1.4):
    """Determine skewness of data in columns and apply
    log transform with it exceeds the specified threshold"""
    numeric_data = data.select_dtypes(include=[float])
    sk = numeric_data.apply(lambda x: skew(x.dropna()), reduce=False)
    columns = sk[sk > threshold].index
    data, new_columns = log_transform(data, columns, inplace=True)
    return columns, new_columns


def cut_quantiles(frame, q):
    """Discretize columns of DataFrame by splitting data into specified number of quantiles"""
    if isinstance(q, pandas.DataFrame):
        labels = ["Q{0}".format(i) for i in range(1, q.shape[0])]
        quantiles = q
    else:
        labels = ["Q{0}".format(i) for i in range(1, q + 1)]
        if isinstance(frame, pandas.DataFrame):
            cols = frame.columns
        else:
            cols = [frame.name]
        quantiles = pandas.DataFrame(index=list(range(q + 1)), columns=cols)

    def _qcut(x):
        bins = x.quantile(numpy.arange(0, 1.0 + 1./q, 1./q)).unique()
        if len(bins) < 3:
            print("{0} has less than 3 unique quartiles".format(x.name))
            return pandas.Series(index=x.index, name=x.name)

        n = len(bins) - 1
        cat_val = pandas.cut(x, bins, labels=labels[:n])
        quantiles.ix[:n, x.name] = bins

        return pandas.Categorical.from_codes(cat_val.cat.codes + 1,
                                             categories=["NONE"] + cat_val.cat.categories.tolist(),
                                             ordered=cat_val.cat.ordered)

    def _cut(x):
        bins = quantiles[x.name].dropna()
        cat_val = pandas.cut(x, bins, labels=labels[:(len(bins) - 1)])
        return pandas.Categorical.from_codes(cat_val.cat.codes + 1,
                                             categories=["NONE"] + cat_val.cat.categories.tolist(),
                                             ordered=cat_val.cat.ordered)

    func = _cut if isinstance(q, pandas.DataFrame) else _qcut

    if isinstance(frame, pandas.DataFrame):
        df_new = frame.apply(func, axis=0, reduce=False)
    else:
        df_new = func(frame)
    df_new.fillna("NONE", inplace=True)

    return df_new, quantiles


def get_missing_values_per_study(table, variables, only_missing=True):
    """Compute percentage of missing value per feature per study"""
    studies = table['STUDYID'].unique()
    df_available = pandas.DataFrame(index=variables, columns=studies)

    for col in variables:
        g = table.loc[:, [col, 'STUDYID']].groupby('STUDYID').apply(lambda x: pandas.isnull(x).sum())
        df_available.loc[col, :] = g[col]

    s = table.groupby('STUDYID').size()
    df_available /= s
    if only_missing:
        df_available = df_available[df_available.sum(axis=1) > 0]
    order = numpy.argsort(-df_available.apply(numpy.sum, axis=1))

    values = df_available.iloc[order, :].astype(float)
    return values
