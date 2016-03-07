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
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.utils import check_X_y

__all__ = ['BalancedGradientBoostingClassifier', 'BalancedRandomForestClassifier']


class BaseBalancedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hybrid_model=False, ensemble_scheme='avg'):
        self.hybrid_model = hybrid_model
        self.ensemble_scheme = ensemble_scheme

    def _create_estimator(self):
        raise NotImplementedError()

    @property
    def classes_(self):
        return self.base_estimators_[0].classes_

    def fit(self, X, y):
        """Balances class and generate class-balanced data set.
           Split the dominating class samples into n_balanced_splits
           and concatenate the minority class samples to each n_balanced_splits
           generated from splitting dominating class samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples]
            Class labels of training samples

        Returns
        -------
        self : an instance of self
        """
        X, y = check_X_y(X, y)

        if self.ensemble_scheme not in {'avg', 'majority_voting'}:
            raise ValueError("ensemble_scheme must be on of 'avg' and 'majority_voting', "
                             "but was %r" % self.ensemble_scheme)

        if len(np.unique(y)) != 2:
            raise ValueError("only binary classification is supported")

        # define new estimators here
        estimator = self._create_estimator()

        # get balanced data splits, dominating class is splitted such that each split has
        # equivalent number of minority class samples. The minority class samples
        # are concatenated into each split generated from majority class
        split_indices_list = self.get_balance_class_dataset_split(y)
        n_balanced_splits = len(split_indices_list)
        if self.hybrid_model:
            n_balanced_splits += 1

        # to save best estimators of each model ensemble
        self.base_estimators_ = np.empty(n_balanced_splits, dtype=np.object)

        # train base model on each balanced dataset
        for i, idx in enumerate(split_indices_list):
            x_split = np.array(X.take(idx, axis=0))
            y_split = np.array(y.take(idx, axis=0))

            gs = clone(estimator).fit(x_split, y_split)
            self.base_estimators_[i] = gs

        if self.hybrid_model:
            # perform grid search CV for unbalanced data also
            gs = clone(estimator).fit(X, y)
            self.base_estimators_[n_balanced_splits - 1] = gs

        self.n_balanced_splits_ = n_balanced_splits
        self.n_features_ = X.shape[1]

        return self

    def predict(self, X):
        """ Prediction .
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples]
            Returns the prediction label/class of each sample in the model.
        """
        if X.shape[1] != self.n_features_:
            raise ValueError('expected %d features, but got %d' % (self.n_features_, X.shape[1]))

        if self.ensemble_scheme == 'majority_voting:':
            base_predictions = np.zeros(shape=(X.shape[0], self.n_balanced_splits_))
            # use optimal grid search parameters from each model
            for i, gs in enumerate(self.base_estimators_):
                base_predictions[:, i] = gs.predict(X)

            # majority voting from ensemble
            y_pred_majority = []
            for i in range(base_predictions.shape[0]):
                y_pred_majority.append(max(k for k, v in Counter(base_predictions[i, :]).items() if v > 1))

            return y_pred_majority
        else:
            proba = self.predict_proba(X)
            predictions = np.argmax(proba, axis=1)
            return predictions

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model.
        """
        if X.shape[1] != self.n_features_:
            raise ValueError('expected %d features, but got %d' % (self.n_features_, X.shape[1]))

        base_predictions = np.zeros(shape=(self.n_balanced_splits_, X.shape[0], 2))

        # use optimal grid search parameters from each model
        for i, gs in enumerate(self.base_estimators_):
            base_predictions[i, :, :] = gs.predict_proba(X)

        y_pred_prob_avg = []
        # averaging probabilities from ensembles
        for i in range(2):
            y_pred_prob_avg.append(np.mean(base_predictions[:, :, i], axis=0))

        return np.transpose(y_pred_prob_avg)

    def get_balance_class_dataset_split(self, Y):
        """Balance the data sets by class distribution.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        Y : array-like, shape = [n_samples]
            Class labels.

        Returns
        -------
        split : List of lists
            balanced data set indices
        """
        n_classes = len(np.unique(Y))
        label_wise_count = np.zeros(shape=n_classes)
        classwise_indices = [[] for _ in range(n_classes)]

        count = 0
        # get indices for each class and count of samples from each class
        for y_label in Y:
            for c in np.unique(Y):
                if c == y_label:
                    label_wise_count[c] += 1
                    classwise_indices[c].append(count)
                    break
            count += 1

        split_ratio = 0.0
        # determine the dominating class to split and split ratio
        if label_wise_count[0] > label_wise_count[1]:
            split_c_class_indx = 0
            non_split_c_class_indx = 1
            split_ratio = label_wise_count[0] / label_wise_count[1]
            if np.abs(np.floor(split_ratio) - split_ratio) <= 0.7:
                # to make last split balanced
                split_ratio = np.floor(split_ratio)
            else:
                split_ratio = np.ceil(split_ratio)
            # number of samples in each split
            samples_per_split = label_wise_count[0] / split_ratio
        else:
            split_ratio = label_wise_count[1] / label_wise_count[0]
            split_c_class_indx = 1
            non_split_c_class_indx = 0
            if np.abs(np.floor(split_ratio) - split_ratio) <= 0.7:
                # to make last split balanced
                split_ratio = np.floor(split_ratio)
            else:
                split_ratio = np.ceil(split_ratio)
            # number of samples in each split
            samples_per_split = label_wise_count[1] / split_ratio

        samples_per_split = int(samples_per_split)
        n_split = int(split_ratio)
        class_indices_to_split = classwise_indices[split_c_class_indx]
        class_indices_to_split_count = len(class_indices_to_split)

        # initialise n_split number of list for balanced splits
        split = [[] for _ in range(n_split)]
        split_count = 0
        i = 0

        # split dominating class samples and concatenate the indices of the minority class samples
        # into each split from dominating class
        while class_indices_to_split_count > i:
            # check if it is the last split
            if split_count + 1 == n_split:
                for j in range(class_indices_to_split_count - i):
                    split[split_count].append(classwise_indices[split_c_class_indx][i])
                    i += 1
            else:
                for j in range(samples_per_split):
                    split[split_count].append(classwise_indices[split_c_class_indx][i])
                    i += 1

            # concatenate the indices of the minority class samples
            # into each split from dominating class
            for t in classwise_indices[non_split_c_class_indx]:
                split[split_count].append(t)

            split_count += 1

        # return n_split number of list of indices, and number of splits
        return split


class BalancedRandomForestClassifier(BaseBalancedClassifier):
    """Ensemble of random forest classifiers trained on data where ratio of classes is close to 50%.

    Multiple datasets are created such that samples of one class appear roughly as often as
    samples of the other class in each dataset. A random forest classifier is trained on
    each balanced dataset.

    Parameters
    ----------
    hybrid_model : bool, optional
        To include unbalanced model scores in balanced ensemble scores.

    ensemble_scheme : {'avg' | 'majority_voting'}
        How to combine probability estimates from ensemble classifiers.
        If `avg` prediction probabilities from base classifiers are averaged.
    """
    def __init__(self, n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 random_state=None, hybrid_model=False, ensemble_scheme='avg'):
        super().__init__(hybrid_model=hybrid_model, ensemble_scheme=ensemble_scheme)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def _create_estimator(self):
        # define new estimators here
        return RandomForestClassifier(n_estimators=self.n_estimators,
                                      criterion=self.criterion,
                                      max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf,
                                      min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                      max_features=self.max_features,
                                      max_leaf_nodes=self.max_leaf_nodes,
                                      random_state=self.random_state,
                                      class_weight="auto")


class BalancedGradientBoostingClassifier(BaseBalancedClassifier):
    """Ensemble of gradient boosting classifiers trained on data where ratio of classes is close to 50%.

    Multiple datasets are created such that samples of one class appear roughly as often as
    samples of the other class in each dataset. A gradient boosting classifier is trained on
    each balanced dataset.

    Parameters
    ----------
    hybrid_model : bool, optional
        To include unbalanced model scores in balanced ensemble scores.

    ensemble_scheme : {'avg' | 'majority_voting'}
        How to combine probability estimates from ensemble classifiers.
        If `avg` prediction probabilities from base classifiers are averaged.
    """
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, init=None, random_state=None,
                 max_features=None, verbose=0,
                 max_leaf_nodes=None,
                 hybrid_model=False, ensemble_scheme='avg'):
        super().__init__(hybrid_model=hybrid_model, ensemble_scheme=ensemble_scheme)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes

    def _create_estimator(self):
        return GradientBoostingClassifier(loss=self.loss, learning_rate=self.learning_rate,
                                          n_estimators=self.n_estimators,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                          max_depth=self.max_depth, init=self.init, subsample=self.subsample,
                                          max_features=self.max_features,
                                          random_state=self.random_state, verbose=self.verbose,
                                          max_leaf_nodes=self.max_leaf_nodes)
