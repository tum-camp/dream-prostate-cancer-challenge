API reference
=============

Linear Models
-------------
.. currentmodule:: survival.linear_model

.. autosummary::
    :toctree: generated/

    CoxPHSurvivalAnalysis


Ensemble Models
---------------
.. currentmodule:: survival.ensemble

.. autosummary::
    :toctree: generated/

    ComponentwiseGradientBoostingSurvivalAnalysis
    GradientBoostingSurvivalAnalysis
    RandomSurvivalForest


Survival Support Vector Machine
-------------------------------
.. currentmodule:: survival.svm

.. autosummary::
    :toctree: generated/

    FastSurvivalSVM
    NaiveSurvivalSVM


Meta Models
-----------
.. currentmodule:: survival.meta

.. autosummary::
    :toctree: generated/

    EnsembleSelection
    EnsembleSelectionRegressor
    Stacking


Kernels
-------

.. currentmodule:: survival.kernels

.. autosummary::
    :toctree: generated/

    clinical_kernel
    ClinicalKernelTransform


Metrics
-------
.. currentmodule:: survival.metrics

.. autosummary::
    :toctree: generated/

    concordance_index_censored


Pre-Processing
--------------
.. currentmodule:: survival.column

.. autosummary::
    :toctree: generated/

    categorical_to_numeric
    encode_categorical
    standardize


I/O Utilities
-------------
.. currentmodule:: survival.io

.. autosummary::
    :toctree: generated/

    loadarff
    writearff

