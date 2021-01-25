|BuildStatus|_ |PyPiVersion|_ |PythonSupport|_

.. |BuildStatus| image:: https://github.com/interpretml/dice/workflows/Python%20package/badge.svg
.. _BuildStatus: https://github.com/interpretml/dice/actions?query=workflow%3A%22Python+package%22

.. |PyPiVersion| image:: https://img.shields.io/pypi/v/dice-ml
.. _PyPiVersion: https://pypi.org/project/dice-ml/

.. |PythonSupport| image:: https://img.shields.io/pypi/pyversions/dice-ml
.. _PythonSupport: https://pypi.org/project/dice-ml/

Multi-objective Optimization for Counterfactual Explanation with Structural Causal Model
======================================================================

*How to explain a machine learning model such that the explanation is truthful to the model and yet interpretable to people?*

`Dung Duong <https://scholar.google.com/citations?user=hoq2nt8AAAAJ&hl=en>`_, `Qian Li <https://scholar.google.com/citations?hl=en&user=yic0QMYAAAAJ>`_, `Guandong Xu <https://scholar.google.com/citations?user=kcrdCq4AAAAJ&hl=en&oi=ao>`_
  
`FAT* '20 paper <https://arxiv.org/abs/1905.07697>`_ | `Docs <https://interpretml.github.io/DiCE/>`_ | Live Jupyter notebook |Binder|_

.. |Binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder:  https://mybinder.org/v2/gh/interpretML/DiCE/master?filepath=docs/source/notebooks


Getting started with DiCE
-------------------------
With DiCE, generating explanations is a simple three-step  process: train
mode and then invoke DiCE to generate counterfactual examples for any input.

.. code:: python

    import dice_ml
    from dice_ml.utils import helpers # helper functions
    # Dataset for training an ML model
    d = dice_ml.Data(dataframe=helpers.load_adult_income_dataset(),
                     continuous_features=['age', 'hours_per_week'],
                     outcome_name='income')
    # Pre-trained ML model
    m = dice_ml.Model(model_path=dice_ml.utils.helpers.get_adult_income_modelpath())
    # DiCE explanation instance
    exp = dice_ml.Dice(d,m)
    