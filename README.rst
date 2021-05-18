Multi-objective Optimization for Counterfactual Explanation with Structural Causal Model
======================================================================

*Our work pay attention to counterfactual explanation with the structural causal model and multiobjective optimization*

This is the code used for the paper `Prototype-based Counterfactual Explanation for Causal Classification <https://arxiv.org/abs/2105.00703>`_. I submitted this paper to IJCAI 2021 and got rejected. This work is still in progress. I appreciate your feedback to improve my work. Contact me at TriDung.Duong@student.uts.edu.au


How to run
-------------------------

Build the classifier model

.. code-block:: console

	python /multiobj-scm-cf/src/model_adult.py
	python /multiobj-scm-cf/src/model_credit.py
	python /multiobj-scm-cf/src/model_simple.py
	python /multiobj-scm-cf/src/model_sangiovese.py


Build the auto-encoder model

.. code-block:: console

	python /multiobj-scm-cf/src/dfencoder_adult.py
	python /multiobj-scm-cf/src/dfencoder_credit.py

Reproduce the results

.. code-block:: console

	python /multiobj-scm-cf/src/run_simplebn.py
	python /multiobj-scm-cf/src/run_adult.py
	python /multiobj-scm-cf/src/run_credit.py
	python /multiobj-scm-cf/src/run_sangiovese.py

Reference:
-------------------------

- Mahajan, D., Tan, C., & Sharma, A. (2019). Preserving causal constraints in counterfactual explanations for machine learning classifiers. arXiv preprint arXiv:1912.03277.
- Van Looveren, A., & Klaise, J. (2019). Interpretable counterfactual explanations guided by prototypes. arXiv preprint arXiv:1907.02584.
- AutoEncoders for DataFrames: https://github.com/AlliedToasters/dfencoder


