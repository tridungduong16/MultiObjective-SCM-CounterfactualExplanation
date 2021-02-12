#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 21:51:45 2020

@author: trduong
"""

import numpy as np
import sys
from helpers import load_adult_income_dataset


def mvdm(X: np.ndarray,
         y: np.ndarray,
         cat_vars: dict,
         alpha: int = 1) -> np.ndarray:
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Modified Value Difference Measure based on Cost et al (1993).
    https://link.springer.com/article/10.1023/A:1022664626993

    Parameters
    ----------
    X
        Batch of arrays.
    y
        Batch of labels or predictions.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.
    alpha
        Power of absolute difference between conditional probabilities.

    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: handle triangular inequality
    # infer number of categories per categorical variable
    n_y = len(np.unique(y))
    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    # conditional probabilities and pairwise distance matrix
    d_pair = {}
    for col, n_cat in cat_vars.items():
        d_pair_col = np.zeros([n_cat, n_cat])
        p_cond_col = np.zeros([n_cat, n_y])
        for i in range(n_cat):
            idx = np.where(X[:, col] == i)[0]
            for i_y in range(n_y):
                p_cond_col[i, i_y] = np.sum(y[idx] == i_y) / (y[idx].shape[0] + 1e-12)

        for i in range(n_cat):
            j = 0
            while j < i:  # symmetrical matrix
                d_pair_col[i, j] = np.sum(np.abs(p_cond_col[i, :] - p_cond_col[j, :]) ** alpha)
                j += 1
        d_pair_col += d_pair_col.T
        d_pair[col] = d_pair_col
    return d_pair



dataset = load_adult_income_dataset()
