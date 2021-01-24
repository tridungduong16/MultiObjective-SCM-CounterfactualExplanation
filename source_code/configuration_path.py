#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:47:54 2020

@author: trduong
"""

BASE_PATH = "/data/trduong/counterfactual-explanation-research.git/{}"
DATA_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/data/{}"
MODEL_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/models/{}"
MODEL_PATH_MAIN = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/models/main/{}"
MODEL_FINAL_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/models/final/{}"
RESULT_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/result/{}"
FINAL_MODEL_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/final_result/model/{}"
FINAL_RESULT_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/final_result/result/{}"
AUTO_ENCODER = FINAL_MODEL_PATH.format('dfencoder.pt')
PRED_MODEL = '/data/trduong/DiCE/dice_ml/utils/sample_trained_models/adult.pth'
LOGGING_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/final_result/logging/{}"
FIGURE_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/final_result/figure/{}"
EVALUATION_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/final_result/result/evaluation/{}"
ORG_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/final_result/result/original/{}"

### Genetic configuration
num_generations = 25 # Number of generations.
num_parents_mating = 25 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 50

init_range_low = 0.
init_range_high = 1.

"""
parent_selection_type="sss": The parent selection type. Supported types are 
sss (for steady-state selection), 
rws (for roulette wheel selection), 
sus (for stochastic universal selection), 
rank (for rank selection), 
random (for random selection), 
and tournament (for tournament selection).
"""
parent_selection_type = "sus"
# parent_selection_type = "rank"

keep_parents = -1

# crossover_type = "single_point" # Type of the crossover operator.
crossover_type = "two_points" # Type of the crossover operator.
# crossover_type = "uniform" # Type of the crossover operator.

mutation_type = "random" # Type of the mutation operator.
# mutation_type = "swap" # Type of the mutation operator.
# mutation_type = "scramble" # Type of the mutation operator.
# mutation_type = "inversion" # Type of the mutation operator.
mutation_probability = 0.95
mutation_percent_genes = 95 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists or when mutation_type is None.

