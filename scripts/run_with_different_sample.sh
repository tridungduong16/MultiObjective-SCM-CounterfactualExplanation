#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/multi_objective_optimization.py --mode different_sample  --n_sample 200
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/simple_bn.py --mode different_sample  --n_sample 200
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/Sangiovese.py --mode different_sample  --n_sample 200
echo "-------------------- Running script done ------------------------------"
 
