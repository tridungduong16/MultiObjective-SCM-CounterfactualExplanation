#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
python /home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/source_code/genetic_cf.py --distance mad
python /home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/source_code/genetic_cf.py --distance certifai
echo "-------------------- Running script done ------------------------------"
