#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_func.py --dataset simple_bn
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_func.py --dataset siag
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_func.py --dataset adult
echo "-------------------- Running script done ------------------------------"
 
