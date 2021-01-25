#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_final.py --dataset simple_bn
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_final.py --dataset siag
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_final.py --dataset adult
echo "-------------------- Running script done ------------------------------"
 
