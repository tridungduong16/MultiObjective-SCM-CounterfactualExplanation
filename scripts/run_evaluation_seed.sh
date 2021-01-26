#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for j in 1 2 3
	do
		python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_func.py --dataset adult --seed $j
		python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_func.py --dataset simple_bn --seed $j
		python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/evaluation_func.py --dataset siag --seed $j
	done

echo "-------------------- Running script done ------------------------------"
 
