#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for j in 1 2 3
	do
		for i in 512 256 128 64 32
		  do
		    echo "Run for adult"
		    echo "Embedding size $i"
		    python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/multi_objective_optimization.py --mode different_size --emb_size $i --seed $j
		    python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/simple_bn.py --mode different_size  --emb_size $i --seed $j
		    python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/Sangiovese.py --mode different_size --emb_size $i --seed $j
		  done
	done

echo "-------------------- Running script done ------------------------------"
 
