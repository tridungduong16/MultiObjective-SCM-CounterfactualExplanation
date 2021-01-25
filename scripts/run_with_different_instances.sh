#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for i in {5..50..5}
  do
    echo "Number of instances $i"
    echo "Run for adult"
    python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/multi_objective_optimization.py --mode different_instance --n_instance $i
    echo "Run for simple_bn"
    python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/simple_bn.py --mode different_instance  --n_instance $i
    echo "Run for siag"
    python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/Sangiovese.py --mode different_instance --n_instance $i
  done

echo "-------------------- Running script done ------------------------------"
 
