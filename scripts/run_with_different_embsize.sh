#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for i in 512 256 128 64 32
  do
    echo "Run for adult"
    echo "Embedding size $i"
    #python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/CF-Adult.py --mode different_size --emb_size $i
    #python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/CF-Sangiovese.py --mode different_size  --emb_size $i
    python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/CF-SimpleBN.py --mode different_size --emb_size $i
  done

echo "-------------------- Running script done ------------------------------"
 
