#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for i in 512 256 128 64 32
  do
    for j in 'full' 'positive' 'negative'
      do
         echo "Embedding size $i and Version $j"
         python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/df_encoder_bn.py --name siag --version $j --emb_size $i
         python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/df_encoder_bn.py --name simple_bn --version $j --emb_size $i
         python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/dencoder_adults_cf.py --name adult --version $j --emb_size $i
      done
  done
echo "-------------------- Running script done ------------------------------"
 
