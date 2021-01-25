#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/genetic_siag.py --name siag --distance mad
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/genetic_siag.py --name siag --distance certifai
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/genetic_siag.py --name simple_bn --distance mad
python /data/trduong/counterfactual-explanation-research.git/my-algorithm/source_code/genetic_siag.py --name simple_bn --distance certifai
echo "-------------------- Running script done ------------------------------"
 
