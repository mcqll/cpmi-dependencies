#!/bin/bash
# Prints some sample tikz files from all scores*.csv files present where run.

# usage 
# scripts/print_tikz_ud.sh results/UD/zh/bert-base-multilingual-cased* '1 2 3' 

SCORES_DIR=${1?Error: no input dir DIR given (where DIR/scores*.csv shuold exist)}
SENTENCE_INDEX=${2?Error: no sentence given}
CONLL_FILE=$(awk '$1 == "conllx_file:" {print $2}' ${SCORES_DIR}/info.txt)
MODEL_SPEC=$(awk '$1 == "model_spec:" {print $2}' ${SCORES_DIR}/info.txt)
UD_NAME=$(basename -- $(dirname -- ${CONLL_FILE}))
UD_NAME="${UD_NAME#*_}" # strip of everything from beginning to first occurence of '_'
# UD_NAME="${UD_NAME#*-}" # strip of everything from beginning to first occurence of '-'

python pmi_accuracy/print_tikz.py \
	--sentence_indices ${SENTENCE_INDEX} \
	--input_file ${SCORES_DIR}/scores*.csv \
	--conllx_file ${CONLL_FILE} \
	--info "${MODEL_SPEC}" \
	--index_info "${UD_NAME}" \
	--edge_types projective.edges.sum nonproj.edges.sum \
	--output_dir pud-comparison-tikz