#!/bin/bash
# Prints some sample tikz files from all scores*.csv files present where run.

PATH_PREFIX=${1?Error: no input filepath given}

for scores_filepath in ${PATH_PREFIX}/scores*.csv; do
	SCORES_FILE="$(basename -- $scores_filepath)"

	IFS='_' # underscore is set as delimiter
	read -ra ARR <<< "$SCORES_FILE" # read into an array as tokens separated by IFS
	MODEL="${ARR[1]}"
	IFS=' ' # reset to default value after usage

	IFS='-' # underscore is set as delimiter
	read -ra ARR <<< "$MODEL" # read into an array as tokens separated by IFS
	MODEL="${ARR[0]}"
	IFS=' ' # reset to default value after usage

    python pmi-accuracy/print_tikz.py \
		--input_file $scores_filepath \
		--edge_types projective.edges.sum \
		--sentence_indices 78 237 381 1132 1081 \
		--info $MODEL \
		--conllx_file ptb3-wsj-data/ptb3-wsj-dev.conllx
done