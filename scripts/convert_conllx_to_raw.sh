#!/bin/bash
#
# Uses AWK script to convert conllx splits to plaintext line-per-sentence
# Usage example: scripts/convert_conllx_to_raw.sh ./ptb3-wsj  

# PATH_PREFIX='./ptb-wsj-data/ptb3-wsj'
PATH_PREFIX=${1?Error: no input filepath given}

for split in train dev test; do
    echo Converting $split split from conllx to txt...
    ./convert_conll_to_raw.awk $PATH_PREFIX-${split}.conllx > $PATH_PREFIX-${split}.txt
done
