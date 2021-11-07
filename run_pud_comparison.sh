#!/bin/bash

# The 20 languages that have PUD treebanks
PUD_LANGS=(Arabic Chinese Czech English Finnish French German Hindi Icelandic Indonesian Italian Japanese Korean Polish Portuguese Russian Spanish Swedish Thai Turkish)

for lang in ${PUD_LANGS[@]}; do
  python pmi_accuracy/main.py --results_dir results/PUD/${lang} --pad 30 --save_npz --model_spec bert-base-multilingual-cased --batch_size 8 --conllx_file ud-treebanks/UD_${lang}-PUD/*_pud-ud-test.conllu > PUD-${lang}.out
done

