# Comparison of CPMI dependency accuracy on multiple languages



## Comparison using `bert-base-multilingual-cased` and PUD treebanks

### Data
Parallel Universal Dependencies (PUD) treebanks for 20 languages, taken from [UDv2.7](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3424) (originally introduced as part of the CoNNL 2017 shared task; Nivre et al, 2017).  In UDv2.7, these PUD treebanks are in directories named `UD_<lang>-PUD/`, where `<lang>` is each of the following languages:

    Arabic, Chinese, Czech, English, Finnish, French, German, Hindi, Icelandic, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish

note: 'Chinese' treebank is for Traditional Chinese

### Language model
[bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) from `transformers`

- multilingual model trained on 104 languages, **including all of the 20 above** (note: Thai was added in *Multilingual Cased (New)* release, [November 23rd, 2018](https://github.com/google-research/bert/commit/332a68723c34062b8f58e5fec3e430db4563320a#diff-8acf157d91703385b49c96621276c6511bf96b92fd6d85883c1bb78f798dc9eb))


Arabic Chinese Czech English Finnish French German Hindi Icelandic Indonesian Italian Japanese Korean Polish Portuguese Russian Spanish Swedish Thai Turkish

language      | total_uuas_nonproj.sum | total_uuas_proj.sum | total_uuas_linear
--------------|------------------------|---------------------|-------------------
Thai          |               0.379905 |            0.421140 |          0.559335
Hindi         |               0.389343 |            0.420417 |          0.514417
Japanese      |               0.391985 |            0.428259 |          0.478688
Chinese       |               0.393194 |            0.420545 |          0.453092
Icelandic     |               0.413035 |            0.439736 |          0.493199
Swedish       |               0.428757 |            0.450573 |          0.443012
English       |               0.433456 |            0.453653 |          0.419746
Finnish       |               0.456309 |            0.478996 |          0.518038
German        |               0.458367 |            0.479440 |          0.418214
Indonesian    |               0.459704 |            0.488976 |          0.558109
Portuguese    |               0.460565 |            0.483913 |          0.449874
Italian       |               0.461425 |            0.481104 |          0.451390
French        |               0.464495 |            0.490085 |          0.452455
Spanish       |               0.471994 |            0.497349 |          0.453991
Arabic        |               0.477601 |            0.508817 |          0.580780
Korean        |               0.477661 |            0.496390 |          0.579884
Turkish       |               0.479394 |            0.499490 |          0.545216
Czech         |               0.479981 |            0.495462 |          0.479981
Russian       |               0.500650 |            0.514631 |          0.506828
Polish        |               0.512592 |            0.533501 |          0.541307

