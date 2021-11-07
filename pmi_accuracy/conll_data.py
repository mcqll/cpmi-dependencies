"""
Reading dependencies from CoNLL files.
"""

from collections import namedtuple

# PTB 'words' that are punctuation marks will be consistently
# excluded from dependency trees. This corresponds to the symbols
# which are UPOS tagged as PUNCT

EXCLUDED_PUNCTUATION = ["", "'", "''", ",", ".", ";",
                        "!", "?", ":", "``",
                        "-LRB-", "-RRB-"]

# Where possible, use this instead, since it is cleaner and works multilingually

EXCLUDED_PUNCTUATION_UPOS = ["PUNCT"]

# Name the columns of CONLL file    CONNL-U fieldnames (https://universaldependencies.org/format.html)
CONLL_COLS = ['ID',               # ID (index): Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0)
              'FORM',             # FORM (sentence): Word form or punctuation symbol
              'LEMMA',            # LEMMA (lemma_sentence): Lemma or stem of word form
              'UPOS',             # UPOS (upos_sentence): Universal part-of-speech tag (https://universaldependencies.org/u/pos/index.html)
              'XPOS',             # XPOS (xpos_sentence): Language-specific part-of-speech tag; underscore if not available.
              'FEATS',            # FEATS (morph): List of morphological features from the universal feature inventory (https://universaldependencies.org/u/feat/index.html) or from a defined language-specific extension (https://universaldependencies.org/ext-feat-index.html); underscore if not available
              'HEAD',             # HEAD (head_indices): Head of the current word, which is either a value of ID or zero (0)
              'DEPREL',           # DEPREL (governance_relations): Universal dependency relation (https://universaldependencies.org/u/dep/index.html) to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
              'DEPS',             # DEPS (secondary_relations): Enhanced dependency graph in the form of a list of HEAD-DEPREL pairs
              'MISC']             # MISC (extra_info): Any other annotation


class CONLLReader():
    def __init__(self, conll_cols, additional_field_name=None):
        if additional_field_name:
            conll_cols += [additional_field_name]
        self.conll_cols = conll_cols
        self.observation_class = namedtuple("Observation", conll_cols)
        self.additional_field_name = additional_field_name

    # Data input
    @staticmethod
    def generate_lines_for_sent(lines, skip_noninteger_ids=True):
        '''Yields batches of lines describing a sentence in conllx.

        Args:
            lines: Each line of a conllx file.
            skip_noninteger_ids: bool for whether to 
                ignore lines that have first column value as a 
                range (e.g. 1-2) or subindex (e.g. 5.1),
                for use with CONLL-U format
        Yields:
            a list of lines describing a single sentence in conllx.
        '''
        buf = []
        for line in lines:
            if line.startswith('#'):
                continue
            if not line.strip():
                if buf:
                    yield buf
                    buf = []
                else:
                    continue
            else:
                if skip_noninteger_ids and line.split('\t')[0].isnumeric():
                    buf.append(line.strip())
        if buf:
            yield buf

    def load_conll_dataset(self, filepath):
        '''Reads in a conllx file; generates Observation objects

        For each sentence in a conllx file, generates a single Observation
        object.

        Args:
            filepath: the filesystem path to the conll dataset
            observation_class: namedtuple for observations

        Returns:
        A list of Observations
        '''
        observations = []
        lines = (x for x in open(filepath))
        for buf in self.generate_lines_for_sent(lines):
            conllx_lines = []
            for line in buf:
                conllx_lines.append(line.strip().split('\t'))
            if self.additional_field_name:
                newfield = [None for x in range(len(conllx_lines))]
                observation = self.observation_class(
                    *zip(*conllx_lines), newfield)
            else:
                observation = self.observation_class(
                    *zip(*conllx_lines))
            observations.append(observation)
        return observations
