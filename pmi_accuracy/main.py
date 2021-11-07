'''
Calculate CPMI scores for a (contextual) embedding model,
on PTB CONLL data.
'''

import os
from datetime import datetime
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import combinations
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import io
from contextlib import redirect_stdout

import task
import parser
import languagemodel
import languagemodel_pos
import embedding
from conll_data import CONLLReader, CONLL_COLS, EXCLUDED_PUNCTUATION_UPOS


# Running and reporting
def score_observation(observation, pmi_matrix, absolute_value=False):
    # Get gold edges distances tensor from conllx file
    # (note 'mst' will always give projective gold edges)
    gold_dist_matrix = task.ParseDistanceTask.labels(observation)
    gold_edges = parser.DepParse(
        'mst', gold_dist_matrix, 
        observation.FORM, observation.UPOS).tree(
            symmetrize_method='none',
            maximum_spanning_tree=False)

    # Make linear-order baseline distances tensor
    linear_dist_matrix = task.LinearBaselineTask.labels(observation)
    baseline_linear_edges = parser.DepParse(
        'mst', linear_dist_matrix, 
        observation.FORM, observation.UPOS).tree(
            symmetrize_method='none',
            maximum_spanning_tree=False)

    # Make random baseline distances tensor
    random_dist_matrix = task.RandomBaselineTask.labels(observation)
    baseline_random_nonproj_edges = parser.DepParse(
        'mst', random_dist_matrix, 
        observation.FORM, observation.UPOS).tree(
            symmetrize_method='none',
            maximum_spanning_tree=True)
    baseline_random_proj_edges = parser.DepParse(
        'projective', random_dist_matrix, 
        observation.FORM, observation.UPOS).tree(
            symmetrize_method='none',
            maximum_spanning_tree=True)

    # Instantiate a parser.DepParse object,
    # with the parsetype 'mst', to get pmi mst parse
    mstparser = parser.DepParse('mst', pmi_matrix, 
        observation.FORM, observation.UPOS)
    pmi_edges = {}
    symmetrize_methods = ['sum', 'triu', 'tril', 'none']
    for symmetrize_method in symmetrize_methods:
        pmi_edges[symmetrize_method] = mstparser.tree(
            symmetrize_method=symmetrize_method,
            maximum_spanning_tree=True,
            absolute_value=absolute_value)

    # Instantiate a parser.DepParse object,
    # with parsetype 'projective', to get pmi projective parse
    projparser = parser.DepParse(
        'projective', pmi_matrix, 
        observation.FORM, observation.UPOS)
    pmi_edges_proj = {}
    for symmetrize_method in symmetrize_methods:
        # note, with Eisner's, symmetrize_method='none' gets a directed parse
        # though this isn't very interpretable as such
        # since the PMI is theoretically symmetric
        pmi_edges_proj[symmetrize_method] = projparser.tree(
            symmetrize_method=symmetrize_method,
            maximum_spanning_tree=True,
            absolute_value=absolute_value)

    print("edges:\ngold       ", gold_edges)
    print("pmi nonproj", pmi_edges)
    print("pmi proj   ", pmi_edges_proj)
    scorer = parser.Accuracy(gold_edges)

    scores = {}
    scores['sentence_length'] = len(observation.FORM)
    scores['number_edges'] = len(gold_edges)
    scores['gold_edges'] = gold_edges
    scores['baseline_linear_n_common'], scores['baseline_linear'] = scorer.uuas(
        baseline_linear_edges)
    scores['baseline_random_nonproj_n_common'], scores['baseline_random_nonproj'] = scorer.uuas(
        baseline_random_nonproj_edges)
    scores['baseline_random_proj_n_common'], scores['baseline_random_proj'] = scorer.uuas(
        baseline_random_proj_edges)
    scores['projective'] = {}
    scores['projective']['edges'] = pmi_edges_proj
    scores['projective']['uuas'] = {}
    scores['projective']['n_common'] = {}
    scores['nonproj'] = {}
    scores['nonproj']['edges'] = pmi_edges
    scores['nonproj']['uuas'] = {}
    scores['nonproj']['n_common'] = {}

    for symmetrize_method in symmetrize_methods:
        scores['nonproj']['n_common'][symmetrize_method], scores['nonproj']['uuas'][symmetrize_method] = scorer.uuas(
            pmi_edges[symmetrize_method])
        scores['projective']['n_common'][symmetrize_method], scores['projective']['uuas'][symmetrize_method] = scorer.uuas(
            pmi_edges_proj[symmetrize_method])

    return scores


def observation_to_df(observation, exclude_punctuation=True):
    """ convert observation to a pandas dataframe (with 0 indexing) 
    returns:
        obs_df (pandas DataFrame)
        too_short (bool) whether the sentence (after possibly removing punctuation) 
            is two words or less
    """
    obs_df = pd.DataFrame(observation).T
    obs_df.columns = CONLL_COLS
    obs_df = obs_df.astype({'ID': 'int32', 'HEAD': 'int32'})
    obs_df[['ID', 'HEAD']] -= 1  # convert to 0-indexing
    obs_df = obs_df.set_index('ID')
    if exclude_punctuation:
        # obs_df = obs_df[~obs_df.FORM.isin(EXCLUDED_PUNCTUATION)]
        obs_df = obs_df[~obs_df.UPOS.isin(EXCLUDED_PUNCTUATION_UPOS)]
    too_short = len(obs_df) <= 2 # 2 words or less is too short
    return obs_df, too_short

class PredictorClass:
    def __init__(self, observation, pmi_matrix, exclude_punctuation=True):
        obs_df, too_short = observation_to_df(observation, exclude_punctuation=exclude_punctuation)
        self.includesentence = not too_short
        if too_short:
            print("PredictorClass says: Sentence too short. Skipping.")
            return

        self.df = pd.DataFrame(combinations(obs_df.index, 2), columns=['i1', 'i2'])

        # make some new columns:
        self.df['lin_dist'] = self.df.apply(
            lambda row: row.i2 - row.i1, axis=1)
        self.df['w1'] = self.df.apply(
            lambda row: obs_df.FORM[row.i1], axis=1)
        self.df['w2'] = self.df.apply(
            lambda row: obs_df.FORM[row.i2], axis=1)
        self.df['UPOS1'] = self.df.apply(
            lambda row: obs_df.UPOS[row.i1], axis=1)
        self.df['UPOS2'] = self.df.apply(
            lambda row: obs_df.UPOS[row.i2], axis=1)
        self.df['XPOS1'] = self.df.apply(
            lambda row: obs_df.XPOS[row.i1], axis=1)
        self.df['XPOS2'] = self.df.apply(
            lambda row: obs_df.XPOS[row.i2], axis=1)

        # whether or not there is a gold arc
        self.df['gold_edge'] = self.df.apply(
            lambda row: obs_df.HEAD[row.i1] == row.i2 or
            obs_df.HEAD[row.i2] == row.i1,
            axis=1)

        # label of gold arc, if one exists
        self.df['relation'] = self.df.apply(
            lambda row: obs_df.DEPREL[row.i1] if obs_df.HEAD[row.i1] == row.i2 else (obs_df.DEPREL[row.i2] if obs_df.HEAD[row.i2] == row.i1 else None),
            axis=1)

        for sym in ['sum', 'triu', 'tril']:
            sym_matrix = pmi_matrix
            if sym == 'sum':
                sym_matrix = sym_matrix + np.transpose(sym_matrix)
            elif sym == 'triu':
                sym_matrix = np.triu(sym_matrix) + np.transpose(np.triu(sym_matrix))
            elif sym == 'tril':
                sym_matrix = np.tril(sym_matrix) + np.transpose(np.tril(sym_matrix))
            self.df[f'pmi_{sym}'] = self.df.apply(lambda row: sym_matrix[row.i1][row.i2], axis=1)

    def add_pmi_edges(self, colname, edges):
        if len(self.df) < 2:
            print(f"Sentence too short. {colname} column not added.")
            return
        self.df[colname] = self.df.apply(
            lambda row: (row.i1, row.i2) in edges, axis=1)


def get_padding(i, observations, threshold):
    '''
    to avoid short sentences on which LM performs less well
    (XLNet in particular, others less so, but padding doesn't hurt)
    gets adjacent observations from PTB to add as padding,
    so total length is at least threshold ptb-tokens long
    (will truncate excessively long padding sentences though)
    input: index and observations
    returns:
        prepadding: list of ptb tokens for before
        postpadding: list of ptb tokens for after
    '''
    j = i
    k = i
    pad_index_set = set()
    total_len = len(observations[i][0])
    while total_len < threshold:
        if j - 1 >= 0 and j - 1 not in pad_index_set:
            j -= 1
            pad_index_set.add(j)
            total_len += len(observations[j][0])
        if total_len >= threshold:
            break
        if k + 1 < len(observations) and k + 1 not in pad_index_set:
            k += 1
            pad_index_set.add(k)
            total_len += len(observations[k][0])
        elif k + 1 == len(observations):
            continue
        else:
            raise ValueError(
                f'Not enough context to pad up to size {threshold}!')
    prepad_index_set = [x for x in sorted(pad_index_set) if x < i]
    postpad_index_set = [x for x in sorted(pad_index_set) if x > i]
    excessive = threshold  # padding sentences longer will be truncated
    prepadding_observations = [observations[x] for x in prepad_index_set]
    prepadding = [i for x in
                  [obs.FORM[:excessive] for obs in prepadding_observations]
                  for i in x]
    postpadding_observations = [observations[x] for x in postpad_index_set]
    postpadding = [i for x in
                   [obs.FORM[:excessive] for obs in postpadding_observations]
                   for i in x]
    # print some explanation
    if pad_index_set != set():
        print(f'Using sentence(s) {sorted(pad_index_set)} as padding for sentence {i}.')
        print(f'|\tprepadding sentence lengths  : {[len(obs.FORM) for obs in prepadding_observations]}')
        for index, obs in zip(prepad_index_set, prepadding_observations):
            if len(obs.FORM) > excessive:
                print(f"|\t\t{index}: truncating at length {excessive}")
        print(f'|\tpostpadding sentence lengths : {[len(obs.FORM) for obs in postpadding_observations]}')
        for index, obs in zip(postpad_index_set, postpadding_observations):
            if len(obs.FORM) > excessive:
                print(f"|\t\ttruncating sentence {index} at length {excessive}")
    return prepadding, postpadding


def write_wordpair(i, obs, pmi_matrix, scores, wordpair_csv, header):
    predictors = PredictorClass(obs, pmi_matrix)
    if predictors.includesentence:
        symmetrize_methods = ['sum', 'triu', 'tril', 'none']
        for symmetrize_method in symmetrize_methods:
            predictors.add_pmi_edges(
                f'pmi_edge_{symmetrize_method}',
                scores['projective']['edges'][symmetrize_method])
            predictors.add_pmi_edges(
                f'pmi_edge_nonproj_{symmetrize_method}',
                scores['nonproj']['edges'][symmetrize_method])
        with open(wordpair_csv, 'a') as f:
            predictors.df.insert(0, 'sentence_index', i)
            predictors.df.to_csv(
                f, mode='a', header=header,
                index=False, float_format='%.7f')


def score(observations, padlen=0, n_obs='all', absolute_value=False,
          write_wordpair_data=True,
          load_npz=False, save_npz=False,
          verbose=False):
    '''get estimates get scores for n (default all) observations'''

    if save_npz and load_npz:
        raise ValueError(
            "Error: load_npz and save_npz both True.\n"
            "Choose one or the other (or neither).")

    all_scores = []

    if save_npz:
        matrices_orddict = OrderedDict()
        loglik_orddict = OrderedDict()

    if load_npz:
        matrices_npz = np.load(os.path.join(NPZ_DIR, 'pmi_matrices.npz'))
        loglik_npz = np.load(os.path.join(NPZ_DIR, 'pseudo_logliks.npz'))

    if write_wordpair_data:
        wordpair_csv = RESULTS_DIR + 'wordpair_' + SUFFIX + '.csv'
        header = True

    if n_obs == 'all':
        n_obs = len(observations)
    
    if CORRUPTION:
        def signal_to_noise_pmi(obs, pmi_matrix):
            '''compute signal to noise ratio matrix, 
            using specified corruption type.
            "random_masking" masks 15% of tokens randomly'''
            if CORRUPTION == 'random_masking':
                pmi_matrix_uncorrupted = pmi_matrix
                n_corruptions = 5 # TODO: parameterize
                pmi_matrices_corrupted = np.tile(
                    np.empty_like(pmi_matrix), reps=(n_corruptions,1,1))
                for i in tqdm(range(0, n_corruptions), leave=False, desc='corruption'):
                    pmi_matrices_corrupted[i], _ = MODEL.ptb_tokenlist_to_pmi_matrix(
                    obs.FORM, add_special_tokens=True,
                    verbose=verbose,  # toggle for troubleshoooting.
                    pad_left=prepadding, pad_right=postpadding,
                    corruption=CORRUPTION)
                pmi_variance = np.var(pmi_matrices_corrupted, axis=0)
                pmi_signal_noise_matrix = pmi_matrix_uncorrupted / (pmi_variance + 1e-15)
                with np.printoptions(precision=1, suppress=True, threshold=np.inf, linewidth=5000):
                    print("pmi_matrix_uncorrupted")
                    print(pmi_matrix_uncorrupted)
                    print(f"pmi_variance (across {n_corruptions} {CORRUPTION} corruptions)")
                    print(pmi_variance)
                    print("pmi_signal_noise_matrix")
                    print(pmi_signal_noise_matrix)
                return pmi_signal_noise_matrix
            else:
                raise NotImplementedError(f"Unimplemented corruption type: {CORRUPTION}")

    for i, obs in enumerate(tqdm(observations[:n_obs], desc=os.path.basename(CONLL_FILE))):
        print(f'_______________\n--> Observation {i} of {n_obs}\n')

        obs_df, _ = observation_to_df(obs, exclude_punctuation=False)
        obs_df_no_punct, too_short = observation_to_df(obs, exclude_punctuation=True)

        if verbose:
            with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None, 'display.width', 5000):
                print(obs_df[['FORM', 'UPOS', 'XPOS', 'HEAD', 'DEPREL']],
                    "\n", sep='')

        if load_npz:
            # check that obs.FORM matches the saved file
            sentence_i = str(' '.join([str(i), *obs.FORM]))
            assert sentence_i == matrices_npz.files[i], \
                f'''Loaded sentence {i} != observed sentence:
                loaded (pmi_matrices.npz) : '{matrices_npz.files[i]}'
                observed (from connl file): '{sentence_i}'
                '''
            assert sentence_i == loglik_npz.files[i], \
                f'''Loaded sentence {i} != observed sentence:
                loaded (pseudo_logliks.npz) : '{loglik_npz.files[i]}'
                observed (from connl file)  : '{sentence_i}'
                '''
            pmi_matrix = matrices_npz[sentence_i]
            pseudo_loglik = loglik_npz[sentence_i]

        else:  # Calculate CPMI scores.
            prepadding, postpadding = get_padding(i, observations, padlen)
            # get a pmi matrix and a pseudo-logprob for the sentence
            if CLI_ARGS.probe_state_dict:
                if POS_SET_TYPE == 'upos':
                    obs_POS = obs.UPOS
                elif POS_SET_TYPE == 'xpos':
                    obs_POS = obs.XPOS
                pmi_matrix, pseudo_loglik = POS_MODEL.ptb_tokenlist_to_pmi_matrix(
                    obs.FORM, obs_POS,
                    add_special_tokens=True,
                    verbose=verbose,  # toggle for troubleshoooting.
                    pad_left=prepadding, pad_right=postpadding)
            else:
                pmi_matrix, pseudo_loglik = MODEL.ptb_tokenlist_to_pmi_matrix(
                    obs.FORM, add_special_tokens=True,
                    verbose=verbose,  # toggle for troubleshoooting.
                    pad_left=prepadding, pad_right=postpadding,
                    corruption=None)
                if CORRUPTION and not too_short:
                    # instead of normal pmi matrix, 
                    # use pmi signal-to-noise ratio matrix
                    # skip if too_short (there's nothing to corrupt)
                    pmi_matrix = signal_to_noise_pmi(obs, pmi_matrix)

        if save_npz:
            sentence_i = str(' '.join([str(i), *obs.FORM]))
            matrices_orddict[sentence_i] = pmi_matrix
            loglik_orddict[sentence_i] = pseudo_loglik

        # if too_short:
        #     print("Sentence too short. Skipping scoring for observation {i}.")
        #     continue

        # calculate score
        scores = score_observation(
            obs, pmi_matrix, absolute_value=absolute_value)

        if write_wordpair_data:
            write_wordpair(i, obs, pmi_matrix,
                           scores, wordpair_csv, header)
            header = False

        scores['pseudo_loglik'] = pseudo_loglik
        all_scores.append(scores)
        print(f"n_edges = {scores['number_edges']}")
        print(f"n_common:\nlinear         : {scores['baseline_linear_n_common']}")
        print(f"random nonproj : {scores['baseline_random_nonproj_n_common']}")
        print(f"random proj    : {scores['baseline_random_proj_n_common']}")
        print(f"nonproj  { {k:round(v,3) for k, v in scores['nonproj']['n_common'].items()}}")
        print(f"proj     { {k:round(v,3) for k, v in scores['projective']['n_common'].items()}}\n")

        print(f"==> uuas (= n_common/n_edges):")
        print(f"linear         : {scores['baseline_linear']}")
        print(f"random nonproj : {scores['baseline_random_nonproj']}")
        print(f"random proj    : {scores['baseline_random_proj']}")
        print(f"nonproj  { {k:round(v,3) for k, v in scores['nonproj']['uuas'].items()}}")
        print(f"proj     { {k:round(v,3) for k, v in scores['projective']['uuas'].items()}}\n")
    print("__________________ \n All scores computed \n")

    if save_npz:
        print("Saving PMI matrices in npz file.")
        write_npz(matrices_orddict, RESULTS_DIR, outfilename="pmi_matrices.npz")
        write_npz(loglik_orddict, RESULTS_DIR, outfilename="pseudo_logliks.npz")
    return all_scores


def write_npz(
        ordered_dict, resultsdir,
        outfilename='saved.npz'):
    save_filepath = os.path.join(resultsdir, outfilename)
    np.savez(save_filepath, **ordered_dict)


def print_uuas_summary_to_file(all_scores, file, show_len_hist=True):
    ''' prints total UUAS and mean per sentence UUAS summaries '''
    mean_uuas_linear = np.nanmean([scores['baseline_linear'] for scores in all_scores])
    mean_uuas_random_nonproj = np.nanmean([scores['baseline_random_nonproj'] for scores in all_scores])
    mean_uuas_random_proj = np.nanmean([scores['baseline_random_proj'] for scores in all_scores])
    mean_uuas_nonproj = {symmethod : np.nanmean(
        [scores['nonproj']['uuas'][symmethod] for scores in all_scores]) for symmethod in ['sum', 'triu', 'tril', 'none']}
    mean_uuas_proj = {symmethod : np.nanmean(
        [scores['projective']['uuas'][symmethod] for scores in all_scores]) for symmethod in ['sum', 'triu', 'tril', 'none']}
    mean_uuas_summary = (f"mean sentence uuas:\n"
        f"\tlinear         :  {mean_uuas_linear:.3}\n"
        f"\trandom nonproj :  {mean_uuas_random_nonproj:.3}\n"
        f"\trandom proj    :  {mean_uuas_random_proj:.3}\n"
        f"\tPMI nonproj    :  { {k:round(v,3) for k, v in mean_uuas_nonproj.items()}}\n"
        f"\tPMI proj       :  { {k:round(v,3) for k, v in mean_uuas_proj.items()}}\n\n")
    
    all_n_edges = [scores['number_edges'] for scores in all_scores]
    total_n_edges = np.sum(all_n_edges)
    mean_n_edges = np.mean(all_n_edges)
    median_n_edges = np.median(all_n_edges)
    std_n_edges = np.std(all_n_edges)

    n_edges_stats = (
        f"n_sentences = {len(all_scores)}, "
        f"n_edges: total = {total_n_edges}\n"
        f"mean = {mean_n_edges:.3}, median = {int(median_n_edges)}, std = {std_n_edges:.3}\n\n")
    
    hist=""
    if show_len_hist:
        import termplotlib as tpl
        a = all_n_edges
        counts, bin_edges = np.histogram(a, bins=max(a))
        fig = tpl.figure()
        fig.hist(counts, bin_edges, orientation="vertical")
        with io.StringIO() as buf, redirect_stdout(buf):
            fig.show()
            print(' '*int(round(np.mean(a))-1)+f'↑ mean {mean_n_edges:.3}')
            print(' '*int(np.median(a)-1)+f'| median {int(median_n_edges)}')
            hist = buf.getvalue() + "\n"


    total_n_common_linear = np.sum([scores['baseline_linear_n_common'] for scores in all_scores])
    total_n_common_random_nonproj = np.sum([scores['baseline_random_nonproj_n_common'] for scores in all_scores])
    total_n_common_random_proj = np.sum([scores['baseline_random_proj_n_common'] for scores in all_scores])
    total_n_common_nonproj = {symmethod : np.sum(
        [scores['nonproj']['n_common'][symmethod] for scores in all_scores]) for symmethod in ['sum', 'triu', 'tril', 'none']}
    total_n_common_proj = {symmethod : np.sum(
        [scores['projective']['n_common'][symmethod] for scores in all_scores]) for symmethod in ['sum', 'triu', 'tril', 'none']}
    total_n_common_summary = (f"total n_common:\n"
        f"\tlinear         :  {total_n_common_linear}\n"
        f"\trandom nonproj :  {total_n_common_random_nonproj}\n"
        f"\trandom proj    :  {total_n_common_random_proj}\n"
        f"\tPMI nonproj    :  { {k:v for k, v in total_n_common_nonproj.items()}}\n"
        f"\tPMI proj       :  { {k:v for k, v in total_n_common_proj.items()}}\n")

    total_uuas_summary = (f"==> total corpus uuas (= n_common / n_edges={total_n_edges}):\n"
        f"\tlinear         :  {total_n_common_linear/float(total_n_edges):.3}\n"
        f"\trandom nonproj :  {total_n_common_random_nonproj/float(total_n_edges):.3}\n"
        f"\trandom proj    :  {total_n_common_random_proj/float(total_n_edges):.3}\n"
        f"\tPMI nonproj    :  { {k:round(v/float(total_n_edges),3) for k, v in total_n_common_nonproj.items()}}\n"
        f"\tPMI proj       :  { {k:round(v/float(total_n_edges),3) for k, v in total_n_common_proj.items()}}\n")

    summary = "\n===== accuracy summary =====\n"
    summary += mean_uuas_summary
    summary += n_edges_stats+hist
    summary += total_n_common_summary
    summary += total_uuas_summary
    print(summary)

    with open(file, mode='a') as file:
        file.write(summary)


def get_info(directory, key):
    ''' gets specified spec value from info.txt'''
    info = os.path.join(directory, 'info.txt')
    with open(info, 'r') as infofile:
        for line in infofile:
            if line.split()[0] == key + ':':
                return(line.split()[1])


if __name__ == '__main__':
    ARGP = ArgumentParser()
    ARGP.add_argument('--probe_state_dict',
                      help='path to saved linear probe.state_dict')
    ARGP.add_argument('--pos_set_type', default='xpos',
                      help='xpos or upos')
    ARGP.add_argument('--n_observations', default='all',
                      help='number of sentences to look at')
    ARGP.add_argument('--model_spec', default='xlnet-base-cased',
                      help='''specify model
                      (e.g. "xlnet-base-cased", "bert-large-cased"),
                      or path for offline''')
    ARGP.add_argument('--model_type_override', default=None,
                      help='''specify model type as e.g. "bert"
                      for bert based models like camembert''')
    ARGP.add_argument('--conllx_file',
                      default='ptb3-wsj-data/ptb3-wsj-dev.conllx',
                      help='''path/to/treebank.conllx or .conllu
                      (ignored if --model_spec = load_npz
                      in which case, reads 'conllx_file' field from info.txt''')
    ARGP.add_argument('--results_dir', default='results/',
                      help='''specify path/to/results/directory/ 
                      (does not need to already exist)''')
    ARGP.add_argument('--model_path',
                      help='''
                      with model, optional:
                        load model state or embeddings from file
                      with --model_spec load_npz:
                        directory where pmi matrices
                        and loglik npz files are''')
    ARGP.add_argument('--batch_size', default=32, type=int)
    ARGP.add_argument('--pad', default=0, type=int,
                      help='(int) pad sentences to be at least this long')
    ARGP.add_argument('--save_npz', action='store_true',
                      help='to save pmi matrices as npz')
    ARGP.add_argument('--absolute_value', action='store_true',
                      help='to treat negative CPMI values as positive')
    ARGP.add_argument('--corruption', default=None,
                      help='''specify corruption method to use
                      in computing signal-to-noise decoding for CPMI.
                      If None, will use standard decoding.''')
    ARGP.add_argument('--info', default=None, 
                      help='optional info suffix to append to results name')
    CLI_ARGS = ARGP.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:',
              round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Cached:   ',
              round(torch.cuda.memory_cached(0) / 1024**3, 1), 'GB')

    SPEC_STRING = str(CLI_ARGS.model_spec).replace('/','+')
    if CLI_ARGS.model_path and CLI_ARGS.model_spec == 'bert-base-uncased':
        # custom naming just for readability, for the checkpointed bert
        import re
        number = re.findall("(\d+)", CLI_ARGS.model_path)
        number = str(int(int(number[-1]) / 1000.0))
        SPEC_STRING = SPEC_STRING + ".ckpt-" + number + "k"

    N_OBS = CLI_ARGS.n_observations
    if N_OBS != 'all':
        N_OBS = int(N_OBS)

    NOW = datetime.now()
    DATE_SUFFIX = NOW.strftime("%Y-%m-%d-%H-%M")

    LOAD_NPZ = False
    if CLI_ARGS.model_spec == 'load_npz':
        if CLI_ARGS.probe_state_dict:
            raise ValueError(
                "Can't use both load_npz and probe_state_dict.")
        if CLI_ARGS.model_path:
            NPZ_DIR = CLI_ARGS.model_path
            LOAD_NPZ = True
        else:
            raise ValueError("No path specified from which to load npz (use --model_path argument).")
        LOADED_MODEL_SPEC = get_info(NPZ_DIR, 'model_spec')
        LOADED_PAD = "pad"+get_info(NPZ_DIR, 'pad')
        SPEC_SUFFIX = 'loaded=' + '_'.join([LOADED_MODEL_SPEC, LOADED_PAD])
        CONLL_FILE = get_info(NPZ_DIR, 'conllx_file')
    else:
        SPEC_SUFFIX = SPEC_STRING + '(' + str(CLI_ARGS.n_observations) + ')' if CLI_ARGS.n_observations != 'all' else SPEC_STRING
        SPEC_SUFFIX += '_pad'+str(CLI_ARGS.pad)
        CONLL_FILE = CLI_ARGS.conllx_file
    SPEC_SUFFIX += '(abs)' if CLI_ARGS.absolute_value else ''
    OPTIONAL_INFO = '_' + str(CLI_ARGS.info) if CLI_ARGS.info else ''
    SUFFIX = SPEC_SUFFIX + OPTIONAL_INFO + '_' + DATE_SUFFIX

    UPOS_TAGSET = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ',
                   'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                   'SCONJ', 'SYM', 'VERB', 'X']
    XPOS_TAGSET = ['#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':',
                   'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
                   'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
                   'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                   'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                   'WDT', 'WP', 'WP$', 'WRB', '``']
    POS_SET_TYPE = CLI_ARGS.pos_set_type  # set 'xpos' or 'upos'
    if POS_SET_TYPE == 'upos':
        POS_TAGSET = UPOS_TAGSET
    elif POS_SET_TYPE == 'xpos':
        POS_TAGSET = XPOS_TAGSET
    if CLI_ARGS.probe_state_dict:
        PROBE_STATE = torch.load(
                CLI_ARGS.probe_state_dict,
                map_location=DEVICE)
        if len(PROBE_STATE) == 2:
            PROBE_TYPE = 'linear'
        elif len(PROBE_STATE) == 4:
            PROBE_TYPE = 'IB'
        else:
            raise NotImplementedError
        SUFFIX = PROBE_TYPE + "_" + POS_SET_TYPE + "_" + SUFFIX
    RESULTS_DIR = os.path.join(CLI_ARGS.results_dir, SUFFIX + '/')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f'RESULTS_DIR: {RESULTS_DIR}\n')

    print('Running with CLI_ARGS:')
    with open(RESULTS_DIR + 'info.txt', mode='w') as infofile:
        for arg, value in sorted(vars(CLI_ARGS).items()):
            argvalue = f"{arg}:\t{value}"
            infofile.write(argvalue+'\n')
            print(argvalue)

    CORRUPTION = CLI_ARGS.corruption
    if CLI_ARGS.corruption:
        # if corruption is not None
        CORRUPTION = str(CLI_ARGS.corruption)

    if not LOAD_NPZ:
        if CLI_ARGS.probe_state_dict:
            if (CLI_ARGS.model_spec.startswith('bert') or
                CLI_ARGS.model_spec.startswith('distilbert')):
                POS_MODEL = languagemodel_pos.BERT(
                    DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size,
                    POS_TAGSET, PROBE_STATE, probe_type=PROBE_TYPE)
            elif CLI_ARGS.model_spec.startswith('xlnet'):
                POS_MODEL = languagemodel_pos.XLNet(
                    DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size,
                    POS_TAGSET, PROBE_STATE, probe_type=PROBE_TYPE)
            else:
                raise NotImplementedError
        else:
            # Instantiate the language model, if not loading from disk
            if CLI_ARGS.model_spec.startswith('xlnet'):
                MODEL = languagemodel.XLNet(
                    DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
            elif (CLI_ARGS.model_spec.startswith('bert') or
                  CLI_ARGS.model_spec.startswith('distilbert') or
                  str(CLI_ARGS.model_type_override) == 'bert'):
                # DistilBERT will work just like BERT
                if CLI_ARGS.model_path:
                    # load checkpointed weights from disk, if specified
                    STATE = torch.load(CLI_ARGS.model_path)
                    MODEL = languagemodel.BERT(
                        DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size,
                        state_dict=STATE)
                else:
                    MODEL = languagemodel.BERT(
                        DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
            elif CLI_ARGS.model_spec.startswith('xlm'):
                MODEL = languagemodel.XLM(
                    DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
            elif CLI_ARGS.model_spec.startswith('bart'):
                MODEL = languagemodel.Bart(
                    DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
            elif CLI_ARGS.model_spec.startswith('gpt2'):
                MODEL = languagemodel.GPT2(
                    DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
            elif CLI_ARGS.model_spec == 'w2v':
                W2V_PATH = CLI_ARGS.model_path
                MODEL = embedding.Word2Vec(
                    DEVICE, CLI_ARGS.model_spec, W2V_PATH)
            elif CLI_ARGS.model_spec == 'linear_baseline':
                MODEL = languagemodel.Baseline('linear_baseline')
            elif CLI_ARGS.model_spec == 'random_baseline':
                MODEL = languagemodel.Baseline('random_baseline')
            else:
                raise ValueError(
                    f'Model spec string {CLI_ARGS.model_spec} not recognized.')

    OBSERVATIONS = CONLLReader(CONLL_COLS).load_conll_dataset(
        CONLL_FILE)

    SCORES = score(OBSERVATIONS, padlen=CLI_ARGS.pad, n_obs=N_OBS,
                   write_wordpair_data=True,
                   load_npz=LOAD_NPZ, save_npz=CLI_ARGS.save_npz,
                   absolute_value=CLI_ARGS.absolute_value,
                   verbose=True)
    print_uuas_summary_to_file(SCORES, RESULTS_DIR + 'info.txt')
    DF = pd.json_normalize(SCORES, sep='.')
    DF.to_csv(path_or_buf=RESULTS_DIR + 'scores_' + SUFFIX + '.csv',
              index_label='sentence_index')
