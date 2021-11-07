'''
Calculate lengthmatched control. 

The lengthmatched control for a given dependency-annotated sentence is a
randomized dependency tree with the same arc-length distribution as the gold
tree. It may be non-projective. This control is computed here by rejection
sampling for a fixed number of iterations, and will fail sometimes on the
longest sentences.

use: 

    python lengthmatch.py > result-lengthmatch.txt

'''

import random
import numpy as np
import networkx as nx
from tqdm import tqdm

from pmi_accuracy.conll_data import CONLLReader, CONLL_COLS, EXCLUDED_PUNCTUATION_UPOS

def is_edge_to_ignore(edge, observation):
    is_d_punct = bool(observation.UPOS[edge[1]-1] in EXCLUDED_PUNCTUATION_UPOS)
    is_h_root = bool(edge[0] == 0)
    return is_d_punct or is_h_root

def score_observation(predicted_edges, observation, convert_from_0index=False):
    # get gold edges (1-indexed)
    gold_edges_list = list(zip(list(map(int, observation.HEAD)),
                               list(map(int, observation.ID)),
                               observation.DEPREL))
    gold_edge_to_label = {(e[0], e[1]): e[2] for e in gold_edges_list
                          if not is_edge_to_ignore(e, observation)}
    # just the unlabeled edges
    gold_edges_set = {tuple(sorted(e)) for e in gold_edge_to_label.keys()}

    # note converting to 1-indexing
    k = 1 if convert_from_0index else 0
    predicted_edges_set = {tuple(sorted((x[0]+k, x[1]+k))) for x in predicted_edges}

    correct_edges = list(gold_edges_set.intersection(predicted_edges_set))
    incorrect_edges = list(predicted_edges_set.difference(gold_edges_set))
    num_correct = len(correct_edges)
    num_total = len(gold_edges_set)
    uuas = num_correct/float(num_total) if num_total != 0 else np.NaN
    return num_total, num_correct, uuas

def Lk(ls):
    """Given a length sequence ls,
    Returns 
        L: the length sequence (sorted)
        k: a list of the counts of each length in L"""
    L = sorted(ls)
    k = [L.count(n) for n in range(L[-1]+1)]
    return L,k

def lengthmatch(observation):
    '''get gold edge set and length sequence'''
    # get gold edges (1-indexed)
    gold_edges_list = list(zip(list(map(int, observation.HEAD)),
                               list(map(int, observation.ID)),
                               observation.DEPREL))
    gold_edge_to_label = {(e[0], e[1]): e[2] for e in gold_edges_list
                          if not is_edge_to_ignore(e, observation)}
    # just the unlabeled edges
    gold_edges_set = {tuple(sorted(e)) for e in gold_edge_to_label.keys()}
    lens = [e[1]-e[0] for e in gold_edges_set]
    return lens, gold_edges_set

def generate_lengthmatch(observation, make_tree=True, iterations=1000):
    '''
    generate length-matched baseline of observation
    after algorithm described in
    https://cs.stackexchange.com/questions/116193
    '''
    lens, ges = lengthmatch(observation)
    print(f"len {len(observation[0])}", end=". ")
    # initialize a graph with the right number of nodes
    for iteration in range(iterations):

        T=nx.Graph()
        T.add_edges_from(ges)
        T.remove_edges_from(ges)
        L,k = Lk(lens)
        
        # for each edge length
        for l in set(L):
            V=T.nodes()
            E=T.edges()

            # generate set P of possible new edges of len l
            eplus = set(tuple(sorted((u,u+l))) for u in V if u+l in V and (u,u+l) not in E)
            eminus= set(tuple(sorted((u,u-l))) for u in V if u-l in V and (u,u-l) not in E)
            P = eplus.union(eminus)
            if len(P)==0:
                raise ValueError("ERROR: no possible edges")
            # sampling k[l] edges of length l from P
            additional_edges = random.sample(P,k[l])
            T.add_edges_from(additional_edges)

            if make_tree and not nx.is_forest(T):
                break
                
        if (make_tree and nx.is_tree(T)) or not make_tree:
            if make_tree: 
                print(f"success after {iteration} iterations")
            return score_observation(T.edges(),observation)
    print(f"FAILED (reached max iterations {iterations})")
    return np.NaN, np.NaN, np.NaN


if __name__ == '__main__':
    CONLL_FILE="ptb3-wsj-data/ptb3-wsj-dev.conllx"
    # CONLL_FILE="ptb3-wsj-data_tatsu/ptb3-wsj-test_10.conllx" # or other conll file.
    OBSERVATIONS = CONLLReader(CONLL_COLS).load_conll_dataset(CONLL_FILE)
    N_ITERATIONS=15000

    n_edges=[]
    n_common=[]
    uuas_scores=[]
    countobs=0
    for i, obs in enumerate(tqdm(OBSERVATIONS)):
        print(str(i).ljust(4),end="")
        if  len(obs[0]) > 2:
            countobs +=1
            n_edg, n_com, uuas = generate_lengthmatch(obs, iterations=N_ITERATIONS)
            uuas_scores.append(uuas)
            n_edges.append(n_edg)
            n_common.append(n_com)
        else:
            print(f"len {len(obs[0])}","(too short/long)")
    failurecount = n_common.count(np.NaN)
    print(f"running lengthmatch.py with iterations={N_ITERATIONS}")
    print(f"{countobs-failurecount} lengthmatched parses sampled successfully (Failure rate: {failurecount}/{countobs} = {failurecount/float(countobs)})")
    print("lengthmatch baseline mean sentence uuas =", np.nanmean(uuas_scores))
    total_uuas = np.nansum(n_common) / float(np.nansum(n_edges))
    print(f'lengthmatch baseline total uuas = {int(np.nansum(n_common))} / {int(np.nansum(n_edges))} = {total_uuas}')
    print('---')
    print(f"n_edges\n{n_edges}")
    print(f"n_common\n{n_common}")
    print(f"uuas_scores\n{uuas_scores}")