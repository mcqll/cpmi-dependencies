"""
Methods to get a tree structure from a distance matrix (numpy array).
-
March 2020
"""

import torch
import numpy as np

from conll_data import EXCLUDED_PUNCTUATION, EXCLUDED_PUNCTUATION_UPOS


class UnionFind:
    '''
    Naive UnionFind (for computing MST with Prim's algorithm).
    '''

    def __init__(self, n):
        self.parents = list(range(n))

    def union(self, i, j):
        '''
        join i and j
        '''
        if self.find(i) != self.find(j):
            i_parent = self.find(i)
            self.parents[i_parent] = j

    def find(self, i):
        '''
        find parent of i
        '''
        i_parent = i
        while True:
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent


class Accuracy:
    '''
    Gets accuracy score for list of edges wrt gold list of edges.
    '''

    def __init__(self, gold_edges):
        self.gold_edges = gold_edges
        self.gold_edges_set = {tuple(sorted(x)) for x in gold_edges}
        self.n_gold = len(gold_edges)

    def uuas(self, prediction_edges):
        '''
        gets uuas accuracy score (num common/num gold)
        returns:
            int n_common 
            float uuas (number between 0 and 1)
        '''
        prediction_edges_set = {tuple(sorted(x)) for x in prediction_edges}
        common = self.gold_edges_set.intersection(prediction_edges_set)
        n_common = len(common)
        n_gold = self.n_gold
        uuas = n_common / float(n_gold) if n_gold != 0 else np.NaN
        return n_common, uuas


class DepParse:
    """Gets tree as MST from matrix of distances"""

    def __init__(self, parsetype, matrix, words, poses):
        '''
        input
            parsetype: a string to specify the decoding/parsing
                       algorithm (e.g. "mst" or "projective")
            matrix: an array of PMIs
            words: a list of tokens
            poses: a list of UPOS tags (used to determine punctuation to exclude)
        '''
        self.parsetype = parsetype
        self.matrix = matrix
        self.words = words
        self.poses = poses

    def tree(self, symmetrize_method='sum',
             maximum_spanning_tree=True,
             absolute_value=False):
        '''
        Gets a Spanning Tree (list of edges) from a nonsymmetric (PMI) matrix,
        using the specified method. and using maximum spanning tree for prims,
        unless otherwise specified
        input:
            symmetrize_method:
                'sum' (default): sums matrix with transpose of matrix;
                'triu': uses only the upper triangle of matrix;
                'tril': uses only the lower triangle of matrix;
                'none': uses the optimum of each unordered pair of edges.
        returns: tree (list of edges)
        '''
        sym_matrix = self.matrix

        if absolute_value:
            sym_matrix = np.absolute(sym_matrix)

        if symmetrize_method == 'sum':
            sym_matrix = sym_matrix + np.transpose(sym_matrix)
        elif symmetrize_method == 'triu':
            sym_matrix = np.triu(sym_matrix) + \
                np.transpose(np.triu(sym_matrix))
        elif symmetrize_method == 'tril':
            sym_matrix = np.tril(sym_matrix) + \
                np.transpose(np.tril(sym_matrix))
        elif symmetrize_method != 'none':
            raise ValueError(
                "Unknown symmetrize_method. Use 'sum' 'triu' 'tril' or 'none'")

        if self.parsetype == "mst":
            edges = self.prims(
                sym_matrix, self.words, self.poses,
                maximum_spanning_tree=maximum_spanning_tree)
        elif self.parsetype == "projective":
            edges = self.eisners(
                sym_matrix, self.words, self.poses)
        else:
            raise ValueError(
                "Unknown parsetype.  Choose 'mst' or 'projective'")
        if self.parsetype == "projective" and not maximum_spanning_tree:
            raise ValueError(
                "Only use Eisner's algorithm for maximum_spanning_tree.")
        return edges

    @staticmethod
    def prims(matrix, words, poses, maximum_spanning_tree=True):
        '''
        Constructs a maximum spanning tree using Prim's algorithm.
            (set maximum_spanning_tree=False for min spanning tree instead).
        Input: matrix (torch tensor of PMIs), words (list of tokens),
            poses (list of UPOS tags)
        Excludes edges to/from punctuation symbols or empty strings,
            and sets np.NaN to -inf
        Returns: tree (list of edges).
        Based on code by John Hewitt.
        '''
        pairs_to_weights = {}
        # excluded = EXCLUDED_PUNCTUATION
        excluded_poses = EXCLUDED_PUNCTUATION_UPOS
        union_find = UnionFind(len(matrix))
        for i_index, line in enumerate(matrix):
            for j_index, dist in enumerate(line):
                # if words[i_index] in excluded:
                if poses[i_index] in excluded_poses:
                    continue
                # if words[j_index] in excluded:
                if poses[j_index] in excluded_poses:
                    continue
                pairs_to_weights[(i_index, j_index)] = dist
        edges = []
        sorted_pairs = sorted(
            pairs_to_weights.items(),
            key=lambda x: float('-inf') if (x[1] != x[1]) else x[1],
            reverse=maximum_spanning_tree)
        for (i_index, j_index), _ in sorted_pairs:
            if union_find.find(i_index) != union_find.find(j_index):
                union_find.union(i_index, j_index)
                edges.append((i_index, j_index))
        return edges

    def eisners(self, matrix, words, poses):
        """
        Parse using Eisner's algorithm.
        entry matrix[head][dep] of matrix is the score for
        the arc from head to dep based on DependencyDecoder class
        from lxmls-toolkit
        https://github.com/LxMLS/lxmls-toolkit/blob/master/lxmls/parsing/dependency_decoder.py
        """

        # with np.printoptions(precision=2, suppress=True):
        #   print(f"raw input matrix for eisners\n{matrix.numpy()}")

        # excluded = EXCLUDED_PUNCTUATION
        excluded_poses = EXCLUDED_PUNCTUATION_UPOS
        # is_word_included = [word not in excluded for word in words]
        is_word_included = [pos not in excluded_poses for pos in poses]
        wordnum_to_index = {}
        counter = 0
        for index, boolean in enumerate(is_word_included):
            if boolean:
                wordnum_to_index[counter] = index
                counter += 1
        # print(f"wordnum_to_index: {wordnum_to_index}")

        # print(f"is_word_included: {is_word_included}")
        matrix = matrix[is_word_included, :][:, is_word_included]
        # with np.printoptions(precision=2, suppress=True):
        #   print(f"input just words matrix for eisners\n{np.array(matrix.numpy())}")

        # add a column and a row of zeros at index 0, for the root of the tree.
        # Note: 0-index is reserved for the root
        # in the algorithm, values in the first column and the main diagonal will be ignored
        # (nothing points to the root and nothing points to itself)
        # I'll fill the first row with a large negative value, to prevent more than one arc from root
        col_zeros = np.zeros((matrix.shape[0], 1))
        matrix_paddedcol = np.concatenate((col_zeros, matrix), 1)
        row_zeros = np.zeros(
            (1, matrix_paddedcol.shape[1])).reshape(1, -1) - 50
        scores = np.concatenate([row_zeros, matrix_paddedcol], 0)

        # ---- begin algorithm ------

        nrows, ncols = np.shape(scores)
        if nrows != ncols:
            raise ValueError("scores must be a nparray with nw+1 rows")

        N = nrows - 1  # Number of words (excluding root).

        # Initialize CKY table.
        complete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
        incomplete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
        # s, t, direction (right=1).
        complete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)
        # s, t, direction (right=1).
        incomplete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)

        incomplete[0, :, 0] -= np.inf

        # Loop from smaller items to larger items.
        for k in range(1, N + 1):
            for s in range(N - k + 1):
                t = s + k

                # First, create incomplete items.
                # left tree
                incomplete_vals0 = complete[s, s:t, 1] + \
                    complete[(s + 1):(t + 1), t, 0] + scores[t, s]
                incomplete[s, t, 0] = np.max(incomplete_vals0)
                incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
                # right tree
                incomplete_vals1 = complete[s, s:t, 1] + \
                    complete[(s + 1):(t + 1), t, 0] + scores[s, t]
                incomplete[s, t, 1] = np.max(incomplete_vals1)
                incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

                # Second, create complete items.
                # left tree
                complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
                complete[s, t, 0] = np.max(complete_vals0)
                complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
                # right tree
                complete_vals1 = incomplete[
                    s, (s + 1):(t + 1), 1] + complete[(s + 1):(t + 1), t, 1]
                complete[s, t, 1] = np.max(complete_vals1)
                complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

        # value = complete[0][N][1]
        heads = -np.ones(N + 1, dtype=int)
        self.eisners_backtrack(incomplete_backtrack,
                               complete_backtrack, 0, N, 1, 1, heads)

        # ---- end algorithm -----------

        edgelist = list(enumerate(heads))
        # Eisner edges, sorted, removing the root node
        # (taking indices [2:] and shifting all values -1)
        sortededges_noroot = sorted({tuple(sorted(tuple(
            [i - 1 for i in edge]))) for edge in edgelist})[2:]
        # Now with indices translated to give word-to-word edges
        # (simply skipping puncuation indices)
        edges = [tuple(wordnum_to_index[w] for w in pair)
                 for pair in sortededges_noroot]
        return edges

    def eisners_backtrack(
            self, incomplete_backtrack, complete_backtrack,
            s, t, direction, complete, heads):
        """
        Backtracking step in Eisner's algorithm.
        - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a
            start position, an end position, and a direction flag (0 means
            left, 1 means right). This array contains the arg-maxes of each
            step in the Eisner algorithm when building *incomplete* spans.
        - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a 
            start position, an end position, and a direction flag (0 means
            left, 1 means right). This array contains the arg-maxes of each
            step in the Eisner algorithm when building *complete* spans.
        - s is the current start of the span
        - t is the current end of the span
        - direction is 0 (left attachment) or 1 (right attachment)
        - complete is 1 if the current span is complete, and 0 otherwise
        - heads is a (NW+1)-sized numpy array of integers which is a
            placeholder for storing the head of each word.
        """
        if s == t:
            return
        if complete:
            r = complete_backtrack[s][t][direction]
            if direction == 0:
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    s, r, 0, 1, heads)
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    r, t, 0, 0, heads)
                return
            else:
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    s, r, 1, 0, heads)
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    r, t, 1, 1, heads)
                return
        else:
            r = incomplete_backtrack[s][t][direction]
            if direction == 0:
                heads[s] = t
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    s, r, 1, 1, heads)
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    r + 1, t, 0, 1, heads)
                return
            else:
                heads[t] = s
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    s, r, 1, 1, heads)
                self.eisners_backtrack(
                    incomplete_backtrack, complete_backtrack,
                    r + 1, t, 0, 1, heads)
                return
