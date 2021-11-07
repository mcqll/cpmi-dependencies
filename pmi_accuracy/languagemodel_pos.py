"""
Getting probability estimates
from a language model with pretrained POS probe on top.

-
JL Hoover
July 2020
"""
import itertools
import numpy as np
from tqdm import tqdm
import torch

import pos_probe
import ib_pos_probe

class LanguageModelPOS:
    """
    Base class for getting probability estimates from
    a pretrained contextual embedding model with linear probe on top.
    """

    def __init__(
            self, device, model_spec, batchsize,
            pos_set, probe_state_dict,
            model_state_dict=None, probe_type='linear'):
        self.device = device
        # self.model = AutoModel.from_pretrained(
        #     pretrained_model_name_or_path=model_spec,
        #     state_dict=model_state_dict).to(device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
        self.model = pos_probe.TransformersModel(model_spec, device)
        self.tokenizer = self.model.tokenizer
        self.batchsize = batchsize
        print(f"Language model '{model_spec}' " +
              f"initialized (batchsize = {batchsize}) on {device}.")
        self.pos_set = pos_set
        self.pos_to_id = {POS: i for i, POS in enumerate(self.pos_set)}
        args = dict(
            hidden_dim=self.model.hidden_size, pos_set=pos_set,
            device=device)
        if probe_type == 'IB':
            pretrained_probe = ib_pos_probe.IBProbe(args).to(device)
        elif probe_type == 'linear':
            pretrained_probe = pos_probe.POSProbe(args).to(device)
        else:
            raise ValueError('Unknown probe type. Use "linear" or "IB".')
        self.probe_type = probe_type
        pretrained_probe.load_state_dict(probe_state_dict)
        self.pretrained_probe = pretrained_probe
        print(f"Pretrained {probe_type} probe loaded to {device}.")

    def _create_pmi_dataset(
            self, ptb_tokenlist, ptb_pos_list,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):
        """Return a torch Dataset and DataLoader
        (override in implementing class)."""
        raise NotImplementedError

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, ptb_pos_list,
            add_special_tokens=True,
            pad_left=None, pad_right=None, verbose=True):
        """Maps tokenlist to PMI matrix, and also returns pseudo log likelihood.
        (override in implementing class)."""
        raise NotImplementedError

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        raise NotImplementedError


class BERT(LanguageModelPOS):
    """Class for using BERT with probe on top"""

    def _create_pmi_dataset(
            self, ptb_tokenlist, ptb_pos_list,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):

        # map each ptb token to a list of spans
        # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
        tokens, ptbtok_to_span = self.make_subword_lists(
            ptb_tokenlist, add_special_tokens=False)

        # map each span to the ptb token position
        # {(0,): 0, (1, 2,): 1, (3,): 2}
        span_to_ptbtok = {}
        for i, span in enumerate(ptbtok_to_span):
            assert span not in span_to_ptbtok
            span_to_ptbtok[span] = i

        # just convert here, tokenizat'n is taken care of by make_subword_lists
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # add special characters add optional padding
        if pad_left:
            pad_left_tokens, _ = self.make_subword_lists(pad_left)
            if add_special_tokens:
                pad_left = [self.tokenizer.cls_token_id]
            pad_left += self.tokenizer.convert_tokens_to_ids(pad_left_tokens)
            if add_special_tokens:
                pad_left += [self.tokenizer.sep_token_id]
        else:
            pad_left = [self.tokenizer.cls_token_id]
        if pad_right:
            pad_right_tokens, _ = self.make_subword_lists(pad_right)
            pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
        else:
            pad_right = []
        if add_special_tokens:
            pad_right += [self.tokenizer.sep_token_id]
        ids = pad_left + ids + pad_right
        n_pad_left = len(pad_left)
        n_pad_right = len(pad_right)

        ptb_pos_ids = [self.pos_to_id[pos] for pos in ptb_pos_list]

        # copy POS id to each subtoken of the word it corresponds to
        # [i for (i, span) in zip(ptb_pos_ids, ptbtok_to_span) for _ in span]
        target_pos_ids = []
        for idx, span in zip(ptb_pos_ids, ptbtok_to_span):
            for _ in span:
                target_pos_ids.append(idx)
        # add padding to the pos id list
        pad_pos_id = -1
        pos_ids = list(itertools.chain(
            [pad_pos_id for _ in pad_left],
            target_pos_ids,
            [pad_pos_id for _ in pad_right]))

        if verbose:
            print(f"PTB tokenlist, ids:\n{list(zip(ptb_tokenlist, ids))}")
            print(f"resulting subword tokens:\n{tokens}")
            print(f"POS list, POS ids:\n{list(zip(ptb_pos_list,ptb_pos_ids))}")
            print(f"ptbtok_to_span: {ptbtok_to_span}")
            print(f"span_to_ptbtok: {span_to_ptbtok}")
            print(f"target_pos_ids: {target_pos_ids}")
            print(f'padleft:{pad_left}\npadright:{pad_right}')
            print(f'input_ids: {ids}')
            print(f"padded pos_ids: {pos_ids}")

        # setup data loader
        dataset = BERTSentenceDataset(
            ids, pos_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_pos_id=-1,
            n_pad_left=n_pad_left, n_pad_right=n_pad_right)
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batchsize,
            collate_fn=BERTSentenceDataset.collate_fn)
        return dataset, loader

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, ptb_pos_list,
            add_special_tokens=True,
            pad_left=None, pad_right=None, verbose=True):
        """Get pmi matrix for given list of PTB tokens and POS tags

        input:
            ptb_tokenlist: PTB-tokenized sentence as list
            ptb_pos_list: corresponding list of POS tags
        return:
            CPMI matrix estimated for that sentence
        """

        # create dataset for observed ptb sentence
        dataset, loader = self._create_pmi_dataset(
            ptb_tokenlist, ptb_pos_list,
            verbose=verbose,
            pad_left=pad_left, pad_right=pad_right,
            add_special_tokens=add_special_tokens)

        # use model to compute PMIs
        results = []
        for batch in tqdm(loader, desc="[getting embeddings]", leave=False):
            embeddings = self.model.get_embeddings(
                batch['input_ids'].to(self.device))
            if self.probe_type=='linear':
                outputs = self.pretrained_probe(embeddings)  # as logprobs
            elif self.probe_type=='IB':
                outputs = self.pretrained_probe(embeddings)
                outputs = outputs[0]  # ignore kld output
            for i, output in enumerate(outputs):
                # the token id we need to predict belongs to target span
                target_pos_id = batch['target_pos_id'][i]
                input_ids = batch['input_ids'][i]
                target_loc = batch['target_loc'][i]
                assert output.size(0) == len(input_ids)
                log_target = output[target_loc, target_pos_id].item()
                result_dict = {}
                result_dict['source_span'] = batch['source_span'][i]
                result_dict['target_span'] = batch['target_span'][i]
                result_dict['log_target'] = log_target
                result_dict['target_pos_id'] = target_pos_id
                results.append(result_dict)

        num_ptbtokens = len(ptb_tokenlist)
        log_p = np.zeros((num_ptbtokens, num_ptbtokens))
        num = np.zeros((num_ptbtokens, num_ptbtokens))
        for result in results:
            log_target = result['log_target']  # predicted log prob
            source_span = result['source_span']  # tuple of indices
            target_span = result['target_span']  # tuple of indices
            ptbtok_source = dataset.span_to_ptbtok[source_span]  # single index
            ptbtok_target = dataset.span_to_ptbtok[target_span]  # single index
            if len(target_span) == 1:
                # sanity check: if target_span is 1 token, then we don't need
                # to accumulate subwords probabilities
                assert log_p[ptbtok_target, ptbtok_source] == 0.
            # we accumulate all log probs for subwords in a given span
            # and get linear mean
            log_p[ptbtok_target, ptbtok_source] = np.logaddexp(
                log_p[ptbtok_target, ptbtok_source], log_target)
            num[ptbtok_target, ptbtok_source] += 1
        # logsumexp summed in logspace, divide by num in log space for mean
        log_p -= np.log(num)

        # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
        # log_p[i, i] is log p(w_i | c)
        # log_p[i, j] is log p(w_i | c \ w_j)
        log_p_wi_I_c = np.diag(log_p)
        pseudo_loglik = np.trace(log_p)
        pmi_matrix = log_p_wi_I_c[:, None] - log_p
        return pmi_matrix, pseudo_loglik

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        '''
        Takes list of items from Penn Treebank tokenized text,
        runs the tokenizer to decompose into the subword tokens expected by XLNet,
        including appending special characters '[CLS]' and '[SEP]', if specified.
        Implements some simple custom adjustments to make the results more like what might be expected.
        [TODO: this could be improved, if it is important.
        For instance, currently it puts an extra space before opening quotes]
        Returns:
            tokens: a flat list of subword tokens
            ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
                where the nth tuple is token indices for the nth ptb word.
        '''
        subword_lists = []
        if add_special_tokens:
            subword_lists.append(['[CLS]'])
        for word in ptb_tokenlist:
            if word == '-LCB-': word = '{'
            elif word == '-RCB-': word = '}'
            elif word == '-LSB-': word = '['
            elif word == '-RSB-': word = ']'
            elif word == '-LRB-': word = '('
            elif word == '-RRB-': word = ')'
            word_tokens = self.tokenizer.tokenize(word)
            subword_lists.append(word_tokens)
        if add_special_tokens:
            subword_lists.append(['[SEP]'])
        # Custom adjustments below
        for i, subword_list_i in enumerate(subword_lists):
            if subword_list_i == ['n', "'", 't'] and i != 0:
                # print(f"{i}: fixing X n ' t => Xn ' t ")
                del subword_list_i[0]
                subword_lists[i-1][-1] += 'n'

        tokens = list(itertools.chain(*subword_lists))  # flattened list
        ptbtok_to_span = []
        pos = 0
        for token in subword_lists:
            ptbtok_to_span.append(())
            for _ in token:
                ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
                pos += 1
        return tokens, ptbtok_to_span


class BERTSentenceDataset(torch.utils.data.Dataset):
    """Dataset class for BERT"""

    def __init__(
            self, input_ids, pos_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=103, pad_pos_id=-1,
            n_pad_left=0, n_pad_right=0):
        self.input_ids = input_ids
        self.n_pad_left = n_pad_left
        self.n_pad_right = n_pad_right
        self.mask_token_id = mask_token_id
        self.ptbtok_to_span = ptbtok_to_span
        self.span_to_ptbtok = span_to_ptbtok
        self.pos_ids = pos_ids
        self.pad_pos_id = pad_pos_id
        self._make_tasks()

    @staticmethod
    def collate_fn(batch):
        """concatenate and prepare batch"""
        tbatch = {}
        tbatch["input_ids"] = torch.LongTensor([b['input_ids'] for b in batch])
        tbatch["target_loc"] = [b['target_loc'] for b in batch]
        tbatch["target_pos_id"] = [b['target_pos_id'] for b in batch]
        tbatch["source_span"] = [b['source_span'] for b in batch]
        tbatch["target_span"] = [b['target_span'] for b in batch]
        return tbatch

    def _make_tasks(self):
        tasks = []
        for source_span in self.ptbtok_to_span:
            for target_span in self.ptbtok_to_span:
                for idx_target, target_pos in enumerate(target_span):
                    # these are the positions of the source span
                    abs_source = [self.n_pad_left + s for s in source_span]
                    # this is the token we want to predict in the target span
                    abs_target_curr = self.n_pad_left + target_pos
                    # these are all the toks we need to mask in the target span
                    abs_target_next = [self.n_pad_left + t
                                       for t in target_span[idx_target:]]
                    # we replace all hidden target tokens with [MASK]
                    input_ids = np.array(self.input_ids)
                    input_ids[abs_target_next] = self.mask_token_id
                    # if the source span is different from target span,
                    # then we need to mask all of its tokens
                    if source_span != target_span:
                        input_ids[abs_source] = self.mask_token_id
                    # the loc in input list to predict (bc bert predicts all)
                    target_loc = abs_target_curr
                    # build all
                    task_dict = {}
                    task_dict["input_ids"] = input_ids
                    task_dict["source_span"] = source_span
                    task_dict["target_span"] = target_span
                    task_dict["target_loc"] = target_loc
                    task_dict["target_pos_id"] = self.pos_ids[abs_target_curr]
                    tasks.append(task_dict)
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]


class XLNet(LanguageModelPOS):
    """Class for using XLNet with probe on top"""

    def _create_pmi_dataset(
            self, ptb_tokenlist, ptb_pos_list,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):

        # map each ptb token to a list of spans
        # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
        tokens, ptbtok_to_span = self.make_subword_lists(
            ptb_tokenlist, add_special_tokens=False)

        # map each span to the ptb token position
        # {(0,): 0, (1, 2,): 1, (3,): 2}
        span_to_ptbtok = {}
        for i, span in enumerate(ptbtok_to_span):
            assert span not in span_to_ptbtok
            span_to_ptbtok[span] = i

        # just convert here, tokenizat'n is taken care of by make_subword_lists
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # add special characters add optional padding
        if pad_left:
            pad_left_tokens, _ = self.make_subword_lists(pad_left)
            pad_left = self.tokenizer.convert_tokens_to_ids(pad_left_tokens)
            if add_special_tokens:
                pad_left += [self.tokenizer.sep_token_id]
        else:
            pad_left = []
        if pad_right:
            pad_right_tokens, _ = self.make_subword_lists(pad_right)
            pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
        else:
            pad_right = []
        if add_special_tokens:
            pad_right += [self.tokenizer.sep_token_id,
                          self.tokenizer.cls_token_id]
        ids = pad_left + ids + pad_right
        n_pad_left = len(pad_left)
        n_pad_right = len(pad_right)

        ptb_pos_ids = [self.pos_to_id[pos] for pos in ptb_pos_list]

        # copy POS id to each subtoken of the word it corresponds to
        # [i for (i, span) in zip(ptb_pos_ids, ptbtok_to_span) for _ in span]
        target_pos_ids = []
        for idx, span in zip(ptb_pos_ids, ptbtok_to_span):
            for _ in span:
                target_pos_ids.append(idx)
        # add padding to the pos id list
        pad_pos_id = -1
        pos_ids = list(itertools.chain(
            [pad_pos_id for _ in pad_left],
            target_pos_ids,
            [pad_pos_id for _ in pad_right]))

        if verbose:
            print(f"PTB tokenlist, ids:\n{list(zip(ptb_tokenlist, ids))}")
            print(f"resulting subword tokens:\n{tokens}")
            print(f"POS list, POS ids:\n{list(zip(ptb_pos_list,ptb_pos_ids))}")
            print(f"ptbtok_to_span: {ptbtok_to_span}")
            print(f"span_to_ptbtok: {span_to_ptbtok}")
            print(f"target_pos_ids: {target_pos_ids}")
            print(f'padleft:{pad_left}\npadright:{pad_right}')
            print(f'input_ids: {ids}')
            print(f"padded pos_ids: {pos_ids}")

        # setup data loader
        dataset = XLNetSentenceDataset(
            ids, pos_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_pos_id=-1,
            n_pad_left=n_pad_left, n_pad_right=n_pad_right)
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batchsize,
            collate_fn=XLNetSentenceDataset.collate_fn)
        return dataset, loader

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, ptb_pos_list,
            add_special_tokens=True,
            pad_left=None, pad_right=None, verbose=True):
        """Get pmi matrix for given list of PTB tokens and POS tags

        input:
            ptb_tokenlist: PTB-tokenized sentence as list
            ptb_pos_list: corresponding list of POS tags
        return:
            CPMI matrix estimated for that sentence
        """

        # create dataset for observed ptb sentence
        dataset, loader = self._create_pmi_dataset(
            ptb_tokenlist, ptb_pos_list,
            verbose=verbose,
            pad_left=pad_left, pad_right=pad_right,
            add_special_tokens=add_special_tokens)

        # use model to compute PMIs
        results = []
        for batch in tqdm(loader, desc="[getting embeddings]", leave=False):
            embeddings = self.model.get_embeddings(
                batch['input_ids'].to(self.device),
                perm_mask=batch['perm_mask'].to(self.device),
                target_mapping=batch['target_map'].to(self.device))
            if self.probe_type=='linear':
                outputs = self.pretrained_probe(embeddings)  # as logprobs
            elif self.probe_type=='IB':
                outputs = self.pretrained_probe(embeddings)
                outputs = outputs[0]  # ignore first component (kld)
            for i, output in enumerate(outputs):
                # the token id we need to predict belongs to target span
                target_pos_id = batch['target_pos_id'][i]
                assert output.size(0) == 1
                log_target = output[0, target_pos_id].item()
                result_dict = {}
                result_dict['source_span'] = batch['source_span'][i]
                result_dict['target_span'] = batch['target_span'][i]
                result_dict['log_target'] = log_target
                result_dict['target_pos_id'] = target_pos_id
                results.append(result_dict)

        num_ptbtokens = len(ptb_tokenlist)
        log_p = np.zeros((num_ptbtokens, num_ptbtokens))
        num = np.zeros((num_ptbtokens, num_ptbtokens))
        for result in results:
            log_target = result['log_target']  # predicted log prob
            source_span = result['source_span']  # tuple of indices
            target_span = result['target_span']  # tuple of indices
            ptbtok_source = dataset.span_to_ptbtok[source_span]  # single index
            ptbtok_target = dataset.span_to_ptbtok[target_span]  # single index
            if len(target_span) == 1:
                # sanity check: if target_span is 1 token, then we don't need
                # to accumulate subwords probabilities
                assert log_p[ptbtok_target, ptbtok_source] == 0.
            # we accumulate all log probs for subwords in a given span
            # and get linear mean
            log_p[ptbtok_target, ptbtok_source] = np.logaddexp(
                log_p[ptbtok_target, ptbtok_source], log_target)
            num[ptbtok_target, ptbtok_source] += 1
        # logsumexp summed in logspace, divide by num in log space for mean
        log_p -= np.log(num)

        # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
        # log_p[i, i] is log p(w_i | c)
        # log_p[i, j] is log p(w_i | c \ w_j)
        log_p_wi_I_c = np.diag(log_p)
        pseudo_loglik = np.trace(log_p)
        pmi_matrix = log_p_wi_I_c[:, None] - log_p
        return pmi_matrix, pseudo_loglik

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        '''
        Takes list of items from Penn Treebank tokenized text,
        runs the tokenizer to decompose into the subword tokens expected by XLNet,
        including appending special characters '<sep>' and '<cls>', if specified.
        Implements some simple custom adjustments to make the results more like what might be expected.
        [TODO: this could be improved, if it is important.
        For instance, currently it puts an extra space before opening quotes]
        Returns:
            tokens: a flat list of subword tokens
            ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
                where the nth tuple is token indices for the nth ptb word. TODO
        '''
        subword_lists = []
        for word in ptb_tokenlist:
            if word == '-LCB-': word = '{'
            elif word == '-RCB-': word = '}'
            elif word == '-LSB-': word = '['
            elif word == '-RSB-': word = ']'
            elif word == '-LRB-': word = '('
            elif word == '-RRB-': word = ')'
            word_tokens = self.tokenizer.tokenize(word)
            subword_lists.append(word_tokens)
        if add_special_tokens:
            subword_lists.append(['<sep>'])
            subword_lists.append(['<cls>'])
        # Custom adjustments below
        for i, subword_list_i in enumerate(subword_lists):
            if subword_list_i[0][0] == '▁' and subword_lists[i-1][-1] in ('(','[','{'):
                # print(f'{i}: removing extra space after character. {subword_list_i[0]} => {subword_list_i[0][1:]}')
                subword_list_i[0] = subword_list_i[0][1:]
                if subword_list_i[0] == '':
                    subword_list_i.pop(0)
            if subword_list_i[0] == '▁' and subword_list_i[1] in (')',']','}',',','.','"',"'","!","?") and i != 0:
                # print(f'{i}: removing extra space before character. {subword_list_i} => {subword_list_i[1:]}')
                subword_list_i.pop(0)
            if subword_list_i == ['▁', 'n', "'", 't'] and i != 0:
                # print(f"{i}: fixing X▁n't => Xn 't ")
                del subword_list_i[0]
                del subword_list_i[0]
                subword_lists[i-1][-1] += 'n'

        tokens = list(itertools.chain(*subword_lists)) # flattened list
        ptbtok_to_span = []
        pos = 0
        for token in subword_lists:
            ptbtok_to_span.append(())
            for _ in token:
                ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
                pos += 1
        return tokens, ptbtok_to_span


class XLNetSentenceDataset(torch.utils.data.Dataset):
    """Dataset class for XLNet"""

    def __init__(
            self, input_ids, pos_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=6, pad_pos_id=-1,
            n_pad_left=0, n_pad_right=0):
        self.input_ids = input_ids
        self.n_pad_left = n_pad_left
        self.n_pad_right = n_pad_right
        self.mask_token_id = mask_token_id
        self.ptbtok_to_span = ptbtok_to_span
        self.span_to_ptbtok = span_to_ptbtok
        self.pos_ids = pos_ids
        self.pad_pos_id = pad_pos_id
        self._make_tasks()

    @staticmethod
    def collate_fn(batch):
        """concatenate and prepare batch"""
        tbatch = {}
        tbatch["input_ids"] = torch.LongTensor([b['input_ids'] for b in batch])
        tbatch["perm_mask"] = torch.FloatTensor([b['perm_mask'] for b in batch])
        tbatch["target_map"] = torch.FloatTensor([b['target_map'] for b in batch])
        tbatch["target_pos_id"] = [b['target_pos_id'] for b in batch]
        tbatch["source_span"] = [b['source_span'] for b in batch]
        tbatch["target_span"] = [b['target_span'] for b in batch]
        return tbatch

    def _make_tasks(self):
        tasks = []
        len_s = len(self.input_ids) # length in subword tokens
        len_t = len(self.ptbtok_to_span) # length in ptb tokens
        for source_span in self.ptbtok_to_span:
            for target_span in self.ptbtok_to_span:
                for idx_target, target_pos in enumerate(target_span):
                    # these are the positions of the source span
                    abs_source = [self.n_pad_left + s for s in source_span]
                    # this is the token we want to predict in the target span
                    abs_target_curr = self.n_pad_left + target_pos
                    # these are all the toks we need to mask in the target span
                    abs_target_next = [self.n_pad_left + t
                                       for t in target_span[idx_target:]]
                    # we replace all hidden target tokens with [MASK]
                    input_ids = np.array(self.input_ids)
                    input_ids[abs_target_next] = self.mask_token_id
                    # create permutation mask
                    perm_mask = np.zeros((len_s, len_s))
                    perm_mask[:, abs_target_next] = 1.
                    # if the source span is different from target span,
                    # then we need to mask all of its tokens
                    if source_span != target_span:
                        input_ids[abs_source] = self.mask_token_id
                        perm_mask[:, abs_source] = 1.
                    # build prediction map
                    target_map = np.zeros((1, len_s))
                    target_map[0, abs_target_curr] = 1.
                    # build all
                    task_dict = {}
                    task_dict["input_ids"] = input_ids
                    task_dict["source_span"] = source_span
                    task_dict["target_span"] = target_span
                    task_dict["target_map"] = target_map
                    task_dict["perm_mask"] = perm_mask
                    task_dict["target_pos_id"] = self.pos_ids[abs_target_curr]
                    tasks.append(task_dict)
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]
