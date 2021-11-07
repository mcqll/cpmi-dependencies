"""Training a linear probe to extract POS embeddings.

-
JL Hoover
July 2020
"""

import os
from functools import partial
from collections import namedtuple
from datetime import datetime
from argparse import ArgumentParser
from contextlib import redirect_stdout
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

import ib_pos_probe


class CONLLReader():
    def __init__(self, conll_cols, additional_field_name=None):
        if additional_field_name:
            conll_cols += [additional_field_name]
        self.conll_cols = conll_cols
        self.observation_class = namedtuple("Observation", conll_cols)
        self.additional_field_name = additional_field_name

    # Data input
    @staticmethod
    def generate_lines_for_sent(lines):
        """Yields batches of lines describing a sentence in conllx.

        Args:
            lines: Each line of a conllx file.
        Yields:
            a list of lines describing a single sentence in conllx.
        """
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
                buf.append(line.strip())
        if buf:
            yield buf

    def load_conll_dataset(self, filepath):
        """Read in a conllx file; generate Observation objects.

        For each sentence in a conllx file, generate a single Observation
        object.

        Args:
            filepath: the filesystem path to the conll dataset
            observation_class: namedtuple for observations

        Returns:
        A list of Observations
        """
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


class POSProbe(nn.Module):
    """Class for linear probe."""

    def __init__(self, args):
        """Args: global args dict."""
        super().__init__()
        self.args = args
        self.hidden_dim = args['hidden_dim']
        self.pos_vocabsize = len(args['pos_set'])
        self.linear = nn.Linear(self.hidden_dim, self.pos_vocabsize)
        self.to(args['device'])

    def forward(self, H):
        """Linear Probe.

        Performs the linear transform W,
        and takes the log_softmax to get a
        log probability distribution over POS tags
        Args:
            H: a batch of sequences, i.e. a tensor of shape
                (batch_size, max_slen, hidden_dim)
        Returns:
            prediction: a batch of predictions, i.e. a tensor of shape
                (batch_size, max_slen, pos_vocabsize)
        """
        WH = self.linear(H)
        prediction = F.log_softmax(WH, dim=-1)
        return prediction


class POSProbeLoss(nn.Module):
    """Cross entropy loss for linear probe."""

    def __init__(self, args):
        """Args: global args dict."""
        super().__init__()
        self.args = args

    def forward(self, prediction_batch, label_batch, length_batch):
        """Get loss (and number of sentences) for batch.

        Gets the Xent loss between the predicted POS distribution and the label
        Args:
            prediction_batch: pytorch batch of softmaxed POS predictions
            label_batch: pytorch batch of true POS label ids (torch.long)
            length_batch: pytorch batch of sentence lengths

        Returns:
            A tuple of:
                batch_loss: average loss in the batch
                number_of_sentences: number of sentences in the batch
        """
        number_of_sentences = torch.sum(length_batch != 0).float()
        device = self.args['device']
        if number_of_sentences > 0:
            prediction_batch = prediction_batch.view(
                -1, len(self.args['pos_set'])).to(device)
            label_batch = label_batch.view(-1).to(device)
            batch_loss = nn.CrossEntropyLoss(
                ignore_index=self.args['pad_pos_id'])(
                    prediction_batch, label_batch)
        else:
            batch_loss = torch.tensor(0.0, device=device)
        return batch_loss, number_of_sentences


class POSDataset(Dataset):
    """PyTorch dataloader for POS from Observations."""

    def __init__(self, args, observations, tokenizer, observation_class):
        """Initialize Dataset for Observation class.

        Args:
            observations: A list of Observations describing a dataset
            tokenizer: an instance of a transformers Tokenizer class
            observation_class: a namedtuple class specifying the fields
            pos_set: the set of POS tags
        """
        self.observations = observations
        self.pos_set_type = args['pos_set_type']
        self.pos_set = args['pos_set']
        self.pad_token_id = args['pad_token_id']
        self.pad_pos_id = args['pad_pos_id']
        self.tokenizer = tokenizer
        self.pos_to_id = {POS: i for i, POS in enumerate(self.pos_set)}
        self.observation_class = observation_class
        self.input_ids, self.pos_ids = self.get_input_ids_and_pos_ids()

    def sentences_to_idlists(self):
        """Replace strings in Observation with lists of integer ids.

        Returns:
            A list of observations with nested integer-lists as sentence fields
        """
        idlist_observations = []
        for obs in tqdm(self.observations, desc="[getting subtoken ids]"):
            idlist = tuple(
                [self.tokenizer.encode(item, add_special_tokens=False)
                 for item in obs.FORM])
            idlist_observations.append(self.observation_class(
                # replace 'FORM' field with nested list of token ids
                obs[0], idlist, *obs[2:]))
        return idlist_observations

    def get_input_ids_and_pos_ids(self):
        """Get flat list of input and POS ids for each observation.

        Returns:
            input_ids: a list containing a list of input ids for each
                observation
            pos_ids: a list containing a list of POS 'ids' for each
                observation, which will repeat when there is more than
                one subtoken per POS tagged word.
        """
        idlist_observations = self.sentences_to_idlists()  # ids replace words
        subtoken_id_lists = [ob.FORM for ob in idlist_observations]
        if self.pos_set_type == 'xpos':
            pos_label_lists = [ob.XPOS for ob in idlist_observations]
        elif self.pos_set_type == 'upos':
            pos_label_lists = [ob.UPOS for ob in idlist_observations]
        input_ids, pos_ids = self.repeat_pos_to_match(
            subtoken_id_lists, pos_label_lists)
        return input_ids, pos_ids

    def repeat_pos_to_match(self, list_id, list_pos):
        """Copy POS tag to each of a word's subword tokens."""
        assert len(list_pos) == len(list_id), "list lengths don't match"
        new_id = []
        new_pos = []
        for i, el_id in enumerate(list_id):
            newlist_id = []
            newlist_pos = []
            for j, elel_id in enumerate(el_id):
                for token_id in elel_id:
                    newlist_id.append(token_id)
                    newlist_pos.append(self.pos_to_id[list_pos[i][j]])
            new_id.append(newlist_id)
            new_pos.append(newlist_pos)
        return new_id, new_pos

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        return self.input_ids[index], self.pos_ids[index]

    @staticmethod
    def collate_fn(batch, pad_token_id, pad_pos_id):
        """Collate_fn for torch DataLoader."""
        seqs = [torch.tensor(b[0]) for b in batch]
        pos_ids = [torch.tensor(b[1]) for b in batch]
        lengths = torch.tensor([len(s) for s in seqs])
        padded_input_ids = nn.utils.rnn.pad_sequence(
            seqs, padding_value=pad_token_id, batch_first=True)
        padded_pos_ids = nn.utils.rnn.pad_sequence(
            pos_ids, padding_value=pad_pos_id, batch_first=True)
        return padded_input_ids, padded_pos_ids, lengths


class TransformersModel:
    """Class wrapper for huggingface transformers model."""

    def __init__(self, spec, device='cpu'):
        """Initialize transformers model and tokenizer on device."""
        self.spec = spec
        self.device = device
        self.model = AutoModel.from_pretrained(spec).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec)
        self.hidden_size = self.model.config.hidden_size
        self.pad_token_id = self.model.config.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_pos_id = -1

    def get_embeddings(self, input_ids_batch, **kwds):
        """Get POS embeddings for batch.

        Input:
            input_ids_batch: a batch of (padded) input ids
            kwds:
                - no kwds expected for BERT;
                - 'perm_mask' and 'target mapping' batches for XLNet
        Return:
            last hidden layer of pretrained model for that batch
        """
        # 0 for MASKED tokens. 1 for NOT MASKED tokens.
        attention_mask = (
            input_ids_batch != self.pad_token_id).type(torch.float)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_batch.to(self.device),
                attention_mask=attention_mask.to(self.device),
                **kwds)
        # the hidden states are the first component of the outputs tuple
        return outputs[0]


def run_train_probe(args, model, probe, loss, train_loader, dev_loader):
    """Train probe."""
    device = args['device']
    use_bottleneck = args['bottleneck']
    pad_pos_id = args['pad_pos_id']
    pos_vocabsize = len(args['pos_set'])
    opt = args['training_options']
    if opt['algorithm'] == 'adam':
        optimizer = torch.optim.Adam(probe.parameters(), **opt['hyperparams'])
    elif opt['algorithm'] == 'sgd':
        optimizer = torch.optim.SGD(probe.parameters(), **opt['hyperparams'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=0)
    max_acc = -100
    max_acc_epoch = -1

    for ep_i in tqdm(range(args['epochs']), desc='training'):
        ep_train_loss = 0
        ep_dev_loss = 0
        ep_train_ep_count = 0
        ep_dev_ep_count = 0
        ep_train_loss_count = 0
        ep_dev_loss_count = 0

        for batch in tqdm(train_loader, desc='train batch', leave=False):
            probe.train()
            optimizer.zero_grad()
            input_ids_batch, label_batch, length_batch = batch
            length_batch = length_batch.to(device)
            mask = (
                label_batch != args['pad_pos_id']).type(torch.float).to(device)
            embedding_batch = model.get_embeddings(input_ids_batch).to(device)
            if use_bottleneck:
                # embedding_batch size ([batchsize, maxsentlen, embeddingsize])
                prediction_batch, kld_batch = probe(embedding_batch)
                # get the mean kld per sentence, ignoring words not to predict
                mean_kld = (kld_batch.to(device) * mask).sum(1) / mask.sum(1)
                train_kld = mean_kld.mean(0)  # mean across batch
                # prediction_batch size ([batchsize, maxsentlen, POSvocabsize])
                batch_loss, count = loss(
                    prediction_batch.to(device),
                    label_batch, length_batch, train_kld)
            else:
                # embedding_batch size ([batchsize, maxsentlen, embeddingsize])
                prediction_batch = probe(embedding_batch)
                # prediction_batch size ([batchsize, maxsentlen, POSvocabsize])
                batch_loss, count = loss(
                    prediction_batch.to(device),
                    label_batch, length_batch)
            prediction_accuracy = get_batch_acc(
                label_batch, prediction_batch, pad_pos_id, pos_vocabsize)
            batch_loss.backward()
            ep_train_loss += (
                batch_loss.detach() * count.detach()).cpu().numpy()
            ep_train_ep_count += 1
            ep_train_loss_count += count.detach().cpu().numpy()
            optimizer.step()

        for batch in tqdm(dev_loader, desc='dev batch', leave=False):
            optimizer.zero_grad()
            probe.eval()
            input_ids_batch, label_batch, length_batch = batch
            length_batch = length_batch.to(device)
            mask = (
                label_batch != args['pad_pos_id']).type(torch.float).to(device)
            embedding_batch = model.get_embeddings(input_ids_batch).to(device)
            if use_bottleneck:
                # embedding_batch size ([batchsize, maxsentlen, embeddingsize])
                prediction_batch, kld_batch = probe(embedding_batch)
                # get the mean kld per sentence, ignoring words not to predict
                mean_kld = (kld_batch.to(device) * mask).sum(1) / mask.sum(1)
                dev_kld = mean_kld.mean(0)  # mean across batch
                # prediction_batch size ([batchsize, maxsentlen, POSvocabsize])
                batch_loss, count = loss(
                    prediction_batch.to(device),
                    label_batch, length_batch, dev_kld)
            else:
                # embedding_batch size ([batchsize, maxsentlen, embeddingsize])
                prediction_batch = probe(embedding_batch)
                # prediction_batch size ([batchsize, maxsentlen, POSvocabsize])
                batch_loss, count = loss(
                    prediction_batch.to(device),
                    label_batch, length_batch)
            dev_accuracy = get_batch_acc(
                label_batch, prediction_batch, pad_pos_id, pos_vocabsize)
            ep_dev_loss += (
                batch_loss.detach() * count.detach()).cpu().numpy()
            ep_dev_loss_count += count.detach().cpu().numpy()
            ep_dev_ep_count += 1
        scheduler.step(ep_dev_loss)
        if use_bottleneck:
            train_kld = f'\ttrain_kld: {train_kld:.3f}'
            dev_kld = f'\t  dev_kld: {dev_kld:.3f}'
        else:
            dev_kld, train_kld = '', ''
        msg = (
            f'[epoch {ep_i}]\n'
            + f'\tper sent train loss:'
            + f'{ep_train_loss/ep_train_loss_count:.3f},'
            + train_kld
            + f'\ttrain acc: {prediction_accuracy*100:.2f} %\n'
            + f'\tper sent   dev loss: {ep_dev_loss/ep_dev_loss_count:.3f},'
            + dev_kld
            + f'\t  dev acc: {dev_accuracy*100:.2f} %')
        if dev_accuracy > max_acc + 0.000001:
            torch.save(
                probe.state_dict(),
                os.path.join(ARGS['results_path'], 'probe.state_dict'))
            max_acc = dev_accuracy
            max_acc_epoch = ep_i
            tqdm.write(msg + '\tSaving probe state_dict')
            write_saved_acc(RESULTS_PATH, ep_i, dev_accuracy)
        elif max_acc_epoch < ep_i - 6:
            tqdm.write(msg + '\tEarly stopping')
            break
        else:
            tqdm.write(msg)


def write_saved_acc(results_path, ep_i, dev_accuracy, msg=""):
    """Append accuracy (and optional message) to info file."""
    with open(results_path + 'info.txt', mode='a') as infof:
        infof.write(
            f'epoch{ep_i:3d} dev acc = {dev_accuracy*100} % ' + msg + '\n')


def get_batch_acc(label_batch, prediction_batch, pad_pos_id, pos_vocabsize):
    """Get prediction accuracy for the positions that aren't padding."""
    # not_pad is the only positions in prediction to care about
    not_padding = label_batch.view(-1).ne(pad_pos_id)
    labels = label_batch.view(-1)[not_padding]
    preds = prediction_batch.view(-1, pos_vocabsize)[not_padding]
    preds = preds.argmax(-1).cpu()
    assert len(preds) == len(labels),\
        "predictions don't align with labels"
    correct = (preds.eq(labels)).sum().float()
    acc = correct / len(labels)
    return acc


def train_probe(args, model, probe, loss, tokenizer):
    """Train linear probe to recover POS embeddings."""
    train_dataset, dev_dataset, _ = load_datasets(args, tokenizer)

    params = {
        'batch_size': args['batch_size'], 'shuffle': True,
        'collate_fn': partial(
            POSDataset.collate_fn,
            pad_token_id=model.pad_token_id,
            pad_pos_id=model.pad_pos_id)}
    train_loader = DataLoader(train_dataset, **params)
    dev_loader = DataLoader(dev_dataset, **params)

    run_train_probe(args, model, probe, loss, train_loader, dev_loader)


def load_datasets(args, tokenizer):
    """Get pytorch Datasets for train, dev, test observations."""
    train_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['train_path'])
    dev_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['dev_path'])
    test_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['test_path'])
    reader = CONLLReader(args['conll_fieldnames'])
    train_obs = reader.load_conll_dataset(train_corpus_path)
    dev_obs = reader.load_conll_dataset(dev_corpus_path)
    test_obs = reader.load_conll_dataset(test_corpus_path)

    obs_class = reader.observation_class
    train_dataset = POSDataset(args, train_obs, tokenizer, obs_class)
    dev_dataset = POSDataset(args, dev_obs, tokenizer, obs_class)
    test_dataset = POSDataset(args, test_obs, tokenizer, obs_class)

    return train_dataset, dev_dataset, test_dataset


def pretty_print_dict(dic, indent=0):
    """Pretty print nested dict to stdout."""
    for key, value in dic.items():
        print('    ' * indent + key, end=': ')
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 1)
        else:
            print(repr(value))


if __name__ == '__main__':
    ARGP = ArgumentParser()
    ARGP.add_argument('--model_spec', default='bert-base-cased',
                      help='''specify model
                      (e.g. "xlnet-base-cased", "bert-large-cased"),
                      or path for offline''')
    ARGP.add_argument('--pos_set_type', default='xpos',
                      help="xpos (PTB's 17 tags) or upos (UD's 45 tags)")
    ARGP.add_argument('--bottleneck', action='store_true',
                      help='''set to run information bottleneck version,
                           higher beta = more compression''')
    ARGP.add_argument('--beta', default=0.01, type=float,
                      help='''higher beta = more compression''')
    ARGP.add_argument('--batch_size', default=32, type=int)
    ARGP.add_argument('--optimizer', default='adam', type=str)
    ARGP.add_argument('--lr', default=0.001, type=float)
    ARGP.add_argument('--weight_decay', default=0.0001, type=float)
    # ARGP.add_argument('--momentum', default=0, type=float)
    ARGP.add_argument('--epochs', default=40, type=int)
    CLI_ARGS = ARGP.parse_args()

    NOW = datetime.now()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    MODEL = TransformersModel(CLI_ARGS.model_spec, DEVICE)
    TOKENIZER = MODEL.tokenizer

    UPOS_TAGSET = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ',
                   'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                   'SCONJ', 'SYM', 'VERB', 'X']

    XPOS_TAGSET = ['#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':',
                   'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
                   'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
                   'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                   'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                   'WDT', 'WP', 'WP$', 'WRB', '``']

    POS_SET_TYPE = CLI_ARGS.pos_set_type  # 'xpos' or 'upos'
    if POS_SET_TYPE == 'upos':
        POS_TAGSET = UPOS_TAGSET
    elif POS_SET_TYPE == 'xpos':
        POS_TAGSET = XPOS_TAGSET
    # TRAIN_OPTS = dict(  # Hewitt uses Adam with lr=0.001
    #     algorithm='adam',
    #     hyperparams=dict(lr=0.1)
    #     )
    # TRAIN_OPTS = dict(
    #     algorithm='sgd',
    #     hyperparams=dict(lr=0.33, weight_decay=5e-4, momentum=0.9)
    #     )
    # IB_TRAIN_OPTS = dict(
    #     algorithm='adam',
    #     hyperparams=dict(lr=1e-3, weight_decay=0.00001, momentum=0.9)
    #     )
    ARGS = dict(
        bottleneck=CLI_ARGS.bottleneck,
        device=DEVICE,
        spec=CLI_ARGS.model_spec,
        batch_size=CLI_ARGS.batch_size,
        epochs=CLI_ARGS.epochs,
        beta=CLI_ARGS.beta,
        hidden_dim=MODEL.hidden_size,
        pad_token_id=MODEL.pad_token_id,
        pad_pos_id=MODEL.pad_pos_id,
        results_path="probe-results/",
        corpus=dict(root='ptb3-wsj-data/',
                    #train_path='CUSTOM2.conllx',
                    #dev_path='CUSTOM.conllx',
                    #test_path='CUSTOM.conllx'),
                    train_path='ptb3-wsj-train.conllx',
                    dev_path='ptb3-wsj-dev.conllx',
                    test_path='ptb3-wsj-test.conllx'),
        conll_fieldnames=[  # Columns of CONLL file
            'ID', 'FORM', 'LEMMA', 'UPOS',
            'XPOS', 'FEATS', 'HEAD',
            'DEPREL', 'DEPS', 'MISC'],
        pos_set_type=POS_SET_TYPE,
        pos_set=POS_TAGSET,
        training_options=dict(
            algorithm=CLI_ARGS.optimizer,
            hyperparams=dict(
                lr=CLI_ARGS.lr,
                # mementum=CLI_ARGS.momentum,
                weight_decay=CLI_ARGS.weight_decay)))

    SPEC_STRING = ARGS['pos_set_type'] + '_' + ARGS['spec']
    if CLI_ARGS.bottleneck:
        SPEC_STRING = 'IB_' + SPEC_STRING
    RESULTS_DIRNAME = SPEC_STRING + '_' + NOW.strftime("%y.%m.%d-%H.%M") + '/'
    RESULTS_PATH = os.path.join(ARGS['results_path'], RESULTS_DIRNAME)
    ARGS['results_path'] = RESULTS_PATH

    os.makedirs(RESULTS_PATH, exist_ok=True)
    print(f"RESULTS_PATH: {ARGS['results_path']}\n")

    with open(RESULTS_PATH + 'info.txt', mode='a') as infofile:
        with redirect_stdout(infofile):
            pretty_print_dict(ARGS)
        infofile.write('')

    if CLI_ARGS.bottleneck:
        PROBE = ib_pos_probe.IBProbe(ARGS)
        LOSS = ib_pos_probe.IBLoss(ARGS)
    else:
        PROBE = POSProbe(ARGS)
        LOSS = POSProbeLoss(ARGS)

    train_probe(ARGS, MODEL, PROBE, LOSS, TOKENIZER)
