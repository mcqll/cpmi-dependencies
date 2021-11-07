"""For pos_probe.py
Training a linear probe to extract POS embeddings,
using information bottleneck.

-
JL Hoover
July 2020
"""

import torch
import torch.nn as nn

import pos_probe


class Bottleneck(nn.Module):
    """Module for bottlenecked representation"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        dim = args["hidden_dim"]
        self.linear = nn.Linear(dim, 2 * dim)
        self.to(args['device'])

    def forward(self, H):
        """Linear layer for bottleneck representation.

        Projects embeddings up to 2*hidden_dim to get parameters
        which can be used as means and covariances for multivariate
        gaussian to define the bottleneck representation z ~ q(z|h).

        Args:
            H: a batch of sequences, i.e. a tensor of shape
                (batch_size, max_slen, hidden_dim)
        Returns:
            tuple (means, log_covariances), torch tensors of
                shape = (batch_size, max_slen, hidden_dim)
        """
        means, log_covariances = torch.chunk(self.linear(H), 2, dim=-1)
        return means, log_covariances


class IBProbe(nn.Module):
    """Given embedding, samples from bottleneck, returns KL and prediction"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.W_encoder = Bottleneck(args)
        self.W_decoder = pos_probe.POSProbe(args)
        self.to(args['device'])

    def forward(self, H):
        """Samples from the bottleneck.
        Args:
            H: a batch of sequences, i.e. a tensor of shape
                (batch_size, max_slen, hidden_dim)
        Returns:
            prediction: a batch of predictions, log_softmaxed, of shape
                (batch_size, max_slen, pos_vocabsize)
            kld: a batch of KL divergences, of shape
                (batch_size, max_slen)

            """
        mus, logsigs = self.W_encoder(H)
        # KLD(q(z | x) || r(z)), where r(z) = N(0,I^d)
        # kld.shape = (batch_size, maxsentlen), so
        # to get loss do kld_loss = kld.mean(1).mean(0)
        kld = 0.5 * (
            logsigs.exp() + mus.pow(2) - 1 - logsigs).sum(2)
        # get logits for (y given z) = Wz using z ~ q(z | x)
        sample = mus + torch.randn_like(logsigs) * torch.exp(0.5 * logsigs)
        prediction = self.W_decoder(sample)
        return prediction, kld


class IBLoss(nn.Module):
    """Calculates the IB loss
    L_IB = Xent(decoder;y) + beta * KL(encoder || gaussian approx)
    """

    def __init__(self, args):
        """Args: global args dict."""
        super().__init__()
        self.args = args
        self.xent_loss = pos_probe.POSProbeLoss(args)

    def forward(self, prediction_batch, label_batch, length_batch, kld):
        """Get IB loss (and number of sentences) for batch.

        Gets the Information Bottleneck loss
        Args:
            prediction_batch: pytorch batch of softmaxed POS predictions
            label_batch: pytorch batch of true POS label ids (torch.long)
            length_batch: pytorch batch of sentence lengths
            kld: KL divergence for batch, float (size 0 tensor)

        Returns:
            A tuple of:
                batch_loss: average loss in the batch
                number_of_sentences: number of sentences in the batch
        """
        batch_xent_loss, number_of_sentences = self.xent_loss(
            prediction_batch, label_batch, length_batch)
        batch_ib_loss = batch_xent_loss + self.args['beta'] * kld
        return batch_ib_loss, number_of_sentences
