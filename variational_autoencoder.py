from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim,
                 hidden_layers_encoder, hidden_layers_decoder, z_sample_count,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputDim = input_dim
        self.embeddingDim = embedding_dim
        self.hiddenLayersEncoder = [input_dim, *hidden_layers_encoder, 2 * embedding_dim]
        self.hiddenLayersDecoder = [embedding_dim, *hidden_layers_decoder, input_dim]
        self.zSampleCount = z_sample_count

        # The network that generates the parameters for the posterior (encoder) q(z|x)
        encoder_layers = OrderedDict()
        for layer_id in range(len(self.hiddenLayersEncoder) - 1):
            encoder_layers["encoder_layer_{0}".format(layer_id)] = torch.nn.Linear(
                in_features=self.hiddenLayersEncoder[layer_id],
                out_features=self.hiddenLayersEncoder[layer_id + 1])
            if layer_id < len(self.hiddenLayersEncoder) - 2:
                encoder_layers["encoder_nonlinearity_{0}".format(layer_id)] = torch.nn.Softplus()
        self.encoder = nn.Sequential(encoder_layers)

        # The network that generates parameters for the distribution p(x|z) (decoder)
        decoder_layers = OrderedDict()
        for layer_id in range(len(self.hiddenLayersDecoder) - 1):
            decoder_layers["decoder_layer_{0}".format(layer_id)] = torch.nn.Linear(
                in_features=self.hiddenLayersDecoder[layer_id],
                out_features=self.hiddenLayersDecoder[layer_id + 1])
            if layer_id < len(self.hiddenLayersEncoder) - 2:
                decoder_layers["decoder_nonlinearity_{0}".format(layer_id)] = torch.nn.Softplus()
        self.decoder = nn.Sequential(decoder_layers)

        self.zGaussian = None

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        rv = multivariate_normal(mu.numpy(), np.power(std.numpy(), 2.0))

        # 2. get the probabilities from the equation
        log_q_z_given_x = q.log_prob(z)
        res = rv.logpdf(z.numpy())
        assert np.allclose(np.sum(log_q_z_given_x.numpy(), axis=1), res)
        log_p_z = p.log_prob(z)

        # kl
        kl = (log_q_z_given_x - log_p_z)

        # sum over last dim to go from single dim distribution to multi-dim
        kl = kl.sum(-1)
        return kl

    # Assumes a diagonal matrix
    def kl_divergence_from_standard_mv_normal(self, mu, std):
        variances = torch.pow(std, 2.0)
        # Trace of diagonal covariance matrix
        trace_sigma = torch.sum(variances)
        dot_prod = torch.dot(mu, mu)
        k = mu.shape[0]
        log_det_sigma = torch.sum(torch.log(variances))
        kl = 0.5 * (trace_sigma + dot_prod - k - log_det_sigma)
        return kl

    # Assumes a diagonal matrix
    def kl_divergence_from_standard_mv_normal_batch(self, mu, std):
        variances = torch.pow(std, 2.0)
        # Trace of diagonal covariance matrix
        trace_sigma = torch.sum(variances, dim=1)
        dot_prod = mu * mu
        dot_prod = torch.sum(dot_prod, dim=1)
        k = mu.shape[1]
        log_det_sigma = torch.sum(torch.log(variances), dim=1)
        kl = trace_sigma + dot_prod - log_det_sigma
        kl = kl - k
        kl = 0.5 * kl
        return kl

    def calculate_loss(self, X):
        # Calculate the parameters for the approximate posterior Q(z|x) for every x in X.
        encoder_params = self.encoder(X)
        mu_q_z_given_x = encoder_params[:, :self.embeddingDim]
        std_q_z_given_x = torch.exp(encoder_params[:, self.embeddingDim:] / 2)

        # Calculate D[Q(z|x)||P(z)]
        # Calculate the KL-Divergence for each X. It can be analytically calculated in close form.
        kl_divergences = self.kl_divergence_from_standard_mv_normal_batch(mu=mu_q_z_given_x,
                                                                          std=std_q_z_given_x)

        # Calculate E_{z ~ Q(z|x)} [log P(x|z)] - The reconstruction loss or the log likelihood.
        Z = self.zGaussian.sample(sample_shape=(X.shape[0], self.zSampleCount))
        Sigma_z = torch.unsqueeze(std_q_z_given_x, dim=1) * Z
        Z_from_q_z_given_x = torch.unsqueeze(mu_q_z_given_x, dim=1) + Sigma_z
        X_mu = self.decoder(Z_from_q_z_given_x)
        p_x_given_z = torch.distributions.Normal(loc=X_mu, scale=1.0)
        log_likelihood = p_x_given_z.log_prob(value=X)
        print("X")

        # kl_divergences_2 = []
        # for idx in range(X.shape[0]):
        #     kl_idx = \
        #         self.kl_divergence_from_standard_mv_normal(mu=mu_q_z_given_x[idx], std=std_q_z_given_x[idx])
        #     kl_divergences_2.append(kl_idx.detach().numpy())
        # kl_divergences_2 = np.array(kl_divergences_2)
        # kl_divergences = kl_divergences.detach().numpy()
        # assert np.allclose(kl_divergences, kl_divergences_2)

        print("X")

    def fit(self, dataset, epoch_count):
        self.zGaussian = torch.distributions.Normal(
            torch.zeros(size=(dataset.dataset.datasetDimensionality, ), dtype=torch.float32),
            torch.ones(size=(dataset.dataset.datasetDimensionality, ), dtype=torch.float32))
        for epoch_id in range(epoch_count):
            for X, y in dataset:
                self.calculate_loss(X=X.to(torch.float32))
