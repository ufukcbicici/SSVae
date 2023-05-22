from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from scipy.stats import multivariate_normal


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim,
                 hidden_layers_encoder, hidden_layers_decoder, z_sample_count):
        super().__init__()
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
        self.logScale = nn.Parameter(torch.Tensor([0.0]))

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

    def calculate_loss_v2(self, X):
        # Calculate the parameters for the approximate posterior Q(z|x) for every x in X.
        encoder_params = self.encoder(X)
        mu_q_z_given_x = encoder_params[:, :self.embeddingDim]
        std_q_z_given_x = torch.exp(encoder_params[:, self.embeddingDim:] / 2)

        # Calculate D[Q(z|x)||P(z)]
        # Calculate the KL-Divergence for each X. It can be analytically calculated in close form.
        kl_divergences = self.kl_divergence_from_standard_mv_normal_batch(mu=mu_q_z_given_x,
                                                                          std=std_q_z_given_x)

        # Calculate E_{z ~ Q(z|x)} [log P(x|z)] - The reconstruction loss or the log likelihood.
        q_z_given_x = torch.distributions.Normal(mu_q_z_given_x, std_q_z_given_x)
        z = q_z_given_x.rsample()
        # Calculate the parameters of the likelihood function P(x|z)
        x_hat = self.decoder(z)
        # Measure the likelihood of the observed X under P(x|z)
        scale = torch.exp(self.logScale)
        mean = x_hat
        p_x_given_z = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_p_x_given_z = p_x_given_z.log_prob(X)
        # Check if log_p_x_given_z is calculated correctly
        # for idx in range(X.shape[0]):
        #     x_s = X[idx].detach().numpy()
        #     rv = multivariate_normal(mean[idx].detach().numpy(), scale.detach().numpy() * np.ones_like(x_s))
        #     log_likelihood_2 = rv.logpdf(x_s)
        #     assert np.allclose(np.sum(log_p_x_given_z[idx].detach().numpy()), log_likelihood_2)
        log_p_x_given_z = torch.sum(log_p_x_given_z, dim=1)

        elbo = (kl_divergences - log_p_x_given_z)
        loss = elbo.mean()
        return loss

    def fit(self, dataset, epoch_count, weight_decay, checkpoint_period):
        self.zGaussian = torch.distributions.Normal(
            torch.zeros(size=(dataset.dataset.datasetDimensionality, ), dtype=torch.float32),
            torch.ones(size=(dataset.dataset.datasetDimensionality, ), dtype=torch.float32))
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=weight_decay)

        for epoch_id in range(epoch_count):
            for i, (X, y) in enumerate(dataset):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    loss = self.calculate_loss_v2(X=X.to(torch.float32))
                    loss.backward()
                    optimizer.step()
                    print("Epoch:{0} Iteration:{1} Loss:{2}".format(epoch_id, i, loss))
            if epoch_id % checkpoint_period == 0:
                self.save_model(epoch=epoch_id)

        self.save_model(epoch=epoch_count + 1)

    def save_model(self, epoch):
        checkpoint_file_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                            "{0}_{1}.pth".format("vae", epoch))
        torch.save({
            "model_state_dict": self.state_dict()
        }, checkpoint_file_path)

    def sample_x(self, sample_count):
        p = torch.distributions.Normal(torch.zeros(size=(self.embeddingDim, )),
                                       torch.ones(size=(self.embeddingDim, )))
        Z = p.sample(sample_shape=torch.Size((sample_count, )))
        X = self.decoder(Z)
        return X

    # def load_model(self, model_path):
    #     pass

    def encode_x(self, x):
        encoder_params = self.encoder(x)
        mu_q_z_given_x = encoder_params[:, :self.embeddingDim]
        return mu_q_z_given_x
