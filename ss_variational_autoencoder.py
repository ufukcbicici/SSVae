from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from scipy.stats import multivariate_normal

from variational_autoencoder import VariationalAutoencoder


class SSVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, m1_model,
                 hidden_layers_q_z_given_x,
                 hidden_layers_q_y_given_x_discriminator,
                 hidden_layers_p_x_given_yz,
                 class_count,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputDim = input_dim
        self.embeddingDim = embedding_dim
        self.m1Model = m1_model

        # Q(z|x)
        self.hiddenLayers_Q_z_given_x = [input_dim, *hidden_layers_q_z_given_x, 2 * embedding_dim]
        self.Q_z_given_x_network = nn.Sequential(
            self.create_mlp(network_name="Q_z_given_x",
                            hidden_layers=self.hiddenLayers_Q_z_given_x))

        # Q(y|x) -> The discriminator
        self.hiddenLayers_Q_y_given_x = [input_dim, *hidden_layers_q_y_given_x_discriminator, class_count]
        self.Q_z_given_y_discriminator_network = nn.Sequential(
            self.create_mlp(network_name="Q_y_given_x",
                            hidden_layers=self.hiddenLayers_Q_y_given_x))

        # The generator P(x|y,z). We will be creating a separate network for every class label y.
        self.classCount = class_count
        self.hiddenLayers_P_x_given_yz = [embedding_dim, *hidden_layers_p_x_given_yz, input_dim]
        self.P_x_given_yz_generator_networks = nn.ModuleList()
        for c in range(self.classCount):
            net = nn.Sequential(
                self.create_mlp(network_name="P_x_given_yz",
                                hidden_layers=self.hiddenLayers_P_x_given_yz))
            self.P_x_given_yz_generator_networks.append(net)

        # The prior p(y)
        self.P_y = torch.distributions.Categorical(torch.tensor([1.0 / self.classCount] * class_count))

        # The prior p(z)
        self.P_z = torch.distributions.Normal(torch.zeros(size=(self.embeddingDim,)),
                                              torch.ones(size=(self.embeddingDim,)))
        self.logScale = nn.Parameter(torch.Tensor([0.0]))

    def create_mlp(self, network_name, hidden_layers):
        layers = OrderedDict()
        for layer_id in range(len(hidden_layers) - 1):
            layers["{0}_layer_{1}".format(network_name, layer_id)] = torch.nn.Linear(
                in_features=hidden_layers[layer_id],
                out_features=hidden_layers[layer_id + 1])
            if layer_id < len(hidden_layers) - 2:
                layers["{0}_nonlinearity_{1}".format(network_name, layer_id)] = torch.nn.Softplus()
        return layers

    def get_labeled_ELBO(self, x, y):
        # Sample z from Q(z|x)
        encoder_params = self.Q_z_given_x_network(x)
        mu_q_z_given_x = encoder_params[:, :self.embeddingDim]
        std_q_z_given_x = torch.exp(encoder_params[:, self.embeddingDim:] / 2)
        q_z_given_x = torch.distributions.Normal(mu_q_z_given_x, std_q_z_given_x)
        z = q_z_given_x.rsample()

        # log P(x|y,z)
        log_probs_arr = []
        for label in range(self.classCount):
            x_hat = self.hiddenLayers_P_x_given_yz[label](z)
            # Measure the likelihood of the observed X under P(x|y,z)
            scale = torch.exp(self.logScale)
            mean = x_hat
            p_x_given_yz = torch.distributions.Normal(mean, scale)
            log_p_x_given_yz = p_x_given_yz.log_prob(x)
            log_p_x_given_yz = torch.sum(log_p_x_given_yz, dim=1)
            log_probs_arr.append(log_p_x_given_yz)
        log_probs_arr = torch.stack(log_probs_arr, dim=1)
        labels_one_hot_arr = torch.nn.functional.one_hot(y, self.classCount)
        final_log_probs = labels_one_hot_arr * log_probs_arr
        final_log_probs = torch.sum(final_log_probs, dim=1)
        mean_likelihood = torch.mean(final_log_probs)

        # log P(y)
        log_p_y = self.P_y.log_prob(value=y)
        log_p_y = torch.mean(log_p_y)

        # log P(z)
        log_p_z = self.P_z.log_prob(value=z)

        # log Q(z|x)
        q_z_given_x.log_prob(value=z)
        print("X")



    def fit(self, labeled_data, unlabeled_data, epoch_count):
        for epoch_id in range(epoch_count):
            for x_l, y in labeled_data:
                # Get unlabeled data
                x_u, _ = unlabeled_data.next()

                # Calculate the ELBO for the labeled data
