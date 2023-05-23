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
        self.Q_y_given_x_discriminator_network = nn.Sequential(
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
        self.logScale = nn.Parameter(torch.Tensor([0.0] * self.classCount))
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()

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
            x_hat = self.P_x_given_yz_generator_networks[label](z)
            # Measure the likelihood of the observed X under P(x|y,z)
            scale = torch.exp(self.logScale[label])
            mean = x_hat
            p_x_given_yz = torch.distributions.Normal(mean, scale)
            log_probs = p_x_given_yz.log_prob(x)
            log_probs = torch.sum(log_probs, dim=1)
            log_probs_arr.append(log_probs)
        log_probs_arr = torch.stack(log_probs_arr, dim=1)
        labels_one_hot_arr = torch.nn.functional.one_hot(y, self.classCount)
        log_p_x_given_yz = labels_one_hot_arr * log_probs_arr
        log_p_x_given_yz = torch.sum(log_p_x_given_yz, dim=1)
        # Assert that the selection logic worked correctly
        assert all([np.array_equal(log_p_x_given_yz.detach().numpy()[idx],
                                   log_probs_arr.detach().numpy()[idx, y.numpy()[idx]])
                    for idx in range(y.shape[0])])
        # log_p_x_given_yz = torch.mean(log_p_x_given_yz)

        # log P(y)
        log_p_y = self.P_y.log_prob(value=y)
        # log_p_y = torch.mean(log_p_y)

        # log P(z)
        log_p_z = self.P_z.log_prob(value=z)
        log_p_z = torch.sum(log_p_z, dim=1)
        # log_p_z = torch.mean(log_p_z)

        # log Q(z|x)
        log_q_z_given_x = q_z_given_x.log_prob(value=z)
        log_q_z_given_x = torch.sum(log_q_z_given_x, dim=1)
        # log_q_z_given_x = torch.mean(log_q_z_given_x)

        labeled_elbo = log_p_x_given_yz + log_p_y + log_p_z - log_q_z_given_x
        return labeled_elbo

    def get_unlabeled_elbo(self, x):
        q_y_given_x_activations = self.Q_y_given_x_discriminator_network(x)
        q_y_given_x_probs = torch.softmax(q_y_given_x_activations, dim=1)
        q_y_given_x_log_probs = torch.log_softmax(q_y_given_x_activations, dim=1)
        label_results = []
        for y in range(self.classCount):
            q_y_given_x_probs_label = q_y_given_x_probs[:, y]
            y_label = torch.tensor([y] * x.shape[0])
            labeled_elbo_y = self.get_labeled_ELBO(x=x, y=y_label)
            q_y_given_x_log_probs_label = q_y_given_x_log_probs[:, y]
            label_result = q_y_given_x_probs_label * labeled_elbo_y
            label_result -= q_y_given_x_probs_label * q_y_given_x_log_probs_label
            label_results.append(label_result)
        label_results = torch.stack(label_results, dim=1)
        unlabeled_elbo = torch.sum(label_results, dim=1)
        return unlabeled_elbo

    def fit(self, labeled_data, unlabeled_data, test_data, epoch_count, weight_decay, validation_period, alpha_coeff):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=weight_decay)

        for epoch_id in range(epoch_count):
            self.train()
            unlabeled_data_loader = iter(unlabeled_data)
            for x_l, y in labeled_data:
                # Get unlabeled data
                try:
                    x_u, _ = next(unlabeled_data_loader)
                except StopIteration:
                    unlabeled_data_loader = iter(unlabeled_data)
                    x_u, _ = next(unlabeled_data_loader)

                optimizer.zero_grad()
                with torch.no_grad():
                    x_l_embedding = self.m1Model.encode_x(x_l.to(torch.float32))
                    x_u_embedding = self.m1Model.encode_x(x_u.to(torch.float32))

                with torch.set_grad_enabled(True):
                    # Calculate the ELBO for the labeled data
                    labeled_elbo = self.get_labeled_ELBO(x=x_l_embedding, y=y.to(torch.int64))
                    labeled_elbo = -1.0 * torch.mean(labeled_elbo)
                    # Calculate the ELBO for the unlabeled data
                    unlabeled_elbo = self.get_unlabeled_elbo(x=x_u_embedding)
                    unlabeled_elbo = -1.0 * torch.mean(unlabeled_elbo)
                    # Calculate the cross entropy loss
                    q_y_given_x_activations = self.Q_y_given_x_discriminator_network(x_l_embedding)
                    classification_loss = self.crossEntropyLoss(q_y_given_x_activations, y.to(torch.int64))
                    total_loss = labeled_elbo + unlabeled_elbo + alpha_coeff * classification_loss
                    total_loss.backward()
                    optimizer.step()

                if epoch_id % validation_period == 0:
                    print("Epoch:{0}".format(epoch_id))
                    print("Classification loss:{0}".format(classification_loss))
                    print("Labeled ELBO:{0}".format(labeled_elbo))
                    print("Unlabeled ELBO loss:{0}".format(unlabeled_elbo))
                    training_accuracy = self.validate(labeled_data=labeled_data)
                    print("Epoch {0} training accuracy:{1}".format(epoch_id, training_accuracy))
                    test_accuracy = self.validate(labeled_data=test_data)
                    print("Epoch {0} test accuracy:{1}".format(epoch_id, test_accuracy))

    def validate(self, labeled_data):
        self.eval()
        predictions = []
        ground_truths = []
        for x, y in labeled_data:
            x_l_embedding = self.m1Model.encode_x(x.to(torch.float32))
            q_y_given_x_activations = self.Q_y_given_x_discriminator_network(x_l_embedding).detach().numpy()
            predicted = np.argmax(q_y_given_x_activations, axis=1)
            ground_truths.append(y.numpy())
            predictions.append(predicted)
        ground_truths = np.concatenate(ground_truths)
        predictions = np.concatenate(predictions)
        accuracy = np.mean(ground_truths == predictions)
        print("Accuracy:{0}".format(accuracy))
        return accuracy

    def fit_only_labeled(self, labeled_data, test_data, epoch_count, weight_decay, validation_period):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=weight_decay)
        for epoch_id in range(epoch_count):
            self.train()
            print("Epoch:{0}".format(epoch_id))
            for x, y in labeled_data:
                optimizer.zero_grad()
                with torch.no_grad():
                    x_l_embedding = self.m1Model.encode_x(x.to(torch.float32))
                with torch.set_grad_enabled(True):
                    q_y_given_x_activations = self.Q_y_given_x_discriminator_network(x_l_embedding)
                    classification_loss = self.crossEntropyLoss(q_y_given_x_activations, y.to(torch.int64))
                    print("Classification loss:{0}".format(classification_loss))
                    classification_loss.backward()
                    optimizer.step()
            if epoch_id % validation_period == 0:
                training_accuracy = self.validate(labeled_data=labeled_data)
                print("Epoch {0} training accuracy:{1}".format(epoch_id, training_accuracy))
                test_accuracy = self.validate(labeled_data=test_data)
                print("Epoch {0} test accuracy:{1}".format(epoch_id, test_accuracy))
