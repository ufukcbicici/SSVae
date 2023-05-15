from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_layers_encoder, hidden_layers_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputDim = input_dim
        self.embeddingDim = embedding_dim
        self.hiddenLayersEncoder = [input_dim, *hidden_layers_encoder, 2 * embedding_dim]
        self.hiddenLayersDecoder = [embedding_dim, *hidden_layers_decoder, input_dim]

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

    def calculate_loss(self, X):
        print("X")

    def fit(self, dataset, batch_size):
        print("X")



