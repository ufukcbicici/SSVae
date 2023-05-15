from variational_autoencoder import VariationalAutoencoder

input_dim = 30
embedding_dim = 8
hidden_layers_encoder = [64, 32]
hidden_layers_decoder = [32, 64]

if __name__ == "__main__":
    vae = VariationalAutoencoder(input_dim=input_dim,
                                 embedding_dim=embedding_dim,
                                 hidden_layers_encoder=hidden_layers_encoder,
                                 hidden_layers_decoder=hidden_layers_decoder)
