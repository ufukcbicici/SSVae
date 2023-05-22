from torch.distributions import Normal

from utils import Utils
from variational_autoencoder import VariationalAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy.stats import norm, multivariate_normal

from vector_dataset import VectorDataset

input_dim = 2
embedding_dim = 1
hidden_layers_encoder = [1024, 512, 256]
hidden_layers_decoder = [256, 512, 1024]

if __name__ == "__main__":
    vae = VariationalAutoencoder(input_dim=input_dim,
                                 embedding_dim=embedding_dim,
                                 hidden_layers_encoder=hidden_layers_encoder,
                                 hidden_layers_decoder=hidden_layers_decoder,
                                 z_sample_count=10)

    samples = Utils.produce_2d_circular_gaussian_data(sample_count=100000)
    plt.plot(samples[:, 0], samples[:, 1], 'x')
    plt.axis('equal')
    plt.show()

    data_loader = torch.utils.data.DataLoader(VectorDataset(X_=samples,
                                                            y_=np.zeros(shape=(samples.shape[0],))),
                                              batch_size=1024, shuffle=True,
                                              num_workers=2, pin_memory=True)
    # vae.fit(dataset=data_loader, epoch_count=10000, weight_decay=0.00005)

    # for i in range(20):
    #     vae_model_checkpoint_path = os.path.join(os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."),
    #                                              "checkpoints", "vae_{0}.pth".format(i * 10))
    #     vae_checkpoint = torch.load(vae_model_checkpoint_path)
    #     vae.load_state_dict(state_dict=vae_checkpoint["model_state_dict"])
    #
    #     X_hat = vae.sample_x(sample_count=10000).detach().numpy()
    #     plt.plot(X_hat[:, 0], X_hat[:, 1], 'x')
    #     plt.axis('equal')
    #     plt.title("Checkpoint {0}".format(i * 10))
    #     plt.show()
    #     print("X")



    # mu = torch.tensor(np.array([1.0, 2.0]))
    # std = torch.tensor(np.array([1.5, 2.5]))
    #
    # m = Normal(mu, std)
    # z = m.sample(sample_shape=torch.Size((1000000, )))
    #
    # kl_div_samples = vae.kl_divergence(mu=mu, std=std, z=z)
    # kl_div_approx = np.mean(kl_div_samples.numpy())
    # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    # q = torch.distributions.Normal(mu, std)
    #
    # with torch.set_grad_enabled(True):
    #     kl_div_2 = torch.distributions.kl_divergence(q=p, p=q)
    #     kl_div_total = torch.sum(kl_div_2)
    #     kl_div_total2 = vae.kl_divergence_from_standard_mv_normal(mu=mu,
    #                                                               std=std)
    #     kl_div_total.backward()

    print("X")
