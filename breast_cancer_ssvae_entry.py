from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Normal

from ss_variational_autoencoder import SSVariationalAutoencoder
from utils import Utils
from variational_autoencoder import VariationalAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy.stats import norm, multivariate_normal

from vector_dataset import VectorDataset
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":
    np.random.seed(67)
    labeled_percentage = 0.25
    test_percentage = 0.1
    normalize_data = True
    input_dim = 8
    embedding_dim = 4
    hidden_layers_encoder = [128, 64, 32]
    hidden_layers_decoder = [32, 64, 128]

    bc_data = load_breast_cancer()
    # X, y = bc_data.data, bc_data.target
    # Normalize data
    if normalize_data:
        min_max_scaler = MinMaxScaler()
        X_ = min_max_scaler.fit_transform(X=bc_data.data)
    else:
        X_ = bc_data.data
    y_ = bc_data.target

    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=test_percentage)
    X_unlabeled, X_labeled, y_unlabeled, y_labeled = train_test_split(X_train, y_train, test_size=labeled_percentage)

    unlabeled_data = torch.utils.data.DataLoader(VectorDataset(X_=X_unlabeled, y_=y_unlabeled),
                                                 batch_size=X_unlabeled.shape[0], shuffle=True,
                                                 num_workers=2, pin_memory=True)
    labeled_data = torch.utils.data.DataLoader(VectorDataset(X_=X_labeled, y_=y_labeled),
                                               batch_size=X_labeled.shape[0], shuffle=True,
                                               num_workers=2, pin_memory=True)
    test_data = torch.utils.data.DataLoader(VectorDataset(X_=X_test, y_=y_test),
                                            batch_size=X_test.shape[0], shuffle=True,
                                            num_workers=2, pin_memory=True)

    vae = VariationalAutoencoder(input_dim=30,
                                 embedding_dim=8,
                                 hidden_layers_encoder=[512, 256, 128],
                                 hidden_layers_decoder=[128, 256, 512],
                                 z_sample_count=1)
    vae_model_checkpoint_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                             "checkpoints", "vae_230000.pth")
    vae_checkpoint = torch.load(vae_model_checkpoint_path)
    vae.load_state_dict(state_dict=vae_checkpoint["model_state_dict"])

    ssvae = SSVariationalAutoencoder(
        class_count=2,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_layers_q_z_given_x=hidden_layers_encoder,
        hidden_layers_q_y_given_x_discriminator=hidden_layers_encoder,
        hidden_layers_p_x_given_yz=hidden_layers_decoder,
        m1_model=vae)

    ssvae.fit(labeled_data=labeled_data, unlabeled_data=unlabeled_data, epoch_count=100000)



    #
    # data_loader = torch.utils.data.DataLoader(VectorDataset(X_=X_, y_=y_),
    #                                           batch_size=X_.shape[0], shuffle=True,
    #                                           num_workers=2, pin_memory=True)

    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X=X_)
    # # Visualize the 2D data
    # plt.plot(X_2d[:, 0], X_2d[:, 1], 'x')
    # plt.axis('equal')
    # plt.show()
    #
    # vae = VariationalAutoencoder(input_dim=input_dim,
    #                              embedding_dim=embedding_dim,
    #                              hidden_layers_encoder=hidden_layers_encoder,
    #                              hidden_layers_decoder=hidden_layers_decoder,
    #                              z_sample_count=1)
    # # vae.fit(dataset=data_loader, epoch_count=100000, weight_decay=0.00005)
    #
    # for i in range(1, 33):
    #     vae_model_checkpoint_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
    #                                              "checkpoints", "vae_{0}.pth".format(i * 5000))
    #     vae_checkpoint = torch.load(vae_model_checkpoint_path)
    #     vae.load_state_dict(state_dict=vae_checkpoint["model_state_dict"])
    #
    #     X_hat = vae.sample_x(sample_count=X_.shape[0]).detach().numpy()
    #     X_hat = pca.transform(X_hat)
    #     plt.plot(X_hat[:, 0], X_hat[:, 1], 'x')
    #     plt.axis('equal')
    #     plt.title("Checkpoint {0}".format(i * 5000))
    #     plt.show()
    #     print("X")
    #
    # # ss_vae = SSVariationalAutoencoder(
    # #     input_dim=X_.shape[1],
    # #     embedding_dim=8,
    # #     hidden_layers_q_z_given_x=[128, 64, 32],
    # #     hidden_layers_q_y_given_x_discriminator=[128, 64, 32],
    # #     hidden_layers_p_x_given_yz=[32, 64, 128],
    # #     class_count=2,
    # #     m1_model=None
    # # )
    # print("X")
