# https://ludwigwinkler.github.io/blog/InducingPoints/
import torch
import torch.distributions
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import scale
import argparse
from IPGP import GP_InducingPoints
from data_util import generate_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-logging', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_samples', type=int, default=200)
    parser.add_argument('-num_inducing_points', type=int, default=6)
    parser.add_argument('-x_noise_std', type=float, default=0.01)
    parser.add_argument('-y_noise_std', type=float, default=0.1)
    parser.add_argument('-zoom', type=int, default=10)
    parser.add_argument('-lr_kernel', type=float, default=0.01)
    parser.add_argument('-lr_ip', type=float, default=0.1)
    parser.add_argument('-num_epochs', type=int, default=100)
    return parser.parse_args()

def main():
    params = get_params()
    X, y = generate_data(params=params)
    X, y = torch.FloatTensor(scale(X)), torch.FloatTensor(scale(y))

    GP_model = GP_InducingPoints(num_inducing_points=params.num_inducing_points, x=X, y=y, dim=1)
    #gp.plot()
    GP_model.plot(title="Initial inducing points")

    optim = torch.optim.Adam([{"params": [GP_model.length_scale, GP_model.noise], "lr": params.lr_kernel},
                              {"params": [GP_model.inducing_x_mu, GP_model.inducing_y_mu, ], "lr": params.lr_ip}])

    train_loader = DataLoader(TensorDataset(X, y), batch_size=params.batch_size, shuffle=True, num_workers=4)

    def train_step(model, data_loader, optim):
        for batch_index, (X, y) in enumerate(data_loader):
            # X, y = X.to(device), y.to(device)
            optim.zero_grad()
            mll = model.NMLL(X, y)
            mll.backward()
            optim.step()
        return mll

    for epoch in range(params.num_epochs):
        mll = train_step(GP_model, train_loader, optim)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch} \t NMLL:{mll:.2f} \t Length Scale: {float(GP_model.length_scale):.2f} \t Noise: {float(GP_model.noise):.2f}')
            # GP_model.plot(title=f"Training Epoch {epoch:.0f}")

    #gp.plot()
    GP_model.plot(title="Inducing points after training")

if __name__ == "__main__":
    main()