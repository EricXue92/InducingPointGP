import torch
import torch.distributions
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", font_scale=1)

class GP_InducingPoints(torch.nn.Module):

    def __init__(self, num_inducing_points, x=None, y=None, dim=1):
        super().__init__()
        assert type(x) != type(None) # some sanity checking
        assert type(y) != type(None), "Input _y should not be None"  # some sanity checking for the correct input
        self.x = x
        self.y = y
        self.num_inducing_points = num_inducing_points
        # initialize inducing points
        inducing_x = torch.linspace(x.min().item(), x.max().item(), self.num_inducing_points).reshape(-1,1) #（6,1）
        self.inducing_x_mu = torch.nn.Parameter( inducing_x + torch.randn_like(inducing_x).clamp(-0.1,0.1) )
        print(f"Inducing points: {self.inducing_x_mu}, shape: {self.inducing_x_mu.shape}")
        self.inducing_y_mu = torch.nn.Parameter( torch.FloatTensor(num_inducing_points, dim).uniform_(-0.5,0.5) )
        print(f"Inducing points y: {self.inducing_y_mu}, shape: {self.inducing_y_mu.shape}")

        self.length_scale = torch.nn.Parameter(torch.scalar_tensor(0.1))
        self.noise = torch.nn.Parameter(torch.scalar_tensor(0.5))
        # jitter for numerical stability
        self.jitter = 1e-6

    def compute_kernel_matrix(self, x1, x2):
        assert x1.shape[1] == x2.shape[1]  # check dimension
        assert x1.numel() >= 0  # sanity check
        assert x2.numel() >= 0  # sanity check
        pdist = (x1 - x2.T) ** 2  # outer difference
        kernel_matrix = torch.exp(-0.5 * 1 / (self.length_scale + 0.001) * pdist)
        return kernel_matrix

    def forward(self, X):
        # compute all the kernel matrices
        self.K_XsX = self.compute_kernel_matrix(X, self.inducing_x_mu)
        self.K_XX = self.compute_kernel_matrix(self.inducing_x_mu, self.inducing_x_mu)
        self.K_XsXs = self.compute_kernel_matrix(X, X)
        # invert K_XX and regularizing it for numerical stability
        self.K_XX_inv = torch.inverse(self.K_XX + self.jitter * torch.eye(self.K_XX.shape[0]))

        # compute mean and covariance for forward prediction
        mu = self.K_XsX @ self.K_XX_inv @ self.inducing_y_mu
        sigma = self.K_XsXs - self.K_XsX @ self.K_XX_inv @ self.K_XsX.T + self.noise * torch.eye(self.K_XsXs.shape[0])

        sigma = sigma + self.jitter * torch.eye(sigma.shape[0])
        sigma = 0.5 * (sigma + sigma.T)
        return mu, torch.diag(sigma)[:, None]

    # Negative Marginal Log Likelihood
    def NMLL(self, X, y):
        self.length_scale.data = self.length_scale.data.clamp(0.00001, 3.0)
        self.noise.data = self.noise.data.clamp(0.000001, 3)

        K_XsXs = self.compute_kernel_matrix(X, X)
        K_XsX = self.compute_kernel_matrix(X, self.inducing_x_mu)
        K_XX = self.compute_kernel_matrix(self.inducing_x_mu, self.inducing_x_mu)
        K_XX_inv = torch.inverse(K_XX + self.jitter * torch.eye(K_XX.shape[0]))

        Q_XX = K_XsXs - K_XsX @ K_XX_inv @ K_XsX.T

        # compute mean and covariance and GP distribution itself
        mu = K_XsX @ K_XX_inv @ self.inducing_y_mu
        Sigma = Q_XX + self.noise ** 2 * torch.eye(Q_XX.shape[0])

        # Add a small positive value to the diagonal for numerical stability
        Sigma = Sigma + self.jitter * torch.eye(Sigma.shape[0])
        # Ensure symmetry
        Sigma = 0.5 * (Sigma + Sigma.T)
        try:
            p_y = MultivariateNormal(mu.squeeze(), covariance_matrix=Sigma)
            mll = p_y.log_prob(y.squeeze())
            mll -= 1 / (2 * self.noise ** 2) * torch.trace(Q_XX)
        except:
            print("Error in NMLL!!")
        return -mll

    def plot(self, title=""):
        x = torch.linspace(self.x.min() * 1.5, self.x.max() * 1.5, 200).reshape(-1, 1)
        with torch.no_grad():
            mu, sigma = self.forward(x)
        x = x.cpu().numpy().squeeze()
        mu = mu.cpu().numpy().squeeze()
        sigma = sigma.cpu().numpy().squeeze()
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.scatter(self.inducing_x_mu.detach().cpu().numpy(), self.inducing_y_mu.detach().cpu().numpy(), label="Inducing points", c='blue')
        plt.scatter(self.x.cpu().numpy(), self.y.cpu().numpy(), alpha=0.2, c='red', label="Data")
        plt.fill_between(x, mu - 3 * sigma, mu + 3 * sigma, alpha=0.1, color='blue', label="Uncertainty (3σ)")
        plt.plot(x, mu, label="Predicted mean", c='green')
        plt.xlim(self.x.min() * 1.5, self.x.max() * 1.5)
        plt.ylim(-3, 3)
        plt.legend()
        #plt.savefig(f"{title}.png", format='png', bbox_inches='tight', pad_inches=0.1, dpi=400)
        plt.savefig(f"{title}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.show()
