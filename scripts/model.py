# Imports as always...
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.nn import SO3Activation

from icoCNN.icoCNN import ConvIco

# Spherical CNN according to https://github.com/jonkhler/s2cnn
# Also see https://github.com/e3nn/e3nn/tree/main/examples/s2cnn


def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3) -> torch.Tensor:
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))


def so3_near_identity_grid(
    max_beta: float = math.pi / 8, max_gamma: float = 2 * math.pi, n_alpha: int = 8, n_beta: int = 3, n_gamma=None
) -> torch.Tensor:
    if n_gamma is None:
        n_gamma = n_alpha  # similar to regular representations
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


def s2_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return torch.cat([(2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)


class S2Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class S2CNN(nn.Module):
    def __init__(self, f_in, b_in, f_out):
        super(S2CNN, self).__init__()

        # Kernel grids.
        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.from_s2 = o3.FromS2Grid((b_in, b_in), lmax=10)

        # Convolutions.
        self.conv1 = S2Convolution(f_in, 16, lmax=10, kernel_grid=grid_s2)
        self.conv2 = SO3Convolution(16, 32, lmax=5, kernel_grid=grid_so3)

        # Activations.
        self.act1 = SO3Activation(lmax_in=10, lmax_out=5, act=torch.relu, resolution=10)
        self.act2 = SO3Activation(lmax_in=5, lmax_out=0, act=torch.relu, resolution=5)

        # Fully connected layer weights.
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, f_out)

    def forward(self, x):
        # Transpose. [batch, features, alpha, beta] -> [batch features, beta, alpha].
        x = x.transpose(-1, -2)

        # From S2. [batch, features, beta, alpha] -> [batch, features, irreps].
        x = self.from_s2(x)

        # Convolutions.
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        # Fully connected classification output.
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class IcoCNN(nn.Module):

    def __init__(self, r, in_channels, out_channels, R_in=1, bias=True, smooth_vertices=False):
        super(IcoCNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.R_in = R_in
        self.bias = bias

        # Convolution layers.
        self.conv1 = ConvIco(r, Cin=in_channels, Cout=16, Rin=R_in, Rout=6, bias=bias, smooth_vertices=smooth_vertices)
        self.conv2 = ConvIco(r, Cin=16, Cout=32, Rin=6, Rout=1, bias=bias, smooth_vertices=smooth_vertices)

        # Fully connected linear layer.
        self.fc1 = nn.Linear(32 * 5 * 2**r * 2**(r+1), 64, bias=bias)
        self.fc2 = nn.Linear(64, out_channels, bias=bias)

    def forward(self, x):
        # Shape going in: [B, in_channels, R_in, 5, 2^r, 2^(r+1)].

        # --- Convolutions ---

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # --- Flatten ---

        x = x.view(x.size(0), -1)

        # --- Fully connected ---

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        # Shape going out: [B, out_channels].

        return x