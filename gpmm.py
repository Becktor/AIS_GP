import numpy as np
import torch
import matplotlib.pyplot as plt
from dataloader import DataLoader, SLDataset, sl_collate, make_relative_meters
from rotate import rotate_origin_only, rotate_via_numpy
from typing import Optional

import gpytorch

torch.set_default_tensor_type(torch.FloatTensor)

import torch


class WeightedLinearKernel(gpytorch.kernels.PolynomialKernel):
    def __init__(self, power: int, **kwargs):
        super(WeightedLinearKernel, self).__init__(power, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        prod = super(WeightedLinearKernel, self).forward(x1, x2, diag=False, last_dim_is_batch=False, **params)
        m = torch.max(prod)
        k = 1 + (prod / m)
        return k


class LinearMean2(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights1", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        self.register_parameter(name="weights2", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias1", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
            self.register_parameter(name="bias2", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        if len(x) > 10:
            re = 1
        x1 = x[:8]
        x2 = x[8:]
        res1 = x1.matmul(self.weights1).squeeze(-1)
        res2 = x2.matmul(self.weights2).squeeze(-1)

        if self.bias1 is not None:
            res1 = res1 + self.bias1
            res2 = res2 + self.bias2
        res = torch.cat((res1, res2), 0)
        return res


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    """
    Spectral Mixture Kernel :
    """

    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        # nm = train_y + train_y[-1]
        # std = torch.std(nm)
        # self.prior = gpytorch.priors.NormalPrior(nm[-1]*2, std)
        #
        # self.mean_module_x = gpytorch.means.ConstantMean(prior=self.prior)

        self.mean_module_x = gpytorch.means.LinearMean(1)
        self.covar_module = gpytorch.kernels.MaternKernel()#gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10)
        # elf.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module_x(x)
        covar_x = self.covar_module(x)
        retval = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return retval


def gp_fit_mixture_x(x_input, x_tar, y_input, y_tar, title='', training_iter=200, model=None,
                     likelihood=None, lr=0.1, route=None, plot=False):
    x_input = torch.tensor(x_input).float()
    y_input = torch.tensor(y_input).float()
    x_tar = torch.tensor(x_tar).float()

    if (model is None) or (likelihood is None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SpectralMixtureGPModel(x_input, y_input, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    i = 0

    while i < training_iter:
        optimizer.zero_grad()
        output = model(x_input)
        loss = -mll(output, y_input)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()), end='\r')
        optimizer.step()
        i += 1
        if loss > 0:
            training_iter += 2
        if loss > -1 and lr < 1e-1:
            training_iter += 2
        if loss > -1.8 and lr < 1e-2:
            training_iter += 2
        if i > 1000:
            break

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    # See https://arxiv.org/abs/1803.06058
    x_predict = x_tar
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(x_predict))
        if plot:
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            ax.plot(x_tar[-len(y_tar):], y_tar, 'r', label='target')
            ax.plot(x_predict, observed_pred.mean, 'g', label='pred')

            ax.plot(x_input, y_input, 'b', label='input', alpha=0.2)
            # Shade between the lower and upper confidence bounds
            ax.fill_between(x_predict, lower.numpy(), upper.numpy(), alpha=0.5)
            ud = upper.numpy() - np.roll(upper.numpy(), -1)
            md = observed_pred.mean.numpy() - np.roll(observed_pred.mean.numpy(), -1)
            # diff = sum((ud - md)[:-1])
            # print(diff)
            ax.legend()
            plt.title(title)
            plt.show()
    return model, likelihood, observed_pred


def getdata(dataloader, sln):
    seq = next(iter(dataloader))

    """"
    Impl mean as linear regression....
    """

    x = np.array(seq)
    x_n = np.array(x)[:, np.where(np.array(x)[-1, :, 1] != -1)].squeeze().reshape(sln, -1, 4)
    x = np.array(x_n)[:, np.where(np.array(x_n)[0, :, 1] != -1)].squeeze().reshape(sln, -1, 4)
    x_rel, y_rel = make_relative_meters(x)
    x_rel = np.flip(x_rel)  # .ravel('F')
    y_rel = np.flip(y_rel)  # .ravel('F')
    # time = np.tile(time, len(x_rel) // sln)
    return x_rel, y_rel


def run_gp_xy(xy_i, xy_t, route, mx=None, lx=None, my=None, ly=None):
    time_input = np.array(range(len(xy_i)))
    time_tar = np.array(range(len(xy_t))) + len(time_input)

    x_i = np.column_stack((time_input, xy_i[:, 0]))
    x_t = np.column_stack((time_tar, xy_t[:, 0]))

    y_i = np.column_stack((time_input, xy_i[:, 1]))
    y_t = np.column_stack((time_tar, xy_t[:, 1]))

    model_x, llh_x, _ = gp_fit_mixture_x(x_i[:, 0], x_t[:, 0], x_i[:, 1], x_t[:, 1], training_iter=100, lr=1e-1)
    model_x, llh_x, _ = gp_fit_mixture_x(x_i[:, 0], x_t[:, 0], x_i[:, 1], x_t[:, 1], training_iter=100,
                                         model=model_x, likelihood=llh_x, lr=1e-2)
    mx, lx, obs_pred_x = gp_fit_mixture_x(x_i[:, 0], x_t[:, 0], x_i[:, 1], x_t[:, 1], training_iter=100,
                                          title='x', model=model_x, likelihood=llh_x, lr=1e-3, plot=True)

    model_y, llh_y, _ = gp_fit_mixture_x(y_i[:, 0], y_t[:, 0], y_i[:, 1], y_t[:, 1], training_iter=100, lr=1e-1)
    model_y, llh_y, _ = gp_fit_mixture_x(y_i[:, 0], y_t[:, 0], y_i[:, 1], y_t[:, 1], training_iter=100,
                                         model=model_y, likelihood=llh_y, lr=1e-2)
    my, ly, obs_pred_y = gp_fit_mixture_x(y_i[:, 0], y_t[:, 0], y_i[:, 1], y_t[:, 1], training_iter=100,
                                          title='y', model=model_y, likelihood=llh_y, lr=1e-3, plot=True)

    x_p = obs_pred_x.mean.numpy()  # rotate_origin_only(obs_pred_x.mean.numpy(), ang1)
    #
    y_p = obs_pred_y.mean.numpy()  # rotate_origin_only(obs_pred_y.mean.numpy(), ang2)

    return x_p, y_p, mx, lx, my, ly


def main():
    sln = 200
    dataset = SLDataset(csv_file='ruteT_maj.csv', seq_len=sln, resample_freq='20s')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=sl_collate)
    x_rel, y_rel = getdata(dataloader, sln)
    split = 20
    high = split + 20

    txy_input = zip(x_rel.T, y_rel.T)
    for inp in txy_input:
        xy_input = np.array(list(inp)).T
        itr = 0
        mx = None
        for x in range(40):
            xy_i = xy_input[itr:split + itr]
            xy_t = xy_input[split + itr:high + itr]
            if mx is None:
                xx, yy, mx, lx, my, ly = run_gp_xy(xy_i, xy_t, xy_input)
            else:
                xx, yy, mx, lx, my, ly = run_gp_xy(xy_i, xy_t, xy_input, mx, lx, my, ly)
            # # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))
            # Plot training data
            ax.plot(xy_input[:, 0], xy_input[:, 1], 'b', label='route', alpha=0.2)
            ax.plot(xy_i[:, 0], xy_i[:, 1], 'b')
            ax.plot(xy_t[:, 0], xy_t[:, 1], 'r', label='target')
            # # Plot predictive means as blue line
            ax.plot(xx, yy, 'g', label='pred')
            ax.legend()
            plt.show()
            itr += 5


if __name__ == "__main__":
    main()


def evaluate(outs, y, std_multiplier=2):
    preds = torch.stack(outs)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    DE = torch.sqrt((y - means)[..., 0] ** 2 + (y - means)[..., 1] ** 2)
    ADE = DE.float().mean().detach()
    FDE = DE[:, -1].float().mean().detach()
    return ic_acc, ADE, FDE
