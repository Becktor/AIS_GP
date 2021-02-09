import sys
import numpy as np
import numpy as np
import torch
from torch import nn, distributions
from scipy.spatial.distance import cdist
from scipy.stats import norm
import matplotlib.pyplot as plt
# from traceback_with_variables import activate_by_import
from dataloader import DataLoader, SLDataset, sl_collate, make_relative_meters
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels

torch.set_default_tensor_type(torch.FloatTensor)


# Kernels

def squared_exponential_kernel(x, y, length_scale, alpha, variance):
    '''
    Function that computes the covariance matrix using a squared-exponential kernel
    '''
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    # pair-wise distances, size: NxM
    sqdist = cdist(x, y, 'sqeuclidean')
    dist = (alpha ** 2) * x @ y.T
    # compute the kernel
    cov_matrix = (dist + variance * np.exp(-0.5 * sqdist * (1 / length_scale)))  # NxM
    return cov_matrix


def abs_kernel(x, y, lengthscale, alpha, variance):
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    absdist = cdist(x, y, 'cityblock')
    k = variance * np.exp(-0.5 * absdist * (1 / lengthscale))  # NxM
    return k


def rational_quadratic_kernel(x, y, lengthscale, alpha, variance):
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    sqdist = cdist(x, y, 'sqeuclidean')
    mid = 1 + (1 / (2 * alpha * lengthscale ** 2)) * sqdist
    k = np.power(variance, 2) * np.power(mid, -alpha)  # NxM
    return k


def matern52_kernel(x, y, lengthscale, alpha, variance):
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    sqdist = cdist(x, y, 'sqeuclidean')
    mid = 1 + np.sqrt(5 * sqdist) / lengthscale + (5 * sqdist) / (3 * lengthscale ** 2)
    k = np.power(variance, 2) * mid * np.exp(-(np.sqrt(5 * sqdist) / lengthscale))  # NxM
    return k


def squared_exponential_kernel_torch(x, y, _lambda, _alpha_param, variance):
    x = x.squeeze(1).expand(x.size(0), y.size(0))
    y = y.squeeze(0).expand(x.size(0), y.size(0))
    sqdist = torch.pow(x - y, 2)
    dist = torch.pow(_alpha_param, 2) * torch.matmul(x, y.T)
    k = (dist + variance * torch.exp(-0.5 * sqdist * (1 / _lambda)))  # NxM
    return k


def abs_kernel_torch(x, y, _lambda, _alpha_param, variance):
    x = x.squeeze(1).expand(x.size(0), y.size(0))
    y = y.squeeze(0).expand(x.size(0), y.size(0))
    absdist = torch.abs(x - y)
    k = variance * torch.exp(-0.5 * absdist * (1 / _lambda))  # NxM
    return k


def rational_quadratic_kernel_torch(x, y, _lambda, _alpha_param, variance):
    x = x.squeeze(1).expand(x.size(0), y.size(0))
    y = y.squeeze(0).expand(x.size(0), y.size(0))
    absdist = torch.pow(x - y, 2)
    mid = 1 + (1 / (2 * _alpha_param * _lambda ** 2)) * absdist
    k = torch.pow(variance, 2) * torch.pow(mid, -_alpha_param)  # NxM
    return k


def matern52_kernel_torch(x, y, _lambda, _alpha_param, variance):
    x = x.squeeze(1).expand(x.size(0), y.size(0))
    y = y.squeeze(0).expand(x.size(0), y.size(0))
    sqdist = torch.pow(x - y, 2)
    mid = 1 + torch.sqrt(5 * sqdist) / _lambda + (5 * sqdist) / (3 * _lambda ** 2)
    k = torch.pow(variance, 2) * mid * torch.exp(-(torch.sqrt(5 * sqdist) / _lambda))  # NxM
    return k


def fit_predictive_gp(x, y, sampling, params, kernel=rational_quadratic_kernel):
    '''
    Function that fit the Gaussian Process. It returns the predictive mean function and
    the predictive covariance function. It follows step by step the algorithm on the lecture
    notes
    '''
    kernel = kernel
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    K = kernel(x, x, params['ls'], params['a'], params['ov'])
    L = np.linalg.cholesky(K + params['a'] + params['nv'] * np.eye(len(x)))

    # compute the mean at our test points.
    Ks = kernel(x, sampling, params['ls'], params['a'], params['ov'])
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  #
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    # compute the variance at our test points.
    Kss = kernel(sampling, sampling, params['ls'], params['a'], params['ov'])
    covariance = Kss - (v.T @ v)
    return mu, covariance


# Here PyTorch is used to define the optimization function, with an ADAM optimizer
# This is likely not a very efficient nor pretty way to do this.


def optimize_gp_hyperparams(x, y, optimization_steps, learning_rate,
                            kernel=rational_quadratic_kernel_torch, params=None):
    """
    Methods that run the optimization of the hyperparams of our GP. We will use
    Gradient Descent because it takes to much time to run grid search at each step
    of bayesian optimization. We use a different definition of the kernel to make the
    optimization more stable

    :param params:
    :param x: training set points
    :param y: training targets
    :param optimization_steps:
    :param learning_rate:
    :param kernel:
    :return: values for length-scale, output_var, noise_var that maximize the log-likelihood
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    N = len(x)

    # tranform our training set in Tensor
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    # we should define our hyperparameters as torch parameters where we keep track of
    # the operations to get hte gradients from them
    m = np.abs(y[-1][0])
    mu = torch.tensor(m).float()
    s = np.var(y)
    sig = torch.tensor(s).float()
    if params:
        lambda_param = nn.Parameter(torch.tensor(params['ls']), requires_grad=True)
        alpha_param = nn.Parameter(torch.tensor(params['a']), requires_grad=True)
        output_variance = nn.Parameter(torch.tensor(params['ov']), requires_grad=True)
        noise_variance = nn.Parameter(torch.tensor(params['nv']), requires_grad=True)
        mu_param = nn.Parameter(torch.tensor(params['mu']), requires_grad=True)
        sig_param = nn.Parameter(torch.tensor(params['sig']), requires_grad=True)

    else:
        lambda_param = nn.Parameter(torch.tensor(1.), requires_grad=True)
        alpha_param = nn.Parameter(torch.tensor(1.), requires_grad=True)
        output_variance = nn.Parameter(torch.tensor(1.), requires_grad=True)
        noise_variance = nn.Parameter(torch.tensor(2.5), requires_grad=True)
        mu_param = nn.Parameter(mu, requires_grad=True)
        sig_param = nn.Parameter(sig, requires_grad=True)

    # we use Adam as optimizer
    optim = torch.optim.Adam([lambda_param, alpha_param, output_variance,
                              noise_variance, mu_param, sig_param], lr=learning_rate)

    # optimization loop using the log-likelihood that involves the cholesky decomposition
    nlls = []
    lambdas = []
    output_variances = []
    noise_variances = []
    iterations = optimization_steps
    for i in range(iterations):
        if lambda_param.item() == lambda_param.item():
            return_dict = {'ls': lambda_param.item(),
                           'a': alpha_param.item(),
                           'ov': output_variance.item(),
                           'nv': noise_variance.item(),
                           'mu': mu_param.item(),
                           'sig': sig_param.item()}
        assert noise_variance >= 0, f"ouch! {i, noise_variance}"
        optim.zero_grad()
        K = kernel(x_tensor, x_tensor, lambda_param, alpha_param,
                   output_variance) + noise_variance * torch.eye(N)
        try:
            cholesky = torch.cholesky(K)
            _alpha_temp, _ = torch.solve(y_tensor, cholesky)
            _alpha, _ = torch.solve(_alpha_temp, cholesky.t())
        except:
            return return_dict
        nll = N / 2 * torch.log(torch.tensor(2 * np.pi)) + 0.5 * torch.matmul(y_tensor.transpose(0, 1), _alpha) + \
              torch.sum(torch.log(torch.diag(cholesky)))

        # we have to add the log-likelihood of the prior
        norm = distributions.Normal(loc=m, scale=s)
        prior_negloglike = torch.log(lambda_param) - torch.log(torch.exp(norm.log_prob(lambda_param)))

        nll += 0.9 * prior_negloglike
        nll.backward()

        nlls.append(nll.item())
        lambdas.append(lambda_param.item())
        output_variances.append(output_variance.item())
        noise_variances.append(noise_variance.item())
        optim.step()

        # projected in the constraints (lengthscale and output variance should be positive)
        for p in [lambda_param, output_variance]:
            p.data.clamp_(min=0.0000001)
        noise_variance.data.clamp_(min=0.001, max=0.05)
        alpha_param.data.clamp_(min=0.001, max=0.1)
        # mu_param.data.clamp_(min=0.01, max=40)
        # sig_param.data.clamp_(min=0.01, max=40)

    return_dict = {'ls': lambda_param.item(),
                   'a': alpha_param.item(),
                   'ov': output_variance.item(),
                   'nv': noise_variance.item(),
                   'mu': mu_param.item(),
                   'sig': sig_param.item()}

    return return_dict


def gaussian_process(x_input, y_input, x_tar, y_tar, sampling,
                     draw=False, title='', kernel=rational_quadratic_kernel_torch):
    # print dataset information
    # also in this case we standardize the data to have zero mean and unit variance

    # we shall also define the test set, that is the range of XTest points we want to
    # use to compute the mean and the variance
    params = optimize_gp_hyperparams(x_input, y_input, 1000, 1e-2, kernel=T_KERNEL)
    params = optimize_gp_hyperparams(x_input, y_input, 1000, 1e-3, kernel=T_KERNEL, params=params)
    print(
        'Optimized parameters: ls: {}, al: {}, ov: {}, nv: {}, mu: {}, sig: {}'.format(params['ls'], params['a'],
                                                                                       params['ov'], params['nv'],
                                                                                       params['mu'], params['sig']))

    # we can fit the GP that use the hyperparameters found above

    # x_input = np.append(x_input, x_input[-1] * 2)
    # y_input= np.append(y_input, y_input[-1] * 2)
    mu, covariance = fit_predictive_gp(x_input, y_input, sampling, params, kernel=KERNEL)
    std = np.sqrt(np.diag(covariance))
    if draw:
        plt.plot(x_tar, y_tar, 'bo', label='Target points', alpha=0.4)
        plt.plot(x_input, y_input, 'ro', label='Training points', alpha=0.4)
        plt.gca().fill_between(sampling.flat, mu.reshape(-1) - 2 * std, mu.reshape(-1) + 2 * std, color='lightblue',
                               alpha=0.5, label=r"$2\sigma$")
        plt.plot(sampling, mu, 'green', label=r"$\mu$", alpha=0.4)
        plt.title(title)
        plt.legend()
        plt.show()
    return mu, std, params


## 0.029 secs to run on jb pc
def ap_f_gp(x_input, y_input, x_tar, y_tar, x_test, params, title='', plot=True):
    params = optimize_gp_hyperparams(x_input, y_input, 50, 1e-1, params=params, kernel=T_KERNEL)
    params = optimize_gp_hyperparams(x_input, y_input, 25, 1e-2, params=params, kernel=T_KERNEL)
    params = optimize_gp_hyperparams(x_input, y_input, 10, 1e-3, params=params, kernel=T_KERNEL)

    mu, covariance = fit_predictive_gp(x_input, y_input, x_test, params, kernel=KERNEL)
    std = np.sqrt(np.diag(covariance))
    if plot:
        plt.plot(x_input, y_input, 'ro', label='Training points', alpha=0.4)
        x = np.concatenate([x_input, x_tar])
        y = np.concatenate([y_input, y_tar])
        plt.plot(x, y, 'bo', label='Target', alpha=0.4)
        plt.gca().fill_between(x_test.flat, mu.reshape(-1) - 2 * std, mu.reshape(-1) + 2 * std, color='lightblue',
                               alpha=0.5, label=r"$2\sigma$")
        plt.plot(x_test, mu, 'green', label=r"$\mu$")
        plt.title(title)
        plt.legend()
        plt.show()
    return mu, std, params


KERNEL = squared_exponential_kernel
T_KERNEL = squared_exponential_kernel_torch


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
    time = np.arange(sln)
    time = time
    # time = np.tile(time, len(x_rel) // sln)
    return x_rel, y_rel, time


def test_params(x, y, params):
    # we can fit the GP that use the hyperparameters found above
    samples = 20
    start = 20
    end = 40
    sample_points = np.linspace(x[0], x[end], samples).reshape(-1, 1)
    mu, covariance = fit_predictive_gp(x[:20], y[:20], sample_points, params, kernel=KERNEL)
    std = np.sqrt(np.diag(covariance))
    plt.plot(x[:start], y[:start], 'bo', label='inp', alpha=0.4)
    plt.plot(x[start:end], y[start:end], 'ro', label='test', alpha=0.4)
    # plt.plot(x_input, y_input, 'ro', label='Training points', alpha=0.4)
    plt.gca().fill_between(sample_points.flat, mu.reshape(-1) - 2 * std, mu.reshape(-1) + 2 * std, color='lightblue',
                           alpha=0.5, label=r"$2\sigma$")
    plt.plot(sample_points, mu, 'green', label=r"$\mu$", alpha=0.4)
    plt.title('title')
    plt.legend()
    plt.show()


def main2():
    sln = 200
    dataset = SLDataset(csv_file='ruteT_maj.csv', seq_len=sln, resample_freq='20s')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=sl_collate)
    data = getdata(dataloader, sln)
    """"
    Impl mean as linear regression....
    """
    params = {'ls': 0.26, 'a': 0.10, 'ov': 0.001, 'nv': 0.0001, 'mu': 0.047, 'sig': 0.010}
    for d in data[0].T:
        test_params(data[2][100:], d[100:], params)
    return 0
    lr = 1e-3
    for d in data[0].T:
        params = optimize_gp_hyperparams(data[2][:120], d[:120], 1000, lr, kernel=T_KERNEL)

        print(
            'Optimizing parameters: \'ls\': {:0.3f}, \'a\': {:0.3f}, \'ov\': {:0.3f}, \'nv\': {:0.3f}, '
            '\'mu\': {:0.3f}, \'sig\': {:0.3f}'.format(params['ls'], params['a'],
                                                       params['ov'], params['nv'],
                                                       params['mu'], params['sig']))

    for x in range(50):
        try:
            data = getdata(dataloader, sln)
        except:
            continue
        if x == 25:
            lr = 1e-3
        if x == 40:
            lr = 1e-3
        for d in data[0].T:
            params = optimize_gp_hyperparams(data[2], d, 10, lr, kernel=T_KERNEL, params=params)
        for d in data[1].T:
            params = optimize_gp_hyperparams(data[2], d, 10, lr, kernel=T_KERNEL, params=params)
        # params = optimize_gp_hyperparams(data[2], data[1], 10, lr, kernel=T_KERNEL, params=params)
        print(
            'Optimizing parameters_{}: \'ls\': {:0.3f}, \'a\': {:0.3f}, \'ov\': {:0.3f}, \'nv\': {:0.3f}, '
            '\'mu\': {:0.3f}, \'sig\': {:0.3f}'.format(x, params['ls'], params['a'],
                                                       params['ov'], params['nv'],
                                                       params['mu'], params['sig']))
    test_params(data[2][100:], d[100:], params)


def rel_coord_pred(x_i, x_t, y_i, y_t):
    time_input = np.arange(len(x_i))
    time_tar = 1 + time_input[-1] + np.arange(len(x_t))
    gp_x, x_p = gp_fit_sklearn(time_input, time_tar, x_i, x_t, title='X')
    gp_y, y_p = gp_fit_sklearn(time_input, time_tar, y_i, y_t, title='Y')
    return x_p, y_p


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module_x = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=8)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module_x(x)
        covar_x = self.covar_module(x)
        retval = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return retval


class SpectralMixtureGPModel2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module_time = gpytorch.means.ConstantMean()
        self.covar_module_x = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module_x.initialize_from_data(train_x, train_y[:, 0])
        self.covar_module_y = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module_y.initialize_from_data(train_x, train_y[:, 1])

    def forward(self, x):
        mean_x = self.mean_module_time(x)
        mean_y = self.mean_module_time(x)
        covar_x = self.covar_module_x(x)
        covar_y = self.covar_module_x(x)
        retval_x = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        retval_y = gpytorch.distributions.MultivariateNormal(mean_y, covar_y)
        return retval_x, retval_y


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=20), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def gp_fit_mixture_x(x_input, x_tar, y_input, y_tar, title='', training_iter=200, model=None,
                     likelihood=None, lr=0.1, route=None, plot=False):
    x_input = torch.tensor(x_input).float()
    y_input = torch.tensor(y_input).float()
    start = x_tar[0] - 1
    end = start + len(x_tar) + 1
    x_tar = torch.tensor(np.arange(start, end)).float()

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
    prev_loss = 0
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_input)
        loss = -mll(output, y_input)
        if i == 0:
            prev_loss = loss.item()
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()), end='\r')
        optimizer.step()
        # check = prev_loss - 0.0001
        # if i > 0 and (loss.item() > check):
        #     break
        # prev_loss = loss.item()

    # Test points every 0.1 between 0 and 5
    test_x = x_tar
    if plot:
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
        # See https://arxiv.org/abs/1803.06058
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions
            observed_pred = likelihood(model(test_x))
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data
            if route.any():
                ax.plot(range(len(route[:, 0])), route[:, 0], 'b', label='route', alpha=0.2)
            ax.plot(x_tar[1:], y_tar, 'r')
            ax.plot(test_x, observed_pred.mean, 'g', label='pred')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            # ax1.legend()
            # plt.title(title)
            plt.show()
    return model, likelihood


def gp_fit_mixture_xy(x_input, x_tar, y_input, y_tar, title='', training_iter=200, model=None,
                      likelihood=None, lr=0.1, route=None):
    x_input = torch.tensor(x_input).float()
    y_input = torch.tensor(y_input).float()
    start = x_tar[0] - 1
    end = start + len(x_tar) + 1
    x_tar = torch.tensor(np.arange(start, end)).float()

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
    prev_loss = 0
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_input)
        loss = -mll(output, y_input)
        if i == 0:
            prev_loss = loss.item()
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()), end='\r')
        optimizer.step()
        # check = prev_loss - 0.0001
        # if i > 0 and (loss.item() > check):
        #     break
        # prev_loss = loss.item()

    # Test points every 0.1 between 0 and 5
    test_x = x_tar

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    # See https://arxiv.org/abs/1803.06058
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred_x, observed_pred_y = likelihood(model(test_x))
        mean_x = observed_pred_x.mean
        mean_y = observed_pred_y.mean
        # Initialize plot
        f, axes = plt.subplots(2, 2, figsize=(4, 3))
        ax1 = axes[0, 0]
        ax2 = axes[1, 0]
        ax3 = axes[1, 1]
        # Get upper and lower confidence bounds
        lower, upper = observed_pred_x.confidence_region()
        # Plot training data
        if route.any():
            ax1.plot(route[:, 0], route[:, 1], 'b', label='route', alpha=0.2)
        ax1.plot(y_input.numpy()[:, 0], y_input.numpy()[:, 1], 'r', label='input')
        # Plot predictive means as blue line
        ax1.plot(y_tar[:, 0], y_tar[:, 1], 'b', alpha=0.5, label='target')
        ax2.plot(x_input, y_input.numpy()[:, 0])
        ax2.plot(test_x, mean_x, 'g')
        ax3.plot(x_input, y_input.numpy()[:, 1])
        ax3.plot(test_x, mean_y, 'g')
        ax1.plot(mean_x, mean_y, 'g', label='pred')
        # Shade between the lower and upper confidence bounds
        # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax1.legend()
        # plt.title(title)
        plt.show()
    return model, likelihood


def gp_fit_sklearn_xy(x_input, x_tar, y_input, y_tar, title='', route=None, gp=None):
    if gp:
        gp1 = gp
    else:
        k1 = kernels.DotProduct(sigma_0=1., sigma_0_bounds=(1e-3, 1e1))
        k3 = kernels.RationalQuadratic(alpha=1.5, length_scale=2.5,
                                       length_scale_bounds=(1e-3, 20), alpha_bounds=(1e-3, 10))
        k4 = kernels.ConstantKernel(1., (1e-3, 1e2))
        k5 = kernels.ConstantKernel(1., (1e-2, 1e2))
        kernel = k1 * k4 + k3 * k5
        gp1 = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=0,
            random_state=0
        )

    x_input = x_input.reshape(-1, 1)
    x_tar = x_tar.reshape(-1, 1)

    gp1.fit(x_input, y_input)
    pred, std = gp1.predict(x_tar, return_std=True)
    if route.any():
        plt.plot(route[:, 0], route[:, 1], 'b', label='Prediction', alpha=0.2)
    plt.plot(y_input[:, 0], y_input[:, 1], 'bo', label='Input', alpha=0.4)
    plt.plot(y_tar[:, 0], y_tar[:, 1], 'go', label='Target', alpha=0.4)
    plt.plot(pred[:, 0], pred[:, 1], 'ro', label='Prediction', alpha=0.4)
    # plt.gca().fill_between(pred[:, 0].reshape(-1) - 2 * std, pred[:, 0].reshape(-1) + 2 * std,
    #                        pred[:, 1].reshape(-1) - 2 * std, pred[:, 1].reshape(-1) + 2 * std, color='lightblue',
    #                        alpha=0.5, label=r"$2\sigma$")

    plt.title(title)
    plt.legend()
    plt.show()
    return gp1, pred


def gp_fit_sklearn(x_input, x_tar, y_input, y_tar, params=None, title=''):
    k1 = kernels.DotProduct(sigma_0=1, sigma_0_bounds=(1e-05, 5))
    k2 = kernels.RBF(length_scale=10, length_scale_bounds=(1e-3, x_tar[-1]))
    k3 = kernels.RationalQuadratic(alpha=1, length_scale=10, length_scale_bounds=(1e-3, x_tar[-1]))

    kernel = k1 * k2 * k3

    gp1 = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        alpha=0
    )
    if params:
        gp1.set_params(params)
    gp1.fit(x_input.reshape(-1, 1), y_input)
    pred, std = gp1.predict(x_tar.reshape(-1, 1), return_std=True)

    plt.plot(x_input, y_input, 'bo', label='Input', alpha=0.4)
    plt.plot(x_tar, y_tar, 'go', label='Target', alpha=0.4)
    plt.plot(x_tar, pred, 'ro', label='Prediction', alpha=0.4)
    plt.gca().fill_between(x_tar, pred.reshape(-1) - 2 * std, pred.reshape(-1) + 2 * std, color='lightblue',
                           alpha=0.5, label=r"$2\sigma$")
    plt.title(title)
    plt.legend()
    plt.show()
    return gp1, pred


def main():
    sln = 200
    dataset = SLDataset(csv_file='ruteT_maj.csv', seq_len=sln, resample_freq='20s')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=sl_collate)
    x_rel, y_rel, time = getdata(dataloader, sln)
    split = 20  # 20
    high = split + 20  # split * 2

    txy_input = zip(x_rel.T, y_rel.T)
    for inp in txy_input:
        xy_input = np.array(list(inp)).T
        itr = 0
        model = None
        for x in range(40):
            time_input = time[itr:split + itr]
            time_tar = time[split + itr:high + itr]

            xy_i = xy_input[itr:split + itr]
            xy_t = xy_input[split + itr:high + itr]
            # if model is None:
            model_x, llh_x = gp_fit_mixture_x(time_input, time_tar, xy_i[:, 0], xy_t[:, 0], training_iter=300, lr=1e-1,
                                          title='x', route=xy_input)
            model_x, llh_x = gp_fit_mixture_x(time_input, time_tar, xy_i[:, 0], xy_t[:, 0], training_iter=100, title='x',
                                          model=model_x, likelihood=llh_x, lr=1e-2, route=xy_input)
            model_x, llh_x = gp_fit_mixture_x(time_input, time_tar, xy_i[:, 0], xy_t[:, 0], training_iter=100, title='x',
                                          model=model_x, likelihood=llh_x, lr=1e-3, route=xy_input, plot=True)

            model_y, llh_y = gp_fit_mixture_x(time_input, time_tar, xy_i[:, 0], xy_t[:, 1], training_iter=300, lr=1e-1,
                                          title='x', route=xy_input)
            model_y, llh_y = gp_fit_mixture_x(time_input, time_tar, xy_i[:, 0], xy_t[:, 1], training_iter=100,
                                              model=model, likelihood=llh_y, lr=1e-2)
            model_y, llh_y = gp_fit_mixture_x(time_input, time_tar, xy_i[:, 0], xy_t[:, 1], training_iter=100,
                                              title='y', model=model, likelihood=llh_y, lr=1e-3, route=xy_input, plot=True)
            # else:
            #     model, llh = gp_fit_mixture_xy(time_input, time_tar, xy_i, xy_t, training_iter=20, title='x',
            #                                    model=model, likelihood=llh, lr=1e-3, route=xy_input)
            # yy = gp_fit_mixture_xy(time_input, time_tar, xy_i[:, 1], xy_t[:, 1], title='y')
            # # Initialize plot
            # f, ax = plt.subplots(1, 1, figsize=(4, 3))
            #
            # ax.plot(xy_i[:, 0], xy_i[:, 1], 'r')
            # ax.plot(xy_t[:, 0], xy_t[:, 1], 'b')
            # # Plot predictive means as blue line
            # ax.plot(xx.mean.numpy(), yy.mean.numpy(), 'g')
            #
            # # Shade between the lower and upper confidence bounds
            # # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            # ax.legend(['Observed Data', 'Target', 'Mean'])
            # plt.show()
            itr += 2


def main22():
    sln = 120
    dataset = SLDataset(csv_file='ruteT_maj.csv', seq_len=sln, resample_freq='20s')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=sl_collate)
    x_rel, y_rel, time = getdata(dataloader, sln)
    split = 20  # 20
    high = split + 20  # split * 2
    time_input = time[:split]
    time_tar = time[split:high]

    x_input = x_rel[:split, 0]
    x_tar = x_rel[split:high, 0]

    y_input = y_rel[:split, 0]
    y_tar = y_rel[split:high, 0]

    samples = 50
    sample_points = time[split:high]  # np.linspace(time_input[0], time_tar[-1], samples).reshape(-1, 1)

    x_p, x_std, params_x = gaussian_process(time_input, x_input, time_tar, x_tar, sample_points, True, 'x')

    y_p, y_std, params_y = gaussian_process(time_input, y_input, time_tar, y_tar, sample_points, True, 'y')

    plt.plot(x_input, y_input, 'bo', label='Input', alpha=0.4)
    plt.plot(x_tar, y_tar, 'g', label='Target', alpha=0.4)
    plt.plot(x_p, y_p, 'r', label='Prediction', alpha=0.4)
    plt.title('Data')
    plt.legend()
    plt.show()
    itr = 0

    for x in range(20):
        itr += 2
        time_input = time[itr:split + itr]
        time_tar = time[split + itr:high + itr]

        x_input = x_rel[itr:split + itr, 0]
        x_tar = x_rel[split + itr:high + itr, 0]

        y_input = y_rel[itr:split + itr, 0]
        y_tar = y_rel[split + itr:high + itr, 0]
        samples = 50 - itr
        sample_points = time[itr:high + itr]  # np.linspace(time_input[0], time_tar[-1], samples).reshape(-1, 1)
        x_p, x_std, params_x = ap_f_gp(time_input, x_input, time_tar, x_tar, sample_points, params_x, 'x')
        print(
            'Updated x_{}: ls: {}, al: {}, ov: {}, nv: {}, mu: {}, sig: {}'.format(x, params_x['ls'],
                                                                                   params_x['a'],
                                                                                   params_x['ov'],
                                                                                   params_x['nv'],
                                                                                   params_x['mu'],
                                                                                   params_x['sig']))
        y_p, y_std, params_y = ap_f_gp(time_input, y_input, time_tar, y_tar, sample_points, params_y, 'y')
        print(
            'Updated y_{}: ls: {}, al: {}, ov: {}, nv: {}, mu: {}, sig: {}'.format(x, params_y['ls'],
                                                                                   params_y['a'],
                                                                                   params_y['ov'],
                                                                                   params_y['nv'],
                                                                                   params_y['mu'],
                                                                                   params_y['sig']))

        plt.plot(x_input, y_input, 'bo', label='Input', alpha=0.4)
        plt.plot(x_tar, y_tar, 'go', label='Target', alpha=0.4)
        plt.plot(x_p, y_p, 'r', label='Prediction', alpha=0.4)
        plt.title('Actual')
        plt.legend()
        plt.show()
        itr = 0


if __name__ == "__main__":
    main()
