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

torch.set_default_tensor_type(torch.FloatTensor)


# Kernels

def squared_exponential_kernel(x, y, lengthscale, alpha, variance):
    '''
    Function that computes the covariance matrix using a squared-exponential kernel
    '''
    # pair-wise distances, size: NxM
    sqdist = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'sqeuclidean')
    # compute the kernel
    cov_matrix = variance * np.exp(-0.5 * sqdist * (1 / lengthscale ** 2))  # NxM
    return cov_matrix


def abs_kernel(x, y, lengthscale, alpha, variance):
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    absdist = cdist(x, y, 'cityblock')
    k = variance * np.exp(-0.5 * absdist * (1 / lengthscale ** 2))  # NxM
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
    k = variance * torch.exp(-0.5 * sqdist * (1 / _lambda ** 2))  # NxM
    return k


def abs_kernel_torch(x, y, _lambda, _alpha_param, variance):
    x = x.squeeze(1).expand(x.size(0), y.size(0))
    y = y.squeeze(0).expand(x.size(0), y.size(0))
    absdist = torch.abs(x - y)
    k = variance * torch.exp(-0.5 * absdist * (1 / _lambda ** 2))  # NxM
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
    K = kernel(x, x, params[0], params[1], params[2])
    L = np.linalg.cholesky(K + params[4] * np.eye(len(x)))

    # compute the mean at our test points.
    Ks = kernel(x, sampling, params[0], params[1], params[2])
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  #
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    # compute the variance at our test points.
    Kss = kernel(sampling, sampling, params[0], params[1], params[2])
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
    mu = torch.tensor(np.abs(y[-1])*2).float()
    sig = torch.tensor(np.var(y)).float()
    if params:
        lambda_param = nn.Parameter(torch.tensor(params[0]), requires_grad=True)
        alpha_param = nn.Parameter(torch.tensor(params[1]), requires_grad=True)
        output_variance = nn.Parameter(torch.tensor(params[2]), requires_grad=True)
        noise_variance = nn.Parameter(torch.tensor(params[3]), requires_grad=True)
        mu_param = nn.Parameter(torch.tensor(params[4]), requires_grad=True)
        sig_param = nn.Parameter(torch.tensor(params[5]), requires_grad=True)
    else:
        lambda_param = nn.Parameter(mu, requires_grad=True)
        alpha_param = nn.Parameter(torch.tensor(1.), requires_grad=True)
        output_variance = nn.Parameter(torch.tensor(1.), requires_grad=True)
        noise_variance = nn.Parameter(torch.tensor(.5), requires_grad=True)
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
        assert noise_variance >= 0, f"ouch! {i, noise_variance}"
        optim.zero_grad()
        K = kernel(x_tensor, x_tensor, lambda_param, alpha_param,
                   output_variance) + noise_variance * torch.eye(N)

        cholesky = torch.cholesky(K)
        _alpha_temp, _ = torch.solve(y_tensor, cholesky)
        _alpha, _ = torch.solve(_alpha_temp, cholesky.t())

        nll = N / 2 * torch.log(torch.tensor(2 * np.pi)) + 0.5 * torch.matmul(y_tensor.transpose(0, 1), _alpha) + \
              torch.sum(torch.log(torch.diag(cholesky)))

        # we have to add the log-likelihood of the prior
        norm = distributions.Normal(loc=mu_param, scale=sig_param)
        prior_negloglike = torch.log(lambda_param) - torch.log(torch.exp(norm.log_prob(lambda_param)))

        nll += 0.9 * prior_negloglike
        nll.backward()

        nlls.append(nll.item())
        lambdas.append(lambda_param.item())
        output_variances.append(output_variance.item())
        noise_variances.append(noise_variance.item())
        optim.step()

        # projected in the constraints (lengthscale and output variance should be positive)
        for p in [lambda_param, output_variance, alpha_param]:
            p.data.clamp_(min=0.0000001)

        noise_variance.data.clamp_(min=1e-5, max=0.05)
        mu_param.data.clamp_(min=0, max=50)
        sig_param.data.clamp_(min=.1, max=20)

    return (lambda_param.item(), alpha_param.item(), output_variance.item(), noise_variance.item(),
            mu_param.item(), sig_param.item())


def gaussian_process(x_input, y_input, x_tar, y_tar, sampling,
                     draw=False, title='', kernel=rational_quadratic_kernel_torch):
    # print dataset information
    # also in this case we standardize the data to have zero mean and unit variance

    # we shall also define the test set, that is the range of XTest points we want to
    # use to compute the mean and the variance

    params = optimize_gp_hyperparams(x_input, y_input, 1500, 1e-3)
    print(
        'Optimized parameters: ls: {}, al: {}, ov: {}, nv: {}, mu: {}, sig: {}'.format(params[0], params[1], params[2],
                                                                                       params[3], params[4], params[5]))

    # we can fit the GP that use the hyperparameters found above
    mu, covariance = fit_predictive_gp(x_input, y_input, sampling, params)
    std = np.sqrt(np.diag(covariance))
    if draw:
        plt.plot(x_tar, y_tar, 'bo', label='test')
        plt.plot(x_input, y_input, 'ro', label='Training points')
        plt.gca().fill_between(sampling.flat, mu.reshape(-1) - 2 * std, mu.reshape(-1) + 2 * std, color='lightblue',
                               alpha=0.5, label=r"$2\sigma$")
        plt.plot(sampling, mu, 'green', label=r"$\mu$")
        plt.title(title)
        plt.legend()
        plt.show()
    return mu, std, params


def ap_f_gp(x_input, y_input, x_tar,  y_tar, x_test, params, title=''):
    params = optimize_gp_hyperparams(x_input, y_input, 10, 1e-3, params=params)

    mu, covariance = fit_predictive_gp(x_input, y_input, x_test, params)
    std = np.sqrt(np.diag(covariance))

    plt.plot(x_input, y_input, 'ro', label='Training points')
    plt.plot(x_tar, y_tar, 'bo', label='test')
    plt.gca().fill_between(x_test.flat, mu.reshape(-1) - 2 * std, mu.reshape(-1) + 2 * std, color='lightblue',
                           alpha=0.5, label=r"$2\sigma$")
    plt.plot(x_test, mu, 'green', label=r"$\mu$")
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    sln = 40
    dataset = SLDataset(csv_file='sep2018.csv', seq_len=sln, resample_freq='20s')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=sl_collate)
    seq = next(iter(dataloader))

    """"
    Impl mean as linear regression....
    """

    x = np.array(seq)
    x_n = np.array(x)[:, np.where(np.array(x)[-1, :, 1] != -1)].squeeze().reshape(sln, -1, 4)
    x = np.array(x_n)[:, np.where(np.array(x_n)[0, :, 1] != -1)].squeeze().reshape(sln, -1, 4)
    x_rel, y_rel = make_relative_meters(x)
    x_rel = np.flip(x_rel)
    y_rel = np.flip(y_rel)

    time = np.array(range(sln))
    time = (time - np.mean(time)) / np.std(time)
    split = int(sln * 0.5)
    time_input = time[:split]
    time_tar = time[split:]

    x_input = x_rel[:split, 0]
    x_tar = x_rel[split:, 0]

    y_input = y_rel[:split, 0]
    y_tar = y_rel[split:, 0]

    samples = 50
    sample_points = np.linspace(time_input[0], time_tar[-1], samples).reshape(-1, 1)

    x_p, x_std, params_x = gaussian_process(time_input, x_input, time_tar, x_tar, sample_points, True, 'x')
    y_p, y_std, params_y = gaussian_process(time_input, y_input, time_tar, y_tar, sample_points, True, 'y')
    plt.plot(x_input, y_input, 'bo', label='Input')
    plt.plot(x_tar, y_tar, 'go', label='Target')
    plt.plot(x_p, y_p, 'ro', label='Prediction')
    plt.title('Data')
    plt.legend()
    plt.show()
    itr = 0
    for x in range(5):
        itr += 2
        time_input = time[itr:split + itr]
        time_tar = time[split:]

        x_input = x_rel[itr:split + itr, 0]
        x_tar = x_rel[split:, 0]

        y_input = y_rel[itr:split + itr, 0]
        y_tar = y_rel[split:, 0]
        samples = 50 - itr
        sample_points = np.linspace(time_input[0], time_tar[-1], samples).reshape(-1, 1)
        ap_f_gp(time_input, x_input, time_tar,  x_tar, sample_points, params_x, 'x')
        ap_f_gp(time_input, y_input, time_tar, y_tar, sample_points, params_y, 'y')


if __name__ == "__main__":
    main()
