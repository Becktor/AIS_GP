import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader, SLDataset, sl_collate, make_relative_meters
import torch
import gpytorch

torch.set_default_tensor_type(torch.FloatTensor)


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    """
    Spectral Mixture Kernel :
    """

    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module_x = gpytorch.means.LinearMean(1)
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module_x(x)
        covar_x = self.covar_module(x)
        retval = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return retval


def gp_pred(x_input, y_input, future_steps=None, training_iter=200, model=None, likelihood=None, lr=0.1):
    x_input = torch.tensor(x_input).float()
    y_input = torch.tensor(y_input).float()

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
    observed_pred = None
    if future_steps is not None:
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        x_predict = torch.tensor(future_steps).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions
            observed_pred = likelihood(model(x_predict))
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


def run_gp_xy(xy_i, timesteps):
    time_input = np.array(range(len(xy_i)))
    time_tar = np.array(range(timesteps))

    x_i = np.column_stack((time_input, xy_i[:, 0]))
    y_i = np.column_stack((time_input, xy_i[:, 1]))

    model_x, llh_x, _ = gp_pred(x_i[:, 0], x_i[:, 1], training_iter=100, lr=1e-1)
    model_x, llh_x, _ = gp_pred(x_i[:, 0], x_i[:, 1], training_iter=100, model=model_x, likelihood=llh_x, lr=1e-2)
    _, _, obs_pred_x = gp_pred(x_i[:, 0], x_i[:, 1], time_tar, training_iter=100, model=model_x, likelihood=llh_x,
                               lr=1e-3)

    model_y, llh_y, _ = gp_pred(y_i[:, 0], y_i[:, 1], training_iter=100, lr=1e-1)
    model_y, llh_y, _ = gp_pred(y_i[:, 0], y_i[:, 1], training_iter=100, model=model_y, likelihood=llh_y, lr=1e-2)
    _, _, obs_pred_y = gp_pred(y_i[:, 0], y_i[:, 1], time_tar, training_iter=100, model=model_y, likelihood=llh_y,
                               lr=1e-3)

    return obs_pred_x, obs_pred_y


def main():
    sln = 200
    dataset = SLDataset(csv_file='ruteT_maj.csv', seq_len=sln, resample_freq='20s')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=sl_collate)
    x_rel, y_rel = getdata(dataloader, sln)
    split = 5
    high = split + 20

    txy_input = zip(x_rel.T, y_rel.T)
    for inp in txy_input:
        xy_input = np.array(list(inp)).T
        itr = 0
        for x in range(40):
            xy_i = xy_input[itr:split + itr]
            xy_t = xy_input[split + itr:high + itr]
            # Gaussian Process
            obs_x, obs_y = run_gp_xy(xy_i, high)
            xx = obs_x.mean.numpy()
            yy = obs_y.mean.numpy()
            x_conf = obs_x.confidence_region()
            y_conf = obs_y.confidence_region()

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

