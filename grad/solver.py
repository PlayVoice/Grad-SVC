# Adapted from https://github.com/thuhcsi/LightGrad

import torch


class NoiseScheduleVP:

    def __init__(self, beta_0=0.05, beta_1=20):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.T = 1.

    def marginal_log_mean_coeff(self, t):
        return -0.25 * t**2 * (self.beta_1 -
                               self.beta_0) - 0.5 * t * self.beta_0

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(
            -2. * lamb,
            torch.zeros((1, )).to(lamb))
        Delta = self.beta_0**2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 -
                                                          self.beta_0)

    def get_time_steps(self, t_T, t_0, N):
        lambda_T = self.marginal_lambda(torch.tensor(t_T))
        lambda_0 = self.marginal_lambda(torch.tensor(t_0))
        logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1)
        return self.inverse_lambda(logSNR_steps)
