import torch


class NoiseScheduleVP:

    def __init__(self, beta_min=0.05, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = 1.
    
    def get_noise(self, t, beta_init, beta_term, cumulative=False):
        if cumulative:
            noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
        else:
            noise = beta_init + (beta_term - beta_init)*t
        return noise

    def marginal_log_mean_coeff(self, t):
        return -0.25 * t**2 * (self.beta_max -
                               self.beta_min) - 0.5 * t * self.beta_min

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        tmp = 2. * (self.beta_max - self.beta_min) * torch.logaddexp(
            -2. * lamb,
            torch.zeros((1, )).to(lamb))
        Delta = self.beta_min**2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_min) / (self.beta_max -
                                                            self.beta_min)

    def get_time_steps(self, t_T, t_0, N):
        lambda_T = self.marginal_lambda(torch.tensor(t_T))
        lambda_0 = self.marginal_lambda(torch.tensor(t_0))
        logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1)
        return self.inverse_lambda(logSNR_steps)
    
    @torch.no_grad()
    def reverse_diffusion(self, estimator, spk, z, mask, mu, n_timesteps, stoc):
        print("use dpm-solver reverse")
        xt = z * mask
        yt = xt - mu
        T = 1
        eps = 1e-3
        time = self.get_time_steps(T, eps, n_timesteps)
        for i in range(n_timesteps):
            s = torch.ones((xt.shape[0], )).to(xt.device) * time[i]
            t = torch.ones((xt.shape[0], )).to(xt.device) * time[i + 1]

            lambda_s = self.marginal_lambda(s)
            lambda_t = self.marginal_lambda(t)
            h = lambda_t - lambda_s

            log_alpha_s = self.marginal_log_mean_coeff(s)
            log_alpha_t = self.marginal_log_mean_coeff(t)

            sigma_t = self.marginal_std(t)
            phi_1 = torch.expm1(h)

            noise_s = estimator(spk, yt + mu, mask, mu, s)
            lt = 1 - torch.exp(-self.get_noise(s, self.beta_min, self.beta_max, cumulative=True))
            a = torch.exp(log_alpha_t - log_alpha_s)
            b = sigma_t * phi_1 * torch.sqrt(lt)
            yt = a * yt + (b * noise_s)
        xt = yt + mu
        return xt


class MaxLikelihood:

    def __init__(self, beta_min=0.05, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max
   
    def get_noise(self, t, beta_init, beta_term, cumulative=False):
        if cumulative:
            noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
        else:
            noise = beta_init + (beta_term - beta_init)*t
        return noise
    
    def get_gamma(self, s, t, beta_init, beta_term):
        gamma = beta_init*(t-s) + 0.5*(beta_term-beta_init)*(t**2-s**2)
        gamma = torch.exp(-0.5*gamma)
        return gamma

    def get_mu(self, s, t):
        gamma_0_s = self.get_gamma(0, s, self.beta_min, self.beta_max)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        gamma_s_t = self.get_gamma(s, t, self.beta_min, self.beta_max)
        mu = gamma_s_t * ((1-gamma_0_s**2) / (1-gamma_0_t**2))
        return mu        

    def get_nu(self, s, t):
        gamma_0_s = self.get_gamma(0, s, self.beta_min, self.beta_max)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        gamma_s_t = self.get_gamma(s, t, self.beta_min, self.beta_max)
        nu = gamma_0_s * ((1-gamma_s_t**2) / (1-gamma_0_t**2))
        return nu

    def get_sigma(self, s, t):
        gamma_0_s = self.get_gamma(0, s, self.beta_min, self.beta_max)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        gamma_s_t = self.get_gamma(s, t, self.beta_min, self.beta_max)
        sigma = torch.sqrt(((1 - gamma_0_s**2) * (1 - gamma_s_t**2)) / (1 - gamma_0_t**2))
        return sigma        

    def get_kappa(self, t, h, noise):
        nu = self.get_nu(t-h, t)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        kappa = (nu*(1-gamma_0_t**2)/(gamma_0_t*noise*h) - 1)
        return kappa

    def get_omega(self, t, h, noise):
        mu = self.get_mu(t-h, t)
        kappa = self.get_kappa(t, h, noise)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        omega = (mu-1)/(noise*h) + (1+kappa)/(1-gamma_0_t**2) - 0.5
        return omega 

    @torch.no_grad()
    def reverse_diffusion(self, estimator, spk, z, mask, mu, n_timesteps, stoc=False):
        print("use MaxLikelihood reverse")
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - i*h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                 device=z.device)            
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self.get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)

            kappa_t_h = self.get_kappa(t, h, noise_t) 
            omega_t_h = self.get_omega(t, h, noise_t)
            sigma_t_h = self.get_sigma(t-h, t)
 
            es = estimator(spk, xt, mask, mu, t)

            dxt = ((0.5+omega_t_h)*(xt - mu) + (1+kappa_t_h) * es)
            dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                   requires_grad=False)
            dxt_stoc = dxt_stoc * sigma_t_h

            dxt = dxt * noise_t * h + dxt_stoc
            xt = (xt + dxt) * mask
        return xt


class GradRaw:

    def __init__(self, beta_min=0.05, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_noise(self, t, beta_init, beta_term, cumulative=False):
        if cumulative:
            noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
        else:
            noise = beta_init + (beta_term - beta_init)*t
        return noise
    
    @torch.no_grad()
    def reverse_diffusion(self, estimator, spk, z, mask, mu, n_timesteps, stoc=False):
        print("use grad-raw reverse")
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * \
                torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self.get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - estimator(spk, xt, mask, mu, t)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - estimator(spk, xt, mask, mu, t))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt
