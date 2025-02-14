import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
import time

import sys
sys.path.append('multivariate-hawkes-inhibition')
from class_and_func.hawkes_process import exp_thinning_hawkes
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_estimator_bfgs
import matplotlib.pyplot as plt

def time_tic(counters=[time.perf_counter, time.process_time, time.thread_time, time.time]):
    return np.r_[[name() for name in counters]]

def time_toc(times, counters=[time.perf_counter, time.process_time, time.thread_time, time.time]):
    return time_tic(counters) - times

def plot_kernels(dic, title=None, bounds=(-0.5, 5.5), sharey=False, fittrue=False, figsize=None, legendpos=(0, 0)):
    t_kernel = np.linspace(bounds[0], bounds[1], num=500)
    for name, model in dic.items():
        fig, axk = plt.subplots(model.p, model.p, sharex=True, sharey=sharey, figsize=figsize)
        break
    axlim = np.empty_like(axk)

    for name, model in dic.items():
        val_kernel = model.kernel(t_kernel)
        for j in range(model.p):
            for l in range(model.p):
                label = rf'Kernel ${j + 1} \leftarrow {l + 1}$' if name=='true' else \
                    (name if (j, l)==legendpos else None)
                linewidth = 2

                axk[j, l].plot(t_kernel, val_kernel[j][l], label=label, linewidth=linewidth)

                if fittrue and name=='true':
                    axlim[j, l] = axk[j, l].get_ylim()
                elif fittrue:
                    axk[j, l].set_ylim(*axlim[j, l])

                axk[j, l].legend()
    fig.suptitle(title)
    return fig, axk

def plot_diagonal_kernels(dic, title=None, bounds=(-0.5, 5.5), sharey=False, fittrue=False, figsize=None, legendpos=0):
    t_kernel = np.linspace(bounds[0], bounds[1], num=500)
    for name, model in dic.items():
        fig, axk = plt.subplots(1, model.p, sharex=True, sharey=sharey, figsize=figsize)
        break
    axlim = np.empty_like(axk)

    for name, model in dic.items():
        val_kernel = model.kernel(t_kernel)
        for j in range(model.p):
            label = rf'Kernel ${j + 1} \leftarrow {j + 1}$' if name=='true' else \
                (name if j==legendpos else None)
            linewidth = 2

            axk[j].plot(t_kernel, val_kernel[j][j], label=label, linewidth=linewidth)

            if fittrue and name=='true':
                axlim[j] = axk[j].get_ylim()
            elif fittrue:
                axk[j].set_ylim(*axlim[j])

            axk[j].legend()
    fig.suptitle(title)
    return fig, axk

def logit(x, t=1.):
    # res = np.log(1+np.exp(t*x))/t
    # No overflow version
    res = np.empty_like(x)
    mask = x > 0
    temp = x[mask]
    res[mask] = temp + np.log(1 + np.exp(-t*temp)) / t
    res[~mask] = np.log(1 + np.exp(t*x[~mask])) / t
    return res + 1e-16

relu = lambda x: np.fmax(0, x)

def der_logit(x, t=1.):
    # res = 1 / (1+np.exp(-t*x))
    # No overflow version
    res = np.empty_like(x)
    mask = x < 0
    temp = np.exp(t*x[mask])
    res[mask] = temp / (1+temp)
    res[~mask] = 1 / (1+np.exp(-t*x[~mask]))
    # res[res<1e-10] = 0
    return res

class MultivariateExponentialHawkes():
    """Miguel's package for simulation and inference of multivariate exponential
    Hawkes processes with inhibition"""
    def __init__(self, mu=None, alpha=None, beta=None):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.p = None if mu is None else np.array(mu).size
        self.res_optim = None
        self.param = None
        self.model_simulate = None
        self.model_fit = None
        self.model_est = None
        self.optimize_time = None
        self.fit_time = None

    def fit(self, times):
        self.fit_time = time_tic()
        if hasattr(times[0], '__iter__'):
            self.p = max([m for t,m in times])
        else:
            times = [(t, 1) for t in times]
            self.p = 1
        model = multivariate_estimator_bfgs(dimension=self.p)
        self.optimize_time = time_tic()
        model.fit(times)
        self.optimize_time = time_toc(self.optimize_time)
        self.mu = model.mu_estim.squeeze()
        self.alpha = model.alpha_estim.squeeze()
        self.beta = model.beta_estim.squeeze()
        self.res_optim = model.res
        self.param = model.res.x
        self.model_fit = model
        self.model_est = HawkesProcess(mu=self.mu, alpha=self.alpha, beta=np.tile(self.beta, (self.beta.size, 1)).T)
        self.fit_time = time_toc(self.fit_time)
        return self

    def simulate(self, size=None, horizon=None):
        if self.mu is None or self.alpha is None or self.beta is None:
            return []
        if self.p==1:
            model = exp_thinning_hawkes(lambda_0=self.mu, alpha=self.alpha, beta=self.beta,
                                        max_jumps=size, max_time=horizon)
        else:
            model = multivariate_exponential_hawkes(mu=np.array(self.mu),
                                                    alpha=np.array(self.alpha),
                                                    beta=np.array(self.beta),
                                                    max_jumps=size, max_time=horizon)
        model.simulate()
        self.model_simulate = model
        return model.timestamps

    def plot_intensity(self, *args, **kwargs):
        self.model_simulate.plot_intensity(*args, **kwargs)

    def kernel(self, *args, **kwargs):
        return self.model_est.kernel(*args, **kwargs)

    def intensity(self, *args, **kwargs):
        return self.model_est.intensity(*args, **kwargs)

    def scores(self, *args, **kwargs):
        return self.model_est.scores(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.model_est.score(*args, **kwargs)

class HawkesProcess():
    def __init__(self, mu=1, kernel='exponential', alpha=0.5, beta=1, approx_support=10, burnin=None, history=None,
                 num_int=1000, link_param=100):
        self.mu = np.r_[mu]
        if kernel=='exponential':
            alpha = np.r_[alpha]
            beta = np.r_[beta]
            # Common beta per row of alpha
            if beta.ndim==1 and alpha.ndim==2:
                beta = beta[:, np.newaxis] * np.ones(beta.shape[0])
            if alpha.ndim==1:
                alpha = alpha[:, np.newaxis]
            # Crappy lambdas - https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
            self.kernel_fun = [[(lambda u,v: (lambda x: alpha[u, v] * np.exp(-beta[u, v]*x)))(j, l)
                                for l in range(alpha.shape[1])] for j in range(alpha.shape[0])]
        else:
            self.kernel_fun = kernel if hasattr(kernel, '__iter__') else [[kernel]]
        self.kernel_bound = None
        self.approx_support = approx_support
        self.p = len(self.mu)
        self.history = [list() for _ in range(self.p)] if history is None else (history if hasattr(history[0], '__iter__')
                                                                                else [list(history)])
        self.burnin = -2 * np.fabs(approx_support) if burnin is None else -np.fabs(burnin)
        self.num_int = num_int
        self.link_param = link_param

    def kernel(self, t):
        return self.kernel_fun[0][0](t) * (t>0) if self.p==1 else [[self.kernel_fun[j][l](t) * (t>0) for l in range(self.p)] for j in range(self.p)]

    def compute_kernel_bound(self):
        t = np.linspace(0, self.approx_support, num=1000)
        self.kernel_bound = np.array([[np.fmax(0, np.max(self.kernel_fun[j][l](t)))*1.1 for l in range(self.p)] for j in range(self.p)])

    def add_history(self, time, mark):
        assert (not self.history[mark]) or (time >= self.history[mark][-1])  # Check that new time is after the last time of Process m
        self.history[mark].append(time)

    def underlying(self, t, history=None):
        t = t if hasattr(t, '__iter__') else [t]  # If t is a float
        t = t if hasattr(t[0], '__iter__') else [t] * self.p  # If t is a list
        u = [np.reshape(row, (-1, 1)) for row in t]
        history = self.history if history is None else history
        res = [np.ones(u[j].size) * self.mu[j] for j in range(self.p)]
        for j in range(self.p):
            for l in range(self.p):
                diff = u[j] - np.array(history[l])
                mask = diff > 0
                res[j] += np.sum(self.kernel_fun[j][l](diff*mask) * mask, axis=1)  # Add mask in kernel_fun to avoir overflows
        return (res[0] if len(res[0])>1 else res[0][0]) if self.p==1 else res

    def intensity(self, t):
        # Float or list or list of lists
        intensities = self.underlying(t)
        is_list_of_list = hasattr(intensities, '__iter__') and hasattr(intensities[0], '__iter__')
        return [relu(int) for int in intensities] if is_list_of_list else relu(intensities)

    def scores(self, times):
        """Log-likelihood of parameters with respect to times"""
        # convert times to list of list of times
        if hasattr(times[0], '__iter__'):
            times = [[t for t, k in times if k == j] for j in range(1, self.p + 1)]
        else:
            times = [times]
        link = lambda x: logit(x, self.link_param)
        T = np.max([np.max(row) for row in times])
        times_int = np.linspace(0, T, num=self.num_int)
        intensities = [link(row) for row in self.underlying(times, history=times)]
        intensities_int = [link(row) for row in self.underlying(times_int, history=times)]
        # log(intensities) - compensator
        logliks = np.array([np.log(intensities[j]).sum() - T * np.mean(intensities_int[j]) for j in range(self.p)])
        return logliks

    def score(self, times):
        """Log-likelihood of parameters with respect to times"""
        return self.scores(times).sum()

    def simulate(self, size=None, horizon=None, clear_history=True, bounds=False):
        size = 1 if size is None and horizon is None else size
        size = np.infty if size is None else size
        horizon = np.infty if horizon is None else horizon

        if self.kernel_bound is None:
            self.compute_kernel_bound()

        if clear_history:
            self.history = [list() for _ in range(self.p)]
            t = self.burnin
            interacting_points = [np.r_[:] for _ in range(self.p)]  # Points in the support of the kernel
        else:
            t = np.max([self.history[j][-1] for j in range(self.p)])
            interacting_points = []  # Points in the support of the kernel
            for j in range(self.p):
                hist_array = np.array(self.history[j])
                interacting_points.append(hist_array[np.logical_and(t-self.approx_support <= hist_array, hist_array <= t)])

        N_interacting_points = np.array([arr.size for arr in interacting_points])
        upperbounds = self.mu + self.kernel_bound @ N_interacting_points
        upperbound = upperbounds.sum()

        n = 0
        path = []
        upperlist = []
        while n < size and t < horizon:
            t += -np.log(np.random.rand()) / upperbound
            intensities = np.array(self.intensity(t))
            mark = np.random.multinomial(1, np.r_[intensities.ravel() / upperbound, 0]).argmax()
            if mark < self.p:
                if t <= horizon:
                    self.add_history(t, mark)
                if 0 <= t <= horizon:
                    path.append(t if self.p==1 else (t, mark+1))
                    n += 1

                # Update points in the support of the kernel
                for j in range(self.p):
                    interacting_points[j] = interacting_points[j][t-self.approx_support <= interacting_points[j]]
                interacting_points[mark] = np.r_[interacting_points[mark], t]
                N_interacting_points = np.array([arr.size for arr in interacting_points])
                upperbounds = self.mu + self.kernel_bound @ N_interacting_points
                upperbound = upperbounds.sum()
                if 0 <= t <= horizon:
                    upperlist.append(upperbounds)
        return (path, upperlist) if bounds else path

class MultivariateKernelHawkes():
    def __init__(self, gamma=1, support=np.inf, reg=0, link='logit', link_param=100, num_int=1000,
                 bfgs_iter=None, interactions=True, simple=False):
        self.times = None
        self.p = None
        self.diff_times = None
        self.mask_diff_times = None
        self.T = None
        self.N = None
        self.ncol_submatrix = None
        self.times_int = None
        self.mu = None
        self.weights = None
        self.intercepts = None
        self.param = None
        self.gamma = gamma
        self.support = support
        self.reg = reg
        self.link_name = link
        self.link = None
        self.der_link = None
        self.link_param = link_param
        self.num_int = num_int
        self.res_optim = None
        self.bfgs_iter = bfgs_iter
        self.interactions = interactions
        self.simple = simple
        self.K = None
        self.Q = None
        self.B = None
        self.E = None
        self.rescale = None
        self.kernel_time = None
        self.optimize_time = None
        self.fit_time = None

    def copy(self, kernel=True):
        model = MultivariateKernelHawkes()
        model.times = None if self.times is None else [row.copy() for row in self.times]  # List of lists
        model.p = self.p  # Integer
        model.diff_times = None if self.diff_times is None else [[arr.copy() for arr in row] for row in self.diff_times]  # List of lists of arrays
        model.mask_diff_times = None if self.mask_diff_times is None else [[arr.copy() for arr in row] for row in self.mask_diff_times]  # List of lists of arrays
        model.T = self.T  # Float
        model.N = None if self.N is None else self.N.copy()  # List of integers
        model.ncol_submatrix = None if self.ncol_submatrix is None else self.ncol_submatrix.copy()  # List of integers
        model.times_int = None if self.times_int is None else [arr.copy() for arr in self.times_int]  # List of arrays
        model.mu = None if self.mu is None else self.mu.copy()  # Array
        model.weights = None if self.weights is None else [arr.copy() for arr in self.weights]  # List of arrays
        model.intercepts = None if self.intercepts is None else [arr.copy() for arr in self.intercepts]  # List of arrays
        model.param = None if self.param is None else self.param.copy()  # Array
        model.gamma = self.gamma  # Float
        model.support = self.support  # Float
        model.reg = self.reg  # Float
        model.link_name = self.link_name  # String
        model.link = self.link  # Function
        model.der_link = self.der_link  # Function
        model.link_param = self.link_param  # Float
        model.num_int = self.num_int  # Float
        model.res_optim = self.res_optim  # ?
        model.bfgs_iter = self.bfgs_iter  # Integer
        model.interactions = self.interactions  # Boolean
        model.simple = self.simple  # Boolean
        model.K = None if (self.K is None or not kernel) else [arr.copy() for arr in self.K]  # List of arrays
        model.Q = None if (self.Q is None or not kernel) else [arr.copy() for arr in self.Q]  # List of arrays
        model.B = None if (self.B is None or not kernel) else [arr.copy() for arr in self.B]  # List of arrays
        model.E = None if (self.E is None or not kernel) else [arr.copy() for arr in self.E]  # List of arrays
        model.rescale = self.rescale  # Float
        model.kernel_time = None if self.kernel_time is None else self.kernel_time.copy()  # Numpy array
        model.optimize_time = None if self.optimize_time is None else self.optimize_time.copy()  # Numpy array
        model.fit_time = None if self.fit_time is None else self.fit_time.copy()  # Numpy array
        return model

    def set_reg(self, reg=None):
        self.reg = reg

    def set_link_param(self, link_param=None):
        self.link_param = link_param

    def fit(self, times, compute_kernel=True, verbose=False):
        # times: list of times or list of (time, mark), mark=1, ..., p
        # compute_kernel: set to false to retrain a model with another regularization coefficient only
        self.fit_time = time_tic()
        if compute_kernel or self.K is None:
            # convert times to list of list of times
            if hasattr(times[0], '__iter__'):
                self.p = int(np.array(times)[:, 1].max())  # Number of subprocesses
                times = [[t for t, k in times if k==j] for j in range(1, self.p+1)]
            else:
                self.p = 1
                times = [times]
            self.times = [np.reshape(row, (len(row), 1)) for row in times]
            self.N = [row.size for row in self.times]
            self.ncol_submatrix = [self.N[j] + 1 for j in range(self.p)]  # Number of columns in each kernel submatrix
            self.diff_times = []
            if verbose:
                print("Compute diff times")
            self.mask_diff_times = []
            for row in self.times:
                u = row.squeeze()
                self.diff_times.append([u - base for base in self.times])
                self.mask_diff_times.append([np.logical_and(t > 0, t <= self.support) for t in self.diff_times[-1]])

            self.T = np.max([row.max() for row in self.times])
            self.times_int = [np.linspace(0, self.T, num=self.num_int)] * self.p
            if verbose:
                print("Compute K")
                K_times = time_tic()
            self.kernel_time = time_tic()
            self.K, self.E = self.kernel_matrix(verbose=verbose)
            if verbose:
                K_times = time_toc(K_times)
                print(f"Compute K (end, {K_times/60} min)")
            if verbose:
                print("Compute Q")
                Q_times = time_tic()
            self.Q, self.B = self.kernel_matrix(self.times_int, verbose=verbose)
            self.kernel_time = time_toc(self.kernel_time)
            if verbose:
                Q_times = time_toc(Q_times)
                print(f"Compute Q (end, {Q_times/60} min)")

            if self.link_name == 'logit':
                self.link = lambda x: logit(x, self.link_param)
                self.der_link = lambda x: der_logit(x, self.link_param)
            else:
                raise ValueError('Unknown link')
        else:
            self.kernel_time = np.zeros_like(self.fit_time)

        n = self.p * (sum(self.N) + self.p)  # Total number of kernel parameters
        assert n == self.p * np.sum(self.ncol_submatrix)
        assert n == sum([len(self.alpha_index(j)) for j in range(self.p)])

        if verbose:
            print(f"Minimize ({self.p+n+self.p**2} params)")
        self.optimize_time = time_tic()
        init = np.r_[np.ones(self.p), np.ones(n)*1e-3, np.ones(self.p**2)] if self.param is None else self.param
        res = minimize(self.negloglik, x0=init, jac=True,
                       bounds=[(0, None)] * self.p + [(None, None)] * (n + self.p**2),
                       options=dict(maxiter=self.bfgs_iter) if self.bfgs_iter else None)
        self.optimize_time = time_toc(self.optimize_time)
        if verbose:
            print(f"Minimize (end, {self.optimize_time/60} min)")
        self.res_optim = res
        self.param = res.x
        self.mu = self.param[:self.p]
        self.weights = [self.param[self.alpha_index(j)] for j in range(self.p)]
        self.intercepts = [self.param[self.intercept_index(j)] for j in range(self.p)]
        self.fit_time = time_toc(self.fit_time)
        return self

    def kernel_matrix(self, t=None, history=None, verbose=False):
        training = (t is None and history is None)
        history = self.times if history is None else [np.reshape(row, (-1, 1)) for row in history]
        if training:
            diff_eval = self.diff_times
            mask_diff_eval = self.mask_diff_times
        else:
            # t: list of list of times
            if not hasattr(t[0], 'squeeze'):
                t = [np.array(row) for row in t]
            diff_eval = []
            mask_diff_eval = []
            for row in t:
                u = row.squeeze()
                diff_eval.append([u - base for base in history])
                mask_diff_eval.append([np.logical_and(dt > 0, dt <= self.support) for dt in diff_eval[-1]])
        K = []
        for k in range(self.p):
            K_k = []
            for l in range(self.p):
                if verbose:
                    print(f" Dim {(k, l)} ({diff_eval[k][l].shape[1]+1} rows)")
                    print(" |", " "*((diff_eval[k][l].shape[1]+1)//20), "|\n  ", end="")
                if k==l or self.interactions:
                    # First row is only for training matrix (computation of RKHS norms)
                    K_kl = np.empty((diff_eval[k][l].shape[1]+1, self.ncol_submatrix[k]))
                    min_T_hist = np.fmin(self.T - history[l].squeeze(), self.support)
                    for i in range(K_kl.shape[0]-1):
                        if verbose and i % 20 == 1:
                            print("-", end="")
                        K_kl[i+1, 0] = (self.erf(min_T_hist - diff_eval[k][l][mask_diff_eval[k][l][:, i], i][:, np.newaxis]) \
                                        + self.erf(diff_eval[k][l][mask_diff_eval[k][l][:, i], i][:, np.newaxis])).sum()
                        for j in range(K_kl.shape[1]-1):
                            K_kl[i+1, j+1] = np.exp(-self.gamma * (diff_eval[k][l][mask_diff_eval[k][l][:, i], i][:, np.newaxis] \
                                                                - self.diff_times[k][l][self.mask_diff_times[k][l][:, j], j])**2).sum()
                    if training:  # First row is only for training matrix (computation of RKHS norms)
                        K_kl[0] = K_kl[:, 0]
                        K_kl[0, 0] = (self.erf_primitive(min_T_hist) * 2 -
                                      self.erf_primitive(min_T_hist - min_T_hist[:, np.newaxis])).sum()
                    else:
                        K_kl[0] = 0
                    if verbose:
                        print()
                    # Rescale for numerical optimization
                    if self.rescale is None and training and k==l==0:
                        self.rescale = np.sqrt(K_kl[0, 0])
                        if verbose:
                            print(" Set rescale factor to", self.rescale)
                    elif self.rescale is None:
                        raise ValueError("Training kernel matrix should be computed before evaluation kernel matrices")
                    K_kl[0] /= self.rescale
                    K_kl[:, 0] /= self.rescale
                    # Removes the integral component r_\ell
                    if self.simple:
                        K_kl[:, 0] = 0
                        K_kl[0, :] = 0
                else:
                    K_kl = np.zeros((diff_eval[k][l].shape[1]+1, self.ncol_submatrix[k]))
                if K_kl.shape[0]==1:
                    K_kl = K_kl.squeeze()
                K_k.append(K_kl if training else K_kl[1:])  # First row is only for training matrix (computation of RKHS norms)
            K.append(np.concatenate(K_k, axis=1))
        # Intercept matrix
        E = [np.concatenate([mask_diff_eval[j][l].sum(axis=0)[:, np.newaxis] if j==l or self.interactions else \
                             np.zeros((diff_eval[j][l].shape[1], 1)) for l in range(self.p)], axis=1)
             for j in range(self.p)]
        return K, E

    def negloglik(self, param):
        # n = (len(param)-self.p) // self.p
        val = 0.
        grad = np.zeros_like(param)
        for j in range(self.p):
            mu = param[j]
            alpha = param[self.alpha_index(j)]
            intercept = param[self.intercept_index(j)]
            prodK = []
            ncol = self.ncol_submatrix[j]
            for l in range(self.p):
                prodK.append(self.K[j][:, l*ncol:(l+1)*ncol] @ alpha[l*ncol:(l+1)*ncol])
            underlying_comp = mu + self.Q[j] @ alpha + self.B[j] @ intercept
            underlying_int = mu + np.sum(prodK, axis=0)[1:] + self.E[j] @ intercept
            link_underlying_int = self.link(underlying_int)
            # Value
            compensator = self.T * np.mean(self.link(underlying_comp))
            intensities = np.log(link_underlying_int).sum()
            prodKconcat = np.concatenate(prodK, axis=0)
            reg = alpha @ prodKconcat
            val += compensator - intensities + self.reg/2 * reg
            # Gradient
            compensator_grad = self.der_link(underlying_comp)
            intensities_grad = self.der_link(underlying_int) / link_underlying_int
            # Gradient wrt mu
            grad[j] = self.T * compensator_grad.mean() - intensities_grad.sum()
            # Gradient wrt alpha
            grad[self.alpha_index(j)] = (self.T / self.Q[j].shape[0] * compensator_grad @ self.Q[j]
                                        - intensities_grad @ self.K[j][1:] + self.reg * prodKconcat)
            # Gradient wrt intercept
            grad[self.intercept_index(j)] = (self.T / self.Q[j].shape[0] * compensator_grad @ self.B[j]
                                             - intensities_grad @ self.E[j])
        # Return value, gradient
        return val, grad

    def underlying(self, t, history=None):
        history = self.times if history is None else history
        t = t if hasattr(t, '__iter__') else [t]
        t = t if hasattr(t[0], '__iter__') else [t] * self.p
        K, E = self.kernel_matrix(t, history)
        res = [self.mu[j] + K[j] @ self.weights[j] + E[j] @ self.intercepts[j] for j in range(self.p)]
        return (res[0] if len(res[0])>1 else res[0][0]) if self.p==1 else res

    def intensity(self, t, history=None):
        # Float or list or list of lists
        intensities = self.underlying(t, history)
        is_list_of_list = hasattr(intensities, '__iter__') and hasattr(intensities[0], '__iter__')
        return [relu(int) for int in intensities] if is_list_of_list else relu(intensities)

    def intensity_link(self, t, history=None):
        # Float or list or list of lists
        intensities = self.underlying(t, history)
        is_list_of_list = hasattr(intensities, '__iter__') and hasattr(intensities[0], '__iter__')
        return [self.link(int) for int in intensities] if is_list_of_list else self.link(intensities)

    def kernel(self, t):
        t = t if hasattr(t, '__iter__') else [t]
        t = t if hasattr(t[0], '__iter__') else [t] * self.p
        K, E = self.kernel_matrix(t, [[0]] * self.p)
        res = []
        for j in range(self.p):
            res.append([])
            ncol = self.ncol_submatrix[j]
            for l in range(self.p):
                subalpha = self.weights[j][l*ncol:(l+1)*ncol]
                subintercept = self.intercepts[j][l]
                res[-1].append(K[j][:, l*ncol:(l+1)*ncol] @ subalpha + E[j][:, l] * subintercept)
        return (res[0][0] if len(res[0][0])>1 else res[0][0][0]) if self.p==1 else res

    def scores(self, times):
        """Log-likelihood values of self.param with respect to times"""
        # times: list of times or list of (time, mark), mark=1, ..., p
        # convert times to list of list of times
        if hasattr(times[0], '__iter__'):
            times = [[t for t, k in times if k == j] for j in range(1, self.p + 1)]
        else:
            times = [times]
        T = np.max([np.max(row) for row in times])
        times_int = np.linspace(0, T, num=self.num_int)
        intensities = self.intensity_link(times, history=times)
        intensities_int = self.intensity_link(times_int, history=times)
        # log(intensities) - compensator
        logliks = np.array([np.log(intensities[j]).sum() - T * np.mean(intensities_int[j]) for j in range(self.p)])
        return logliks

    def score(self, times):
        """Log-likelihood of self.param with respect to times"""
        return self.scores(times).sum()

    def alpha_index(self, j):
        c = self.p  # c first parameters are baselines
        return range(c + self.p * sum(self.ncol_submatrix[:j]), c + self.p * sum(self.ncol_submatrix[:j+1]))

    def intercept_index(self, j):
        c = self.p + self.p * sum(self.ncol_submatrix)  # First parameters are baselines and kernel parameters
        return range(c + self.p * j, c + self.p * (j + 1))

    def erf(self, x):
        a = np.sqrt(self.gamma)
        return np.sqrt(np.pi) / 2 * erf(a * x) / a

    def erf_primitive(self, x):
        return x * self.erf(x) + ( np.exp(-self.gamma * x**2) - 1) / (2 * self.gamma)

class MultivariateBasisHawkes(MultivariateKernelHawkes):
    def __init__(self, basis='gaussian', num_basis=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basis_name = basis.lower() if isinstance(basis, str) else 'custom'
        self.basis = None if isinstance(basis, str) else basis
        self.num_basis = num_basis
        self.time_basis = None

    def copy(self, kernel=True):
        model = MultivariateBasisHawkes()
        model.times = None if self.times is None else [row.copy() for row in self.times]  # List of lists
        model.p = self.p  # Integer
        model.diff_times = None if self.diff_times is None else [[arr.copy() for arr in row] for row in self.diff_times]  # List of lists of arrays
        model.mask_diff_times = None if self.mask_diff_times is None else [[arr.copy() for arr in row] for row in self.mask_diff_times]  # List of lists of arrays
        model.T = self.T  # Float
        model.N = None if self.N is None else self.N.copy()  # List of integers
        model.ncol_submatrix = None if self.ncol_submatrix is None else self.ncol_submatrix.copy()  # List of integers
        model.times_int = None if self.times_int is None else [arr.copy() for arr in self.times_int]  # List of arrays
        model.mu = None if self.mu is None else self.mu.copy()  # Array
        model.weights = None if self.weights is None else [arr.copy() for arr in self.weights]  # List of arrays
        model.intercepts = None if self.intercepts is None else [arr.copy() for arr in self.intercepts]  # List of arrays
        model.param = None if self.param is None else self.param.copy()  # Array
        model.gamma = self.gamma  # Float
        model.support = self.support  # Float
        model.reg = self.reg  # Float
        model.link_name = self.link_name  # String
        model.link = self.link  # Function
        model.der_link = self.der_link  # Function
        model.link_param = self.link_param  # Float
        model.num_int = self.num_int  # Float
        model.res_optim = self.res_optim  # ?
        model.bfgs_iter = self.bfgs_iter  # Integer
        model.interactions = self.interactions  # Boolean
        model.simple = self.simple  # Boolean
        model.K = None if (self.K is None or not kernel) else [arr.copy() for arr in self.K]  # List of arrays
        model.Q = None if (self.Q is None or not kernel) else [arr.copy() for arr in self.Q]  # List of arrays
        model.B = None if (self.B is None or not kernel) else [arr.copy() for arr in self.B]  # List of arrays
        model.E = None if (self.E is None or not kernel) else [arr.copy() for arr in self.E]  # List of arrays
        model.rescale = self.rescale  # Float
        model.kernel_time = None if self.kernel_time is None else self.kernel_time.copy()  # Numpy array
        model.optimize_time = None if self.optimize_time is None else self.optimize_time.copy()  # Numpy array
        model.fit_time = None if self.fit_time is None else self.fit_time.copy()  # Numpy array
        model.basis_name = self.basis_name  # String
        model.basis = self.basis  # Lambda function
        model.num_basis = self.num_basis  # Integer
        model.time_basis = self.time_basis.copy()   # Arange
        return model

    def fit(self, times, compute_kernel=True, verbose=False):
        # times: list of times or list of (time, mark), mark=1, ..., p
        # compute_kernel: set to false to retrain a model with another regularization coefficient only
        self.fit_time = time_tic()
        if compute_kernel or self.K is None:
            # convert times to list of list of times
            if hasattr(times[0], '__iter__'):
                self.p = int(np.array(times)[:, 1].max())  # Number of subprocesses
                times = [[t for t, k in times if k == j] for j in range(1, self.p + 1)]
            else:
                self.p = 1
                times = [times]
            self.times = [np.reshape(row, (len(row), 1)) for row in times]
            self.N = [row.size for row in self.times]
            self.ncol_submatrix = [self.num_basis for j in range(self.p)]  # Number of columns in each kernel submatrix
            self.diff_times = []
            if verbose:
                print("Compute diff times")
            self.mask_diff_times = []
            for row in self.times:
                u = row.squeeze()
                self.diff_times.append([u - base for base in self.times])
                self.mask_diff_times.append([np.logical_and(t > 0, t <= self.support) for t in self.diff_times[-1]])
            self.T = np.max([row.max() for row in self.times])
            self.times_int = [np.linspace(0, self.T, num=self.num_int)] * self.p

            if self.support != np.infty and self.basis_name != 'bernstein':
                self.time_basis = np.linspace(0, self.support, num=self.num_basis)
            elif self.basis_name != 'bernstein':
                raise ValueError("support cannot be infinity")

            if self.basis_name == 'bernstein':
                # R. Lemonnier and N. Vayatis. Nonparametric Markovian Learning of Triggering Kernels for Mutu-
                # ally Exciting and Mutually Inhibiting Multivariate Hawkes Processes. In T. Calders, F. Esposito,
                # E. Hüllermeier, and R. Meo, editors, Machine Learning and Knowledge Discovery in Databases,
                # pages 161–176, Berlin, Heidelberg, 2014. Springer.
                self.basis = lambda x,k: np.exp(-self.gamma * x * k)
                self.time_basis = np.arange(1, self.num_basis+1)
                self.support = np.infty
            elif self.basis_name == 'gaussian':
                # H. Xu, M. Farajtabar, and H. Zha. Learning Granger Causality for Hawkes Processes. In Proceedings
                # of The 33rd International Conference on Machine Learning, pages 1717–1726. PMLR, 2016.
                self.basis = lambda x,y: np.exp(-self.gamma * (x - y)**2)
            elif not hasattr(self.basis, '__call__'):
                raise ValueError("basis should be 'bernstein', 'gaussian' or a callable (float,float) -> float")

            if self.link_name == 'logit':
                self.link = lambda x: logit(x, self.link_param)
                self.der_link = lambda x: der_logit(x, self.link_param)
            else:
                raise ValueError('Unknown link')

            if verbose:
                print("Compute K")
                K_times = time_tic()
            self.kernel_time = time_tic()
            self.K, self.E = self.kernel_matrix(verbose=verbose)
            if verbose:
                K_times = time_toc(K_times)
                print(f"Compute K (end, {K_times / 60} min)")
            if verbose:
                print("Compute Q")
                Q_times = time_tic()
            self.Q, self.B = self.kernel_matrix(self.times_int, verbose=verbose)
            self.kernel_time = time_toc(self.kernel_time)
            if verbose:
                Q_times = time_toc(Q_times)
                print(f"Compute Q (end, {Q_times / 60} min)")

        n = self.p * self.p * self.num_basis  # Total number of kernel parameters
        assert n == self.p * np.sum(self.ncol_submatrix)
        assert n == sum([len(self.alpha_index(j)) for j in range(self.p)])

        if self.basis_name == 'gaussian':
            bounds = [(0, None)] * (self.p + n)
        else:
            bounds = [(0, None)] * self.p + [(None, None)] * n

        if verbose:
            print(f"Minimize ({self.p + n + self.p ** 2} params)")
        self.optimize_time = time_tic()
        init = np.r_[np.ones(self.p), np.ones(n) * 1e-3] if self.param is None else self.param
        res = minimize(self.negloglik, x0=init, jac=True,
                       bounds=bounds,
                       options=dict(maxiter=self.bfgs_iter) if self.bfgs_iter else None)
        self.optimize_time = time_toc(self.optimize_time)
        if verbose:
            print(f"Minimize (end, {self.optimize_time/60} min)")
        self.res_optim = res
        self.param = res.x
        self.mu = self.param[:self.p]
        self.weights = [self.param[self.alpha_index(j)] for j in range(self.p)]
        self.intercepts = [np.zeros(self.p) for _ in range(self.p)]
        self.fit_time = time_toc(self.fit_time)
        return self

    def kernel_matrix(self, t=None, history=None, verbose=False):
        training = (t is None and history is None)
        history = self.times if history is None else [np.reshape(row, (-1, 1)) for row in history]
        if training:
            diff_eval = self.diff_times
            mask_diff_eval = self.mask_diff_times
        else:
            # t: list of list of times
            if not hasattr(t[0], 'squeeze'):
                t = [np.array(row) for row in t]
            diff_eval = []
            mask_diff_eval = []
            for row in t:
                u = row.squeeze()
                diff_eval.append([u - base for base in history])
                mask_diff_eval.append([np.logical_and(dt > 0, dt <= self.support) for dt in diff_eval[-1]])
        K = []
        for k in range(self.p):
            K_k = []
            for l in range(self.p):
                if verbose:
                    print(f" Dim {(k, l)} ({diff_eval[k][l].shape[1]} rows)")
                    print(" |", " "*(diff_eval[k][l].shape[1]//20), "|\n  ", end="")
                if k==l or self.interactions:
                    # First row is only for training matrix (computation of RKHS norms)
                    K_kl = np.empty((diff_eval[k][l].shape[1], self.ncol_submatrix[k]))
                    for i in range(K_kl.shape[0]):
                        if verbose and i % 20 == 1:
                            print("-", end="")
                        for j in range(K_kl.shape[1]):
                            K_kl[i, j] = self.basis(diff_eval[k][l][mask_diff_eval[k][l][:, i], i],
                                                    self.time_basis[j]).sum()
                    if verbose:
                        print()
                else:
                    K_kl = np.zeros((diff_eval[k][l].shape[1], self.ncol_submatrix[k]))
                if K_kl.shape[0]==1:
                    K_kl = K_kl.squeeze()
                K_k.append(K_kl)
            K.append(np.concatenate(K_k, axis=1))
        # Intercept matrix
        E = [np.concatenate([np.zeros((diff_eval[j][l].shape[1], 1)) for l in range(self.p)], axis=1)
             for j in range(self.p)]
        return K, E

    def negloglik(self, param):
        val = 0.
        grad = np.zeros_like(param)
        for j in range(self.p):
            mu = param[j]
            alpha = param[self.alpha_index(j)]
            underlying_comp = mu + self.Q[j] @ alpha
            underlying_int = mu + self.K[j] @ alpha
            link_underlying_int = self.link(underlying_int)
            # Value
            compensator = self.T * np.mean(underlying_comp)
            intensities = np.log(link_underlying_int).sum()
            reg = alpha @ alpha
            val += compensator - intensities + self.reg / 2 * reg
            # Gradient
            compensator_grad = np.ones_like(underlying_comp)
            intensities_grad = self.der_link(underlying_int) / link_underlying_int
            # Gradient wrt mu
            grad[j] = self.T * compensator_grad.mean() - intensities_grad.sum()
            # Gradient wrt alpha
            grad[self.alpha_index(j)] = (self.T / self.Q[j].shape[0] * compensator_grad @ self.Q[j]
                                         - intensities_grad @ self.K[j] + self.reg * alpha)
        # Return value, gradient
        return val, grad


if __name__=='__main__':
    pass
