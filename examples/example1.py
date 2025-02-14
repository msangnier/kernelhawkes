import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from kernelhawkes import HawkesProcess, MultivariateKernelHawkes, MultivariateExponentialHawkes, plot_kernels

mpl.style.use('seaborn-v0_8')
np.set_printoptions(precision=2)


# Simulation
fun_diag1 = lambda x: (8*x**2-1)*(x <= 1/2) + np.exp(-2.5*(x-1/2))*(x > 1/2)
fun_diag2 = lambda x: (8*x**2-1)*(x <= 1/2) + np.exp(-(x-1/2))*(x > 1/2)
hawkes_kernels = [
    [fun_diag1, lambda x: np.exp(-10*(x-1)**2), lambda x: -0.6*np.exp(-3*x**2) - 0.4*np.exp(-3*(x-1)**2)],
    [lambda x: 2**(-5*x), fun_diag2, lambda x: -np.exp(-2*(x-3)**2)],
    [lambda x: -np.exp(-5*(x-2)**2), lambda x: (1+np.cos(np.pi*x)) * np.exp(-x)/2, fun_diag1]
]

hp = HawkesProcess(mu=0.05 * np.ones(len(hawkes_kernels)), kernel=hawkes_kernels)
times = hp.simulate(size=500)

# Estimation (exponential model)
model = MultivariateExponentialHawkes().fit(times)
print("Neg-log-lik Exponential model:", model.res_optim.fun)

# Estimation (RKHS)
multi_rkhs = MultivariateKernelHawkes(link_param=100, support=5, gamma=1e0, reg=1e0)
multi_rkhs.fit(times)
print("Neg-log-lik RKHS model:", multi_rkhs.negloglik(multi_rkhs.param)[0])

# Plot
plot_kernels(dict(true=hp, Exponential=model, RKHS=multi_rkhs))
plt.show()
