.. -*- mode: rst -*-

kernelhawkes
===========

kernelhawkes is a Python library for nonparametric estimation of nonlinear multivariate Hawkes processes, where the interaction functions are assumed to lie in a reproducing kernel Hilbert space (RKHS).

Example
-------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from seaborn import set_theme    
    from kernelhawkes import HawkesProcess, MultivariateKernelHawkes, plot_kernels
    
    set_theme()
    
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
    
    # Estimation
    multi_rkhs = MultivariateKernelHawkes(link_param=100, support=5, gamma=1e0, reg=1e0)
    multi_rkhs.fit(times)
    print("Neg-log-lik:", multi_rkhs.negloglik(multi_rkhs.param)[0])
    
    # Plot
    plot_kernels(dict(true=hp, RKHS=multi_rkhs))

Dependencies
------------

optboosting needs Python >= 3, setuptools and Numpy.

Installation
------------

To install kernelhawkes from pip, type::

    pip install https://github.com/msangnier/kernelhawkes/archive/master.zip

To install kernelhawkes from source, type::

    git clone https://github.com/msangnier/kernelhawkes.git
    cd kernelhawkes
    sudo python setup.py install

Authors
-------

Maxime Sangnier

References
----------

- Nonparametric estimation of Hawkes processes with RKHSs (2025), A. Bonnet and M. Sangnier. AISTATS.
                                                                          
