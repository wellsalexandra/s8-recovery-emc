import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt

from astropy.table import Table
from nautilus import Prior, Sampler
from nautilus.bounds.basic import Ellipsoid
from scipy.special import logsumexp
import weightedstats as ws


def skew_normal_logpdf(mu, cov, skew, x):

    import numpy as np
    from scipy.stats import multivariate_normal, norm

    mu = np.atleast_1d(mu)
    cov = np.atleast_2d(cov)
    skew = np.atleast_1d(skew)

    # eq. (1.3)
    delta = skew / np.sqrt(1 + skew**2)
    sigma = np.sqrt(np.diagonal(cov))
    gamma = skew * sigma
    pre = np.linalg.inv(cov)

    # eq. (2.4)
    delta_c = np.sqrt(1 - delta**2)
    alpha = (np.dot(gamma, pre) / delta_c /
             np.sqrt(1 + np.dot(np.dot(gamma, pre), gamma)))

    # eq. (2.5)
    omega = np.outer(delta_c, delta_c) * (cov + np.outer(gamma, gamma))

    log_pdf = multivariate_normal(
        cov=omega, mean=mu, allow_singular=True).logpdf(x)
    log_cdf = norm.logcdf(np.sum(alpha * (x - mu), axis=-1))

    return np.log(2.0) + log_pdf + log_cdf


def convert_param_dict(param_dict):

    import numpy as np
    
    mu = np.array([param_dict['mu_s8'], param_dict['mu_Om'],
                   param_dict['mu_w0']])
    sigma = np.array([param_dict['sigma_tilde_s8'],
                      param_dict['sigma_tilde_Om'],
                      param_dict['sigma_tilde_w0']])
    sigma = sigma / (1 - sigma)
    skew = np.array([param_dict['delta_s8'], 0, 0])
    cov = np.outer(sigma, sigma)
    cov[0, 1] *= param_dict['r_s8_Om']
    cov[1, 0] *= param_dict['r_s8_Om']
    cov[0, 2] *= param_dict['r_s8_w0']
    cov[2, 0] *= param_dict['r_s8_w0']
    cov[1, 2] *= param_dict['r_s8_Om'] * param_dict['r_s8_w0']
    cov[2, 1] *= param_dict['r_s8_Om'] * param_dict['r_s8_w0']
    skew = skew / np.sqrt(1 - skew**2)
    return mu, cov, skew


def likelihood(cosmo, log_l_sim, param_dict):

    import numpy as np
    from scipy.stats import multivariate_normal, norm

    mu, cov, skew = convert_param_dict(param_dict)
    log_l_mod = skew_normal_logpdf(mu, cov, skew, cosmo)
    if not np.all(np.isfinite(log_l_mod)):
        return -1e99
    log_l_mod = log_l_mod - np.average(log_l_mod - log_l_sim)

    return np.sum(norm.logpdf(
        log_l_sim, loc=log_l_mod, scale=param_dict['log_l_err']))

# %%

table = Table.read('cosmo_logl_wp_ds_a_b_with_aDS.csv')
table.remove_row(23)
cosmo = np.vstack([table['s8'], table['Om'], table['w0']]).T

prior = Prior()
prior.add_parameter('mu_s8', (np.amin(cosmo[:, 0]), np.amax(cosmo[:, 0])))
prior.add_parameter('mu_Om', (np.amin(cosmo[:, 1]), np.amax(cosmo[:, 1])))
prior.add_parameter('mu_w0', (np.amin(cosmo[:, 2]), np.amax(cosmo[:, 2])))
prior.add_parameter('sigma_tilde_s8', (1e-5, 1))
prior.add_parameter('sigma_tilde_Om', (1e-5, 1))
prior.add_parameter('sigma_tilde_w0', (1e-5, 1))
prior.add_parameter('r_s8_Om', (-1, +1))
prior.add_parameter('r_s8_w0', (-1, +1))
prior.add_parameter('delta_s8', (-1, +1))
prior.add_parameter('log_l_err', (0, 100)) ### Constant error


def sample_points(sampler, cosmo, n_points=100000, n_dist=100):

    ell = Ellipsoid.compute(cosmo, enlarge_per_dim=1.0)
    points = ell.sample(n_points)
    w = np.zeros(n_points)

    dists, log_p = sampler.posterior(return_as_dict=True)[:2]
    p = np.exp(log_p - logsumexp(log_p))

    for i in np.random.choice(len(p), size=n_dist, p=p):
        param_dict = dict()
        for key in dists.keys():
            param_dict[key] = dists[key][i]
        mu, cov, skew = convert_param_dict(param_dict)
        log_l = skew_normal_logpdf(mu, cov, skew, points)
        w += np.exp(log_l - logsumexp(log_l)) / n_dist

    return points, w



sampler = Sampler(prior, likelihood, 
                        likelihood_args=(cosmo, table['log_l']), pool=4)
sampler.run(verbose=True, discard_exploration=True)

points, w = sample_points(sampler, cosmo)

i = 0
plt.hist(points[:, i], weights=w, bins=100, color='blue', alpha=0.5, histtype='step')
plt.xlabel(r'$S_8$')
print(np.sqrt(np.cov(points[:, i], aweights=w)))
print(ws.weighted_median(points[:, i], weights=w))

plt.axvline(x=0.8267, ymin=0, ymax=50, color='black', label='True Value', linestyle='--')
plt.annotate("$S_8 = 0.8267$", [0.80, 0.055])
plt.xlim(0.71, 0.85)
plt.legend()
plt.show()


