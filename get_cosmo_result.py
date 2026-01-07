#%%
import sys
pip install numpy pandas matplotlib astroML halotools pytest-astropy nautilus-sampler wquantiles corner
pip install --upgrade --force-reinstall git+https://github.com/johannesulf/TabCorr.git

#%%
import argparse
import corner
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tabcorr import TabCorr
from tabcorr import database
from astropy.table import Table
from nautilus import Prior, Sampler
from nautilus.bounds.basic import Ellipsoid
from scipy.spatial import ConvexHull
from scipy.spatial import Rectangle
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
import wquantiles as wq

# %%
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = 3.33, 2.5
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['backend'] = 'GTK3Agg'
print(mpl.rcParams['backend'].__init__)
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['legend.framealpha'] = 0.1
# mpl.rcParams['legend.edgecolor'] = 0.0
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

#%%
fit = 'rsd' # options: 'wpds' or 'rsd'
mock_type = 'mass' # options: 'mass+conc' or 'mass' 

#%%
table = Table.read('./cosmo_logl/cosmo_logl_'+str(fit)+'_'+str(mock_type)+'.csv')

x = np.vstack([table['S_8'], table['O_m'], table['n_s']]).T
log_l_sim = table['log L_max']
chi_sq = -2 * table['log L_max']

prior = Prior()
prior.add_parameter('mu_s8', (np.amin(x[:, 0]), np.amax(x[:, 0])))
prior.add_parameter('mu_Om', (np.amin(x[:, 1]), np.amax(x[:, 1])))
prior.add_parameter('mu_ns', (np.amin(x[:, 2]), np.amax(x[:, 2])))
prior.add_parameter('log_sigma_s8', (np.log(np.ptp(x[:, 0])) - 5,
                                     np.log(np.ptp(x[:, 0])) + 2.5))
prior.add_parameter('log_sigma_Om', (np.log(np.ptp(x[:, 1])) - 5,
                                     np.log(np.ptp(x[:, 1])) + 2.5))
prior.add_parameter('log_sigma_ns', (np.log(np.ptp(x[:, 2])) - 5,
                                     np.log(np.ptp(x[:, 2])) + 2.5))
prior.add_parameter('r_s8_Om', (-1, +1))
prior.add_parameter('r_s8_ns', (-1, +1))
prior.add_parameter('delta_s8', (-1, +1))
prior.add_parameter('r_sim', (0, 10))

# load mock
mock_measurements = pd.read_csv('./mocks/mock_model1_mass.csv', sep = ',')

config = database.configuration('efficient')
s_bins = np.sqrt(config['s_bins'][1:] * config['s_bins'][:-1])
rp_wp_bins = np.sqrt(config['rp_wp_bins'][1:] * config['rp_wp_bins'][:-1])
rp_ds_bins = np.sqrt(config['rp_ds_bins'][1:] * config['rp_ds_bins'][:-1])

rp_lim_ds = 2.5
rp_lim_xi = 2.5
rp_lim_wp = 1.0

wp_mock = mock_measurements['wp'].values
ds_mock = mock_measurements['ds'].values

all_xi_mock = []
all_xi_mock = all_xi_mock + list(mock_measurements['xi0'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi2'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi4'])
all_xi_mock = pd.Series(all_xi_mock).values

# apply scale
tripled_rp_ave = pd.Series(list(s_bins) + list(s_bins) + list(s_bins))
tripled_rp_mask = tripled_rp_ave > rp_lim_xi

# First cut 14th element from lensing, for rp_ave_ds and ds_mock
ds_mock_cut = ds_mock[:13]

rp_mask_ds = rp_ds_bins > rp_lim_ds
rp_mask_wp = rp_wp_bins > rp_lim_wp

rp_ave_ds = rp_ds_bins[rp_mask_ds]
rp_ave_wp = rp_wp_bins[rp_mask_wp]

wp_mock = wp_mock[rp_mask_wp]
ds_mock = ds_mock_cut[rp_mask_ds]
all_xi_mock = all_xi_mock[tripled_rp_mask]

if(fit == 'rsd'):
    n_data = len(all_xi_mock)
elif(fit == 'wpds'):
    n_data = len(wp_mock) + len(ds_mock)

def skew_normal_logpdf(mu, cov, skew, x):

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
    mu = np.array([param_dict['mu_s8'], param_dict['mu_Om'],
                   param_dict['mu_ns']])
    sigma = np.array([param_dict['log_sigma_s8'],
                      param_dict['log_sigma_Om'],
                      param_dict['log_sigma_ns']])
    sigma = np.exp(sigma)
    skew = np.array([param_dict['delta_s8'], 0, 0])
    cov = np.outer(sigma, sigma)
    cov[0, 1] *= param_dict['r_s8_Om']
    cov[1, 0] *= param_dict['r_s8_Om']
    cov[0, 2] *= param_dict['r_s8_ns']
    cov[2, 0] *= param_dict['r_s8_ns']
    cov[1, 2] *= param_dict['r_s8_Om'] * param_dict['r_s8_ns']
    cov[2, 1] *= param_dict['r_s8_Om'] * param_dict['r_s8_ns']
    skew = skew / np.sqrt(1 - skew**2)
    return mu, cov, skew


#%%
def likelihood(param_dict):

    
    mu, cov, skew = convert_param_dict(param_dict)
    log_l_mod = skew_normal_logpdf(mu, cov, skew, x)
    if not np.all(np.isfinite(log_l_mod)):
        return -1e99
    log_l_err = np.sqrt(param_dict['r_sim']**2 * n_data / 2 +
                        param_dict['r_sim'] * chi_sq)
    log_l_mod = log_l_mod - np.average(log_l_mod - log_l_sim,
                                       weights=log_l_err**(-2))

    return np.sum(norm.logpdf(log_l_sim, loc=log_l_mod, scale=log_l_err))


#%%
sampler = Sampler(prior, likelihood, pool=None)
sampler.run(verbose=True, discard_exploration=True)

# %%

points, log_w, log_l = sampler.posterior(return_as_dict=True)
param_dict = dict()
for key in points.keys():
    param_dict[key] = points[key][np.argmax(log_l)]
mu, cov, skew = convert_param_dict(param_dict)
log_l_mod = skew_normal_logpdf(mu, cov, skew, x)
log_l_err = np.sqrt(param_dict['r_sim']**2 * n_data / 2 +
                    param_dict['r_sim'] * chi_sq)
log_l_mod = log_l_mod - np.average(log_l_mod - log_l_sim,
                                   weights=log_l_err**(-2))

# %%
def sample_points(sampler, cosmo, n_points=10000000, n_dist=300):

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


points, w = sample_points(sampler, x)

#%%

x_min, x_max = np.amin(points, axis=0), np.amax(points, axis=0)
x_min -= 0.05 * (x_max - x_min)
x_max += 0.05 * (x_max - x_min)

table = Table.read('./cosmo_logl/cosmo_logl_'+str(fit)+'_'+str(mock_type)+'.csv')
chi_sq_wpds = -2 * table['log L_max']

fig = plt.figure(figsize=(3.33, 3.33))

fig = corner.corner(
    points, 100, weights=w, color='royalblue', levels=(0.68, 0.95, 0.99),
    hist_kwargs=dict(density=True, color='royalblue', lw=0),
    plot_datapoints=False, max_n_ticks=4, plot_density=False, labelpad=0.1,
    labels=[r'$S_8$', r'$\Omega_{\rm m}$', r'$n_s$'],
    fig=fig, range=[(x1, x2) for x1, x2 in zip(x_min, x_max)],
    contour_kwargs=dict(linewidths=0), fill_contours=True)

axes = np.array(fig.axes).reshape((3, 3))

# Plot the 1-d histograms.
for i in range(3):
    h, bins = np.histogram(points[:, i], density=True, bins=100, weights=w)
    axes[i, i].plot(0.5 * (bins[1:] + bins[:-1]), h, color='royalblue')

# Plot the prior.
for i in range(3):
    h, bins = np.histogram(points[:, i], density=True, bins=100)
    axes[i, i].plot(0.5 * (bins[1:] + bins[:-1]), h, color='grey', ls='--',
                    zorder=-100)
    for k in range(i):
        hull = ConvexHull(points[:, [i, k]])
        axes[i, k].plot(points[hull.vertices, k], points[hull.vertices, i],
                        color='grey', ls='--', zorder=-100)

for i in range(3):
    for k in range(i):
        im = axes[i, k].scatter(
            x[:, k], x[:, i], c=chi_sq_wpds, norm=mpl.colors.LogNorm(), s=5, lw=0,
            zorder=10)

cax = fig.add_axes([0.5, 0.9, 0.4, 0.05])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label(r'$\chi^2$', labelpad=0)

truth = [0.8147 * np.sqrt(0.3089 / 0.3), 0.3089, 0.9667]
for i in range(3):
    axes[i, i].axvline(truth[i], color='tomato', lw=1)
    for k in range(i):
        axes[i, k].scatter(truth[k], truth[i], color='tomato', marker='*',
                           lw=0, zorder=11, s=15)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
if(fit == 'wpds'):
    fig.text(0.74, 0.55, r'$w_p$ + $\Delta\Sigma$', fontsize=14)
elif(fit == 'rsd'):
    fig.text(0.8, 0.55,'RSD', fontsize=14)

plt.show()


#%%

def cosmo_results(param_idx): # param is int: 0 = s8, 1 = om, 2 = ns 
    
    # calc median
    
    median = wq.median(points[:, param_idx], weights=w)
    
    # calc 68 percentile
    
    p_16 = wq.quantile(points[:, param_idx], w, 0.16)
    p_84 = wq.quantile(points[:, param_idx], w, 0.84)
    
    max = p_84 - median
    min = p_16 - median
    
    # calc 1 sigma
    range_68 = max - min
    one_sigma = range_68 / 2
    
    print(str(fit)+ " for " + str(mock_type) + " mock: " + str(median) + " +/- " + str(one_sigma))
    
    # calc sigma difference from truth
    
    truth = [0.8267, 0.3098, 0.9667]
    truth_val = truth[param_idx]

    sigma_diff = (truth_val - median) / one_sigma
    
    print("Delta" + str(sigma_diff))
    
    return [median, one_sigma, sigma_diff]
    
    
#%%
s8_res = cosmo_results(0)
print(" ")
om_res = cosmo_results(1)
print(" ")
ns_res = cosmo_results(2)