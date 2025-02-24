# attempt at random corner plot:
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import corner
import matplotlib as mpl

# table = Table.read('sampler_post_wp_ds.csv')
table = Table.read('sampler_post_rsd.csv')
log_w = table['log_w']
w = np.exp(log_w)
points = np.asarray([table['logM1'], table['logM0'], table['alpha'], table['logMmin'], 
                   table['sigma_logM']]).T

labels = [r'$logM_1$', r'$logM_0$', r'$\alpha$', r'$logM_{min}$', r'$\sigma_{logM}$']


x_min, x_max = np.amin(points, axis=0), np.amax(points, axis=0)
x_min -= 0.05 * (x_max - x_min)
x_max += 0.05 * (x_max - x_min)

fig = plt.figure(figsize=(3.33, 3.33))

fig = corner.corner(
    points, weights=w, 
    labels=labels,  
    levels=(0.68, 0.95, 0.99), 
    range=[(x1, x2) for x1, x2 in zip(x_min, x_max)],
    color='royalblue', fill_contours=True,
    plot_datapoints=False, max_n_ticks=4, plot_density=False, labelpad=0.1,
    hist_kwargs=dict(density=True, color='royalblue', lw=0),
    contour_kwargs=dict(linewidths=0), label_kwargs=dict(fontsize=25)
)


axes = np.array(fig.axes).reshape((5, 5))
for i in range(5):
    h, bins = np.histogram(points[:, i], density=True, bins=100, weights=w)
    axes[i, i].plot(0.5 * (bins[1:] + bins[:-1]), h, color='royalblue')


plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# fig.suptitle(r'$w_p+\Delta\Sigma$', fontsize=35)
fig.suptitle(r'RSD', fontsize=35)
plt.show()
