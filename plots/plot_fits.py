import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from tabcorr import database
from scipy.interpolate import interp1d

import jerboa

for assembly_bias_mock, assembly_bias_model in zip(
        [True, False, True], [True, True, False]):

    fig, axs = plt.subplots(
        figsize=(7, 2), ncols=3, nrows=2, height_ratios=[3.5, 1.0],
        sharex='col')

    config = database.configuration('efficient')
    obs = jerboa.observations('ggl', assembly_bias_mock)
    errorbar_kwargs = dict(fmt='.', capsize=2, ms=3, lw=1, capthick=1)

    for tpcf, color, (ax1, ax2), bins, exp, offset, label in zip(
            ['wp', 'ds', 'xi0', 'xi2', 'xi4'],
            ['royalblue', 'tomato', 'seagreen', 'darkviolet', 'orange'],
            [axs[:, 0], axs[:, 1], axs[:, 2], axs[:, 2], axs[:, 2]],
            ['rp_wp_bins', 'rp_ds_bins', 's_bins', 's_bins', 's_bins'],
            [1, 1, 1.5, 1.5, 1.5],
            [1, 1, 1, np.exp(0.1), np.exp(0.2)],
            [r"$w_\mathrm{p}$", r"$\Delta\Sigma$", r"$\xi_0$", r"$\xi_2$",
             r"$\xi_4$"]):

        stats = jerboa.statistics('ggl' if tpcf in ['wp', 'ds'] else 'rsd',
                                  assembly_bias_mock, assembly_bias_model)
        stats = dict(stats[np.argmax(stats['log L_max'])])

        x = np.sqrt(config[bins][1:] * config[bins][:-1])
        y = x**exp * stats[tpcf]
        x_plot = np.geomspace(x[0], x[-1], 10000)
        c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(mc.cnames[color])))
        # https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
        ax1.plot(x_plot * offset, interp1d(x, y, kind='cubic')(x_plot),
                 color=colorsys.hls_to_rgb(c[0], 1 - 0.5 * (1 - c[1]), c[2]),
                 zorder=0)
        ax1.errorbar(x * offset, x**exp * obs[tpcf],
                     yerr=x * obs[f'{tpcf}_err'],
                     color=color, zorder=1, label=label, **errorbar_kwargs)
        ax2.errorbar(x * offset,
                     (obs[tpcf] - stats[tpcf]) / obs[f'{tpcf}_err'],
                     yerr=1, color=color, zorder=1, **errorbar_kwargs)

    for ax in axs[0]:
        ax.set_xscale('log')

    for ax in axs[1]:
        ax.set_ylim(-3.5, +3.5)
        ax.set_yticks([-3, 0, +3])
        ax.set_yticks([], minor=True)
        ax.axhline(0, ls='--', color='black', zorder=-1)
        ax.set_ylabel(r'$\delta$')

    axs[0, 2].legend(loc='lower left', frameon=False, ncols=3,
                     handletextpad=-0.5, columnspacing=-0.5, borderaxespad=0)

    axs[0, 0].set_ylabel(r"$r_\mathrm{p} w_\mathrm{p} \, [h^{-2} \, "
                         r"\mathrm{Mpc}^2]$")
    axs[0, 1].set_ylabel(r"$r_\mathrm{p} \Delta\Sigma \, [10^6 \, M_\odot "
                         r"/ \mathrm{pc}]$")
    axs[0, 2].set_ylabel(r"$s^{1.5} \xi \, [h^{-1.5}\mathrm{Mpc}^{1.5}]$")
    for ax in [axs[1, 0], axs[1, 1]]:
        ax.set_xlabel(r"$r_\mathrm{p} \, [h^{-1} \, \mathrm{Mpc}]$")
    axs[1, 2].set_xlabel(r"$s \, [h^{-1} \, \mathrm{Mpc}]$")

    # limit y-range in case the model prediction on small scales is high
    axs[0, 0].set_ylim(ymax=240)

    for (ax1, ax2), tpcf_min in zip(
            [axs[:, 0], axs[:, 1], axs[:, 2]],
            [jerboa.WP_MIN, jerboa.DS_MIN, jerboa.XI_MIN]):
        for ax in [ax1, ax2]:
            ax.axvspan(0, tpcf_min, color='lightgrey', zorder=-2)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(hspace=0, wspace=0.25)
    fname_base = (f"best_fit{'_mock_gab' if assembly_bias_mock else ''}"
                  f"{'_model_gab' if assembly_bias_model else ''}")
    plt.savefig(f'{fname_base}.pdf')
    plt.savefig(f'{fname_base}.png', dpi=300)
    plt.close()
