"""
SI Fig 2 — (b) Setup-cost scaling and (c) Mobility-model scaling.

Reproduces the publication PDF `benchmark_combined_bc.pdf`.

Inputs (in ./data/):
  - benchmark_setup_results_4.pkl    : EBSTN v4 vs Optimised TN/ABM setup phases
  - benchmark_mobility_results.pkl   : Mobility-model PDF + Positions timings

Run:
    python plot_si2.py
"""

import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_STEM = os.path.join(HERE, 'benchmark_combined_bc')

mpl.rcParams.update({
    'font.family':        'serif',
    'font.size':          12,
    'axes.labelsize':     12,
    'axes.titlesize':     12,
    'legend.fontsize':    12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.5,
    'ytick.minor.width':  0.5,
    'lines.linewidth':    1.8,
    'figure.dpi':         300,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})


def fit_powerlaw(Ns, ms, k=4):
    k = min(k, len(Ns))
    Ns_f, ms_f = Ns[-k:], ms[-k:]
    logN, logm = np.log(Ns_f), np.log(ms_f)
    slope, intercept, r, _, se = stats.linregress(logN, logm)
    n  = len(Ns_f)
    df = max(n - 2, 1)
    t_crit       = stats.t.ppf(0.975, df=df)
    ci_lo, ci_hi = slope - t_crit * se, slope + t_crit * se
    resid = logm - (intercept + slope * logN)
    s_res = np.sqrt((resid**2).sum() / df) if n > 2 else np.nan
    return dict(slope=slope, intercept=intercept, se=se, r2=r**2,
                ci_lo=ci_lo, ci_hi=ci_hi, s_res=s_res,
                logN_fit=logN, Ns_fit=Ns_f, k=k)


def fit_left_edge(Ns, k):
    n_all = len(Ns)
    if n_all > k:
        return np.exp(0.5 * (np.log(Ns[-(k + 1)]) + np.log(Ns[-k])))
    return Ns[-k]


def draw_panel_b(ax):
    with open(os.path.join(DATA_DIR, 'benchmark_setup_results_4.pkl'), 'rb') as f:
        data = pickle.load(f)

    results     = data['results']
    N_agents    = data['N_agents']
    N_agents_su = data['N_agents_su']

    VARIANTS = {
        'Optimised TN / ABM': ['build', 'initialise', 'agent_lists'],
        'EBSTN v4':           ['build', 'initialise', 'agent_lists', 'S_lists'],
    }
    DISPLAY = {'Optimised TN / ABM': 'Optimised TN/ABM', 'EBSTN v4': 'EBSTN'}
    COLORS  = {'Optimised TN / ABM': '#e08214', 'EBSTN v4': '#1a9850'}
    MARKERS = {'Optimised TN / ABM': 's', 'EBSTN v4': 'o'}

    def key_to_N(key):
        if isinstance(key, int):
            return N_agents[key]
        return N_agents_su[key]

    def get_data(phases):
        Ns, ms = [], []
        for key, phdict in results.items():
            Ns.append(key_to_N(key))
            ms.append(sum(phdict[p] for p in phases))
        order = np.argsort(Ns)
        return np.array(Ns)[order], np.array(ms)[order]

    KLAST = 3
    variant_data = {}
    for vname, phases in VARIANTS.items():
        Ns, ms = get_data(phases)
        fit    = fit_powerlaw(Ns, ms, k=KLAST)
        variant_data[vname] = dict(Ns=Ns, ms=ms, fit=fit)

    def draw_variant(Ns, ms, fit, vname):
        color  = COLORS[vname]
        label  = DISPLAY[vname]
        marker = MARKERS[vname]
        k      = fit['k']
        n_all  = len(Ns)
        if n_all > k:
            ax.loglog(Ns[:-k], ms[:-k], marker, color=color,
                      markersize=5, alpha=0.65, zorder=5,
                      markeredgecolor=color, markeredgewidth=1.0,
                      linestyle='none')
        ax.loglog(Ns[-k:], ms[-k:], marker, color=color,
                  markersize=6.5, zorder=6, label=label,
                  markeredgecolor='white', markeredgewidth=0.6,
                  linestyle='none')
        N_lo_fit = fit_left_edge(Ns, k)
        N_line   = np.logspace(np.log10(N_lo_fit), np.log10(Ns[-1]), 300)
        T_fit    = np.exp(fit['intercept']) * N_line ** fit['slope']
        ax.loglog(N_line, T_fit, '-', color=color, lw=2.2, label='_nolegend_')

    for vname in VARIANTS:
        d = variant_data[vname]
        draw_variant(d['Ns'], d['ms'], d['fit'], vname)

    ref_Ns = variant_data['Optimised TN / ABM']['Ns']
    ref_ms = variant_data['Optimised TN / ABM']['ms']
    all_Ns = np.concatenate([variant_data[v]['Ns'] for v in VARIANTS])
    N_lo   = all_Ns.min() * 0.7
    N_hi   = all_Ns.max() * 1.4
    N_ref  = np.logspace(np.log10(N_lo), np.log10(N_hi), 400)

    N_c1 = np.exp(np.mean(np.log(ref_Ns)))
    T_c1 = np.exp(np.mean(np.log(ref_ms)))
    ax.loglog(N_ref, T_c1 * (N_ref / N_c1) ** 1.0,
              '--', color='#444444', lw=1.3, alpha=0.65,
              label=r'$\mathcal{O}(N^1)$ reference')

    ax.set_xlabel(r'Total number of actors in systems $(N)$')
    ax.set_ylabel('Computation time (ms)')
    ax.set_title('(a) Setup scaling: EBSTN vs Optimised TN/ABM', pad=6)
    ax.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
    ax.grid(True, which='minor', ls=':',  alpha=0.18, lw=0.5)
    ax.minorticks_on()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left',
              framealpha=0.93, edgecolor='#cccccc',
              borderpad=0.7, labelspacing=0.35,
              handlelength=1.8, frameon=False)


def draw_panel_c(ax):
    with open(os.path.join(DATA_DIR, 'benchmark_mobility_results.pkl'), 'rb') as f:
        results = pickle.load(f)

    divs_sorted = sorted(results.keys(), reverse=True)
    Ns     = np.array([results[d]['N']            for d in divs_sorted])
    ms_pdf = np.array([results[d]['t_pdf'].mean() for d in divs_sorted]) * 1e3
    ms_pos = np.array([results[d]['t_pos'].mean() for d in divs_sorted]) * 1e3
    ms_mob = ms_pdf + ms_pos

    KLAST   = 3
    fit_pdf = fit_powerlaw(Ns, ms_mob, k=KLAST)
    k_pdf   = fit_pdf['k']

    COLOR_PDF = '#1a9850'

    if len(Ns) > k_pdf:
        ax.loglog(Ns[:-k_pdf], ms_mob[:-k_pdf], 'o', color=COLOR_PDF,
                  markersize=5, alpha=0.65, zorder=5,
                  markeredgecolor=COLOR_PDF, markeredgewidth=1.0,
                  linestyle='none')
    ax.loglog(Ns[-k_pdf:], ms_mob[-k_pdf:], 'o', color=COLOR_PDF,
              markersize=6.5, zorder=6, label='Mobility model',
              markeredgecolor='white', markeredgewidth=0.6,
              linestyle='none')

    N_lo_fit = fit_left_edge(Ns, k_pdf)
    N_line   = np.logspace(np.log10(N_lo_fit), np.log10(Ns[-1]), 300)
    T_fit    = np.exp(fit_pdf['intercept']) * N_line ** fit_pdf['slope']
    ax.loglog(N_line, T_fit, '-', color=COLOR_PDF, lw=2.2, zorder=5,
              label='_nolegend_')

    N_lo  = Ns.min() * 0.6
    N_hi  = Ns.max() * 1.5
    N_ref = np.logspace(np.log10(N_lo), np.log10(N_hi), 400)

    N_c1 = np.exp(np.mean(np.log(Ns)))
    T_c1 = np.exp(np.mean(np.log(ms_mob)))
    ax.loglog(N_ref, T_c1 * (N_ref / N_c1) ** 1.0,
              '--', color='#444444', lw=1.3, alpha=0.60,
              label=r'$\mathcal{O}(N^1)$ reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Total number of actors in systems $(N)$')
    ax.set_ylabel('Computation time (ms)')
    ax.set_title('(b) Mobility model scaling', pad=6)
    ax.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
    ax.grid(True, which='minor', ls=':',  alpha=0.18, lw=0.5)
    ax.minorticks_on()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left',
              framealpha=0.93, edgecolor='#cccccc',
              borderpad=0.7, labelspacing=0.35,
              handlelength=1.8, frameon=False)


fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(13, 4.8))
draw_panel_b(ax_b)
draw_panel_c(ax_c)
fig.tight_layout(pad=0.5)
fig.savefig(f'{OUT_STEM}.pdf', bbox_inches='tight')
fig.savefig(f'{OUT_STEM}.png', dpi=300, bbox_inches='tight')
print(f'Saved {OUT_STEM}.pdf and {OUT_STEM}.png')
