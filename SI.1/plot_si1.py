"""
SI Fig 1 — Computational scaling: EBSTN vs TN/ABM variants.

Reproduces the publication PDF `benchmark_combined_a.pdf`.

Inputs (in ./data/):
  - benchmark_results_v18.pkl  : main scaling results (EBSTN v4, Real ABM)
  - benchmark_results_v14.pkl  : separate run containing Naive ABM timings

Run:
    python plot_si1.py
"""

import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_STEM = os.path.join(HERE, 'benchmark_combined_a')

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


def draw_panel_a(ax):
    with open(os.path.join(DATA_DIR, 'benchmark_results_v18.pkl'), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'benchmark_results_v14.pkl'), 'rb') as f:
        results_naive = pickle.load(f)

    ABM_DUMB = '⑥v4 Dumb ABM (np.where scan)'
    for key in results_naive:
        if ABM_DUMB in results_naive[key]:
            results.setdefault(key, {})[ABM_DUMB] = results_naive[key][ABM_DUMB]

    EBSTN_V4 = 'Ⓒv4 EBSTN (O(1) S_flat)'
    ABM_REAL = '⑥ Real ABM (dyn. moves)'

    DISPLAY = {EBSTN_V4: 'EBSTN', ABM_REAL: 'Real ABM', ABM_DUMB: 'Naive ABM'}
    COLORS  = {EBSTN_V4: '#1a9850', ABM_REAL: '#e08214', ABM_DUMB: '#d73027'}
    MARKERS = {EBSTN_V4: 'o', ABM_REAL: 's', ABM_DUMB: 'D'}

    def key_to_N(key):
        if isinstance(key, int):
            return int(166549 * 100 / key)
        return int(166549 * int(str(key).lstrip('×')))

    def get_data(vname, div_only=False):
        Ns, ms = [], []
        for key, vdict in results.items():
            if vname not in vdict:
                continue
            if div_only and not isinstance(key, int):
                continue
            Ns.append(key_to_N(key))
            ms.append(vdict[vname]['times'].mean() * 1e3)
        order = np.argsort(Ns)
        return np.array(Ns)[order], np.array(ms)[order]

    KLAST = 3
    ebstn_N,  ebstn_ms  = get_data(EBSTN_V4)
    abm_N,    abm_ms    = get_data(ABM_REAL)
    dumb_N_all, dumb_ms_all = get_data(ABM_DUMB, div_only=True)
    n_dumb = min(3, len(dumb_N_all))
    dumb_N, dumb_ms = dumb_N_all[-n_dumb:], dumb_ms_all[-n_dumb:]

    ebstn_fit = fit_powerlaw(ebstn_N, ebstn_ms, k=KLAST)
    abm_fit   = fit_powerlaw(abm_N,   abm_ms,   k=KLAST) if len(abm_N) > 0 else None
    dumb_fit  = fit_powerlaw(dumb_N,  dumb_ms,  k=KLAST) if len(dumb_N) > 0 else None

    def draw_variant(Ns, ms, fit, vname):
        if len(Ns) == 0 or fit is None:
            return
        color  = COLORS[vname]
        label  = DISPLAY[vname]
        marker = MARKERS.get(vname, 'o')
        k      = fit['k']
        n_all  = len(Ns)
        if n_all > k:
            ax.loglog(Ns[:-k], ms[:-k], marker, color=color,
                      markersize=5, alpha=0.65, zorder=5,
                      markeredgecolor=color, markeredgewidth=1.0)
        if label == 'Real ABM':
            ax.loglog(Ns[-k:], ms[-k:], marker, color=color,
                      markersize=6.5, zorder=6, label='Optimised TN/ABM',
                      markeredgecolor='white', markeredgewidth=0.6)
        elif label == 'Naive ABM':
            ax.loglog(Ns[-k:], ms[-k:], marker, color=color,
                      markersize=6.5, zorder=6, label='Straightforward TN/ABM',
                      markeredgecolor='white', markeredgewidth=0.6)
        else:
            ax.loglog(Ns[-k:], ms[-k:], marker, color=color,
                      markersize=6.5, zorder=6, label=label,
                      markeredgecolor='white', markeredgewidth=0.6)
        N_lo_fit = fit_left_edge(Ns, k)
        N_line   = np.logspace(np.log10(N_lo_fit), np.log10(Ns[-1]), 300)
        T_fit    = np.exp(fit['intercept']) * N_line ** fit['slope']
        ax.loglog(N_line, T_fit, '-', color=color, lw=2.2, label='_nolegend_')

    draw_variant(ebstn_N, ebstn_ms, ebstn_fit, EBSTN_V4)
    draw_variant(abm_N,   abm_ms,   abm_fit,   ABM_REAL)
    draw_variant(dumb_N,  dumb_ms,  dumb_fit,  ABM_DUMB)

    ref_Ns  = abm_N if len(abm_N) > 0 else ebstn_N
    ref_ms_ = abm_ms if len(abm_ms) > 0 else ebstn_ms
    all_Ns  = [a for a in [ebstn_N, abm_N, dumb_N] if len(a) > 0]
    all_N   = np.concatenate(all_Ns) if all_Ns else np.array([1e4, 1e6])
    N_lo, N_hi = all_N.min() * 0.7, all_N.max() * 1.4
    N_ref = np.logspace(np.log10(N_lo), np.log10(N_hi), 400)

    N_c1 = np.exp(np.mean(np.log(ref_Ns)))
    T_c1 = np.exp(np.mean(np.log(ref_ms_)))
    ax.loglog(N_ref, T_c1 * (N_ref / N_c1) ** 1.0,
              '--', color='#444444', lw=1.3, alpha=0.65,
              label=r'$\mathcal{O}(N^1)$ reference')

    if len(dumb_N) > 0:
        N_c2 = np.exp(np.mean(np.log(dumb_N)))
        T_c2 = np.exp(np.mean(np.log(dumb_ms)))
    else:
        N_c2 = N_c1
        T_c2 = T_c1 * 1e2
    ax.loglog(N_ref, T_c2 * (N_ref / N_c2) ** 2.0,
              ':', color='#b35806', lw=1.5, alpha=0.70,
              label=r'$\mathcal{O}(N^2)$ reference')

    ax.set_xlabel(r'Total number of actors in systems $(N)$')
    ax.set_ylabel('Mean calculation time per timestep  (ms)')
    ax.set_title('Computational scaling: EBSTN vs TN/ABM variants', pad=6)
    ax.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
    ax.grid(True, which='minor', ls=':',  alpha=0.18, lw=0.5)
    ax.minorticks_on()

    handles, labels = ax.get_legend_handles_labels()
    ref_idx   = [i for i, l in enumerate(labels) if 'mathcal' in l]
    model_idx = [i for i, l in enumerate(labels) if 'mathcal' not in l]
    reordered = [model_idx[j] for j in reversed(range(len(model_idx)))] + ref_idx
    ax.legend([handles[i] for i in reordered],
              [labels[i] for i in reordered],
              loc='upper left',
              framealpha=0.93, edgecolor='#cccccc',
              borderpad=0.7, labelspacing=0.35,
              handlelength=1.8, frameon=False)


fig, ax = plt.subplots(figsize=(6.5, 4.8))
draw_panel_a(ax)
fig.tight_layout(pad=0.5)
fig.savefig(f'{OUT_STEM}.pdf', bbox_inches='tight')
fig.savefig(f'{OUT_STEM}.png', dpi=300, bbox_inches='tight')
print(f'Saved {OUT_STEM}.pdf and {OUT_STEM}.png')
