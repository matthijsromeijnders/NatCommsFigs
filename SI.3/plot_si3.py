"""
SI Fig 3 — Hot-loop memory: Naive TN/ABM vs Optimised TN/ABM vs EBSTN.

Reproduces the publication PDF `benchmark_memory_combined.pdf` from cached
tracemalloc measurements.

Inputs (in ./data/):
  - benchmark_memory_combined_data.pkl   : (naive_res, opt_res) tuple

Run:
    python plot_si3.py
"""

import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
CACHE    = os.path.join(DATA_DIR, 'benchmark_memory_combined_data.pkl')
OUT_STEM = os.path.join(HERE, 'benchmark_memory_combined')

N_MUNIS  = 355
N_GROUPS = 11
N_DAYS   = 7
N_HOURS  = 24
N_SLOTS  = N_DAYS * N_HOURS                    # 168
N_SLOTS_NIGHT = N_DAYS * (N_HOURS - 8 + 1)     # 119
N_MAX    = 5_000_000

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


def fmt_x(x, _):
    if x >= 1e6:
        v = x / 1e6
        return f'{v:.0f}M' if v == int(v) else f'{v:.1f}M'
    if x >= 1e3:
        v = x / 1e3
        return f'{v:.0f}k' if v == int(v) else f'{v:.1f}k'
    return f'{int(x)}'


def fmt_y(x, _):
    if x < 0.01:  return f'{x:.3f}'
    if x < 1:     return f'{x:.2f}'
    if x < 1024:  return f'{x:.0f}'
    return f'{x/1024:.1f} GB'


def naive_analytic_mb(N):
    """Analytical Positions memory: (N_DAYS*N_HOURS*N) int16, in MB."""
    return N_SLOTS * 2 * np.asarray(N, dtype=float) / 1024**2


with open(CACHE, 'rb') as f:
    naive_res, opt_res = pickle.load(f)

N_naive        = np.array([r['N']              for r in naive_res])
mb_naive       = np.array([r['mb']             for r in naive_res])
N_opt          = np.array([r['N']              for r in opt_res])
mb_abm         = np.array([r['abm_mb']         for r in opt_res])
mb_ebstn       = np.array([r['ebstn_mb']       for r in opt_res])
mb_ebstn_ovlp  = np.array([r['ebstn_ovlp_mb']  for r in opt_res])

mb_naive_w    = mb_naive
mb_abm_w      = mb_abm
mb_ebstn_w    = mb_ebstn
mb_ebstn_nw   = mb_ebstn_ovlp * N_SLOTS_NIGHT / N_SLOTS

N_extrap  = np.logspace(np.log10(N_naive[-1]), np.log10(N_MAX), 200).astype(int)
mb_extrap = naive_analytic_mb(N_extrap)

fig, ax = plt.subplots(figsize=(6, 4.8))

ax.loglog(N_naive, mb_naive_w,  color='#d62728', lw=2.2,
          marker='D', ms=6, label='Straightforward TN/ABM')
ax.loglog(N_extrap, mb_extrap,  color='#d62728', lw=2.2, ls='--')
ax.loglog(N_opt, mb_abm_w,      color='#ff7f0e', lw=2.2,
          marker='s', ms=6, label='Optimised TN/ABM')
ax.loglog(N_opt, mb_ebstn_w,    color='#1f77b4', lw=2.2,
          marker='o', ms=6, label='EBSTN')
ax.loglog(N_opt, mb_ebstn_nw,   color='#2ca02c', lw=2.2,
          marker='^', ms=6, label='EBSTN (overlapping events)')

ax.set_xlabel('Total number of actors in system $(N)$')
ax.set_ylabel('Memory allocation per timestep (MB)')
ax.legend(loc='upper left', frameon=False)
ax.set_xlim(N_naive[0] * 0.8, N_MAX * 1.2)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
ax.grid(True, which='both', ls=':', alpha=0.4)

fig.tight_layout()
fig.savefig(f'{OUT_STEM}.pdf', bbox_inches='tight')
fig.savefig(f'{OUT_STEM}.png', dpi=300, bbox_inches='tight')
print(f'Saved {OUT_STEM}.pdf and {OUT_STEM}.png')
