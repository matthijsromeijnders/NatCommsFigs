"""
SI Fig 5 — Infectious actors over time (15 runs) with spatial maps for the
most and least infectious runs on day 17.

Reproduces the publication PDF `RevisionFigSIB_1.pdf`.

Inputs (in ./data/):
  - Status_5_v3_{run}.npz                            : 15 SEIR status CSR files
  - infections_by_infector_Div100_loc=126_{run}_v3.npy : transmission graphs
                                                          (used to refine which
                                                           run is highest/lowest)
  - agents_homes.npy                                 : (n_agents,) int32, home
                                                          muni-index per agent
  - uniLocs.npy                                      : (n_muni,) muni GM-codes
  - gemeente_2019_v3.{shp,shx,dbf,prj,CPG}           : NL muni shapefile

Run:
    python plot_si5.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from mapclassify import NaturalBreaks

# ── CLI defaults match plot_stnstudyv3.py ─────────────────────────────────────
NRUNS = 15
DAY   = 17

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, 'data')
OUT_STEM  = os.path.join(HERE, 'RevisionFigSIB_1')

mpl.rcParams.update({
    'font.family':        'serif',
    'font.size':          14,
    'axes.labelsize':     14,
    'axes.titlesize':     14,
    'legend.fontsize':    14,
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.5,
    'ytick.minor.width':  0.5,
    'lines.linewidth':    1.8,
    'figure.dpi':         150,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# ── Load cumulative E / I / EI per run from sparse Status arrays ──────────────
print('Loading status files ...')
results = {}
for run in range(NRUNS):
    npz_path = os.path.join(DATA_DIR, f'Status_5_v3_{run}.npz')
    if not os.path.exists(npz_path):
        continue
    sp = np.load(npz_path, allow_pickle=True)
    data, indices, indptr = sp['data'], sp['indices'], sp['indptr']
    T = indptr.shape[0] - 1
    seen_1, seen_2 = set(), set()
    elist  = np.empty(T, dtype=np.int32)
    ilist  = np.empty(T, dtype=np.int32)
    eilist = np.empty(T, dtype=np.int32)
    for t in range(T):
        s, e   = indptr[t], indptr[t + 1]
        data_t = data[s:e]
        ag_t   = indices[s:e]
        if ag_t.size:
            a1 = ag_t[data_t == 1]
            a2 = ag_t[data_t == 2]
            if a1.size: seen_1.update(a1.tolist())
            if a2.size: seen_2.update(a2.tolist())
        elist[t]  = len(seen_1)
        ilist[t]  = len(seen_2)
        eilist[t] = len(seen_1 | seen_2)
    results[run] = {'E': elist, 'I': ilist, 'EI': eilist}
res = results

# ── Identify highest / lowest run (refine with infection counts if available) ─
final_I = {k: res[k]['E'][-1] for k in res}
highest = max(final_I, key=final_I.get)
lowest  = min(final_I, key=final_I.get)

print('Loading infection-by-infector files ...')
all_runs = {}
for run in range(NRUNS):
    fpath = os.path.join(DATA_DIR, f'infections_by_infector_Div100_loc=126_{run}_v3.npy')
    if not os.path.exists(fpath):
        continue
    all_runs[run] = np.load(fpath, allow_pickle=True).item()

if all_runs:
    total_inf = {run: sum(arr.size for arr in d['infected_lists'])
                 for run, d in all_runs.items()}
    sorted_runs = sorted(total_inf, key=total_inf.get, reverse=True)
    highest = sorted_runs[0]
    lowest  = sorted_runs[-1]

# ── Per-muni infectious counts on day=17 for the hi/lo runs ───────────────────
def infectious_per_municipality(npz_path, agents_homes, T_use=504, n_muni=355):
    sp = np.load(npz_path, allow_pickle=True)
    data, indices, indptr = sp['data'], sp['indices'].astype(np.int32), sp['indptr']
    T_use = min(T_use, indptr.shape[0] - 1)
    out = np.zeros((T_use, n_muni), dtype=np.int32)
    for t in range(T_use):
        s, e = indptr[t], indptr[t + 1]
        inf_ag = indices[s:e][data[s:e] == 1]
        inf_ag = inf_ag[(inf_ag >= 0) & (inf_ag < agents_homes.size)]
        if inf_ag.size:
            out[t] = np.bincount(agents_homes[inf_ag], minlength=n_muni).astype(np.int32)
    return out

agents_homes = np.load(os.path.join(DATA_DIR, 'agents_homes.npy'))

npz_hi_path = os.path.join(DATA_DIR, f'Status_5_v3_{highest}.npz')
npz_lo_path = os.path.join(DATA_DIR, f'Status_5_v3_{lowest}.npz')
print('Computing per-municipality infectious counts ...')
data_hi = infectious_per_municipality(npz_hi_path, agents_homes)
data_lo = infectious_per_municipality(npz_lo_path, agents_homes)
t_idx       = DAY * 24 - 1
map_vals_hi = data_hi[t_idx]
map_vals_lo = data_lo[t_idx]

# ── Shapefile ────────────────────────────────────────────────────────────────
print('Loading map shapefile ...')
mapdf = gpd.read_file(os.path.join(DATA_DIR, 'gemeente_2019_v3.shp'))
mapdf = mapdf[mapdf['H2O'] == 'NEE']
mapdf.sort_values('GM_NAAM', inplace=True)
mapdf = mapdf.reset_index(drop=True)

# ── Plot ─────────────────────────────────────────────────────────────────────
print('Plotting Fig 1 ...')

cmap_red  = LinearSegmentedColormap.from_list('grey_red',  [(0.85, 0.85, 0.85, 1), (1, 0, 0, 1)], N=256)
cmap_blue = LinearSegmentedColormap.from_list('grey_blue', [(0.85, 0.85, 0.85, 1), (0, 0, 1, 1)], N=256)

pos_vals = np.concatenate([map_vals_hi, map_vals_lo])
pos_vals = pos_vals[pos_vals > 0]
if pos_vals.size == 0:
    bounds = [0, 1]
else:
    nb = NaturalBreaks(pos_vals, k=6)
    bounds = [0] + nb.bins.tolist()

fig1, axs1 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
ax_ts, ax_map_hi, ax_map_lo = axs1

# (a) time series
dt_days = 1 / 24
T_plot  = DAY * 24
for k in res:
    y      = res[k]['I']
    t_days = (np.arange(len(y)) * dt_days)
    if k == highest:
        ax_ts.plot(t_days[:T_plot], y[:T_plot], color='red',  linewidth=2.0)
    elif k == lowest:
        ax_ts.plot(t_days[:T_plot], y[:T_plot], color='blue', linewidth=2.0)
    else:
        ax_ts.plot(t_days[:T_plot], y[:T_plot], color='grey', alpha=0.3, linewidth=0.9)
ax_ts.set_xlim(0, 17)
ax_ts.set_yscale('log')
ax_ts.set_title('(a) Infectious actors over time (15 runs)')
ax_ts.set_xlabel('time (days)')
ax_ts.set_ylabel('number of infectious actors')
ax_ts.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
ax_ts.grid(True, which='minor', ls=':',  alpha=0.18, lw=0.5)
ax_ts.minorticks_on()
ax_ts.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
ax_ts.legend(handles=[
    mlines.Line2D([], [], color='red',  linewidth=2.0, label=f'run {highest} (most infectious)'),
    mlines.Line2D([], [], color='blue', linewidth=2.0, label=f'run {lowest} (least infectious)'),
    mlines.Line2D([], [], color='grey', linewidth=1.5, alpha=0.5, label='other runs'),
], loc='upper left', frameon=False)

# (b) & (c) maps
maxx = map_vals_hi.max()
for ax_m, cmap_m, mv, label_run in [
    (ax_map_hi, cmap_red,  map_vals_hi, f'(b) run {highest}, day {DAY}'),
    (ax_map_lo, cmap_blue, map_vals_lo, f'(c) run {lowest}, day {DAY}'),
]:
    df_m = mapdf.copy()
    df_m['DATA'] = mv
    df_m.plot(ax=ax_m, column='DATA', cmap=cmap_m, edgecolor='black',
              linewidth=0.25, vmin=0, vmax=maxx)
    ax_m.set_title(label_run)
    ax_m.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap_m, norm=plt.Normalize(vmin=0, vmax=maxx))
    sm._A = []
    cax = ax_m.inset_axes([0.1, 0.005, 0.8, 0.03])
    cb = fig1.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('infectious resident actors')

fig1.suptitle(f'Infectious disease dynamics and spatial distribution on day {DAY}')

fig1.savefig(f'{OUT_STEM}.pdf', bbox_inches='tight')
fig1.savefig(f'{OUT_STEM}.png', dpi=300, bbox_inches='tight')
print(f'Saved {OUT_STEM}.pdf and {OUT_STEM}.png')
