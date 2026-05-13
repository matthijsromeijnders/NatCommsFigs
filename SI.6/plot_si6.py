"""
SI Fig 6 — Secondary-infection PMFs for the most/least infectious runs,
exposed-actor time series, and daily superspreading index.

Reproduces the publication PDF `RevisionFigSIB_2.pdf`.

Inputs (in ./data/):
  - Status_5_v3_{run}.npz                            : 15 SEIR status CSR files
  - infections_by_infector_Div100_loc=126_{run}_v3.npy : transmission graphs
  - agents_homes.npy / uniLocs.npy                   : agent→muni mapping
  - gemeente_2019_v3.{shp,shx,dbf,prj,CPG}           : NL muni shapefile
                                                          (for annotation names)

Run:
    python plot_si6.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import geopandas as gpd

# ── Defaults match plot_stnstudyv3.py ────────────────────────────────────────
NRUNS     = 15
T_CUT     = 400
DAY       = 17
SS_METRIC = 'max'

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_STEM = os.path.join(HERE, 'RevisionFigSIB_2')

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

# ── Identify highest / lowest run ────────────────────────────────────────────
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

# ── Helpers ──────────────────────────────────────────────────────────────────
def pdf_offspring_up_to(run_dict, t_cut, cut0s=False):
    """PMF of secondary infections per eligible infector (infected before t_cut)."""
    infector_ids  = run_dict['infector_ids']
    infected_list = run_dict['infected_lists']
    t_list        = run_dict['t_lists']
    infection_time = {}
    for children, times in zip(infected_list, t_list):
        if children is None or len(children) == 0:
            continue
        for child, t in zip(children, times):
            child, t = int(child), float(t)
            if child not in infection_time or t < infection_time[child]:
                infection_time[child] = t
    k_full = np.zeros(len(infector_ids), dtype=np.int32)
    keep   = np.zeros(len(infector_ids), dtype=bool)
    for j, inf in enumerate(infector_ids):
        k_full[j] = 0 if infected_list[j] is None else int(len(infected_list[j]))
        keep[j] = infection_time.get(inf, t_cut + 1) < t_cut
    k = k_full[keep]
    if cut0s:
        k = k[k > 0]
    if k.size == 0:
        return np.array([1.0]), np.array([1])
    pmf = np.bincount(k)
    return pmf / pmf.sum(), pmf


def daytime_location_events_scatter_normalized_by_infectious(
        run_dict, status_sp, start_h=9, end_h=14,
        infectious_status=1, T=504):
    """(day, location) event size normalized by #infectious individuals at that hour."""
    event_counter = {}
    event_hour    = {}
    for times, locs in zip(run_dict['t_lists'], run_dict['loc_lists']):
        if times is None or len(times) == 0:
            continue
        th   = np.floor(np.asarray(times)).astype(np.int32)
        locs = np.asarray(locs).astype(np.int32)
        mask = (th % 24 >= start_h) & (th % 24 < end_h)
        for tt, d, l in zip(th[mask], th[mask] // 24, locs[mask]):
            key = (int(d), int(l))
            event_counter[key] = event_counter.get(key, 0) + 1
            if key not in event_hour or tt < event_hour[key]:
                event_hour[key] = int(tt)
    if not event_counter:
        return np.array([]), np.array([]), np.array([]), np.array([])
    data    = status_sp['data']
    indices = status_sp['indices']
    indptr  = status_sp['indptr']
    T_use   = min(T, indptr.shape[0] - 1)
    exposed_per_hour = np.zeros(T_use, dtype=np.int32)
    for t in range(T_use):
        s, e = indptr[t], indptr[t + 1]
        exposed_per_hour[t] = int(np.sum(data[s:e] == infectious_status))
    days, norm_sizes, locs_out, hours_out = [], [], [], []
    for (d, l), c in event_counter.items():
        h = event_hour[(d, l)]
        if h < 0 or h >= T_use:
            continue
        I = exposed_per_hour[h]
        if I <= 0:
            continue
        days.append(d); locs_out.append(l); hours_out.append(h)
        norm_sizes.append(c / I)
    return (np.array(days), np.array(norm_sizes),
            np.array(locs_out), np.array(hours_out))


def raw_transmissions_by_loc_day(run_dict, start_h=8, end_h=17):
    counter = {}
    for times, locs in zip(run_dict['t_lists'], run_dict['loc_lists']):
        if times is None or len(times) == 0:
            continue
        th       = np.floor(np.asarray(times)).astype(np.int32)
        locs_arr = np.asarray(locs).astype(np.int32)
        mask     = (th % 24 >= start_h) & (th % 24 < end_h)
        for d, l in zip(th[mask] // 24, locs_arr[mask]):
            key = (int(d), int(l))
            counter[key] = counter.get(key, 0) + 1
    return counter


def daily_ss_index(days, vals, metric='p95'):
    if days.size == 0:
        return np.array([]), np.array([])
    dmax = int(days.max())
    x    = np.arange(dmax + 1)
    ss   = np.full(dmax + 1, np.nan)
    for d in x:
        v = vals[days == d]
        if v.size:
            ss[d] = v.max() if metric == 'max' else np.percentile(v, 95)
        else:
            ss[d] = 0.0
    return x, ss


def _peak_events_by_loc(days, sizes, locs, daily_x, daily_mn, n_muni=355):
    if daily_x.size == 0:
        return np.zeros(n_muni), -1
    _valid = np.where(np.isfinite(daily_mn) & (daily_mn > 0))[0]
    if not _valid.size:
        return np.zeros(n_muni), -1
    peak_day = int(daily_x[_valid[np.argmax(daily_mn[_valid])]])
    mask = (days == peak_day)
    agg = np.zeros(n_muni)
    for l, s in zip(locs[mask].astype(int), sizes[mask]):
        if 0 <= l < n_muni:
            agg[l] += s
    return agg, peak_day


# ── Compute data ─────────────────────────────────────────────────────────────
print('Computing offspring PMFs ...')
_, pmf_hi = pdf_offspring_up_to(all_runs[highest], T_CUT)
_, pmf_lo = pdf_offspring_up_to(all_runs[lowest],  T_CUT)

print('Computing normalized event scatter ...')
npz_hi_path = os.path.join(DATA_DIR, f'Status_5_v3_{highest}.npz')
npz_lo_path = os.path.join(DATA_DIR, f'Status_5_v3_{lowest}.npz')
status_hi_sp = np.load(npz_hi_path, allow_pickle=True)
status_lo_sp = np.load(npz_lo_path, allow_pickle=True)

days_hi_norm, sizes_hi_norm, locs_hi_norm, _ = daytime_location_events_scatter_normalized_by_infectious(
    all_runs[highest], status_hi_sp, start_h=8, end_h=17, infectious_status=2, T=504)
days_lo_norm, sizes_lo_norm, locs_lo_norm, _ = daytime_location_events_scatter_normalized_by_infectious(
    all_runs[lowest],  status_lo_sp, start_h=8, end_h=17, infectious_status=2, T=504)

raw_hi = raw_transmissions_by_loc_day(all_runs[highest])
raw_lo = raw_transmissions_by_loc_day(all_runs[lowest])

# ── Shapefile (for muni-name annotations) ────────────────────────────────────
print('Loading map shapefile ...')
uniLocs = np.load(os.path.join(DATA_DIR, 'uniLocs.npy'), allow_pickle=True)
mapdf = gpd.read_file(os.path.join(DATA_DIR, 'gemeente_2019_v3.shp'))
mapdf = mapdf[mapdf['H2O'] == 'NEE']
mapdf.sort_values('GM_NAAM', inplace=True)
mapdf = mapdf.reset_index(drop=True)
gm_to_name = dict(zip(mapdf['GM_CODE'].str.strip(), mapdf['GM_NAAM']))


def _loc_idx_to_name(loc_idx):
    raw = uniLocs[loc_idx]
    key = str(raw).strip()
    if key in gm_to_name:
        return gm_to_name[key]
    try:
        gm_key = f"GM{int(raw):04d}"
        if gm_key in gm_to_name:
            return gm_to_name[gm_key]
    except (ValueError, TypeError):
        pass
    if 0 <= loc_idx < len(mapdf):
        return mapdf['GM_NAAM'].iloc[loc_idx]
    return key


# ── Plot ─────────────────────────────────────────────────────────────────────
print('Plotting Fig 2 ...')

xlim_pmf = max(pmf_hi.size, pmf_lo.size)
ylim_pmf = max(pmf_hi.max(), pmf_lo.max()) * 1.5

x_hi, p95_hi = daily_ss_index(days_hi_norm, sizes_hi_norm, metric=SS_METRIC)
x_lo, p95_lo = daily_ss_index(days_lo_norm, sizes_lo_norm, metric=SS_METRIC)

T_plot = DAY * 24

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
fig2.set_constrained_layout_pads(hspace=0.15, wspace=0.12)
ax_a, ax_b = axes2[0, 0], axes2[0, 1]
ax_c, ax_d = axes2[1, 0], axes2[1, 1]

# (a) PMF — most infectious
ax_a.bar(np.arange(pmf_hi.size), pmf_hi, color='red', alpha=0.8)
ax_a.set_title(f'(a) Superspreader actors in run {highest}')
ax_a.set_xlabel('secondary infections per infectious actor')
ax_a.set_ylabel('infectious actor count')
ax_a.set_yscale('log')
ax_a.set_xlim(-0.4, xlim_pmf)
ax_a.set_ylim(0.7, ylim_pmf)
ax_a.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
ax_a.minorticks_on()

# (c) exposed actors over time
for k in res:
    y      = res[k]['E']
    t_days = np.arange(len(y)) / 24
    if k == highest:
        ax_b.plot(t_days[:T_plot], y[:T_plot], color='red',  linewidth=2.0)
    elif k == lowest:
        ax_b.plot(t_days[:T_plot], y[:T_plot], color='blue', linewidth=2.0)
    else:
        ax_b.plot(t_days[:T_plot], y[:T_plot], color='grey', alpha=0.3, linewidth=0.9)
ax_b.set_xlim(0, DAY)
ax_b.set_yscale('log')
ax_b.set_title('(c) Exposed actors over time (15 runs)')
ax_b.set_xlabel('time (days)')
ax_b.set_ylabel('number of exposed actors')
ax_b.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
ax_b.grid(True, which='minor', ls=':',  alpha=0.18, lw=0.5)
ax_b.minorticks_on()
ax_b.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
ax_b.legend(handles=[
    mlines.Line2D([], [], color='red',  linewidth=2.0, label=f'run {highest}'),
    mlines.Line2D([], [], color='blue', linewidth=2.0, label=f'run {lowest}'),
    mlines.Line2D([], [], color='grey', linewidth=1.5, alpha=0.5, label='other runs'),
], loc='upper left', frameon=False)

# (b) PMF — least infectious
ax_c.bar(np.arange(pmf_lo.size), pmf_lo, color='blue', alpha=0.8)
ax_c.set_title(f'(b) Superspreader actors in run {lowest}')
ax_c.set_xlabel('secondary infections per infectious actor')
ax_c.set_ylabel('infectious actor count')
ax_c.set_yscale('log')
ax_c.set_xlim(-0.4, xlim_pmf)
ax_c.set_ylim(0.7, ylim_pmf)
ax_c.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
ax_c.minorticks_on()

# Pre-compute peak-day aggregates for panel (d) annotations
agg_hi, peak_hi = _peak_events_by_loc(days_hi_norm, sizes_hi_norm, locs_hi_norm, x_hi, p95_hi)
agg_lo, peak_lo = _peak_events_by_loc(days_lo_norm, sizes_lo_norm, locs_lo_norm, x_lo, p95_lo)
_top_name_hi = _loc_idx_to_name(int(np.argmax(agg_hi))) if agg_hi.any() else '?'
_top_name_lo = _loc_idx_to_name(int(np.argmax(agg_lo))) if agg_lo.any() else '?'

# (d) daily superspreading index
ax_d.plot(x_hi + 0.5, p95_hi, linewidth=2.0, color='red',  label=f'run {highest}')
ax_d.plot(x_lo + 0.5, p95_lo, linewidth=2.0, color='blue', label=f'run {lowest}')
ax_d.set_title('(d) Accounting for superspreading')
ax_d.set_xlabel('time (days)')
ax_d.set_ylabel('daily superspreading index')
ax_d.set_xlim(0, DAY)
ax_d.grid(True, which='major', ls='--', alpha=0.35, lw=0.6)
ax_d.minorticks_on()
ax_d.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
ax_d.legend(frameon=False)

_annot_kw = dict(fontsize=10, ha='center', va='bottom',
                 arrowprops=dict(arrowstyle='->', color='0.1', lw=2))
if peak_hi >= 0 and peak_hi < p95_hi.size and np.isfinite(p95_hi[peak_hi]):
    _top_loc_hi = int(np.argmax(agg_hi))
    _n_hi = raw_hi.get((peak_hi, _top_loc_hi), 0)
    ax_d.annotate(f'{_top_name_hi}\n(Work, #exposures = {_n_hi})',
                  xy=(peak_hi + 0.5, p95_hi[peak_hi]),
                  xytext=(peak_hi + 5, p95_hi[peak_hi] - 0.08),
                  color='red', **_annot_kw)
if peak_lo >= 0 and peak_lo < p95_lo.size and np.isfinite(p95_lo[peak_lo]):
    _top_loc_lo = int(np.argmax(agg_lo))
    _n_lo = raw_lo.get((peak_lo, _top_loc_lo), 0)
    ax_d.annotate(f'{_top_name_lo}\n(Work, #exposures = {_n_lo})',
                  xy=(peak_lo + 0.5, p95_lo[peak_lo]),
                  xytext=(peak_lo + 5, p95_lo[peak_lo] - 0.06),
                  color='blue', **_annot_kw)

# Dotted grey vertical line aligning peak of (d) with (b)
_peak_day = None
if x_hi.size > 0 and np.any(np.isfinite(p95_hi)):
    _valid = np.where(np.isfinite(p95_hi))[0]
    _peak_day = float(x_hi[_valid[np.argmax(p95_hi[_valid])]])
elif x_lo.size > 0 and np.any(np.isfinite(p95_lo)):
    _valid = np.where(np.isfinite(p95_lo))[0]
    _peak_day = float(x_lo[_valid[np.argmax(p95_lo[_valid])]])

if _peak_day is not None:
    _vline_kw = dict(color='0.25', linewidth=1.2, linestyle=(0, (4, 4)), zorder=100)
    ax_b.axvline(_peak_day + 0.5, **_vline_kw)
    ax_d.axvline(_peak_day + 0.5, **_vline_kw)

    fig2.canvas.draw()
    inv = fig2.transFigure.inverted()
    x_fig = inv.transform(ax_b.transData.transform([_peak_day + 0.5, 0]))[0]
    bb_b_fig = inv.transform(ax_b.get_window_extent())
    bb_d_fig = inv.transform(ax_d.get_window_extent())
    y_start = bb_b_fig[0, 1]
    y_end   = bb_d_fig[1, 1] - 0.02
    gap_line = mlines.Line2D(
        [x_fig, x_fig], [y_end, y_start],
        transform=fig2.transFigure, clip_on=False, **_vline_kw,
    )
    fig2.add_artist(gap_line)

fig2.suptitle('Secondary infection distribution and superspreading event dynamics')

fig2.savefig(f'{OUT_STEM}.pdf', bbox_inches='tight')
fig2.savefig(f'{OUT_STEM}.png', dpi=300, bbox_inches='tight')
print(f'Saved {OUT_STEM}.pdf and {OUT_STEM}.png')
