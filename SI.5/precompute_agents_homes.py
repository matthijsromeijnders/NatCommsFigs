"""
One-shot preprocessing for SI.5 and SI.6.

Reads PeopleDF.pkl + ExtraPeopleDF.pkl (Seed_febDiv100_0) and Gemeenten.pkl,
derives the (n_agents,) array mapping each agent to its home muni-index,
and writes:
  - data/agents_homes.npy    (n_agents,) int32
  - data/uniLocs.npy         (n_muni,) — unique muni GM-codes in their canonical order

These small derived files replace ~500 MB of pickles in the SI.5/SI.6 data
folders. Run once; SI.5/SI.6 plotting scripts only need the .npy outputs.
"""

import os
import numpy as np
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
REPO     = os.path.normpath(os.path.join(HERE, '..', '..'))
SEED_DIR = os.path.join(REPO, 'STNRevision', 'Seed_febDiv100_0')
GEMEENTEN = os.path.join(REPO, 'Data', 'Model_V1', 'Data', 'Gemeenten.pkl')
OUT_DIR   = os.path.join(HERE, 'data')

os.makedirs(OUT_DIR, exist_ok=True)

print('Loading PeopleDF.pkl ...')
people = pd.read_pickle(os.path.join(SEED_DIR, 'PeopleDF.pkl'))
print('Loading ExtraPeopleDF.pkl ...')
extra  = pd.read_pickle(os.path.join(SEED_DIR, 'ExtraPeopleDF.pkl'))
print('Loading Gemeenten.pkl ...')
uniLocs = np.array(pd.read_pickle(GEMEENTEN)).T[0]
loc_to_id = {v: i for i, v in enumerate(uniLocs)}

homes = np.concatenate([np.asarray(people['Home']), np.asarray(extra['Home'])])
agents_homes = np.vectorize(lambda x: loc_to_id[x])(homes).astype(np.int32)

np.save(os.path.join(OUT_DIR, 'agents_homes.npy'), agents_homes)
np.save(os.path.join(OUT_DIR, 'uniLocs.npy'), uniLocs)
print(f'Saved agents_homes.npy ({agents_homes.shape}, {agents_homes.nbytes/1024:.1f} KB)')
print(f'Saved uniLocs.npy      ({uniLocs.shape})')
