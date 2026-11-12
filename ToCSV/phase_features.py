"""Shared helpers for PprToCSV.py and OprToCSV.py.

Turns one per-sample flight dataframe (already labelled with an
'extrafeatures' sub-phase column) into a single feature row:
median and IQR of every numeric column plus the duration of each sub-phase,
flattened as '<SubPhase>_<feature>' and followed by the Cost_Index and
Max_Payload labels parsed from the file name.
"""
import argparse
import os
import re
import sys

import pandas as pd
from scipy.stats import iqr

# 3 sub-phases per flight block (climb, cruise, descent)
N_SUBPHASES = 9


def summarize_flight(df, posix, feature_cols, file_name):
    """df: per-sample frame with 'extrafeatures' + feature_cols.
    posix: timestamps aligned with df. Returns (row, number of sub-phases)."""
    grouped = df.groupby('extrafeatures')[feature_cols]

    df_median = grouped.median().add_suffix('-median')
    df_iqr = grouped.agg(iqr).add_suffix('-iqr')
    comb_df = df_median.join(df_iqr)

    times = posix.groupby(df['extrafeatures'])
    comb_df['duration[s]'] = times.max() - times.min()
    comb_df = comb_df[~comb_df.index.str.startswith('Drop')]

    s = comb_df.stack()
    s.index = [f"{x}_{y}" for x, y in s.index]
    out = s.to_frame('').T

    name = os.path.basename(file_name)
    out['Cost_Index'] = float(re.findall(r'CI_(\d+)', name)[0])
    out['Max_Payload'] = float(re.findall(r'MPL_(\d+.\d)', name)[0])
    return out, len(comb_df)


def datastore_prep(data_prep, path, csv_name):
    """Run data_prep on every file in `path` and write one row per flight."""
    rows = []
    skipped = 0
    for file_name in sorted(os.listdir(path)):
        out, n_phases = data_prep(os.path.join(path, file_name))
        if n_phases != N_SUBPHASES:
            # A missing sub-phase would shift every following column.
            print(f'Skipping {file_name}: {n_phases}/{N_SUBPHASES} sub-phases', file=sys.stderr)
            skipped += 1
            continue
        rows.append(out)

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(csv_name, index=False)
    print(f'{csv_name}: {len(out_df)} flights, {out_df.shape[1] - 2} features, {skipped} skipped')


def cli(data_prep, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--in-dir', required=True, help='folder with one raw CSV per flight')
    parser.add_argument('--out-csv', required=True, help='output feature CSV')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    datastore_prep(data_prep, args.in_dir, args.out_csv)
