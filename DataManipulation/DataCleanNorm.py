"""Clean and min-max normalise the feature CSVs produced by ToCSV/*ToCSV.py.

Reads  <data-dir>/<Set>Train.csv and <data-dir>/<Set>Test.csv
Writes <data-dir>/<Set>TrainNorm.csv and <data-dir>/<Set>TestNorm.csv

Normalisation statistics (min, max, fill values) come from the train set only.
The last two columns (Cost_Index, Max_Payload) are labels and are not normalised.

Usage:
    python DataManipulation/DataCleanNorm.py --set ppr --data-dir sample_data/processed
"""
import argparse
import os

import pandas as pd

LABELS = ['Cost_Index', 'Max_Payload']
# 9 sub-phases x features per sub-phase
N_FEATURES = {'ppr': 9 * 17, 'opr': 9 * 23}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', required=True, choices=sorted(N_FEATURES))
    parser.add_argument('--data-dir', default='sample_data/processed')
    args = parser.parse_args()

    prefix = os.path.join(args.data_dir, args.set.capitalize())
    train_df = pd.read_csv(prefix + 'Train.csv')
    test_df = pd.read_csv(prefix + 'Test.csv')

    features = [c for c in train_df.columns if c not in LABELS]
    assert len(features) == N_FEATURES[args.set], \
        f'expected {N_FEATURES[args.set]} features, found {len(features)}'
    assert list(test_df.columns) == list(train_df.columns), 'train/test columns differ'

    # drop incomplete train rows; fill test gaps with the train mean
    n_before = len(train_df)
    train_df = train_df.dropna().reset_index(drop=True)
    test_df[features] = test_df[features].fillna(train_df[features].mean())
    test_df = test_df.dropna(subset=LABELS).reset_index(drop=True)

    # min-max scaling with train statistics; constant columns map to 0
    train_min = train_df[features].min()
    span = train_df[features].max() - train_min
    span = span.where(span != 0, 1)

    for df, split in ((train_df, 'Train'), (test_df, 'Test')):
        df[features] = (df[features] - train_min) / span
        out = prefix + split + 'Norm.csv'
        df.to_csv(out, index=False)
        print(f'{out}: {len(df)} rows, {len(features)} features')

    print(f'Dropped {n_before - len(train_df)} train rows with missing values')


if __name__ == '__main__':
    main()
