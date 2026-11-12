"""Split raw trajectory files into train and test folders by SIMBAD flight ID.

A flight goes to the test set when a file with the same SIMBAD_<id> exists in
--test-list-dir (the folder of reference test trajectories); every other
flight goes to the train set.

Writes <out-dir>/train/ and <out-dir>/test/.

Usage:
    python DataManipulation/TrainTestSplit.py --src-dir data/PprVariables \
        --test-list-dir data/Tests/FDR_V6_tests --out-dir data/PprSplit
"""
import argparse
import os
import re
import shutil

ID_PATTERN = re.compile(r'SIMBAD_(\d+)')


def flight_id(file_name):
    match = ID_PATTERN.search(file_name)
    return match.group(1) if match else None


def test_ids(test_list_dir):
    return {fid for fid in map(flight_id, os.listdir(test_list_dir)) if fid}


def train_test_split(src_dir, test_set, out_dir, copy=False):
    transfer = shutil.copy2 if copy else shutil.move
    counts = {'train': 0, 'test': 0, 'skipped': 0}

    for split in ('train', 'test'):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    for file_name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, file_name)
        fid = flight_id(file_name)
        if fid is None or not os.path.isfile(src):
            counts['skipped'] += 1
            continue
        split = 'test' if fid in test_set else 'train'
        transfer(src, os.path.join(out_dir, split, file_name))
        counts[split] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src-dir', required=True, help='folder with all raw flight CSVs')
    parser.add_argument('--test-list-dir', required=True,
                        help='folder whose SIMBAD_<id> file names define the test flights')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--copy', action='store_true', help='copy files instead of moving them')
    args = parser.parse_args()

    counts = train_test_split(args.src_dir, test_ids(args.test_list_dir), args.out_dir, args.copy)
    print(f"train: {counts['train']}, test: {counts['test']}, skipped (no SIMBAD id): {counts['skipped']}")


if __name__ == '__main__':
    main()
