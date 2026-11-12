"""Run the whole pipeline on synthetic sample data with one command.

Steps, stopping at the first failure:
  1. generate raw trajectories        -> sample_data/raw/{ppr,opr}/{train,test}/
  2. ToCSV: one feature row per flight -> sample_data/processed/<Set>{Train,Test}.csv
  3. DataCleanNorm: clean + normalise  -> sample_data/processed/<Set>{Train,Test}Norm.csv
  4. train the GCNN                    -> sample_data/runs/<set>_<target>_loss.png, .weights.h5

Usage:
    python run_sample.py                                  # both sets, CI, 50 epochs
    python run_sample.py --targets ci payload --skip-generate  # both targets, reuse raw data
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join('sample_data', 'raw')
PROCESSED = os.path.join('sample_data', 'processed')
RUNS = os.path.join('sample_data', 'runs')


def run(step, args):
    print(f'\n=== {step}', flush=True)
    start = time.time()
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2', PYTHONIOENCODING='utf-8')
    result = subprocess.run([sys.executable] + args, cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f'\nFAILED at "{step}" (exit code {result.returncode})')
    print(f'--- done in {time.time() - start:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sets', nargs='+', default=['ppr', 'opr'], choices=['ppr', 'opr'])
    parser.add_argument('--targets', nargs='+', default=['ci'], choices=['ci', 'payload'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n-train', type=int, default=200)
    parser.add_argument('--n-test', type=int, default=50)
    parser.add_argument('--skip-generate', action='store_true',
                        help='reuse existing raw files in sample_data/raw')
    args = parser.parse_args()

    if not args.skip_generate:
        run('Generate sample trajectories',
            [os.path.join('sample_data', 'generate_sample_data.py'), '--out-dir', RAW,
             '--n-train', str(args.n_train), '--n-test', str(args.n_test)])

    for var_set in args.sets:
        name = var_set.capitalize()
        for split in ('train', 'test'):
            run(f'{name} {split}: raw -> features',
                [os.path.join('ToCSV', f'{name}ToCSV.py'),
                 '--in-dir', os.path.join(RAW, var_set, split),
                 '--out-csv', os.path.join(PROCESSED, f'{name}{split.capitalize()}.csv')])

        run(f'{name}: clean + normalise',
            [os.path.join('DataManipulation', 'DataCleanNorm.py'),
             '--set', var_set, '--data-dir', PROCESSED])

        for target in args.targets:
            run(f'{name}: train GCNN on {target} ({args.epochs} epochs)',
                [os.path.join('models', 'train_gcnn.py'), '--set', var_set, '--target', target,
                 '--epochs', str(args.epochs), '--data-dir', PROCESSED, '--out-dir', RUNS])

    print(f'\nAll steps finished. Plots and weights are in {RUNS}')


if __name__ == '__main__':
    main()
