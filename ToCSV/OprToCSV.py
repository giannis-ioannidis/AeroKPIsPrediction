"""Convert raw Opr trajectories (one CSV per flight) into one feature row per flight.

Usage:
    python ToCSV/OprToCSV.py --in-dir sample_data/raw/opr/train --out-csv sample_data/processed/OprTrain.csv
"""
import pandas as pd

from phase_features import cli, summarize_flight

FEATURES = ['h[ft]', 'Temp[ºC]', 'Press[hPa]', 'Wn[kt]', 'We[kt]', 'Ws[kt]', 'Wx[kt]',
            'Lat[º]', 'Lon[º]', 'vdot[kt/s]', 'hdot[ft/min]']


#function for general categorization
def extrafeatures(row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2):
  if row['Flight_Block'] == 'climb' and row['h[ft]'] < cli_lim1:
    return 'Climbing(-)'
  elif row['Flight_Block'] == 'climb' and row['h[ft]'] >= cli_lim1 and row['h[ft]'] < cli_lim2:
    return 'Climbing'
  elif row['Flight_Block'] == 'climb' and row['h[ft]'] >= cli_lim2:
    return 'Climbing(+)'
  elif row['Flight_Block'] == 'cruise' and row['h[ft]'] < cru_lim1:
    return 'Cruising(-)'
  elif row['Flight_Block'] == 'cruise' and row['h[ft]'] >= cru_lim1 and row['h[ft]'] < cru_lim2:
    return 'Crusing'
  elif row['Flight_Block'] == 'cruise' and row['h[ft]'] >= cru_lim2:
    return 'Cruising(+)'
  elif row['Flight_Block'] == 'descent' and row['h[ft]'] < desc_lim1:
    return 'Descending(-)'
  elif row['Flight_Block'] == 'descent' and row['h[ft]'] >= desc_lim1 and row['h[ft]'] < desc_lim2:
    return 'Descending'
  elif row['Flight_Block'] == 'descent' and row['h[ft]'] >= desc_lim2:
    return 'Descending(+)'


#final concatenated dataframe
def data_prep(name):

  pre_df = pd.read_csv(name)
  pre_df = pre_df.loc[:, ~pre_df.columns.str.startswith('reserved')]
  pre_df = pre_df.replace(to_replace='CRUISE_.*', value='CRUISE', regex=True)

  df = pre_df[['Flight_Block', 'Phase'] + FEATURES].copy()

  def limits(block):
    h = df.loc[df['Flight_Block'] == block, 'h[ft]']
    delta = (h.max() - h.min()) / 3
    return h.min() + delta, h.min() + 2 * delta

  cli_lim1, cli_lim2 = limits('climb')
  cru_lim1, cru_lim2 = limits('cruise')
  desc_lim1, desc_lim2 = limits('descent')

  df['extrafeatures'] = df.apply(lambda row: extrafeatures(
      row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2), axis=1)

  return summarize_flight(df, pre_df['posix[s]'], FEATURES, name)


if __name__ == '__main__':
  cli(data_prep, __doc__)
