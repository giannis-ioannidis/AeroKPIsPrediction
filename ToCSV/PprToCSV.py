"""Convert raw Ppr trajectories (one CSV per flight) into one feature row per flight.

Usage:
    python ToCSV/PprToCSV.py --in-dir sample_data/raw/ppr/train --out-csv sample_data/processed/PprTrain.csv
"""
from math import sin, cos, sqrt, atan2, radians

import numpy as np
import pandas as pd

from phase_features import cli, summarize_flight

FEATURES = ['h[ft]', 'Temp[oC]', 'v-Wn[kt]', 'u-We[kt]',
            'Lat[o]', 'Lon[o]', 'vdot[kt/s]', 'hdot[ft/min]']


#function for general categorization
def extrafeatures33(row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2):
  if row['phase'] == 'climbing_BELOW-FL100':
    return 'Drop1'
  elif row['phase'] == 'climbing_ABOVE-FL100' and row['h[ft]'] < cli_lim1:
    return 'Climbing(-)'
  elif row['phase'] == 'climbing_ABOVE-FL100' and row['h[ft]'] >= cli_lim1 and row['h[ft]'] < cli_lim2:
    return 'Climbing'
  elif row['phase'] == 'climbing_ABOVE-FL100' and row['h[ft]'] >= cli_lim2:
    return 'Climbing(+)'
  elif row['phase'] == 'descending_ABOVE-FL100' and row['h[ft]'] < desc_lim1:
    return 'Descending(-)'
  elif row['phase'] == 'descending_ABOVE-FL100' and row['h[ft]'] >= desc_lim1 and row['h[ft]'] < desc_lim2:
    return 'Descending'
  elif row['phase'] == 'descending_ABOVE-FL100' and row['h[ft]'] >= desc_lim2:
    return 'Descending(+)'
  elif row['phase'] == 'descending_BELOW-FL100':
    return 'Drop2'
  elif row['phase'] == 'cruising' and row['h[ft]'] < cru_lim1:
    return 'Crusing(-)'
  elif row['phase'] == 'cruising' and row['h[ft]'] >= cru_lim1 and row['h[ft]'] < cru_lim2:
    return 'Cruising'
  elif row['phase'] == 'cruising' and row['h[ft]'] >= cru_lim2:
    return 'Cruising(+)'


#calculating distance using longitude and latidute
def getDist(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * \
        cos(rLat2) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c  # Distance in km


#calculating velocity
def getVel(dist_km, time_start, time_end):
    """Return 0 if time_start == time_end, avoid dividing by 0"""
    return dist_km / (time_end - time_start) if time_end > time_start else 0


#adding distances column
def Dist_col(df):
  lat, lon = df['latitude'].to_numpy(), df['longitude'].to_numpy()
  distances = [0] + [1000 * getDist(lat[i-1], lon[i-1], lat[i], lon[i])
                     for i in range(1, len(df))]
  df = df.assign(**{'Distance[m]': distances})
  return df.iloc[1:].reset_index(drop=True)


#adding velocities column
def Vel_col(df):
  dist, posix = df['Distance[m]'].to_numpy(), df['posix'].to_numpy()
  velocities = [0] + [getVel(dist[i], posix[i-1], posix[i]) for i in range(1, len(df))]
  df = df.assign(**{'Velocities[m/s]': velocities})
  return df.iloc[1:].reset_index(drop=True)


#adding velocities and geometric altitude derivatives columns
def vhdots_col(df):
  vel = df['Velocities[m/s]'].to_numpy(dtype=float)
  h = df['h[ft]'].to_numpy(dtype=float)
  dt = np.diff(df['posix'].to_numpy(dtype=float))
  with np.errstate(divide='ignore', invalid='ignore'):
    vdots = np.concatenate([[0], 1.9438444924423 * np.diff(vel) / dt])
    hdots = np.concatenate([[0], 60 * np.diff(h) / dt])
  df = df.assign(**{'vdot[kt/s]': vdots, 'hdot[ft/min]': hdots})
  return df.iloc[1:].reset_index(drop=True)


#cleaning rows including infinities
def drop_inf(df):
  return df[np.isfinite(df['vdot[kt/s]'])].reset_index(drop=True)


#final concatenated dataframe
def data_prep(name):

  pre_df = pd.read_csv(name)
  pre_df = Dist_col(pre_df)
  pre_df = Vel_col(pre_df)
  pre_df = vhdots_col(pre_df)
  pre_df = drop_inf(pre_df)

  to_keep = ['phase', 'h[ft]', 'AirTemperature(C)', 'v_wind_component',
             'u_wind_component', 'latitude', 'longitude', 'vdot[kt/s]', 'hdot[ft/min]']

  df = pre_df[to_keep].copy()
  df.columns = ['phase'] + FEATURES

  def limits(phase):
    h = df.loc[df['phase'] == phase, 'h[ft]']
    delta = (h.max() - h.min()) / 3
    return h.min() + delta, h.min() + 2 * delta

  cli_lim1, cli_lim2 = limits('climbing_ABOVE-FL100')
  cru_lim1, cru_lim2 = limits('cruising')
  desc_lim1, desc_lim2 = limits('descending_ABOVE-FL100')

  df['extrafeatures'] = df.apply(lambda row: extrafeatures33(
      row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2), axis=1)

  return summarize_flight(df, pre_df['posix'], FEATURES, name)


if __name__ == '__main__':
  cli(data_prep, __doc__)
