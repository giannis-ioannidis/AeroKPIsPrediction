"""Generate synthetic raw flight trajectories in the formats expected by
ToCSV/PprToCSV.py and ToCSV/OprToCSV.py.

Each flight is a climb -> cruise (with two step-climbs) -> descent profile
sampled every 10 s. Cost Index (CI) and Max Payload (MPL) are encoded in the
file name (SIMBAD_<id>_CI_<ci>_MPL_<mpl>.csv) and influence climb/descent
rates, speeds, cruise level and cruise length, so the model has a learnable
signal.

Output layout:
    <out-dir>/ppr/{train,test}/SIMBAD_00001_CI_45_MPL_15.3.csv
    <out-dir>/opr/{train,test}/SIMBAD_00001_CI_45_MPL_15.3.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

DT = 10  # seconds between samples
START_ALT = 1500.0  # ft
FL100 = 10000.0  # ft


def build_profile(ci, mpl, rng):
    """Return a dict of numpy arrays describing one flight."""
    climb_rate = 1800 + 15 * ci - 60 * (mpl - 10) + rng.normal(0, 80)  # ft/min
    desc_rate = 1500 + 12 * ci + rng.normal(0, 80)  # ft/min
    cruise_alt = 36000 - 400 * (mpl - 10) - 20 * ci + rng.normal(0, 300)
    cruise_steps = int(max(60, 150 - 0.6 * ci + rng.normal(0, 10)))
    seg_len = cruise_steps // 3

    h, block = [], []

    # climb
    alt = START_ALT
    while alt < cruise_alt:
        h.append(alt)
        block.append('climb')
        alt += climb_rate / 60 * DT * (1 + rng.normal(0, 0.03))

    # cruise: three flight levels separated by 1000 ft step-climbs
    for level in range(3):
        level_alt = cruise_alt + 1000 * level
        for k in range(seg_len):
            if level > 0 and k < 5:  # short transition into the new level
                h.append(level_alt - 1000 + 200 * (k + 1))
            else:
                h.append(level_alt + rng.normal(0, 15))
            block.append('cruise')

    # descent
    alt = h[-1]
    while alt > START_ALT:
        alt -= desc_rate / 60 * DT * (1 + rng.normal(0, 0.03))
        h.append(max(alt, START_ALT - 100))
        block.append('descent')

    h = np.asarray(h)
    block = np.asarray(block)
    n = len(h)
    posix = 1_600_000_000 + rng.integers(0, 10_000_000) + DT * np.arange(n)

    # true airspeed (kt), depends on phase and CI
    tas = np.where(block == 'climb', 280 + 0.5 * ci,
                   np.where(block == 'cruise', 440 + 0.8 * ci, 300 + 0.4 * ci))
    tas = tas * (1 + rng.normal(0, 0.01, n))

    # position: fly roughly east from a random origin
    lat0, lon0 = rng.uniform(35, 55), rng.uniform(-5, 20)
    heading = np.radians(rng.uniform(60, 120))
    step_nm = tas * DT / 3600
    lat = lat0 + np.cumsum(step_nm * np.cos(heading) / 60)
    lon = lon0 + np.cumsum(step_nm * np.sin(heading) / (60 * np.cos(np.radians(lat))))

    # atmosphere (ISA + offset) and winds
    temp = 15 - 1.98 * h / 1000 + rng.normal(0, 3) + rng.normal(0, 0.3, n)
    press = 1013.25 * (1 - 6.8756e-6 * h) ** 5.2559
    wn = rng.normal(0, 20) + rng.normal(0, 3, n)
    we = rng.normal(15, 20) + rng.normal(0, 3, n)

    return dict(h=h, block=block, posix=posix, tas=tas, lat=lat, lon=lon,
                temp=temp, press=press, wn=wn, we=we, heading=heading)


def to_ppr(p):
    phase = np.empty(len(p['h']), dtype=object)
    below = p['h'] < FL100
    phase[(p['block'] == 'climb') & below] = 'climbing_BELOW-FL100'
    phase[(p['block'] == 'climb') & ~below] = 'climbing_ABOVE-FL100'
    phase[p['block'] == 'cruise'] = 'cruising'
    phase[(p['block'] == 'descent') & ~below] = 'descending_ABOVE-FL100'
    phase[(p['block'] == 'descent') & below] = 'descending_BELOW-FL100'
    return pd.DataFrame({
        'phase': phase,
        'posix': p['posix'],
        'h[ft]': p['h'],
        'AirTemperature(C)': p['temp'],
        'v_wind_component': p['wn'],
        'u_wind_component': p['we'],
        'latitude': p['lat'],
        'longitude': p['lon'],
    })


def to_opr(p):
    n = len(p['h'])
    hdot = np.gradient(p['h'], DT) * 60  # ft/min
    vdot = np.gradient(p['tas'], DT)  # kt/s
    # along-track / cross-track wind components
    ws = p['wn'] * np.cos(p['heading']) + p['we'] * np.sin(p['heading'])
    wx = -p['wn'] * np.sin(p['heading']) + p['we'] * np.cos(p['heading'])

    phase_name = np.empty(n, dtype=object)
    phase_name[p['block'] == 'climb'] = 'CLIMB'
    phase_name[p['block'] == 'descent'] = 'DESCENT'
    cruise_idx = np.flatnonzero(p['block'] == 'cruise')
    for k, idx in enumerate(np.array_split(cruise_idx, 3)):
        phase_name[idx] = f'CRUISE_{k + 1}'

    return pd.DataFrame({
        'Flight_Block': p['block'],
        'Phase': phase_name,
        'posix[s]': p['posix'],
        'h[ft]': p['h'],
        'Temp[ºC]': p['temp'],
        'Press[hPa]': p['press'],
        'Wn[kt]': p['wn'],
        'We[kt]': p['we'],
        'Ws[kt]': ws,
        'Wx[kt]': wx,
        'Lat[º]': p['lat'],
        'Lon[º]': p['lon'],
        'vdot[kt/s]': vdot,
        'hdot[ft/min]': hdot,
        'reserved_1': 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'raw'))
    parser.add_argument('--n-train', type=int, default=200)
    parser.add_argument('--n-test', type=int, default=50)
    parser.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    splits = [('train', args.n_train), ('test', args.n_test)]
    for var_set in ('ppr', 'opr'):
        for split, _ in splits:
            os.makedirs(os.path.join(args.out_dir, var_set, split), exist_ok=True)

    flight_id = 0
    for split, count in splits:
        for _ in range(count):
            flight_id += 1
            ci = int(rng.integers(0, 101))
            mpl = round(float(rng.uniform(10.0, 20.0)), 1)
            name = f'SIMBAD_{flight_id:05d}_CI_{ci}_MPL_{mpl:.1f}.csv'
            profile = build_profile(ci, mpl, rng)
            to_ppr(profile).to_csv(os.path.join(args.out_dir, 'ppr', split, name), index=False)
            to_opr(profile).to_csv(os.path.join(args.out_dir, 'opr', split, name), index=False)

    print(f'Wrote {flight_id} flights per variable set to {args.out_dir}')


if __name__ == '__main__':
    main()
