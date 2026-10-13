"""Demo of the source adapter: writes a fake vendor export (Fahrenheit, W, SOC fraction, IST, split asset id).

    python vendor_demo.py --out vendor_export.csv
    python pipeline.py --input vendor_export.csv --mapping examples/vendor_mapping.json --out vendor_run
"""
import argparse

import pandas as pd

from pipeline import generate


def to_vendor(raw):
    """Re-express the canonical schema the way a hypothetical new source would publish it."""
    f = lambda c: raw[c] * 9 / 5 + 32
    ts = pd.to_datetime(raw.timestamp, utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    site_block = raw.asset_id.str.split('-', n=1, expand=True)
    return pd.DataFrame({'Time_IST': ts.dt.strftime('%Y-%m-%d %H:%M:%S'), 'Site': site_block[0], 'Block': site_block[1],
        'AmbTemp_F': f('ambient_c'), 'POA_kWm2': raw.irradiance_wm2 / 1000, 'ModTemp_F': f('module_c'),
        'PV_Power_W': raw.pv_kw * 1000, 'IGBT_Temp_F': f('igbt_c'), 'PCS_Power_MW': raw.pcs_kw / 1000,
        'PCS_Temp_K': raw.pcs_c + 273.15, 'Container_Temp_F': f('container_c'),
        'SOC_Min_frac': raw.soc_min_pct / 100, 'SOC_Max_frac': raw.soc_max_pct / 100})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='vendor_export.csv')
    parser.add_argument('--days', type=int, default=1)
    args = parser.parse_args()
    raw, _ = generate(args.days)
    to_vendor(raw).to_csv(args.out, index=False)
