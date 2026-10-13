"""Reproducible synthetic PV/BESS monitoring demonstration. No control commands."""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
KINDS = ['inverter_overheat', 'pcs_overheat', 'hvac_abnormal', 'soc_imbalance', 'pv_underperformance']
REQUIRED = ['timestamp', 'asset_id', 'ambient_c', 'irradiance_wm2', 'module_c',
            'pv_kw', 'igbt_c', 'pcs_kw', 'pcs_c', 'container_c', 'soc_min_pct', 'soc_max_pct']
NUMERIC = REQUIRED[2:]


def generate(days=7, seed=42):
    """Independent seeded noise plus labelled interventions; labels never enter detector."""
    rng = np.random.default_rng(seed)
    rows, events = [], []
    time = pd.date_range('2026-09-01', periods=days * 1440, freq='min', tz='UTC')
    for asset in ['HYBRID-01', 'HYBRID-02', 'HYBRID-03']:
        hour = time.hour.to_numpy() + time.minute.to_numpy() / 60
        sun = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
        irr = 900 * sun * rng.uniform(.85, 1, len(time))
        ambient = 27 + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
        module = ambient + .025 * irr
        pv = np.minimum(850, 1000 * irr / 1000 * (1 - .004 * (module - 25)) * .97)
        pv *= rng.normal(1, .015, len(time))
        pcs = 200 * np.sin(2 * np.pi * hour / 24)
        df = pd.DataFrame(dict(timestamp=time, asset_id=asset, ambient_c=ambient,
            irradiance_wm2=irr, module_c=module, pv_kw=pv,
            igbt_c=ambient + .03 * pv + rng.normal(0, 1.2, len(time)),
            pcs_kw=pcs, pcs_c=ambient + .06 * abs(pcs) + rng.normal(0, 1.2, len(time)),
            container_c=ambient + 2 + rng.normal(0, .7, len(time)),
            soc_min_pct=50 + 15 * np.sin(2 * np.pi * hour / 24),
            soc_max_pct=52 + 15 * np.sin(2 * np.pi * hour / 24)))
        # One 60-minute event of each kind per asset per day, separated in time.
        for day in range(days):
            for kind, start_hour in zip(KINDS, [9, 11, 13, 15, 16]):
                start = day * 1440 + start_hour * 60
                idx = df.index[start:start + 60]
                column, amount = {'inverter_overheat': ('igbt_c', 20),
                    'pcs_overheat': ('pcs_c', 20), 'hvac_abnormal': ('container_c', 15),
                    'soc_imbalance': ('soc_max_pct', 12),
                    'pv_underperformance': ('pv_kw', .5)}[kind]
                if kind == 'pv_underperformance':
                    df.loc[idx, column] *= amount
                else:
                    df.loc[idx, column] += amount
                events.append(dict(asset_id=asset, kind=kind,
                    start=time[start], end=time[start + 59]))
        rows.append(df)
    raw = pd.concat(rows, ignore_index=True)
    # Explicit data-quality challenges outside injected event windows.
    raw.loc[20, 'igbt_c'] = np.nan
    raw.loc[25, 'soc_min_pct'] = -5
    raw = pd.concat([raw, raw.iloc[[30]]], ignore_index=True)
    raw = raw.drop(index=[35])
    return raw.sample(frac=1, random_state=seed), pd.DataFrame(events)


def clean(raw):
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = raw[REQUIRED].copy()
    df['timestamp'] = pd.to_datetime(df.timestamp, utc=True, errors='coerce')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    invalid = df.timestamp.isna() | df.asset_id.isna() | df.asset_id.astype(str).str.strip().eq('')
    duplicates = df.duplicated(['asset_id', 'timestamp'])
    quality = {'input_rows': len(df), 'invalid_keys': int(invalid.sum()),
               'duplicate_rows': int((duplicates & ~invalid).sum())}
    df = df.loc[~invalid & ~duplicates].sort_values(['asset_id', 'timestamp']).copy()
    ranges = {'soc_min_pct': (0, 100), 'soc_max_pct': (0, 100),
        'irradiance_wm2': (0, 1500), 'pv_kw': (0, 2000), 'module_c': (-40, 120),
        'ambient_c': (-40, 70), 'igbt_c': (-40, 180), 'pcs_c': (-40, 150),
        'container_c': (-40, 100), 'pcs_kw': (-2000, 2000)}
    for col, (lo, hi) in ranges.items():
        df.loc[~df[col].between(lo, hi), col] = np.nan
    bad_soc = df.soc_min_pct > df.soc_max_pct
    df.loc[bad_soc, ['soc_min_pct', 'soc_max_pct']] = np.nan
    quality['rows_with_invalid_or_missing_values'] = int(df[NUMERIC].isna().any(axis=1).sum())
    gaps = df.groupby('asset_id').timestamp.diff().dt.total_seconds()
    quality['non_one_minute_intervals'] = int((gaps.notna() & gaps.ne(60)).sum())
    quality['accepted_rows'] = len(df)
    # Do not interpolate safety-relevant readings or bridge telemetry gaps.
    return df, quality


def detect(df, cfg):
    result = df.copy()
    result['expected_igbt_c'] = df.ambient_c + .03 * df.pv_kw
    result['expected_pcs_c'] = df.ambient_c + .06 * df.pcs_kw.abs()
    result['expected_container_c'] = df.ambient_c + 2
    result['expected_pv_kw'] = (cfg['pv_nameplate_kw'] * df.irradiance_wm2 / 1000 *
        (1 + cfg['pv_temperature_coefficient'] * (df.module_c - 25)) *
        cfg['inverter_efficiency']).clip(lower=0, upper=cfg['inverter_ac_rating_kw'])
    residuals = {'inverter_overheat': df.igbt_c - result.expected_igbt_c,
        'pcs_overheat': df.pcs_c - result.expected_pcs_c,
        'hvac_abnormal': df.container_c - result.expected_container_c,
        'soc_imbalance': df.soc_max_pct - df.soc_min_pct}
    thresholds = [cfg[k] for k in ['inverter_residual_c', 'pcs_residual_c',
                                  'container_residual_c', 'soc_spread_pct']]
    conditions = {}
    for (kind, residual), threshold in zip(residuals.items(), thresholds):
        result[kind + '_residual'] = residual
        conditions[kind] = residual > threshold
    conditions['pv_underperformance'] = (df.irradiance_wm2 > cfg['minimum_irradiance_wm2']) & (
        df.pv_kw < cfg['underperformance_ratio'] * result.expected_pv_kw)
    alerts = []
    for asset, group in df.groupby('asset_id', sort=False):
        for kind, condition in conditions.items():
            active, streak, previous = None, 0, None
            for idx in group.index:
                t = df.at[idx, 'timestamp']
                contiguous = previous is not None and (t - previous).total_seconds() == 60
                if not contiguous or not bool(condition.at[idx]):
                    if active is not None:
                        alerts.append(active)
                    active, streak = None, 0
                if bool(condition.at[idx]):
                    streak += 1
                    if streak == cfg['persistence_minutes']:
                        active = dict(asset_id=asset, kind=kind,
                            onset=t - pd.Timedelta(minutes=streak - 1), detected_at=t, end=t)
                    elif active is not None:
                        active['end'] = t
                previous = t
            if active is not None:
                alerts.append(active)
    return result, pd.DataFrame(alerts, columns=['asset_id', 'kind', 'onset', 'detected_at', 'end'])


def evaluate(alerts, events):
    """One-to-one event overlap matching; delayed detections must occur during event."""
    matched, delays, tp = set(), [], 0
    for alert in alerts.itertuples():
        candidates = events[(events.asset_id == alert.asset_id) & (events.kind == alert.kind)
            & (events.start <= alert.detected_at) & (events.end >= alert.detected_at)]
        candidates = candidates.loc[~candidates.index.isin(matched)]
        if len(candidates):
            i = candidates.index[0]
            matched.add(i)
            tp += 1
            delays.append((alert.detected_at - events.at[i, 'start']).total_seconds() / 60)
    fp, fn = len(alerts) - tp, len(events) - tp
    return dict(true_positive_events=tp, false_positive_events=fp, missed_events=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        mean_detection_delay_minutes=float(np.mean(delays)) if delays else None,
        evaluation_scope='Synthetic injected events only; no real-world generalization claim')


def run(raw, out, cfg, events=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, quality = clean(raw)
    features, alerts = detect(df, cfg)
    with sqlite3.connect(out / 'monitoring.sqlite') as conn:
        features.to_sql('telemetry', conn, if_exists='replace', index=False)
        alerts.to_sql('alerts', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX telemetry_asset_time ON telemetry(asset_id,timestamp)')
    features.to_csv(out / 'standardized.csv', index=False)
    alerts.to_csv(out / 'alerts.csv', index=False)
    summary = {'data_quality': quality, 'alerts_by_type': alerts.kind.value_counts().to_dict()}
    if events is not None:
        summary['validation'] = evaluate(alerts, events)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    # Self-contained evidence report, rather than a hosted application.
    html = '<html><meta charset="utf-8"><title>RenewableWatch</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#16312c}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd}pre{background:#eef5f1;padding:20px}</style><h1>RenewableWatch</h1><p>PV + battery equipment monitoring. Synthetic demonstration, advisory alerts only.</p>'
    html += '<h2>Run summary</h2><pre>' + json.dumps(summary, indent=2) + '</pre><h2>Alert evidence</h2>'
    html += alerts.to_html(index=False) + '</html>'
    (out / 'report.html').write_text(html)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, help='CSV in documented schema; omit for synthetic demo')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=Path, default=BASE / 'config.json')
    args = parser.parse_args()
    if args.days < 1:
        parser.error('--days must be positive')
    cfg = json.loads(args.config.read_text())
    if not isinstance(cfg['persistence_minutes'], int) or cfg['persistence_minutes'] < 1:
        parser.error('persistence_minutes must be a positive integer')
    if args.input:
        run(pd.read_csv(args.input), args.out, cfg)
    else:
        raw, events = generate(args.days, args.seed)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        raw.to_csv(Path(args.out) / 'raw.csv', index=False)
        events.to_csv(Path(args.out) / 'events.csv', index=False)
        run(raw, args.out, cfg, events)
