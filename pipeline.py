"""Reproducible PV/BESS monitoring pipeline with a synthetic demonstration. Advisory alerts only; no control commands."""
import argparse
import html
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from adapter import SCHEMA as REQUIRED, extract_sql, standardize

BASE = Path(__file__).resolve().parent
KINDS = ['inverter_overheat', 'pcs_overheat', 'hvac_abnormal', 'soc_imbalance', 'pv_underperformance']
NUMERIC = REQUIRED[2:]
FLATLINE_COLUMNS = ['ambient_c', 'igbt_c', 'pcs_c', 'container_c']
THERMAL = {'igbt_k': .03, 'pcs_k': .06, 'container_offset': 2.}
QUALITY_LIMITS = {'max_invalid_key_fraction': .001, 'max_duplicate_fraction': .01,
                  'max_invalid_row_fraction': .01, 'max_irregular_interval_fraction': .01,
                  'min_asset_completeness': .95, 'max_flatline_fraction': .01}


def generate(days=7, seed=42, severity=1., ramp=False, inject=True, thermal=None):
    """Independent seeded noise plus labelled interventions; labels never enter detector.

    severity scales every injected fault (1 = the original large faults); ramp grows a fault linearly over its
    60 minutes instead of a step; inject=False gives healthy telemetry; thermal overrides the true coefficients
    so the detector's assumed baseline can be made wrong on purpose.
    """
    th = {**THERMAL, **(thermal or {})}
    rng = np.random.default_rng(seed)
    rows, events = [], []
    time = pd.date_range('2026-09-01', periods=days * 1440, freq='min', tz='UTC')
    profile = np.arange(1, 61) / 60 if ramp else np.ones(60)
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
            igbt_c=ambient + th['igbt_k'] * pv + rng.normal(0, 1.2, len(time)),
            pcs_kw=pcs, pcs_c=ambient + th['pcs_k'] * abs(pcs) + rng.normal(0, 1.2, len(time)),
            container_c=ambient + th['container_offset'] + rng.normal(0, .7, len(time)),
            soc_min_pct=50 + 15 * np.sin(2 * np.pi * hour / 24),
            soc_max_pct=52 + 15 * np.sin(2 * np.pi * hour / 24)))
        # One 60-minute event of each kind per asset per day, separated in time.
        for day in range(days if inject else 0):
            for kind, start_hour in zip(KINDS, [9, 11, 13, 15, 16]):
                start = day * 1440 + start_hour * 60
                idx = df.index[start:start + 60]
                column, amount = {'inverter_overheat': ('igbt_c', 20),
                    'pcs_overheat': ('pcs_c', 20), 'hvac_abnormal': ('container_c', 15),
                    'soc_imbalance': ('soc_max_pct', 12),
                    'pv_underperformance': ('pv_kw', .5)}[kind]
                if kind == 'pv_underperformance':
                    df.loc[idx, column] *= 1 - (1 - amount) * severity * profile
                else:
                    df.loc[idx, column] += amount * severity * profile
                events.append(dict(asset_id=asset, kind=kind,
                    start=time[start], end=time[start + 59]))
        rows.append(df)
    raw = pd.concat(rows, ignore_index=True)
    # Explicit data-quality challenges outside injected event windows.
    raw.loc[20, 'igbt_c'] = np.nan
    raw.loc[25, 'soc_min_pct'] = -5
    raw = pd.concat([raw, raw.iloc[[30]]], ignore_index=True)
    raw = raw.drop(index=[35])
    return raw.sample(frac=1, random_state=seed), pd.DataFrame(events, columns=['asset_id', 'kind', 'start', 'end'])


def _flatline_samples(series, n):
    """Samples inside runs of >= n identical consecutive readings (stuck sensor / frozen feed)."""
    run = (series != series.shift()).cumsum()
    return int(((series.groupby(run).transform('size') >= n) & series.notna()).sum())


def clean(raw, interval_seconds=60, flatline_samples=60):
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
    df = df.loc[~invalid & ~duplicates].sort_values(['asset_id', 'timestamp']).reset_index(drop=True)
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
    # Key name is historical: counts intervals different from the configured sampling interval.
    quality['non_one_minute_intervals'] = int((gaps.notna() & gaps.ne(interval_seconds)).sum())
    span = df.groupby('asset_id').timestamp.agg(lambda t: (t.max() - t.min()).total_seconds())
    count = df.groupby('asset_id').size()
    quality['asset_completeness'] = {a: round(float(count[a] / (span[a] / interval_seconds + 1)), 5) for a in count.index}
    flat = {c: sum(_flatline_samples(g[c], flatline_samples) for _, g in df.groupby('asset_id')) for c in FLATLINE_COLUMNS}
    quality['flatline_samples'] = sum(flat.values())
    quality['flatline_columns'] = sorted(c for c, n in flat.items() if n)
    quality['accepted_rows'] = len(df)
    # Do not interpolate safety-relevant readings or bridge telemetry gaps.
    return df, quality


def quality_gate(quality, cfg):
    """Automated data-quality checks. Returns {'passed': bool, 'failures': [...]} so a run can be gated in CI."""
    limit = {**QUALITY_LIMITS, **cfg.get('quality_limits', {})}
    rows, total = max(quality['accepted_rows'], 1), max(quality['input_rows'], 1)
    checks = [
        ('invalid_keys', quality['invalid_keys'] / total, limit['max_invalid_key_fraction'], 'max'),
        ('duplicate_rows', quality['duplicate_rows'] / total, limit['max_duplicate_fraction'], 'max'),
        ('rows_with_invalid_or_missing_values', quality['rows_with_invalid_or_missing_values'] / rows,
         limit['max_invalid_row_fraction'], 'max'),
        ('irregular_intervals', quality['non_one_minute_intervals'] / rows,
         limit['max_irregular_interval_fraction'], 'max'),
        ('flatline_samples', quality['flatline_samples'] / rows, limit['max_flatline_fraction'], 'max'),
        ('min_asset_completeness', min(quality['asset_completeness'].values(), default=0),
         limit['min_asset_completeness'], 'min')]
    failures = [f'{name} = {value:.4f} violates {kind} {bound}' for name, value, bound, kind in checks
                if (value > bound if kind == 'max' else value < bound)]
    if quality['accepted_rows'] == 0:
        failures.append('no accepted rows')
    return {'passed': not failures, 'failures': failures}


def _fit_through_origin(x, y):
    """Least squares y = k*x with one robust (MAD) outlier-trim pass; returns k and residual sd."""
    ok = x.notna() & y.notna() & x.ne(0)
    x, y = x[ok], y[ok]
    k = (x * y).sum() / (x * x).sum()
    r = y - k * x
    keep = (r - r.median()).abs() <= 4 * 1.4826 * (r - r.median()).abs().median()
    k = (x[keep] * y[keep]).sum() / (x[keep] ** 2).sum()
    return float(k), float((y[keep] - k * x[keep]).std())


def calibrate(df, cfg):
    """Fit thermal baselines on HEALTHY telemetry (already cleaned). Freeze the result before evaluating later data."""
    k_igbt, sd_igbt = _fit_through_origin(df.pv_kw, df.igbt_c - df.ambient_c)
    k_pcs, sd_pcs = _fit_through_origin(df.pcs_kw.abs(), df.pcs_c - df.ambient_c)
    offset = (df.container_c - df.ambient_c).dropna()
    return {**cfg, 'igbt_k_c_per_kw': round(k_igbt, 5), 'pcs_k_c_per_kw': round(k_pcs, 5),
            'container_offset_c': round(float(offset.median()), 3),
            'calibration': {'rows': len(df), 'igbt_residual_sd_c': round(sd_igbt, 3),
                            'pcs_residual_sd_c': round(sd_pcs, 3), 'container_residual_sd_c': round(float(offset.std()), 3)}}


def _runs(cond, contiguous):
    """Start/end positions of every run of consecutive True samples."""
    continuing = cond & np.r_[False, cond[:-1]] & contiguous
    starts = np.flatnonzero(cond & ~continuing)
    ends = np.flatnonzero(cond & ~np.r_[continuing[1:], False])
    return starts, ends


def _spans(starts, ends, ts, persistence, window):
    """Alert spans as (onset, detected_at, end) positions.

    An alert opens on the first run lasting >= persistence samples; any later run starting within `window`
    extends it, so a marginal fault flickering around the threshold is one alert rather than many.
    """
    spans = []
    for s, e in zip(starts, ends):
        if spans and ts[s] - ts[spans[-1][2]] <= window:
            spans[-1][2] = e
        elif e - s + 1 >= persistence:
            spans.append([s, s + persistence - 1, e])
    return spans


def detect(df, cfg):
    interval = cfg.get('sample_interval_seconds', 60)
    window = pd.Timedelta(minutes=cfg.get('merge_gap_minutes', 0))
    persistence = max(1, round(cfg['persistence_minutes'] * 60 / interval))
    result = df.copy()
    result['expected_igbt_c'] = df.ambient_c + cfg.get('igbt_k_c_per_kw', THERMAL['igbt_k']) * df.pv_kw
    result['expected_pcs_c'] = df.ambient_c + cfg.get('pcs_k_c_per_kw', THERMAL['pcs_k']) * df.pcs_kw.abs()
    result['expected_container_c'] = df.ambient_c + cfg.get('container_offset_c', THERMAL['container_offset'])
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
        conditions[kind] = residual > threshold  # NaN compares False: missing data never alarms
    conditions['pv_underperformance'] = (df.irradiance_wm2 > cfg['minimum_irradiance_wm2']) & (
        df.pv_kw < cfg['underperformance_ratio'] * result.expected_pv_kw)
    alerts = []
    for asset, group in df.groupby('asset_id', sort=False):
        group = group.sort_values('timestamp')
        ts = group.timestamp.reset_index(drop=True)
        # Only an exact sampling interval continues a run: gaps and missing values reset persistence.
        contiguous = (ts.diff().dt.total_seconds() == interval).to_numpy()
        for kind, condition in conditions.items():
            cond = condition.loc[group.index].to_numpy(dtype=bool)
            for onset, detected, end in _spans(*_runs(cond, contiguous), ts, persistence, window):
                alerts.append(dict(asset_id=asset, kind=kind, onset=ts[onset], detected_at=ts[detected], end=ts[end]))
    return result, pd.DataFrame(alerts, columns=['asset_id', 'kind', 'onset', 'detected_at', 'end'])


def evaluate(alerts, events):
    """One-to-one event overlap matching; delayed detections must occur during event."""
    matched, delays, tp = set(), [], 0
    for alert in alerts.itertuples() if len(events) else []:
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


CSS = '''
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--grid:#e3e2de;--line:#2a78d6;--alert:#eb6834;--panel:#eef5f1}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--grid:#3a3937;--line:#3987e5;--alert:#d95926;--panel:#232826}}
body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 16px;background:var(--bg);color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:6px 8px;border:1px solid var(--grid);text-align:left}
pre{background:var(--panel);padding:16px;overflow:auto;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}figure{margin:0}
figcaption{font-size:14px;color:var(--muted)}svg{width:100%;height:auto}
svg text{font:11px system-ui;fill:var(--muted)}.gate{padding:8px 12px;border:1px solid var(--grid);font-weight:600}
'''


def _chart(ts, y, threshold, onset, detected, end, unit, below=False, w=520, h=190):
    """Inline-SVG evidence chart: signal, dashed threshold, shaded alert window, detection marker."""
    pl, pr, pt, pb = 44, 12, 12, 24
    t0, secs = ts.iloc[0], max((ts.iloc[-1] - ts.iloc[0]).total_seconds(), 1)
    fx = lambda t: pl + (t - t0).total_seconds() / secs * (w - pl - pr)
    lo, hi = min(np.nanmin(y), threshold), max(np.nanmax(y), threshold)
    pad = (hi - lo or 1) * .08
    lo, hi = lo - pad, hi + pad
    fy = lambda v: pt + (hi - v) / (hi - lo) * (h - pt - pb)
    d, pen = [], False
    for t, v in zip(ts, y):
        if np.isnan(v):
            pen = False
            continue
        d.append(f'{"L" if pen else "M"}{fx(t):.1f},{fy(v):.1f}')
        pen = True
    grid = ''.join(f'<line x1="{pl}" x2="{w - pr}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{pl - 6}" y="{fy(v) + 4:.1f}" text-anchor="end">{v:.3g}</text>'
                   for v in np.linspace(lo + pad, hi - pad, 3))
    ty = fy(threshold)
    side = 'alert below' if below else 'alert above'
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(unit)} around an alert">{grid}'
        f'<rect x="{fx(onset):.1f}" y="{pt}" width="{max(fx(end) - fx(onset), 1):.1f}" height="{h - pt - pb}" '
        f'fill="var(--alert)" opacity=".16"><title>Alert window {onset:%H:%M}-{end:%H:%M} UTC</title></rect>'
        f'<line x1="{pl}" x2="{w - pr}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="var(--muted)" stroke-dasharray="5 4"/>'
        f'<text x="{w - pr}" y="{ty - 4:.1f}" text-anchor="end">threshold {threshold:g} ({side})</text>'
        f'<path d="{" ".join(d)}" fill="none" stroke="var(--line)" stroke-width="2"/>'
        f'<line x1="{fx(detected):.1f}" x2="{fx(detected):.1f}" y1="{pt}" y2="{h - pb}" stroke="var(--alert)" stroke-width="2">'
        f'<title>Detected {detected:%H:%M} UTC</title></line>'
        f'<text x="{pl}" y="{h - 6}">{ts.iloc[0]:%d %b %H:%M}</text>'
        f'<text x="{w - pr}" y="{h - 6}" text-anchor="end">{ts.iloc[-1]:%H:%M} UTC</text></svg>')


def build_report(summary, features, alerts, cfg):
    gate = summary['quality_gate']
    page = [f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>RenewableWatch</title><style>{CSS}</style><body><h1>RenewableWatch</h1>'
            '<p>PV + battery equipment monitoring. Advisory alerts only; no control commands. '
            'Synthetic data unless a real input was supplied.</p>',
            f'<p class="gate">Data-quality gate: {"PASS" if gate["passed"] else "FAIL"}'
            + ('' if gate['passed'] else ' &mdash; ' + html.escape('; '.join(gate['failures']))) + '</p>',
            '<h2>Alert evidence</h2><p>First alert of each type: residual (measured minus physical baseline) with the '
            'threshold, the alert window, and the moment persistence was satisfied.</p><div class="grid">']
    interval = pd.Timedelta(seconds=cfg.get('sample_interval_seconds', 60))
    specs = {'inverter_overheat': ('IGBT temp. residual', 'C above baseline', cfg['inverter_residual_c']),
             'pcs_overheat': ('PCS temp. residual', 'C above baseline', cfg['pcs_residual_c']),
             'hvac_abnormal': ('Container temp. residual', 'C above baseline', cfg['container_residual_c']),
             'soc_imbalance': ('SOC spread (max - min)', 'percentage points', cfg['soc_spread_pct']),
             'pv_underperformance': ('PV measured / expected', 'ratio', cfg['underperformance_ratio'])}
    for kind, (title, unit, threshold) in specs.items():
        hit = alerts[alerts.kind == kind].head(1)
        if hit.empty:
            continue
        a = hit.iloc[0]
        win = features[(features.asset_id == a.asset_id) & (features.timestamp >= a.onset - 90 * pd.Timedelta(minutes=1))
                       & (features.timestamp <= a.end + 90 * pd.Timedelta(minutes=1))].sort_values('timestamp')
        if kind == 'pv_underperformance':
            y = (win.pv_kw / win.expected_pv_kw).where(win.irradiance_wm2 > cfg['minimum_irradiance_wm2']).to_numpy()
        else:
            y = win[kind + '_residual'].to_numpy()
        if np.isnan(y).all():
            continue
        page.append(f'<figure>{_chart(win.timestamp, y, threshold, a.onset, a.detected_at, a.end, unit, kind == "pv_underperformance")}'
                    f'<figcaption><b>{title}</b>, {a.asset_id}, {unit}. Onset {a.onset:%H:%M}, detected {a.detected_at:%H:%M} UTC.</figcaption></figure>')
    page.append('</div><h2>Run summary</h2><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>'
                f'<h2>Alerts ({len(alerts)})</h2>')
    if len(alerts) > 200:
        page.append('<p>Showing the first 200; the full list is in alerts.csv.</p>')
    page.append(alerts.head(200).to_html(index=False) + '</body></html>')
    return ''.join(page)


def run(raw, out, cfg, events=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    interval = cfg.get('sample_interval_seconds', 60)
    df, quality = clean(raw, interval, cfg.get('flatline_samples', 60))
    features, alerts = detect(df, cfg)
    with closing(sqlite3.connect(out / 'monitoring.sqlite')) as conn, conn:
        features.to_sql('telemetry', conn, if_exists='replace', index=False)
        alerts.to_sql('alerts', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX telemetry_asset_time ON telemetry(asset_id,timestamp)')
    features.to_csv(out / 'standardized.csv', index=False)
    alerts.to_csv(out / 'alerts.csv', index=False)
    summary = {'data_quality': quality, 'quality_gate': quality_gate(quality, cfg),
               'alerts_by_type': alerts.kind.value_counts().to_dict()}
    if events is not None:
        summary['validation'] = evaluate(alerts, events)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    # Self-contained evidence report, rather than a hosted application.
    (out / 'report.html').write_text(build_report(summary, features, alerts, cfg), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def load_raw(args):
    """Extract: SQLite query, CSV, or the synthetic demo; then optionally map a vendor schema onto ours."""
    if args.db:
        if not args.query:
            raise SystemExit('--db requires --query')
        raw = extract_sql(args.db, args.query)
    else:
        raw = pd.read_csv(args.input, dtype=str)  # keep IDs like '01' intact; numerics are coerced later
    return standardize(raw, json.loads(args.mapping.read_text())) if args.mapping else raw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='CSV in documented schema; omit for synthetic demo')
    parser.add_argument('--db', type=Path, help='SQLite database to extract from (read-only); needs --query')
    parser.add_argument('--query', help='SELECT statement returning the source columns')
    parser.add_argument('--mapping', type=Path, help='JSON mapping from the source schema/units to the pipeline schema')
    parser.add_argument('--calibrate-input', type=Path, help='HEALTHY telemetry (CSV) to fit thermal baselines on; '
                        'writes calibrated_config.json to --out and exits')
    parser.add_argument('--strict', action='store_true', help='exit with status 2 if the data-quality gate fails')
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
    if args.calibrate_input:
        healthy, _ = clean(pd.read_csv(args.calibrate_input, dtype=str), cfg.get('sample_interval_seconds', 60))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        calibrated = calibrate(healthy, cfg)
        (Path(args.out) / 'calibrated_config.json').write_text(json.dumps(calibrated, indent=2))
        print(json.dumps({k: calibrated[k] for k in ['igbt_k_c_per_kw', 'pcs_k_c_per_kw', 'container_offset_c', 'calibration']}, indent=2))
        sys.exit(0)
    if args.input or args.db:
        summary = run(load_raw(args), args.out, cfg)
    else:
        raw, events = generate(args.days, args.seed)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        raw.to_csv(Path(args.out) / 'raw.csv', index=False)
        events.to_csv(Path(args.out) / 'events.csv', index=False)
        summary = run(raw, args.out, cfg, events)
    if args.strict and not summary['quality_gate']['passed']:
        sys.exit(2)
