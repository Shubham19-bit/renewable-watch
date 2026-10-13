"""Harder validation scenarios. All synthetic: they probe detector behaviour, not real-world accuracy.

Run: python benchmark.py --out outputs   (writes benchmark.json and benchmark.md)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline import BASE, KINDS, calibrate, clean, detect, evaluate, generate

# A baseline that is wrong on purpose: the detector assumes THERMAL defaults, the "plant" differs.
MISMATCH = {'igbt_k': .045, 'pcs_k': .12, 'container_offset': 6.}


def score(cfg, raw, events):
    df, _ = clean(raw)
    _, alerts = detect(df, cfg)
    total = evaluate(alerts, events)
    per_kind = {}
    for kind in KINDS:
        m = evaluate(alerts[alerts.kind == kind], events[events.kind == kind]) if len(events) else {}
        per_kind[kind] = {'recall': m.get('recall'), 'delay_min': m.get('mean_detection_delay_minutes')}
    days = df.timestamp.nunique() / 1440
    total['false_alerts_per_asset_day'] = round(total['false_positive_events'] / (df.asset_id.nunique() * days), 3)
    total.pop('evaluation_scope')
    return total, per_kind


def run_all(cfg):
    out = {}
    raw, events = generate(14, seed=7)
    out['baseline_full_size_faults'], _ = score(cfg, raw, events)

    sweep = {}
    for sev in [1., .8, .7, .6, .55, .5, .4]:
        raw, events = generate(14, seed=7, severity=sev)
        total, per_kind = score(cfg, raw, events)
        sweep[str(sev)] = {'overall': total, 'per_kind': per_kind}
    out['severity_sweep'] = sweep

    raw, events = generate(14, seed=7, ramp=True)
    total, per_kind = score(cfg, raw, events)
    out['ramping_faults_full_size'] = {'overall': total, 'per_kind': per_kind}

    raw, events = generate(14, seed=11, inject=False)
    out['healthy_only'], _ = score(cfg, raw, events)

    healthy, _ = generate(7, seed=100, inject=False, thermal=MISMATCH)
    calibrated = calibrate(clean(healthy)[0], cfg)
    raw, events = generate(7, seed=101, thermal=MISMATCH)  # time-separated: different seed, frozen coefficients
    uncal, _ = score(cfg, raw, events)
    cal, _ = score(calibrated, raw, events)
    out['model_mismatch'] = {'true_coefficients': MISMATCH,
        'fitted_coefficients': {k: calibrated[k] for k in ['igbt_k_c_per_kw', 'pcs_k_c_per_kw', 'container_offset_c']},
        'uncalibrated': uncal, 'calibrated_on_separate_healthy_week': cal}
    return out


def markdown(res):
    f = lambda v: '-' if v is None else f'{v:.2f}'
    lines = ['## Severity sweep (14 days, 3 assets; 1.0 = original fault size)', '',
             '| severity | overall recall | precision | ' + ' | '.join(KINDS) + ' (recall) |', '|---|---|---|' + '---|' * len(KINDS)]
    for sev, r in res['severity_sweep'].items():
        o = r['overall']
        lines.append(f'| {sev} | {f(o["recall"])} | {f(o["precision"])} | '
                     + ' | '.join(f(r['per_kind'][k]['recall']) for k in KINDS) + ' |')
    ramp = res['ramping_faults_full_size']
    lines += ['', '## Ramping faults (grow linearly over 60 min)', '',
              '| kind | recall | mean delay (min) |', '|---|---|---|']
    lines += [f'| {k} | {f(v["recall"])} | {f(v["delay_min"])} |' for k, v in ramp['per_kind'].items()]
    mm = res['model_mismatch']
    lines += ['', '## Wrong thermal baseline vs calibrated baseline', '',
              '| detector | recall | precision | false alerts |', '|---|---|---|---|']
    for name, key in [('uncalibrated defaults', 'uncalibrated'), ('calibrated on healthy week', 'calibrated_on_separate_healthy_week')]:
        m = mm[key]
        lines.append(f'| {name} | {f(m["recall"])} | {f(m["precision"])} | {m["false_positive_events"]} |')
    h = res['healthy_only']
    lines += ['', f'Healthy-only run: {h["false_positive_events"]} false alerts over 14 days x 3 assets.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--config', type=Path, default=BASE / 'config.json')
    args = parser.parse_args()
    results = run_all(json.loads(args.config.read_text()))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'benchmark.json').write_text(json.dumps(results, indent=2))
    (Path(args.out) / 'benchmark.md').write_text(markdown(results))
    print(markdown(results))
