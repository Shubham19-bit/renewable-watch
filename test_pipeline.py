import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from adapter import extract_sql, standardize
from pipeline import BASE, calibrate, clean, detect, evaluate, generate, quality_gate, run
from vendor_demo import to_vendor


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((BASE / 'config.json').read_text())
        raw, self.events = generate(days=1, seed=91)
        self.df, self.quality = clean(raw)

    def test_quality_challenges(self):
        self.assertEqual(self.quality['duplicate_rows'], 1)
        self.assertEqual(self.quality['rows_with_invalid_or_missing_values'], 2)
        self.assertEqual(self.quality['non_one_minute_intervals'], 1)

    def test_held_out_seed_events(self):
        _, alerts = detect(self.df, self.cfg)
        metrics = evaluate(alerts, self.events)
        self.assertEqual(metrics['true_positive_events'], 15)
        self.assertEqual(metrics['missed_events'], 0)
        self.assertEqual(metrics['false_positive_events'], 0)
        self.assertEqual(metrics['mean_detection_delay_minutes'], 4)

    def test_spike_does_not_trigger(self):
        df = self.df[self.df.asset_id.eq('HYBRID-01')].iloc[:15].copy()
        df.loc[df.index[3], 'igbt_c'] += 40
        _, alerts = detect(df, self.cfg)
        self.assertTrue(alerts.empty)

    def test_gap_breaks_persistence(self):
        df = self.df[self.df.asset_id.eq('HYBRID-01')].iloc[:6].copy()
        df['igbt_c'] += 40
        df = df.drop(df.index[3])
        _, alerts = detect(df, self.cfg)
        self.assertTrue(alerts.empty)

    def test_missing_temperature_breaks_persistence(self):
        df = self.df[self.df.asset_id.eq('HYBRID-01')].iloc[:6].copy()
        df['igbt_c'] += 40
        df.loc[df.index[3], 'igbt_c'] = float('nan')
        _, alerts = detect(df, self.cfg)
        self.assertTrue(alerts.empty)

    def test_duplicate_alert_is_false_positive(self):
        _, alerts = detect(self.df, self.cfg)
        repeated = pd.concat([alerts, alerts.iloc[[0]]], ignore_index=True)
        self.assertEqual(evaluate(repeated, self.events)['false_positive_events'], 1)

    def test_required_schema(self):
        with self.assertRaises(ValueError):
            clean(self.df.drop(columns=['pcs_c']))


class DetectionBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((BASE / 'config.json').read_text())
        raw, self.events = generate(days=1, seed=91)
        self.df, _ = clean(raw)
        self.one = self.df[self.df.asset_id.eq('HYBRID-01')].iloc[:40].copy()

    def test_alert_boundaries(self):
        df = self.one
        df.loc[df.index[3:13], 'igbt_c'] += 40
        _, alerts = detect(df, self.cfg)
        ts = df.timestamp.reset_index(drop=True)
        self.assertEqual(len(alerts), 1)
        a = alerts.iloc[0]
        self.assertEqual((a.onset, a.detected_at, a.end), (ts[3], ts[7], ts[12]))

    def test_flicker_is_one_alert_with_merge_window_and_many_without(self):
        df = self.one
        df['igbt_c'] += 40
        df.loc[df.index[[12, 24]], 'igbt_c'] -= 40  # single-sample dips below threshold
        self.assertEqual(len(detect(df, self.cfg)[1]), 1)
        self.assertGreater(len(detect(df, {**self.cfg, 'merge_gap_minutes': 0})[1]), 1)

    def test_five_minute_cadence_needs_configured_interval(self):
        df = self.df[self.df.timestamp.dt.minute % 5 == 0].reset_index(drop=True)
        cfg = {**self.cfg, 'sample_interval_seconds': 300, 'persistence_minutes': 15}
        _, alerts = detect(df, cfg)
        metrics = evaluate(alerts, self.events)
        self.assertEqual((metrics['true_positive_events'], metrics['false_positive_events']), (15, 0))
        self.assertEqual(metrics['mean_detection_delay_minutes'], 10)
        self.assertTrue(detect(df, self.cfg)[1].empty)  # default 60 s cadence cannot see 5-minute data

    def test_healthy_telemetry_raises_no_alerts(self):
        raw, events = generate(days=2, seed=5, inject=False)
        self.assertTrue(events.empty)
        self.assertTrue(detect(clean(raw)[0], self.cfg)[1].empty)

    def test_faults_below_threshold_are_not_detected(self):
        raw, events = generate(days=2, seed=6, severity=.4)
        metrics = evaluate(detect(clean(raw)[0], self.cfg)[1], events)
        self.assertEqual(metrics['true_positive_events'], 0)
        self.assertEqual(metrics['false_positive_events'], 0)

    def test_ramping_fault_is_detected_late(self):
        raw, events = generate(days=2, seed=6, ramp=True)
        metrics = evaluate(detect(clean(raw)[0], self.cfg)[1], events)
        self.assertEqual(metrics['recall'], 1)
        self.assertGreater(metrics['mean_detection_delay_minutes'], 20)

    def test_calibration_recovers_wrong_baseline(self):
        wrong = {'igbt_k': .045, 'pcs_k': .12, 'container_offset': 6.}
        healthy, _ = generate(days=3, seed=100, inject=False, thermal=wrong)
        fitted = calibrate(clean(healthy)[0], self.cfg)
        self.assertAlmostEqual(fitted['igbt_k_c_per_kw'], .045, delta=.001)
        self.assertAlmostEqual(fitted['pcs_k_c_per_kw'], .12, delta=.002)
        self.assertAlmostEqual(fitted['container_offset_c'], 6, delta=.1)
        raw, events = generate(days=3, seed=101, thermal=wrong)  # different seed: time-separated evaluation
        df, _ = clean(raw)
        uncalibrated = evaluate(detect(df, self.cfg)[1], events)
        calibrated = evaluate(detect(df, fitted)[1], events)
        self.assertGreater(uncalibrated['false_positive_events'], 0)
        self.assertEqual(calibrated['false_positive_events'], 0)
        self.assertEqual(calibrated['recall'], 1)


class DataQualityGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((BASE / 'config.json').read_text())
        self.raw, _ = generate(days=1, seed=91)
        self.df, self.quality = clean(self.raw)

    def test_demo_data_passes(self):
        self.assertTrue(quality_gate(self.quality, self.cfg)['passed'])

    def test_many_invalid_values_fail(self):
        raw = self.raw.copy()
        raw.iloc[::10, raw.columns.get_loc('igbt_c')] = np.nan
        gate = quality_gate(clean(raw)[1], self.cfg)
        self.assertFalse(gate['passed'])
        self.assertTrue(any('invalid_or_missing' in f for f in gate['failures']))

    def test_flatline_is_flagged_and_fails(self):
        df = self.df.copy()
        df.loc[df.index[100:400], 'igbt_c'] = 25.0
        _, quality = clean(df)
        self.assertGreaterEqual(quality['flatline_samples'], 300)
        self.assertIn('igbt_c', quality['flatline_columns'])
        self.assertFalse(quality_gate(quality, self.cfg)['passed'])

    def test_missing_telemetry_lowers_completeness(self):
        hours = pd.to_datetime(self.raw.timestamp).dt.hour
        raw = self.raw[~((self.raw.asset_id == 'HYBRID-02') & hours.between(6, 11))]  # 6 h outage mid-day
        self.assertLess(clean(raw)[1]['asset_completeness']['HYBRID-02'], .8)

    def test_run_writes_report_with_charts(self):
        raw, events = generate(days=1, seed=91)
        with tempfile.TemporaryDirectory() as out:
            summary = run(raw, out, self.cfg, events)
            page = (Path(out) / 'report.html').read_text(encoding='utf-8')
        self.assertEqual(page.count('<svg'), 5)
        self.assertIn('Data-quality gate: PASS', page)
        self.assertTrue(summary['quality_gate']['passed'])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.mapping = json.loads((BASE / 'examples' / 'vendor_mapping.json').read_text())
        self.raw, _ = generate(days=1, seed=3)

    def test_vendor_export_round_trips_to_canonical_schema(self):
        vendor = to_vendor(self.raw).astype(str)  # as read_csv(dtype=str) delivers it
        out = standardize(vendor, self.mapping)
        keys = ['asset_id', 'timestamp']
        expected = self.raw.sort_values(keys, kind='stable').reset_index(drop=True)
        got = out.sort_values(keys, kind='stable').reset_index(drop=True)
        self.assertEqual(list(got.asset_id), list(expected.asset_id))  # leading zeros survive
        self.assertTrue((got.timestamp == pd.to_datetime(expected.timestamp, utc=True)).all())
        for col in expected.columns[2:]:
            np.testing.assert_allclose(got[col], expected[col], rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=col)

    def test_wrong_unit_missing_column_and_partial_mapping_are_rejected(self):
        vendor = to_vendor(self.raw)
        with self.assertRaises(ValueError):
            standardize(vendor, {**self.mapping, 'igbt_c': {'source': 'IGBT_Temp_F', 'unit': 'W'}})
        with self.assertRaises(ValueError):
            standardize(vendor.drop(columns=['SOC_Max_frac']), self.mapping)
        with self.assertRaises(ValueError):
            standardize(vendor, {k: v for k, v in self.mapping.items() if k != 'pcs_c'})

    def test_sql_extraction_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'source.sqlite'
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute('CREATE TABLE src (a INTEGER)')
                conn.execute('INSERT INTO src VALUES (1), (2)')
            self.assertEqual(len(extract_sql(db, 'SELECT * FROM src')), 2)
            with self.assertRaises(ValueError):
                extract_sql(db, 'DROP TABLE src')
            with self.assertRaises(Exception):
                extract_sql(db, 'WITH t AS (SELECT 1) DELETE FROM src')
            self.assertEqual(len(extract_sql(db, 'SELECT * FROM src')), 2)


if __name__ == '__main__':
    unittest.main()
