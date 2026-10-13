import unittest
import pandas as pd
from pipeline import BASE, clean, detect, evaluate, generate
import json


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


if __name__ == '__main__':
    unittest.main()
