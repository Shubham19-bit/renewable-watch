# RenewableWatch: Physics-Based Solar and Battery Equipment Monitoring

Runnable Python portfolio prototype aligned with solar PV and battery-storage operations analytics. It demonstrates ingestion, quality checks, physical residuals, persistent alerts, SQL storage and labelled-event validation. It is independently developed; it uses no DESRI proprietary code or operational data.

## Run

Python 3.10+ recommended. From this directory:

```bash
python -m pip install -r requirements.txt
python pipeline.py --days 7 --out outputs
python -m unittest -v
```

Open `outputs/report.html` in a browser. The included outputs are an example run. The core pipeline needs no cloud service, API key or GPU.

## Pipeline

1. Generate reproducible one-minute telemetry for three hybrid assets, or ingest your CSV.
2. Convert timestamps to UTC, coerce numeric values, remove duplicate keys, flag missing samples and mask out-of-range values.
3. Compute expected power/temperature and SOC spread.
4. Require consecutive one-minute violations; missing observations and telemetry gaps reset persistence.
5. Write standardised telemetry and alerts to SQLite and CSV, and generate a self-contained HTML evidence report.
6. Evaluate alerts against separate injected-event labels using one-to-one matching.

## Detection rules

| Use case | Signal | Default trigger |
|---|---|---|
| Inverter IGBT overheating | Measured temperature minus ambient/load baseline | Residual >12 C |
| Battery PCS overheating | Measured temperature minus ambient/absolute-load baseline | Residual >12 C |
| Container HVAC abnormality | Container temperature minus ambient+2 C | Residual >8 C |
| SOC imbalance | Maximum minus minimum container SOC | Spread >8 percentage points |
| Solar underperformance | Measured AC power / expected AC power | <0.75 with irradiance >150 W/m2 |

All alerts require five consecutive abnormal samples by default. `onset` is the first abnormal sample; `detected_at` is when persistence is satisfied. Five consecutive samples therefore produce a four-minute delay from onset.

The PV baseline uses a simplified irradiance- and module-temperature-dependent DC power equation followed by constant efficiency and AC clipping. See Sandia PVPMC's [PVWatts model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/point-value-models/pvwatts/) for the equation background. This is not a full PVWatts implementation.

The thermal baselines are illustrative steady-state engineering surrogates: `T_igbt = T_ambient + 0.03*P_pv` and `T_pcs = T_ambient + 0.06*abs(P_pcs)`. They approximate load-dependent heating; they do not estimate junction temperature from validated thermal impedances. HVAC detection identifies unusual heat, not its root cause. SOC differences alone do not diagnose battery damage.

## CSV schema and conventions

One row per hybrid asset per minute. Required columns:

```text
timestamp,asset_id,ambient_c,irradiance_wm2,module_c,pv_kw,igbt_c,pcs_kw,pcs_c,container_c,soc_min_pct,soc_max_pct
```

UTC ISO timestamp; temperatures in Celsius; plane-of-array irradiance in W/m2; powers in kW; SOC in percent. Positive PCS power means discharge; negative means charge. SOC min/max must describe comparable battery containers in the same asset. Real deployments should separate device identities and use a site-specific mapping layer.

```bash
python pipeline.py --input your_telemetry.csv --out real_run
```

Labels are never required or consumed by the detector. The custom-input command does not compute labelled-event metrics. Invalid values stay missing rather than being interpolated. First duplicate wins. Gaps are counted, not filled. There is no streaming or incremental database append: each run replaces the SQLite tables.

## Validation and honest reporting

The generator injects one 60-minute event of each type per asset per day. It also introduces a missing reading, invalid SOC, duplicate and missing timestamp. Synthetic events deliberately exceed the thresholds; excellent synthetic scores verify mechanics, not real-world effectiveness. A separate random seed is used in the regression tests, but it shares the same simulator and cannot establish external validity.

Do not call this predictive maintenance: it detects sustained anomalies after onset. Production readiness would require real historian/SCADA data, asset-specific calibration, curtailment and maintenance masks, operational labels, drift handling, false-alert analysis and measured lead time before failures.

## SQL examples

```sql
SELECT kind, COUNT(*) AS alerts FROM alerts GROUP BY kind;
SELECT asset_id, timestamp, pv_kw, expected_pv_kw
FROM telemetry WHERE irradiance_wm2 > 150 AND pv_kw < 0.75*expected_pv_kw;
```

## Next development steps

1. Add an adapter for a real telemetry export, including units and device IDs.
2. Calibrate thermal baselines on healthy training periods; freeze them before time-separated evaluation.
3. Add curtailment, charge/discharge mode and planned-maintenance context.
4. Compare physical residual rules with an unsupervised anomaly model on the same evaluation period.
5. Package incremental ingestion, logging, CI and a dashboard after the core data/model validation.

## Resume wording after you run and understand the project

"Built a Python/SQL prototype for one-minute solar PV and battery telemetry, implementing data-quality checks and persistent physics-based alerts for inverter/PCS overheating, HVAC abnormalities and SOC imbalance; evaluated against labelled synthetic events."

Report synthetic metrics explicitly as synthetic. Complete independent interview preparation; the DESRI notice prohibits AI assistance during its assessment/interview.
