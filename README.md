# RenewableWatch: Physics-Based Solar and Battery Equipment Monitoring

![tests](https://github.com/Shubham19-bit/renewable-watch/actions/workflows/tests.yml/badge.svg)

Runnable Python prototype of a solar PV and battery-storage monitoring pipeline: extract, sanitise, standardise, transform with first-principles physics, alert, store in SQL, and validate against labelled events. It is independently developed; it uses no DESRI proprietary code or operational data. Advisory alerts only; nothing here sends control commands.

## Run

Python 3.10+ recommended. No cloud service, API key or GPU is needed (pandas + numpy only).

```bash
python -m pip install -r requirements.txt
python pipeline.py --days 7 --out outputs        # synthetic demo + report
python benchmark.py --out outputs                # harder validation scenarios
python -m unittest -v                            # 22 tests
```

Open `outputs/report.html`: data-quality verdict, one evidence chart per alert type (residual, threshold, alert window, detection time), run summary and alert table.

## Pipeline

1. **Extract**: synthetic generator, a CSV, or a read-only SQL query (`--db source.sqlite --query "SELECT ..."`; only `SELECT`/`WITH`, opened `mode=ro`).
2. **Standardise** (`adapter.py`): a JSON mapping renames columns, joins identifier parts, converts units (F/K to C, W/MW to kW, SOC fraction to percent, kW/m2 to W/m2) and converts local time to UTC. Wrong units, missing columns and partial mappings are rejected. See `examples/vendor_mapping.json` and `vendor_demo.py` for a worked example of an export in a different schema.
3. **Sanitise** (`clean`): UTC timestamps, numeric coercion, first-duplicate-wins, range masking (invalid values become missing, never interpolated), gap counting, per-asset completeness, flatline (stuck sensor) detection.
4. **Data-quality gate** (`quality_gate`): thresholds in `config.json`; `--strict` exits with status 2 on failure, so CI or a scheduler can stop bad data before it produces alerts.
5. **Physics residuals**: expected power/temperature and SOC spread (below).
6. **Persistence + debounce**: an alert opens after `persistence_minutes` of consecutive abnormal samples; missing samples and telemetry gaps reset persistence; a later run within `merge_gap_minutes` extends the same alert instead of opening a new one.
7. **Store and report**: SQLite (`telemetry`, `alerts`), CSVs, self-contained HTML report.
8. **Validate**: alerts vs separate injected-event labels with one-to-one matching (labels never reach the detector).

## Detection rules

| Use case | Signal | Default trigger |
|---|---|---|
| Inverter IGBT overheating | Measured temperature minus ambient/load baseline | Residual >12 C |
| Battery PCS overheating | Measured temperature minus ambient/absolute-load baseline | Residual >12 C |
| Container HVAC abnormality | Container temperature minus ambient+offset | Residual >8 C |
| SOC imbalance | Maximum minus minimum container SOC | Spread >8 percentage points |
| Solar underperformance | Measured AC power / expected AC power | <0.75 with irradiance >150 W/m2 |

`onset` is the first abnormal sample; `detected_at` is when persistence is satisfied (four-minute delay at 1-minute cadence and 5-minute persistence); `end` is the last abnormal sample. `persistence_minutes` is a duration: with `sample_interval_seconds: 300` and `persistence_minutes: 15`, three consecutive samples are required. Data at a cadence other than `sample_interval_seconds` produces no alerts, by design (and the cadence mismatch shows up in the quality report).

The PV baseline uses a simplified irradiance- and module-temperature-dependent DC power equation followed by constant efficiency and AC clipping. See Sandia PVPMC's [PVWatts model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/point-value-models/pvwatts/). This is not a full PVWatts implementation.

The thermal baselines are illustrative steady-state surrogates: `T_igbt = T_ambient + k_igbt*P_pv`, `T_pcs = T_ambient + k_pcs*abs(P_pcs)`, `T_container = T_ambient + offset`, with defaults `k_igbt=0.03`, `k_pcs=0.06`, `offset=2` in `config.json`. They do not estimate junction temperature from validated thermal impedances. HVAC detection identifies unusual heat, not its root cause. SOC differences alone do not diagnose battery damage.

### Calibration

Real equipment will not match the defaults. Fit the coefficients on a healthy period, freeze them, then evaluate on a later period:

```bash
python pipeline.py --calibrate-input healthy_period.csv --out cal      # writes cal/calibrated_config.json
python pipeline.py --input later_period.csv --config cal/calibrated_config.json --out run
```

The fit is least squares through the origin with one MAD-based outlier trim. It assumes the training window is mostly healthy.

## CSV schema and conventions

One row per asset per sampling interval. Required columns:

```text
timestamp,asset_id,ambient_c,irradiance_wm2,module_c,pv_kw,igbt_c,pcs_kw,pcs_c,container_c,soc_min_pct,soc_max_pct
```

UTC ISO timestamp; temperatures in Celsius; plane-of-array irradiance in W/m2; powers in kW; SOC in percent. Positive PCS power means discharge; negative means charge. SOC min/max must describe comparable battery containers in the same asset. CSVs are read as text so identifiers such as `01` keep their leading zeros.

```bash
python pipeline.py --input your_telemetry.csv --out real_run                      # canonical schema
python pipeline.py --input vendor.csv --mapping examples/vendor_mapping.json --out real_run   # other schema/units
```

Labels are never required or consumed by the detector, so custom-input runs report no labelled-event metrics. There is no streaming or incremental database append: each run replaces the SQLite tables.

## Validation and honest reporting

All numbers below are **synthetic**. The detector's baseline equations are the same ones the generator uses for healthy data, so a perfect score on full-size faults only verifies mechanics. `benchmark.py` therefore also probes behaviour that is not trivially perfect (14 days, 3 assets, results in `outputs/benchmark.md`):

| Scenario | Result |
|---|---|
| Full-size faults (about 1.7x the threshold) | recall 1.00, precision 1.00, mean delay 4 min |
| Faults scaled to 0.7x of that | recall 1.00, precision 1.00 |
| 0.6x | recall 0.82 (IGBT and PCS about 0.55; HVAC, SOC, PV still 1.00), precision 0.98 |
| 0.5x | recall 0.13: faults near or below the threshold are simply not seen |
| Faults that grow over 60 min instead of stepping | recall 1.00 but mean delay 34-43 min: it alerts only once the residual crosses the threshold |
| Healthy telemetry, 14 days x 3 assets | 0 false alerts |
| Plant differs from assumed baseline (IGBT k 0.045, PCS k 0.12, container offset 6) | uncalibrated: precision 0.70 (45 false alerts in 7 days); calibrated on a separate healthy week: precision 1.00, recall 1.00, fitted coefficients within 0.001-0.002 of truth |

What this does and does not show: a threshold on a physical residual has a sensitivity floor and cannot warn early about slow degradation, and a wrong baseline is more dangerous than a noisy one, so calibration matters. It does not show real-world accuracy. The generator's noise is white and Gaussian, there is no curtailment, soiling, maintenance, sensor drift or correlated weather error, and the same simulator produces both training and test data.

Do not call this predictive maintenance: it detects sustained anomalies after onset. Production readiness would require real historian/SCADA data, asset-specific calibration, curtailment and maintenance masks, operational labels, drift handling, false-alert analysis on real healthy history, and measured lead time before failures.

## SQL examples

```sql
SELECT kind, COUNT(*) AS alerts FROM alerts GROUP BY kind;
SELECT asset_id, timestamp, pv_kw, expected_pv_kw
FROM telemetry WHERE irradiance_wm2 > 150 AND pv_kw < 0.75*expected_pv_kw;
```

## Repository

| File | Purpose |
|---|---|
| `pipeline.py` | generator, cleaning, quality gate, calibration, detection, evaluation, report, CLI |
| `adapter.py` | schema/unit mapping and read-only SQL extraction |
| `benchmark.py` | severity sweep, ramping faults, healthy run, wrong-baseline vs calibrated |
| `vendor_demo.py`, `examples/` | fake vendor export and its mapping |
| `test_pipeline.py` | 22 unit and end-to-end tests |
| `.github/workflows/tests.yml` | CI: tests, gated synthetic run, gated vendor-adapter run |

## Next development steps

1. Test the adapter and thresholds on a real telemetry export, including device identity mapping.
2. Add curtailment, charge/discharge mode and planned-maintenance masks.
3. Replace the surrogate thermal baselines with load- and weather-aware models fitted per asset; handle seasonal drift.
4. Compare the physical-residual rules with an unsupervised anomaly model on the same time-separated evaluation period.
5. Incremental ingestion, logging and a scheduled run once the data and model validation hold.
