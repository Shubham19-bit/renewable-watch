"""Source adapter: map a vendor/historian export onto the pipeline schema. Read-only; no control commands."""
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

SCHEMA = ['timestamp', 'asset_id', 'ambient_c', 'irradiance_wm2', 'module_c',
          'pv_kw', 'igbt_c', 'pcs_kw', 'pcs_c', 'container_c', 'soc_min_pct', 'soc_max_pct']

# Allowed source units per target column, as functions into the pipeline's canonical unit.
_TEMP = {'C': lambda s: s, 'F': lambda s: (s - 32) * 5 / 9, 'K': lambda s: s - 273.15}
_POWER = {'kW': lambda s: s, 'W': lambda s: s / 1000, 'MW': lambda s: s * 1000}
_SOC = {'pct': lambda s: s, 'fraction': lambda s: s * 100}
_IRR = {'W/m2': lambda s: s, 'kW/m2': lambda s: s * 1000}
UNITS = {'ambient_c': _TEMP, 'module_c': _TEMP, 'igbt_c': _TEMP, 'pcs_c': _TEMP, 'container_c': _TEMP,
         'pv_kw': _POWER, 'pcs_kw': _POWER, 'soc_min_pct': _SOC, 'soc_max_pct': _SOC,
         'irradiance_wm2': _IRR}


def _spec(value):
    return value if isinstance(value, dict) else {'source': value}


def _timestamp(series, tz):
    """Parse to UTC. Naive stamps are localised to `tz` (default UTC); ambiguous/nonexistent DST stamps become NaT."""
    parsed = pd.to_datetime(series, errors='coerce')
    if not hasattr(parsed, 'dt'):
        return pd.to_datetime(series, errors='coerce', utc=True)
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(tz or 'UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    return parsed.dt.tz_convert('UTC')


def standardize(raw, mapping):
    """Rename, join and unit-convert a source table into the canonical schema.

    mapping: {target: "source_col" | {"source": col|[cols], "unit": u, "join": sep, "timezone": tz}}.
    Values are never interpolated or clipped here; range masking stays in pipeline.clean.
    """
    unknown = set(mapping) - set(SCHEMA)
    unmapped = set(SCHEMA) - set(mapping)
    if unknown or unmapped:
        raise ValueError(f'Mapping must cover the schema exactly; unknown={sorted(unknown)} missing={sorted(unmapped)}')
    out = pd.DataFrame(index=raw.index)
    for target in SCHEMA:
        spec = _spec(mapping[target])
        sources = spec['source'] if isinstance(spec['source'], list) else [spec['source']]
        absent = [c for c in sources if c not in raw.columns]
        if absent:
            raise ValueError(f'{target}: source columns not in input: {absent}')
        if target == 'timestamp':
            out[target] = _timestamp(raw[sources[0]], spec.get('timezone'))
        elif target == 'asset_id':
            out[target] = raw[sources].astype(str).agg(spec.get('join', '-').join, axis=1)
        else:
            unit = spec.get('unit', next(iter(UNITS[target])))
            if unit not in UNITS[target]:
                raise ValueError(f'{target}: unit {unit!r} not supported; choose from {sorted(UNITS[target])}')
            out[target] = UNITS[target][unit](pd.to_numeric(raw[sources[0]], errors='coerce'))
    return out


def extract_sql(db_path, query):
    """Run a read-only SELECT against a SQLite database (stand-in for the new source database)."""
    if not query.lstrip().lower().startswith(('select', 'with')):
        raise ValueError('Only SELECT/WITH queries are allowed; the pipeline is read-only.')
    uri = Path(db_path).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        return pd.read_sql_query(query, conn)
