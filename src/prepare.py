"""Reproducible M5 preparation. Raw records never enter public artifacts."""
from __future__ import annotations
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_START = pd.Timestamp('2014-03-29')
TRAIN_END = pd.Timestamp('2016-03-27')
VALID_END = pd.Timestamp('2016-04-24')
TEST_END = pd.Timestamp('2016-05-22')
ID_COLS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
SOURCES = ['sales_train_evaluation.csv', 'calendar.csv', 'sell_prices.csv']

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)

class Audit:
    """Application access ledger. Logs metadata, never field values or credentials."""
    def __init__(self, path, run_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
    def record(self, action, resource, **details):
        event = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                     run_id=self.run_id, actor_role='academic_analyst',
                     action=action, resource=resource, **details)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, default=str) + '\n')

def read_inputs(source, manifest_path, expected_path, audit):
    """Read an authorised local zip or directory and verify source identity."""
    source = Path(source)
    audit.record('read', 'cohort_manifest', purpose='fixed Module 2 scope')
    manifest = pd.read_csv(manifest_path)
    require(len(manifest) == 100 and manifest.id.is_unique, 'Manifest must contain 100 unique series')
    audit.record('read', 'source_hash_manifest')
    expected = json.loads(Path(expected_path).read_text())['source_sha256']
    archive = zipfile.ZipFile(source) if source.is_file() else None
    def open_source(name):
        return archive.open(name) if archive else (source / name).open('rb')
    hashes = {}
    for name in SOURCES:
        audit.record('read', name, purpose='source identity verification')
        digest = hashlib.sha256()
        with open_source(name) as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        hashes[name] = digest.hexdigest()
        require(hashes[name] == expected[name], f'Source hash mismatch: {name}')
    audit.record('read', 'calendar.csv', purpose='date and event integration')
    with open_source('calendar.csv') as f:
        calendar = pd.read_csv(f, parse_dates=['date'])
    require(calendar.d.is_unique and calendar.date.is_unique, 'Duplicate calendar keys')
    required_cal = ['d','date','wm_yr_wk','wday','month','year','snap_CA','snap_TX','snap_WI']
    require(calendar[required_cal].notna().all().all(), 'Missing calendar key or date feature')
    require(calendar.date.sort_values().diff().dropna().eq(pd.Timedelta(days=1)).all(), 'Calendar gaps')
    wanted = set(manifest.id)
    selected, metadata, keys = [], [], []
    totals = dict(source_series=0, missing_sales=0, negative_sales=0, noninteger_sales=0)
    audit.record('read', 'sales_train_evaluation.csv', purpose='chunked sales validation and cohort extraction')
    with open_source('sales_train_evaluation.csv') as f:
        for chunk in pd.read_csv(f, chunksize=1000):
            days = [c for c in chunk.columns if re.fullmatch(r'd_\d+', c)]
            require(len(days) == 1941, 'Unexpected sales day schema')
            arr = chunk[days].to_numpy()
            totals['source_series'] += len(chunk)
            totals['missing_sales'] += int(pd.isna(arr).sum())
            totals['negative_sales'] += int((arr < 0).sum())
            totals['noninteger_sales'] += int(((arr % 1) != 0).sum())
            keys.append(chunk[['item_id','store_id']])
            train = arr[:, :1885]
            meta = chunk[ID_COLS].copy()
            meta['eligible_sales'] = (meta.cat_id.eq('HOUSEHOLD') &
                ((train[:, -730:] > 0).mean(axis=1) >= .8) &
                (train[:, :730] > 0).any(axis=1))
            metadata.append(meta)
            selected.append(chunk[chunk.id.isin(wanted)].copy())
    require(pd.concat(keys).duplicated().sum() == 0, 'Duplicate product-store source keys')
    require(sum(totals[k] for k in ['missing_sales','negative_sales','noninteger_sales']) == 0,
            'Invalid source sales; no silent correction is permitted')
    wide = pd.concat(selected, ignore_index=True)
    require(len(wide) == 100 and set(wide.id) == wanted, 'Selected series absent or duplicated')
    require(wide[ID_COLS].sort_values('id').reset_index(drop=True).equals(
        manifest[ID_COLS].sort_values('id').reset_index(drop=True)), 'Manifest identifiers changed')
    audit.record('read', 'sell_prices.csv', purpose='selected-series weekly price extraction')
    price_parts = []
    with open_source('sell_prices.csv') as f:
        for chunk in pd.read_csv(f, chunksize=100000):
            require(chunk[['store_id','item_id','wm_yr_wk','sell_price']].notna().all().all(), 'Missing price source values')
            require(chunk.sell_price.gt(0).all(), 'Nonpositive selling price')
            price_parts.append(chunk.merge(manifest[['item_id','store_id']], on=['item_id','store_id'], how='inner', validate='many_to_one'))
    prices = pd.concat(price_parts, ignore_index=True)
    require(not prices.duplicated(['item_id','store_id','wm_yr_wk']).any(), 'Duplicate selected weekly price keys')
    if archive:
        archive.close()
    audit.record('transform', 'source_checks', **totals)
    return wide, calendar, prices, pd.concat(metadata, ignore_index=True), hashes, totals

def integrate(wide, calendar, prices):
    keep_days = calendar.loc[(calendar.date >= TRAIN_START - pd.Timedelta(days=28)) &
                             (calendar.date <= TEST_END), 'd'].tolist()
    long = wide.melt(id_vars=ID_COLS, value_vars=keep_days, var_name='d', value_name='units')
    joined = long.merge(calendar, on='d', validate='many_to_one', how='left')
    require(joined.date.notna().all(), 'Unmatched calendar dates')
    joined = joined.merge(prices, on=['store_id','item_id','wm_yr_wk'],
                          validate='many_to_one', how='left')
    require(len(joined) == len(long), 'Join multiplied or removed records')
    return joined.sort_values(['id','date']).reset_index(drop=True)

def prepare_features(frame):
    df = frame.sort_values(['id','date']).copy()
    require(not df.duplicated(['id','date']).any(), 'Duplicate daily grain')
    require(df.groupby('id').date.diff().dropna().eq(pd.Timedelta(days=1)).all(), 'Series date gaps')
    # Remove all post-cutoff actuals before calculating features. Targets remain labels only.
    history = df.units.where(df.date <= TRAIN_END)
    groups = history.groupby(df.id, sort=False)
    df['lag_7'] = groups.shift(7)
    df['lag_28'] = groups.shift(28)
    df['rolling_mean_28'] = groups.transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    df['historical_sell_price'] = df.sell_price.where(df.date <= TRAIN_END)
    df['price_missing'] = df.sell_price.isna()
    df['zero_sales_cause'] = np.where(df.units.eq(0), 'unknown', 'not_zero')
    df['split'] = np.select([df.date <= TRAIN_END, df.date <= VALID_END], ['train','validation'], default='test')
    # SNAP is a state calendar flag, not an individual customer attribute.
    df['snap'] = np.select([df.state_id.eq('CA'), df.state_id.eq('TX')], [df.snap_CA, df.snap_TX], default=df.snap_WI)
    df = df[df.date >= TRAIN_START].copy()
    columns = ID_COLS + ['date','d','wm_yr_wk','units','wday','month','year',
        'event_name_1','event_type_1','event_name_2','event_type_2','snap',
        'lag_7','lag_28','rolling_mean_28','historical_sell_price','price_missing','zero_sales_cause','split']
    return df[columns].reset_index(drop=True)

PII_COLUMNS = {'name','full_name','email','phone','address','customer_id','national_id','passport','ip_address'}
def anonymize(frame):
    """Remove known PII columns and reject contact-like strings in retained text."""
    removed = [c for c in frame.columns if c.lower() in PII_COLUMNS]
    clean = frame.drop(columns=removed).copy()
    for col in clean.select_dtypes(include=['object','string']).columns:
        require(not clean[col].dropna().astype(str).str.contains(
            r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:\+\d[\d ()-]{8,}\d)', regex=True).any(),
            f'Potential contact information in retained column: {col}')
    return clean, removed

def assert_output(frame):
    require(len(frame) == 78600, 'Expected 100 series x 786 dates')
    require(frame.groupby('split').size().to_dict() == {'test':2800,'train':73000,'validation':2800}, 'Unexpected chronological split')
    train = frame[frame.split.eq('train')]
    require(train[['lag_7','lag_28','rolling_mean_28','historical_sell_price']].notna().all().all(), 'Incomplete training features or prices')
    require(frame.units.ge(0).all() and frame.units.mod(1).eq(0).all(), 'Invalid labels')
    require(frame.loc[frame.split.ne('train'),'historical_sell_price'].isna().all(), 'Future prices leaked')
    require(frame.loc[frame.date > TRAIN_END + pd.Timedelta(days=28), ['lag_7','lag_28','rolling_mean_28']].isna().all().all(), 'Holdout actuals leaked into features')

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
