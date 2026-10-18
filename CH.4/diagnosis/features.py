"""입력 칼럼을 명시적으로 허용하고, 예측 시각 이전 관측만 요약한다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROFILES = {
    'A': ('P1-CR1-CAL01', ['GRADE','LOAD','AMBIENT','SET_TEMP','TC-Z3','OP-Z3','RUNNING']),
    'B': ('P1-CR1-WLD01', ['GRADE','THICKNESS','SPD','PWR','SET-PWR','RESISTANCE','RUNNING']),
    'C': ('P1-CC1-INS01', ['LOAD','WD-SPD','WD-LOD','OSC-FRQ','OSC-STK','MLD-LVL','CONTROL_DELAY','RUNNING']),
    'D': ('P1-CR1-SPM01', ['LOAD','VIB-TOP_RMS','VIB-TOP_STD','VIB-TOP_KUR','VIB-TOP_CRF','VIB-BOT_RMS','CUR-MTR_RMS']),
    'E1': ('P1-SP1-EXF01', ['LOAD','RPM','VIB-MDE_RMS','VIB-MDE_STD','VIB-MDE_KUR','VIB-MDE_CRF','VIB-MNDE_RMS','VIB-FDE_RMS','VIB-FNDE_RMS']),
    'E2': ('P1-SP2-EXF01', ['LOAD','RPM','VIB-MDE_RMS','VIB-MDE_STD','VIB-MDE_KUR','VIB-MDE_CRF','VIB-MNDE_RMS','VIB-FDE_RMS','VIB-FNDE_RMS']),
    'F': ('P1-SP1-AGT01', ['LOAD','VISCOSITY','VIB-AGT_RMS','VIB-AGT_STD','VIB-AGT_KUR','VIB-AGT_CRF']),
}
STATES = ['normal','degraded','failed','maintenance']
TRUTH = ['GT_HEALTH_STATE','GT_CAUSE_PART','GT_FAULT_MODE']
FEATURE_VERSION = 1

def manifest_at(raw):
    return json.loads((Path(raw)/'manifest.json').read_text(encoding='utf-8'))

def make_features(frame, equipment):
    """Dense A/C: right-closed 5min bins; B: same-coil-time aggregation; others: burst features.

    D/E/F bursts become available N/FS_HZ seconds after their recorded start.
    No labels, IDs, elapsed scenario time, maintenance or outlier flags are inputs.
    """
    frame = frame.copy()
    columns = PROFILES[equipment][1]
    frame['MEAS_DT'] = pd.to_datetime(frame['MEAS_DT'],utc=True)
    # Missing measurements may already have been filled by the upstream export.
    # Mask them conservatively so this model does not depend on its gap-length look-ahead.
    numeric = frame[columns].apply(pd.to_numeric,errors='coerce')
    missing = frame.get('IS_MISSING',pd.Series(0,index=frame.index)).fillna(1).astype(bool)
    if 'QUALITY_FLAG' in frame:
        missing |= frame.QUALITY_FLAG.ne('ok')
    preserve = {'RUNNING','LOAD','GRADE','AMBIENT','SET_TEMP','THICKNESS','SET-PWR','VISCOSITY','RPM'}
    for col in columns:
        if col not in preserve: numeric.loc[missing,col] = np.nan
    if equipment=='A': numeric['TEMP_ERROR'] = numeric['SET_TEMP']-numeric['TC-Z3']
    if equipment=='B': numeric['POWER_ERROR'] = numeric['SET-PWR']-numeric['PWR']
    numeric['MISSING_RATE'] = missing.astype(float)
    numeric.index = pd.DatetimeIndex(frame.MEAS_DT)
    numeric = numeric.sort_index(kind='stable')
    if equipment in ['D','E1','E2','F']:
        duration = pd.to_numeric(frame.N)/pd.to_numeric(frame.FS_HZ)
        numeric.index = pd.DatetimeIndex(frame.MEAS_DT + pd.to_timedelta(duration,unit='s'))
        numeric = numeric.sort_index(kind='stable')
    if equipment in ['A','C']:
        last_time = numeric.index.max()
        groups = numeric.resample('5min',closed='right',label='right',origin='start_day')
        base = groups.mean()
        spread = groups.std(ddof=0).add_suffix('__bin_std')
        base = base.join(spread)
        base = base.loc[base.index<=last_time]
    else:
        # B has 13 spatial records at the same time. Do not treat them as 13 time steps.
        base = numeric.groupby(level=0,sort=True).mean()
        if equipment=='B':
            base = base.join(numeric.groupby(level=0).std(ddof=0).add_suffix('__position_std'))
    features = base.add_suffix('__now')
    # 1h for dense signals; 6h for 30min/2h sampled features.
    hours = 6 if equipment in ['E1','E2','F'] else 1
    roll = base.rolling(f'{hours}h',closed='right',min_periods=1)
    features = features.join(roll.mean().add_suffix(f'__mean_{hours}h'))
    features = features.join(roll.std(ddof=0).add_suffix(f'__std_{hours}h'))
    features = features.join(base.diff().add_suffix('__change'))
    features = features.replace([np.inf,-np.inf],np.nan).astype('float32')
    assert features.index.is_monotonic_increasing and features.index.is_unique
    return features

def attach_truth(features, intervals, eqp, scenario):
    result = features.copy()
    for col in TRUTH: result[col] = ''
    result['DIAG_INTERVAL'] = ''
    if intervals is None: return result
    d = intervals.loc[intervals.EQP_CD.eq(eqp)].sort_values('START_AT').copy()
    if d.empty: raise ValueError(f'{scenario}: no diagnosis for {eqp}')
    starts=pd.DatetimeIndex(pd.to_datetime(d.START_AT,utc=True)).as_unit('ns').asi8
    ends=pd.DatetimeIndex(pd.to_datetime(d.END_AT,utc=True)).as_unit('ns').asi8
    if np.any(ends<=starts) or np.any(starts[1:]<ends[:-1]):
        raise ValueError('Invalid/overlapping diagnosis intervals')
    times=features.index.as_unit('ns').asi8
    pos=np.searchsorted(starts,times,side='right')-1
    valid=(pos>=0)&(times<ends[np.maximum(pos,0)])
    if not valid.all(): raise ValueError(f'Uncovered prediction times in {scenario}/{eqp}')
    for raw,col in zip(['HEALTH_STATE','CAUSE_PART','FAULT_MODE'],TRUTH):
        result[col]=d[raw].fillna('').to_numpy()[pos]
    result['DIAG_INTERVAL']=[f'{scenario}:{eqp}:{j}' for j in pos]
    return result

def feature_columns(table):
    # Only engineered names are allowed; never select all numeric columns.
    cols=[c for c in table.columns if '__' in c and not c.startswith('GT_')]
    banned={'IS_MAINTENANCE','IS_OUTLIER','ROW_ID','T_SEC','SCENARIO_ID','SOURCE_ID',
            'FAULT_MODE','CAUSE_PART','HEALTH_STATE'}
    # VIB-AGT_RMS is a real F sensor: substring "GT_" does not make it ground truth.
    assert cols and all(not c.startswith('GT_') and c.split('__')[0] not in banned for c in cols)
    return cols
