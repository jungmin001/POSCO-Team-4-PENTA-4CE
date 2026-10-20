"""누수·시간 경계·추론에 중요한 검증. CH.4에서 python -m unittest diagnosis.test_integrity."""
import unittest
import numpy as np
import pandas as pd
from .features import make_features,attach_truth,feature_columns,PROFILES
from .model import temperature_scale

class IntegrityTests(unittest.TestCase):
    def casting(self):
        t=pd.date_range('2025-01-01',periods=241,freq='5s',tz='UTC')
        data={c:np.ones(len(t)) for c in PROFILES['C'][1]}
        return pd.DataFrame({'MEAS_DT':t,**data,'IS_MISSING':0,'QUALITY_FLAG':'ok'})

    def test_future_changes_do_not_change_past_features(self):
        a=self.casting();b=a.copy()
        cutoff=pd.Timestamp('2025-01-01 00:10:00',tz='UTC')
        b.loc[b.MEAS_DT>cutoff,'CONTROL_DELAY']=999
        pd.testing.assert_frame_equal(make_features(a,'C').loc[:cutoff],make_features(b,'C').loc[:cutoff])

    def test_gt_and_maintenance_flags_cannot_change_features(self):
        a=self.casting();b=a.copy()
        b['GT_HEALTH_STATE']='failed';b['GT_FAULT_MODE']='fake';b['IS_MAINTENANCE']=1;b['ROW_ID']=999
        pd.testing.assert_frame_equal(make_features(a,'C'),make_features(b,'C'))
        self.assertTrue(all(not c.startswith('GT_') for c in feature_columns(make_features(b,'C'))))

    def test_interval_end_is_exclusive_and_time_units_match(self):
        f=make_features(self.casting(),'C')
        diagnosis=pd.DataFrame({'EQP_CD':['x','x'],'START_AT':['2025-01-01T00:00:00.000Z','2025-01-01T00:10:00.000Z'],
            'END_AT':['2025-01-01T00:10:00.000Z','2025-01-01T00:30:00.000Z'],
            'HEALTH_STATE':['normal','degraded'],'CAUSE_PART':['','valve'],'FAULT_MODE':['','sticking']})
        result=attach_truth(f,diagnosis,'x','S1')
        self.assertEqual(result.loc['2025-01-01T00:05:00Z','GT_HEALTH_STATE'],'normal')
        self.assertEqual(result.loc['2025-01-01T00:10:00Z','GT_HEALTH_STATE'],'degraded')

    def test_b_spatial_rows_are_not_time_steps(self):
        a=pd.DataFrame({c:np.arange(13,dtype=float) for c in PROFILES['B'][1]})
        a['MEAS_DT']='2025-01-01T00:00:00Z';a['IS_MISSING']=0
        f=make_features(a,'B')
        self.assertEqual(len(f),1);self.assertEqual(f.iloc[0]['PWR__now'],6)

    def test_burst_features_only_available_after_acquisition(self):
        a=pd.DataFrame({c:[1.] for c in PROFILES['D'][1]})
        a['MEAS_DT']='2025-01-01T00:00:00Z';a['N']=18;a['FS_HZ']=10;a['IS_MISSING']=0
        f=make_features(a,'D')
        self.assertEqual(f.index[0],pd.Timestamp('2025-01-01T00:00:01.8Z'))

    def test_temperature_preserves_ranking_and_probability_sum(self):
        p=np.array([[.1,.2,.7],[.6,.3,.1]])
        q=temperature_scale(p,2)
        np.testing.assert_allclose(q.sum(axis=1),1)
        np.testing.assert_equal(p.argmax(axis=1),q.argmax(axis=1))

    def test_all_equipment_sensor_allowlists_are_accepted(self):
        for eq,(_,sensors) in PROFILES.items():
            data=pd.DataFrame({c:[1.,2.] for c in sensors})
            data['MEAS_DT']=['2025-01-01T00:00:00Z','2025-01-01T02:00:00Z']
            data['IS_MISSING']=0;data['N']=128;data['FS_HZ']=256
            self.assertTrue(feature_columns(make_features(data,eq)))

if __name__=='__main__':unittest.main()
