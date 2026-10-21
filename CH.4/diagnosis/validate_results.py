"""저장 결과를 독립적으로 재집계: python -m diagnosis.validate_results."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from .features import PROFILES,feature_columns,make_features
from .model import probabilities

def main():
    out=Path(__file__).resolve().parents[1]/'Results/Diagnosis'
    run=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    groups=[set(run[k]) for k in ['fit_scenarios','calibration_scenarios','validation_scenarios','test_scenarios']]
    assert [len(g) for g in groups]==[162,54,72,72]
    for i,a in enumerate(groups):
        for b in groups[i+1:]:assert not a&b
    audit=pd.read_csv(out/'tables/input_audit.csv')
    assert len(audit)==360*7
    checks=[]
    for eq in PROFILES:
        blob=joblib.load(out/'features'/f'{eq}.joblib');table=blob['table']
        assert blob['signature']==run['equipment'][eq]['input_signature']
        cols=feature_columns(table)
        assert cols==run['equipment'][eq]['columns']
        assert table.loc[table.split.eq('test'),'GT_HEALTH_STATE'].eq('').all()
        metrics=json.loads((out/'tables'/f'{eq}_metrics.json').read_text(encoding='utf-8'))
        for task in ['state','cause']:
            bundle=joblib.load(out/'models'/f'{eq}_{task}.joblib')
            assert bundle['columns']==cols
            assert set(bundle['calibration_scenarios'])<=groups[1]
            pred=pd.read_csv(out/'predictions'/f'{eq}_{task}_validation.csv.gz',keep_default_na=False)
            m=metrics[task]
            assert set(pred.SCENARIO_ID)<=groups[2] and len(pred)==m['rows']
            np.testing.assert_allclose(accuracy_score(pred.truth,pred.prediction),m['accuracy'])
            np.testing.assert_allclose(f1_score(pred.truth,pred.prediction,average='macro',zero_division=0),m['macro_f1'])
            np.testing.assert_array_equal(confusion_matrix(pred.truth,pred.prediction,labels=m['labels']),m['confusion'])
            probs=[c for c in pred if c.startswith('probability_')]
            np.testing.assert_allclose(pred[probs].sum(axis=1),1,atol=1e-7)
            assert pred[probs].ge(0).all().all() and pred[probs].le(1).all().all()
            if task=='cause':assert pred.GT_HEALTH_STATE.isin(['degraded','failed']).all()
            latest=pd.read_csv(out/'tables'/f'{eq}_latest_test.csv')
            assert len(latest)==72 and set(latest.SCENARIO_ID)==groups[3]
            checks.append(dict(equipment=eq,task=task,validation_rows=len(pred),status='passed'))
    # Rebuild one scenario from sensor columns only and run the persisted model independently.
    sid=run['test_scenarios'][0]
    allowed=set(PROFILES['A'][1])|{'MEAS_DT','IS_MISSING','QUALITY_FLAG','N','FS_HZ'}
    sensor=pd.read_csv(Path(run['scenario_root'])/sid/'A.csv',usecols=lambda c:c in allowed)
    independent=make_features(sensor,'A')
    model=joblib.load(out/'models/A_state.joblib')
    p=probabilities(model,independent[model['columns']])
    batch=pd.read_csv(out/'predictions/A_dashboard_test.csv.gz')
    batch=batch.loc[batch.SCENARIO_ID.eq(sid)]
    np.testing.assert_allclose(p.max(axis=1),batch.state_probability,atol=1e-12)
    np.testing.assert_array_equal(np.asarray(model['classes'])[p.argmax(axis=1)],batch.state_prediction)
    result=dict(status='passed',scenario_disjointness=True,input_truth_leakage_check=True,
        all_2520_inputs_audited=True,saved_metrics_recomputed=True,probability_sums_checked=True,
        saved_model_inference_verified=f'A/{sid}, {len(batch)} rows, batch and independent sensor-only inference matched',checks=checks)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
