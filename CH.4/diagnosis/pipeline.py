"""시나리오별 특징 생성 → 7개 설비 학습 → 검증/시험 예측 → 설명 보고서."""
import argparse
from pathlib import Path
import hashlib
import html
import json
import platform
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .features import PROFILES,STATES,TRUTH,FEATURE_VERSION,manifest_at,make_features,attach_truth,feature_columns
from .model import train_bundle,probabilities,uncertainty,evaluate,explain

BASE=Path(__file__).resolve().parents[1]
DEFAULT_OUT=BASE/'Results'/'Diagnosis'
STATE_KO={'normal':'정상','degraded':'성능 저하','failed':'고장','maintenance':'정비'}
MODE_KO={'friction':'진동 구동 마찰','sticking':'밸브 고착','control-delay':'밸브 제어 지연',
    'background-wear':'배경 열화','heater-loss':'히터 성능 저하','sensor-drift':'온도 센서 드리프트',
    'electrode-wear':'전극 마모','power-loss':'전원 성능 저하','roll-wear':'롤 마모',
    'bearing-damage':'베어링 손상','imbalance':'불평형','looseness':'체결부 풀림',
    'deposit':'부착','wear':'마모','rotor-damage':'회전체 손상'}

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v),allow_nan=False),encoding='utf-8')

def fingerprint(scenario_root,raw,manifest,equip):
    h=hashlib.sha256(f'{FEATURE_VERSION}:{equip}'.encode())
    h.update((raw/'manifest.json').read_bytes())
    for s in manifest['scenarios']:
        p=scenario_root/s['id']/f'{equip}.csv'
        stat=p.stat();h.update(f'{p}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
        d=raw/'diagnostics'/f"{s['id']}.csv"
        if d.exists(): h.update(d.read_bytes())
    return h.hexdigest()

def build_equipment(equip,scenario_root,raw,manifest,out,reuse):
    cache=out/'features'/f'{equip}.joblib'
    signature=fingerprint(scenario_root,raw,manifest,equip)
    if reuse and cache.exists():
        blob=joblib.load(cache)
        if blob['signature']==signature:
            print(f'[{equip}] cache valid',flush=True);return blob['table'],blob['audit']
    parts=[];audit=[];start=time.time()
    for i,s in enumerate(manifest['scenarios']):
        sid=s['id'];path=scenario_root/sid/f'{equip}.csv'
        allowed=set(PROFILES[equip][1])|{'MEAS_DT','IS_MISSING','QUALITY_FLAG','N','FS_HZ'}
        # GT columns are not even read into the feature function.
        df=pd.read_csv(path,usecols=lambda c:c in allowed)
        f=make_features(df,equip)
        diagnosis_path=raw/'diagnostics'/f'{sid}.csv'
        truth=pd.read_csv(diagnosis_path,keep_default_na=False) if s['split']!='test' else None
        if s['split']=='test' and diagnosis_path.exists():
            raise ValueError('Unexpected public test truth: check split before proceeding')
        f=attach_truth(f,truth,PROFILES[equip][0],sid)
        f.insert(0,'MEAS_DT',f.index)
        f.insert(0,'SCENARIO_ID',sid)
        f['EQP_CD']=PROFILES[equip][0];f['split']=s['split'];f['equipment']=equip
        parts.append(f.reset_index(drop=True))
        audit.append(dict(equipment=equip,scenario=sid,split=s['split'],input_rows=len(df),prediction_rows=len(f),truth_rows=int(f.GT_HEALTH_STATE.ne('').sum())))
        if (i+1)%60==0: print(f'[{equip}] features {i+1}/360 ({time.time()-start:.0f}s)',flush=True)
    table=pd.concat(parts,ignore_index=True)
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(signature=signature,table=table,audit=audit),cache,compress=3)
    return table,audit

def plot_confusion(metrics,path,title):
    labels=metrics['labels'];cm=np.array(metrics['confusion']);counts=cm.sum(axis=1)
    norm=np.divide(cm,counts[:,None],out=np.zeros_like(cm,dtype=float),where=counts[:,None]!=0)
    fig,ax=plt.subplots(figsize=(max(6,len(labels)*1.4),max(4.8,len(labels)*1.05)))
    ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    short=[STATE_KO.get(x,x.replace('::','\n')) for x in labels]
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=short,yticklabels=short,
           xlabel='예측',ylabel='정답',title=title)
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
    for r in range(len(labels)):
        for c in range(len(labels)):
            ax.text(c,r,f'{cm[r,c]:,}\n{norm[r,c]:.0%}',ha='center',va='center',fontsize=9,color='white' if norm[r,c]>.55 else 'black')
    fig.tight_layout();fig.savefig(path,dpi=135);plt.close(fig)

def grouped_importance(bundle,x,y,seed=42):
    # Shuffle complete sensor-feature groups together; preserve within-sensor feature relations.
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(x),min(1500,len(x)),replace=False)
    x=x.iloc[idx].copy();y=np.asarray(y)[idx]
    classes=bundle['classes'];base=evaluate_simple(y,classes[probabilities(bundle,x).argmax(axis=1)])
    groups={}
    for c in x:groups.setdefault(c.split('__')[0],[]).append(c)
    rows=[]
    for sensor,cols in groups.items():
        values=[]
        for _ in range(2):
            alt=x.copy();alt.loc[:,cols]=x[cols].iloc[rng.permutation(len(x))].to_numpy()
            values.append(base-evaluate_simple(y,classes[probabilities(bundle,alt).argmax(axis=1)]))
        rows.append(dict(observation=sensor,macro_f1_drop=float(np.mean(values)),repeat_spread=float(np.std(values)),evaluation_rows=len(x)))
    return pd.DataFrame(rows).sort_values('macro_f1_drop',ascending=False)

def evaluate_simple(y,pred):
    from sklearn.metrics import f1_score
    return f1_score(y,pred,average='macro',zero_division=0)

def result_table(bundle,table,cols,task):
    p=probabilities(bundle,table[cols]);classes=np.asarray(bundle['classes'])
    order=np.argsort(-p,axis=1);reasons,margin=uncertainty(bundle,table[cols],p)
    out=table[['SCENARIO_ID','EQP_CD','MEAS_DT','equipment','split',*TRUTH,'DIAG_INTERVAL']].copy()
    out['task']=task;out['prediction']=classes[order[:,0]]
    out['candidate_2']=classes[order[:,1]] if len(classes)>1 else ''
    out['confidence']=p[np.arange(len(p)),order[:,0]]
    out['candidate_2_probability']=p[np.arange(len(p)),order[:,1]] if len(classes)>1 else 0.
    out['margin']=margin;out['uncertainty_reasons']=reasons
    out['needs_review']=reasons!=''
    out['truth']=table.GT_HEALTH_STATE if task=='state' else (table.GT_CAUSE_PART+'::'+table.GT_FAULT_MODE)
    out['correct']=out.truth.eq(out.prediction) if task=='state' else out.truth.eq(out.prediction)&table.GT_HEALTH_STATE.isin(['degraded','failed'])
    for j,cls in enumerate(classes):out['probability_'+cls]=p[:,j]
    return out,p

def select_cases(result,table,cols,bundle,normal_reference,task,limit=9):
    rows=[]
    # One example per scenario in each category; include confident mistakes and uncertain cases.
    candidates=[('confident_error',result.loc[~result.correct].sort_values('confidence',ascending=False)),
                ('uncertain',result.loc[result.needs_review].sort_values('margin')),
                ('correct_abnormal',result.loc[result.correct & result.GT_HEALTH_STATE.isin(['degraded','failed'])].sort_values('confidence',ascending=False))]
    for kind,frame in candidates:
        for idx,row in frame.drop_duplicates('SCENARIO_ID').head(limit//3).iterrows():
            evidence=explain(bundle,table.loc[[idx],cols],row.prediction,normal_reference)
            values=row.to_dict();values['kind']=kind;values['observations']=evidence
            rows.append(values)
    return rows

def process_equipment(equip,table,out,calibration_ids,predict_test):
    print(f'[{equip}] train/evaluate',flush=True)
    cols=feature_columns(table)
    fit=table.split.eq('train')&~table.SCENARIO_ID.isin(calibration_ids)
    cal=table.split.eq('train')&table.SCENARIO_ID.isin(calibration_ids)
    val=table.split.eq('validation');test=table.split.eq('test')
    assert not set(table.loc[fit,'SCENARIO_ID'])&set(table.loc[cal,'SCENARIO_ID'])
    assert not set(table.loc[fit|cal,'SCENARIO_ID'])&set(table.loc[val|test,'SCENARIO_ID'])
    normal_reference=table.loc[fit&table.GT_HEALTH_STATE.eq('normal'),cols].median().fillna(0)
    distributions=table.groupby(['split','GT_HEALTH_STATE'],dropna=False).size().reset_index(name='rows')
    distributions.to_csv(out/'tables'/f'{equip}_class_support.csv',index=False,encoding='utf-8-sig')
    metrics={};bundles={};case_rows=[];state_val=None;cause_val=None;state_test=None;cause_test=None
    for task in ['state','cause']:
        eligible=pd.Series(True,index=table.index) if task=='state' else (table.GT_HEALTH_STATE.isin(['degraded','failed'])&table.GT_CAUSE_PART.ne('')&table.GT_FAULT_MODE.ne(''))
        y=table.GT_HEALTH_STATE if task=='state' else table.GT_CAUSE_PART+'::'+table.GT_FAULT_MODE
        train_mask=fit&eligible;cal_mask=cal&eligible;val_mask=val&eligible
        if train_mask.sum()==0 or cal_mask.sum()==0 or val_mask.sum()==0:
            metrics[task]=dict(unavailable='Insufficient labeled fit/calibration/validation observations');continue
        bundle=train_bundle(table.loc[train_mask,cols],y[train_mask],table.loc[cal_mask,cols],y[cal_mask])
        bundle['equipment']=equip;bundle['task']=task;bundle['normal_reference']=normal_reference
        bundle['calibration_scenarios']=sorted(set(table.loc[cal_mask,'SCENARIO_ID']))
        joblib.dump(bundle,out/'models'/f'{equip}_{task}.joblib',compress=3);bundles[task]=bundle
        result,p=result_table(bundle,table.loc[val_mask],cols,task)
        m=evaluate(y[val_mask],p,bundle['classes'],table.loc[val_mask,'SCENARIO_ID'])
        majority=y[train_mask].value_counts().index[0]
        m['majority_baseline_macro_f1']=float(evaluate_simple(y[val_mask],np.repeat(majority,val_mask.sum())))
        m['train_rows']=int(train_mask.sum());m['calibration_rows']=int(cal_mask.sum())
        m['train_scenarios']=int(table.loc[train_mask,'SCENARIO_ID'].nunique())
        m['temperature']=bundle['temperature'];m['review_confidence_threshold']=bundle['threshold']
        accepted=~result.needs_review
        m['accepted_coverage']=float(accepted.mean())
        m['accepted_accuracy']=float(result.loc[accepted,'correct'].mean()) if accepted.any() else None
        m['calibration']=bundle['calibration']
        m['validation_intervals']=int(table.loc[val_mask,'DIAG_INTERVAL'].nunique())
        interval_result=result.groupby('DIAG_INTERVAL').correct.mean()
        m['mean_interval_accuracy']=float(interval_result.mean())
        if task=='cause':
            m['top2_accuracy']=float((result.truth.eq(result.prediction)|result.truth.eq(result.candidate_2)).mean())
            m['part_accuracy']=float((result.truth.str.split('::').str[0]==result.prediction.str.split('::').str[0]).mean())
        metrics[task]=m
        result.to_csv(out/'predictions'/f'{equip}_{task}_validation.csv.gz',index=False,compression='gzip')
        plot_confusion(m,out/'figures'/f'{equip}_{task}_confusion.png',f'{equip} {"상태" if task=="state" else "원인"} 진단 · 검증')
        importance=grouped_importance(bundle,table.loc[val_mask,cols],y[val_mask])
        importance.to_csv(out/'tables'/f'{equip}_{task}_observation_importance.csv',index=False,encoding='utf-8-sig')
        case_rows.extend(select_cases(result,table,cols,bundle,normal_reference,task))
        if task=='state':state_val=result
        else:cause_val=result
        if predict_test:
            r,_=result_table(bundle,table.loc[test],cols,task)
            r=r.drop(columns=TRUTH+['DIAG_INTERVAL','truth','correct'])
            r.to_csv(out/'predictions'/f'{equip}_{task}_test.csv.gz',index=False,compression='gzip')
            if task=='state':state_test=r
            else:cause_test=r
        print(f'[{equip}/{task}] macro-F1={m["macro_f1"]:.3f}; review={1-m["accepted_coverage"]:.1%}',flush=True)
    if state_val is not None and cause_val is not None:
        # Oracle-conditioned cause performance above is not the full-system performance.
        state_for_cause=state_val.loc[cause_val.index]
        gate=state_for_cause.prediction.isin(['degraded','failed'])
        metrics['end_to_end']=dict(true_abnormal_cause_rows=len(cause_val),
            abnormal_detection_recall=float(gate.mean()),
            cause_top1_with_state_gate_accuracy=float((cause_val.correct & gate).mean()))
    if state_test is not None:
        dashboard=state_test.rename(columns={'prediction':'state_prediction','confidence':'state_probability'})
        dashboard=dashboard[['SCENARIO_ID','EQP_CD','MEAS_DT','equipment','state_prediction','state_probability','needs_review','uncertainty_reasons']].copy()
        if cause_test is not None:
            mask=dashboard.state_prediction.isin(['degraded','failed'])
            dashboard['cause_candidate']=cause_test.prediction.where(mask,'')
            dashboard['cause_candidate_2']=cause_test.candidate_2.where(mask,'')
            dashboard['cause_probability_conditional']=cause_test.confidence.where(mask,np.nan)
            dashboard['cause_needs_review']=cause_test.needs_review.where(mask,False)
            dashboard['cause_uncertainty_reasons']=cause_test.uncertainty_reasons.where(mask,'')
        dashboard.to_csv(out/'predictions'/f'{equip}_dashboard_test.csv.gz',index=False,compression='gzip')
        # A small, inspectable latest-status sample for the future dashboard integration.
        dashboard.groupby('SCENARIO_ID',sort=False).tail(1).to_csv(out/'tables'/f'{equip}_latest_test.csv',index=False,encoding='utf-8-sig')
    save_json(out/'tables'/f'{equip}_metrics.json',metrics)
    save_json(out/'tables'/f'{equip}_cases.json',case_rows)
    return metrics,case_rows

def make_report(all_metrics,all_cases,audits,out,run):
    rows=[]
    for eq,metrics in all_metrics.items():
        for task in ['state','cause']:
            m=metrics.get(task,{})
            if 'macro_f1' not in m:continue
            rows.append(dict(equipment=eq,task=task,validation_rows=m['rows'],validation_scenarios=m['scenarios'],
                accuracy=m['accuracy'],macro_f1=m['macro_f1'],baseline_macro_f1=m['majority_baseline_macro_f1'],
                macro_f1_ci_low=m['macro_f1_scenario_bootstrap_95ci'][0],macro_f1_ci_high=m['macro_f1_scenario_bootstrap_95ci'][1],
                accepted_coverage=m['accepted_coverage'],accepted_accuracy=m['accepted_accuracy'],ece=m['probability_quality']['ece']))
    summary=pd.DataFrame(rows);summary.to_csv(out/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(audits).to_csv(out/'tables'/'input_audit.csv',index=False,encoding='utf-8-sig')
    # Summary plot separates the easier state task from the harder joint-cause task.
    eqs=list(all_metrics)
    fig,ax=plt.subplots(figsize=(10,4.8));positions=np.arange(len(eqs))
    for task,offset,color,label in [('state',-.18,'#376ca8','상태'),('cause',.18,'#d68843','부품·원인')]:
        values=[all_metrics[eq].get(task,{}).get('macro_f1',np.nan) for eq in eqs]
        bars=ax.bar(positions+offset,values,.34,color=color,label=label)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set(xticks=positions,xticklabels=eqs,ylim=(0,1.14),ylabel='검증 Macro-F1',title='설비별 상태·원인 진단 성능')
    ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'figures'/'performance_summary.png',dpi=145);plt.close(fig)
    lines=['# 설비 상태·원인 진단 결과','',
        '센서 관측만으로 현재 상태와 원인 후보를 추정하는 첫 기준 모델입니다. 원인 예측은 관측과 정답의 연관성이며 물리적인 인과관계의 증명은 아닙니다.','',
        '![설비별 성능](figures/performance_summary.png)','',
        '## 어떤 순서로 분석했나','',
        '1. Scenarios의 A·B·C·D·E1·E2·F 파일을 시나리오 하나씩 읽었습니다. GT_ 정답, 행 번호, 경과 시간, 정비·이상치 플래그를 입력에서 제외했습니다.',
        '2. A/C는 과거 5분 구간을 요약하고 B는 같은 시각의 13개 공간 관측을 묶었습니다. D/E/F는 제공된 파형 특징을 썼습니다. 파형은 N/FS_HZ초 뒤에 이용 가능하다고 보수적으로 가정했습니다.',
        '3. 현재 특징과 과거 1시간(A/B/C/D), 6시간(E/F)의 평균·변동·직전 변화량을 만들었습니다. 미래 관측이나 다른 시나리오는 섞지 않았습니다.',
        '4. Raw/diagnostics의 START_AT ≤ 예측 시각 < END_AT 조건으로 상태·부품·유형 정답을 연결했습니다. 원인 공란은 임의로 채우지 않았습니다.',
        '5. 원래 train 216개 중 162개를 학습, 54개를 확률 보정·보류 기준 설정에 사용했습니다. 원래 validation 72개는 별도 평가에만 사용했고 test 72개는 정답 없이 예측했습니다.',
        '6. 설비별 ExtraTrees 상태 모델과 부품::고장유형 원인 모델을 학습했습니다. 원인 모델은 원인이 알려진 degraded/failed 표본으로 학습·평가했습니다.',
        '7. 상태의 오분류·원인 후보·판단 보류·관측 설명을 기록했습니다. 기존 Main.py와 전처리 파일은 변경하지 않았습니다.','',
        '## 성능 읽는 법','',
        '- Macro-F1은 상태·원인별 F1을 동일 비중으로 평균한 값입니다. 단순 정확도보다 드문 이상을 놓치는 문제를 잘 드러냅니다. 비교 기준은 학습 자료의 최빈 클래스를 항상 예측하는 모델입니다.',
        '- 신뢰구간은 독립적인 검증 시나리오를 200회 재추출한 구간입니다. 센서 행이 많아도 독립 고장 사건이 많다는 뜻은 아닙니다.',
        '- 원인 점수는 실제 이상이고 원인이 알려진 시점에 한정된 조건부 성능입니다. 아래 설비별 end_to_end는 상태 모델이 이상을 감지한 경우까지 포함해 평가합니다.',
        '- 자동 채택은 확률·후보 간격·관측 결측·학습 범위 경고를 통과한 비율입니다. 채택 정확도는 검증에서 측정한 실제 값이며 보장 수치가 아닙니다.',
        '- ECE는 예측 확률과 실제 정답률의 차이입니다. 작을수록 좋습니다. 온도 보정은 완전한 확률 신뢰성을 보장하지 않습니다.','',
        '| 설비 | 목표 | 검증 행 | 정확도 | Macro-F1 | 최빈 기준 F1 | F1 95% 구간 | 자동 채택 비율 | 채택 정확도 |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:
        accepted='판단 보류만 있음' if r['accepted_accuracy'] is None else f"{r['accepted_accuracy']:.1%}"
        lines.append(f"| {r['equipment']} | {'상태' if r['task']=='state' else '원인'} | {r['validation_rows']:,} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | {r['baseline_macro_f1']:.3f} | {r['macro_f1_ci_low']:.3f}~{r['macro_f1_ci_high']:.3f} | {r['accepted_coverage']:.1%} | {accepted} |")
    lines+=['','## 불확실성은 어떻게 표시하나','',
        '- low_confidence: 학습 내부 보정 자료에서 정한 확률 기준 미달. 기준을 만족할 자료가 부족하면 모두 보류합니다.',
        '- close_candidates: 1위·2위 확률 차이가 0.15 미만. 이 차이는 사전 지정한 시작 기준입니다.',
        '- missing_observations: 현재 특징 중 20% 초과가 결측. 초기 과거창 부족 때문에 생긴 변화량 NaN과 구분합니다.',
        '- outside_training_envelope: 현재 특징 중 10% 초과가 학습의 0.5~99.5백분위 범위 밖. 새로운 고장을 검출한다는 보장은 없는 보수적 경고입니다.',
        '- 원인 후보 확률은 이상 상태라는 조건 아래의 점수입니다. 상태가 정상·정비면 대시보드 원인 후보를 숨깁니다. 상태 또는 원인에 보류가 있으면 원인 확정 문구를 쓰지 않습니다.','',
        '## 설비별 관측 근거와 혼동 사례','']
    html_sections=[]
    for eq,metrics in all_metrics.items():
        lines += [f'### 설비 {eq}','']
        for task in ['state','cause']:
            m=metrics.get(task,{})
            if 'macro_f1' not in m:
                lines += [f'{task}: {m.get("unavailable","결과 없음")}',''];continue
            lines += [f'![{eq} {task} 혼동행렬](figures/{eq}_{task}_confusion.png)','']
            support=', '.join(f'{label}: {int(m["report"][label]["support"]):,}행' for label in m['labels'])
            lines += [f'- {task} 평가 클래스: {support}.','']
            imp=pd.read_csv(out/'tables'/f'{eq}_{task}_observation_importance.csv').head(3)
            lines += [f'- {task}에서 섞었을 때 F1이 가장 감소한 관측: '+', '.join(f'{r.observation} ({r.macro_f1_drop:+.3f})' for r in imp.itertuples())+'.','']
            if task=='cause':lines += [f'- 원인 상위 2개 후보 포함률: {m["top2_accuracy"]:.1%}. 부품 정확도: {m["part_accuracy"]:.1%}.','']
        if 'end_to_end' in metrics:
            end=metrics['end_to_end'];lines += [f'- 상태 모델까지 포함한 이상 감지율: {end["abnormal_detection_recall"]:.1%}. 실제 이상 시점에서 상태 감지와 원인 1위가 함께 맞은 비율: {end["cause_top1_with_state_gate_accuracy"]:.1%}.','']
        cases=[r for r in all_cases if r['equipment']==eq]
        lines += ['| 종류 | 목표 | 상황·시각 | 정답 | 예측 / 2위 후보 | 점수 | 관측 설명 |','|---|---|---|---|---|---:|---|']
        for r in cases:
            evidence=' / '.join(f"{e['observation']}: {e['current_value']} (정상 참조 {e['normal_reference']:.3g}, 점수 차이 {e['score_drop_to_reference']:+.3f})" for e in r['observations'])
            lines.append(f"| {r['kind']} | {r['task']} | {r['SCENARIO_ID']} {r['MEAS_DT']} | {r['truth']} | {r['prediction']} / {r['candidate_2']} | {r['confidence']:.2f} | {evidence} |")
        lines += ['']
        html_sections.append(f'<h2>설비 {eq}</h2>'+''.join(f'<img alt="{eq} {task} confusion" src="figures/{eq}_{task}_confusion.png">' for task in ['state','cause'] if 'macro_f1' in metrics.get(task,{})))
    lines += ['## 관측 설명의 의미와 한계','',
        '관측 중요도는 검증 표본 최대 1,500행에서 센서별 특징 묶음을 두 번 섞어 Macro-F1이 얼마나 감소하는지 확인한 값입니다. 상관된 센서끼리는 중요도가 나뉠 수 있습니다. 검증 중요도는 사후 설명용이며 이번 모델의 특징 선택에 사용하지 않았습니다.',
        '사례 설명은 한 센서의 특징 묶음을 학습 정상 시점의 중앙값으로 바꿨을 때 예측 점수가 얼마나 감소하는지 보여줍니다. 정상 참조는 부하를 정확히 일치시킨 비교가 아니며, 그 값을 실제로 바꾸면 고장이 해결된다는 인과 설명이 아닙니다. 음수면 그 관측이 해당 판단을 지지하지 않는 방향입니다.',
        'A/C의 __now는 마지막 5분 구간의 평균입니다. B의 __now는 같은 시각의 위치별 평균입니다. 표준편차·오차 특징의 단위는 원 관측 또는 해당 파생량의 단위입니다.',
        '공개된 diagnostics는 합성 엔진의 상태·원인 정답입니다. 실제 설비와 고장 원인을 확정하는 성능으로 일반화할 수 없습니다. 특히 failed가 없거나 적은 설비의 고장 진단은 별도 검증이 필요합니다.',
        'D/E/F는 관측 간격이 각각 약 5분/2시간/30분입니다. 출력이 없는 시각을 새 측정으로 간주하지 말고 마지막 관측 시각과 경과 시간을 화면에 표시해야 합니다.',
        '## 저장된 결과','',
        '- metrics_summary.csv: 설비·목표별 핵심 성능.',
        '- tables/*_metrics.json: 클래스별 성능, 확률 보정 결과, 전체 혼동행렬.',
        '- tables/*_cases.json: 대표 혼동·불확실·정답 사례와 관측 설명.',
        '- predictions/*_validation.csv.gz: 검증 정답과 예측 비교. gzip CSV이므로 pandas.read_csv로 바로 읽을 수 있습니다.',
        '- predictions/*_dashboard_test.csv.gz: 정답 없는 시험 자료의 화면 연동용 출력.',
        '- tables/*_latest_test.csv: 시험 상황별 마지막 관측의 작은 예측 표.',
        '- models/*_state.joblib, *_cause.joblib: 재사용 가능한 모델과 보정·보류 기준.',
        '- features/*.joblib: 재실행 속도를 위한 특징 캐시. 센서 입력과 정답 칼럼은 분리됩니다.',
        '- run_manifest.json: 실행 환경, 분할, 입력 서명, 명시적 특징 목록.','',
        '## 다시 실행','',
        'CH.4 폴더에서 `python Diagnosis.py`를 실행하면 전체를 생성합니다. 입력이 바뀌지 않았다면 `python Diagnosis.py --reuse-features`로 특징 생성 시간을 줄일 수 있습니다.',
        '`python Diagnosis.py --equipment C --output Results/Diagnosis_C`로 설비 하나만 별도 실행할 수 있습니다. 기존 Main.py는 전처리용이므로 학습을 위해 실행할 필요가 없습니다.',
        '기존 모델로 새 시나리오를 진단하려면 `python Diagnosis.py --predict-scenario S0289 --equipment C`를 실행합니다. 원인 후보는 상태 결과로 게이트되며 정답 칼럼을 읽지 않습니다.','']
    (out/'분석결과_과정설명.md').write_text('\n'.join(lines),encoding='utf-8')
    overview=summary.copy()
    for col in ['accuracy','accepted_coverage','accepted_accuracy']:
        overview[col]=overview[col].map(lambda x:'보류' if pd.isna(x) else f'{x:.1%}')
    overview['task']=overview['task'].map({'state':'상태','cause':'부품·원인'})
    overview=overview.rename(columns={'equipment':'설비','task':'목표','validation_rows':'평가 행','validation_scenarios':'평가 상황',
        'accuracy':'정확도','macro_f1':'Macro-F1','baseline_macro_f1':'기준 F1','macro_f1_ci_low':'F1 하한','macro_f1_ci_high':'F1 상한',
        'accepted_coverage':'자동 채택률','accepted_accuracy':'채택 정확도','ece':'확률 오차'})
    body='<h1>설비 상태·원인 진단</h1><p>독립 검증 72개 상황. 원인은 실제 이상·원인 확인 시점의 조건부 평가입니다. 상세 과정·혼동 사례는 같은 폴더의 분석결과_과정설명.md를 확인하세요.</p><img style="width:90%;max-width:1000px" src="figures/performance_summary.png">'+overview.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')+''.join(html_sections)
    document='<!doctype html><html lang="ko"><meta charset="utf-8"><title>설비 진단 분석 결과</title><style>body{font-family:Malgun Gothic,sans-serif;margin:32px;color:#172237;max-width:1400px}table{border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid #ddd}th{background:#edf2f7}img{width:48%;vertical-align:top}p{line-height:1.7}</style>'+body+'</html>'
    (out/'결과_한눈에보기.html').write_text(document,encoding='utf-8')
    write_first_read(all_metrics,out,run)
    save_json(out/'run_manifest.json',run)
    return summary

def write_first_read(all_metrics,out,run):
    names={'A':'소둔로','B':'용접기','C':'연주','D':'조질압연','E1':'배풍기 1','E2':'배풍기 2','F':'교반기'}
    lines=['# 상태·원인 진단: 먼저 읽기','',
        '현재 상태를 알아보는 모델과 원인 후보를 좁히는 모델을 7개 설비별로 만들었습니다. 센서와 운전조건만 입력하고 진단 정답은 채점에만 사용합니다.','',
        '![검증 성능 비교](figures/performance_summary.png)','',
        '| 설비 | 상태 F1 | 부품·원인 F1 | 원인 평가 상황 수 | 원인 후보 2개 포함률 | 원인 자동 채택률 |',
        '|---|---:|---:|---:|---:|---:|']
    for eq,metrics in all_metrics.items():
        s=metrics.get('state',{});c=metrics.get('cause',{})
        if 'macro_f1' not in s or 'macro_f1' not in c:continue
        lines.append(f'| {eq} {names[eq]} | {s["macro_f1"]:.3f} | {c["macro_f1"]:.3f} | {c["scenarios"]} | {c["top2_accuracy"]:.1%} | {c["accepted_coverage"]:.1%} |')
    lines+=['','## 이 결과로 무엇을 말할 수 있나','',
        '상태 F1이 높아도 세부 원인이 정확하다는 뜻은 아닙니다. 원인 F1과 후보 2개 포함률, 실제 평가 상황 수를 함께 봐야 합니다. 자동 채택률이 0이면 원인 후보는 저장하되 확정 진단은 모두 보류한다는 뜻입니다.','',
        '자동 채택은 검증 결과를 보고 높인 점수가 아닙니다. 원래 학습 상황 중 별도로 남긴 54개 상황에서 정한 기준을 적용했습니다. 채택 정확도 역시 모든 실제 설비에서 보장되는 값은 아닙니다.','',
        '## 특히 조심할 고장 진단 범위','',
        '| 설비 | 검증에 실제 등장한 상태 | 고장 failed 평가 행 |','|---|---|---:|']
    for eq,metrics in all_metrics.items():
        s=metrics.get('state',{})
        if 'labels' not in s:continue
        failed=s['report'].get('failed',{}).get('support',0)
        lines.append(f'| {eq} | {", ".join(STATE_KO.get(v,v) for v in s["labels"])} | {int(failed):,} |')
    lines+=['',
        '고장 평가 행이 0인 설비는 이번 점수로 고장 진단 성능을 주장할 수 없습니다. 행 수는 서로 독립적인 고장 사건 수가 아닙니다. 특히 E1/E2는 2시간마다 관측하므로 짧은 상태를 놓칠 수 있습니다.','',
        '## 현재 화면에 연결할 결과','',
        'predictions/*_dashboard_test.csv.gz는 정답 없는 시험 상황의 예측입니다. 상태, 원인 후보 2개, 확률, 상태·원인 각각의 추가 점검 여부가 들어 있습니다. tables/*_latest_test.csv는 상황별 마지막 관측만 남긴 작은 표입니다. 실제 서비스에서는 마지막 관측 시각과 경과 시간을 함께 보여줘야 합니다.','',
        '원인 확률은 이상 상태라는 조건 아래의 점수입니다. 상태 또는 원인에 검토 필요 표시가 있으면 원인 확정 대신 후보/추가 점검 필요로 표시합니다. 학습하지 않은 새로운 고장을 정확히 분류하는 기능은 아직 없습니다.','',
        '## 과정과 혼동 사례를 보는 순서','',
        '1. 이 문서의 상태·원인 성능 차이를 확인합니다.',
        '2. 분석결과_과정설명.md에서 설비별 혼동행렬과 사례를 확인합니다. 행은 실제 정답, 열은 예측입니다.',
        '3. tables/*_cases.json에서 confident_error(확신했지만 틀림), uncertain(불확실), correct_abnormal(맞힌 이상)을 봅니다.',
        '4. 사례의 observations는 센서 특징을 정상 학습 참조값으로 바꾸었을 때 예측 점수가 변하는 정도입니다. 실제 부품 고장의 인과 증명은 아닙니다.','',
        '## 다음 개선 우선순위','',
        '혼동이 큰 원인끼리는 같은 부하·소재 조건에서 센서 추이를 비교하고, 정상 운전과의 차이를 나타내는 조건별 잔차 특징을 검토합니다. 고장·원인별 독립 시나리오가 적으면 추가 데이터가 우선입니다. 다음 모델의 효과를 평가하려면 새 검증 분할 또는 train 내부 교차검증을 사용하고, 같은 validation에 반복해서 맞추지 않습니다.','',
        '코드 실행 방법은 ../../diagnosis/README.md에 있습니다. 모델은 첫 기준 모델이며 생산 환경 적용을 완료한 서비스는 아닙니다.','',
        '원인 유형 뜻: '+', '.join(f'{k}={v}' for k,v in MODE_KO.items())+'.','']
    (out/'먼저읽기_진단결과.md').write_text('\n'.join(lines),encoding='utf-8')

def export_validation_dashboard(args):
    # 저장된 모델과 특징 캐시로 validation 시나리오 전체의 화면용 예측을 만든다 (재학습 없음).
    # 원인은 정답이 아니라 "예측된 상태가 degraded/failed인 행"에만 붙인다 (test 화면용 예측과 같은 규칙).
    # actual_* 열은 validation 정답이라 시연에서 예측과 비교하는 용도다.
    for eq in args.equipment:
        table=joblib.load(args.output/'features'/f'{eq}.joblib')['table']
        val=table[table.split.eq('validation')]
        res={}
        for task in ['state','cause']:
            model=joblib.load(args.output/'models'/f'{eq}_{task}.joblib')
            res[task],_=result_table(model,val,model['columns'],task)
        dashboard=res['state'].rename(columns={'prediction':'state_prediction','confidence':'state_probability'})
        dashboard=dashboard[['SCENARIO_ID','EQP_CD','MEAS_DT','equipment','state_prediction','state_probability','needs_review','uncertainty_reasons']].copy()
        cause=res['cause'];mask=dashboard.state_prediction.isin(['degraded','failed'])
        dashboard['cause_candidate']=cause.prediction.where(mask,'')
        dashboard['cause_candidate_2']=cause.candidate_2.where(mask,'')
        dashboard['cause_probability_conditional']=cause.confidence.where(mask,np.nan)
        dashboard['cause_needs_review']=cause.needs_review.where(mask,False)
        dashboard['cause_uncertainty_reasons']=cause.uncertainty_reasons.where(mask,'')
        dashboard['actual_state']=val.GT_HEALTH_STATE.to_numpy()
        dashboard['actual_cause']=(val.GT_CAUSE_PART.fillna('')+'::'+val.GT_FAULT_MODE.fillna('')).where(val.GT_CAUSE_PART.fillna('').ne(''),'').to_numpy()
        save=args.output/'predictions'/f'{eq}_dashboard_validation.csv.gz'
        dashboard.to_csv(save,index=False,compression='gzip');print(save,len(dashboard),flush=True)

def predict_scenario(args,manifest):
    # Independent inference path: does not read any GT or diagnostics files.
    for eq in args.equipment:
        path=args.scenarios/args.predict_scenario/f'{eq}.csv'
        allowed=set(PROFILES[eq][1])|{'MEAS_DT','IS_MISSING','QUALITY_FLAG','N','FS_HZ'}
        table=make_features(pd.read_csv(path,usecols=lambda c:c in allowed),eq)
        result=pd.DataFrame({'MEAS_DT':table.index,'SCENARIO_ID':args.predict_scenario,'EQP_CD':PROFILES[eq][0]})
        for task in ['state','cause']:
            model=joblib.load(args.output/'models'/f'{eq}_{task}.joblib')
            p=probabilities(model,table[model['columns']]);order=np.argsort(-p,axis=1)
            reason,_=uncertainty(model,table[model['columns']],p)
            result[task+'_prediction']=np.asarray(model['classes'])[order[:,0]]
            result[task+'_candidate_2']=np.asarray(model['classes'])[order[:,1]] if p.shape[1]>1 else ''
            result[task+'_probability']=p.max(axis=1)
            result[task+'_review_reason']=reason
        mask=~result.state_prediction.isin(['degraded','failed'])
        result.loc[mask,['cause_prediction','cause_candidate_2','cause_review_reason']]=''
        result.loc[mask,'cause_probability']=np.nan
        save=args.output/'predictions'/f'{eq}_{args.predict_scenario}_inference.csv'
        result.to_csv(save,index=False,encoding='utf-8-sig');print(save,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equipment',nargs='+',choices=list(PROFILES),default=list(PROFILES))
    parser.add_argument('--scenarios',type=Path,default=BASE/'Data/Processed/Scenarios')
    parser.add_argument('--raw',type=Path,default=BASE/'Data/Raw')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--reuse-features',action='store_true')
    parser.add_argument('--skip-test',action='store_true')
    parser.add_argument('--predict-scenario')
    parser.add_argument('--validation-dashboard',action='store_true',help='저장된 모델로 validation 시나리오의 화면용 예측 생성 (재학습 없음)')
    parser.add_argument('--report-only',action='store_true',help='저장된 평가 결과로 보고서만 다시 생성')
    args=parser.parse_args()
    for folder in ['features','models','figures','tables','predictions']:(args.output/folder).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    manifest=manifest_at(args.raw)
    if args.predict_scenario:return predict_scenario(args,manifest)
    if args.validation_dashboard:return export_validation_dashboard(args)
    if args.report_only:
        run=json.loads((args.output/'run_manifest.json').read_text(encoding='utf-8'))
        metrics={};cases=[]
        audits=[];run['equipment']={}
        for eq in args.equipment:
            metrics[eq]=json.loads((args.output/'tables'/f'{eq}_metrics.json').read_text(encoding='utf-8'))
            cases.extend(json.loads((args.output/'tables'/f'{eq}_cases.json').read_text(encoding='utf-8')))
            cache=joblib.load(args.output/'features'/f'{eq}.joblib')
            audits.extend(cache['audit'])
            run['equipment'][eq]=dict(columns=feature_columns(cache['table']),input_signature=cache['signature'])
        make_report(metrics,cases,audits,args.output,run)
        return
    train_ids=[s['id'] for s in manifest['scenarios'] if s['split']=='train']
    rng=np.random.default_rng(20261015);calibration_ids=set(rng.choice(train_ids,54,replace=False))
    all_metrics={};all_cases=[];audits=[]
    run=dict(seed=20261015,feature_version=FEATURE_VERSION,python=platform.python_version(),sklearn=sklearn.__version__,pandas=pd.__version__,
        dataset_version=manifest['datasetVersion'],raw_root=str(args.raw.resolve()),scenario_root=str(args.scenarios.resolve()),
        fit_scenarios=sorted(set(train_ids)-calibration_ids),calibration_scenarios=sorted(calibration_ids),
        validation_scenarios=[s['id'] for s in manifest['scenarios'] if s['split']=='validation'],
        test_scenarios=[s['id'] for s in manifest['scenarios'] if s['split']=='test'],equipment={},
        calibration_target='at most 10% error among calibration accepted rows, minimum max(20,10% of calibration rows)',
        uncertainty_heuristics=dict(candidate_margin=.15,current_missing_fraction=.2,outside_training_fraction=.1,training_quantiles=[.005,.995]))
    for eq in args.equipment:
        table,audit=build_equipment(eq,args.scenarios,args.raw,manifest,args.output,args.reuse_features)
        metrics,cases=process_equipment(eq,table,args.output,calibration_ids,not args.skip_test)
        all_metrics[eq]=metrics;all_cases.extend(cases);audits.extend(audit)
        run['equipment'][eq]=dict(columns=feature_columns(table),input_signature=fingerprint(args.scenarios,args.raw,manifest,eq))
        # Save progress so a later equipment failure does not discard completed work.
        save_json(args.output/'partial_metrics.json',all_metrics)
    summary=make_report(all_metrics,all_cases,audits,args.output,run)
    print(summary.to_string(index=False),flush=True)
    print(f'Report: {args.output / "분석결과_과정설명.md"}',flush=True)
