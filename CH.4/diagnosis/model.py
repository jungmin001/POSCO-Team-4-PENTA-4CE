"""Class-balanced trees, independent calibration, uncertainty and explanations."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline

def temperature_scale(probabilities,temperature):
    z=np.log(np.clip(probabilities,1e-8,1))/temperature
    z-=z.max(axis=1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(axis=1,keepdims=True)

def probabilities(bundle,x):
    return temperature_scale(bundle['pipeline'].predict_proba(x),bundle['temperature'])

def confidence_metrics(y,p,classes):
    y=np.asarray(y); classes=np.asarray(classes)
    pred=classes[p.argmax(axis=1)]; conf=p.max(axis=1)
    correct=pred==y
    ece=0.; bins=[]
    for low in np.arange(0,1,.1):
        mask=(conf>=low)&(conf<low+.1 if low<.9 else conf<=1)
        if mask.any():
            acc=float(correct[mask].mean()); avg=float(conf[mask].mean())
            ece+=mask.mean()*abs(acc-avg)
            bins.append(dict(lower=float(low),count=int(mask.sum()),confidence=avg,accuracy=acc))
    known=np.isin(y,classes)
    loss=float(log_loss(y[known],p[known],labels=classes)) if known.any() and len(classes)>1 else None
    return dict(ece=float(ece),log_loss_known_classes=loss,unseen_truth_rows=int((~known).sum()),bins=bins)

def train_bundle(x,y,cal_x,cal_y,seed=42):
    pipeline=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
        ExtraTreesClassifier(n_estimators=80,max_depth=18,min_samples_leaf=4,
            max_features=.8,class_weight='balanced',n_jobs=2,random_state=seed))
    pipeline.fit(x,y)
    classes=pipeline.classes_
    raw=pipeline.predict_proba(cal_x)
    known=np.isin(np.asarray(cal_y),classes)
    # This is held-out training data, not the validation set.
    temp=1.
    if known.sum()>=20 and len(classes)>1:
        opt=minimize_scalar(lambda t:log_loss(np.asarray(cal_y)[known],temperature_scale(raw[known],t),labels=classes),bounds=(.3,4.),method='bounded')
        temp=float(opt.x)
    p=temperature_scale(raw,temp)
    top=np.sort(p,axis=1)
    margin=top[:,-1]-top[:,-2] if len(classes)>1 else np.ones(len(p))
    pred=classes[p.argmax(axis=1)]
    # Select confidence rule on calibration only. Require useful support, target <=10% error.
    threshold=1.01
    for threshold_candidate in np.arange(.4,1.001,.025):
        m=(p.max(axis=1)>=threshold_candidate)&(margin>=.15)&known
        if m.sum()>=max(20,int(.1*len(cal_y))) and (pred[m]!=np.asarray(cal_y)[m]).mean()<=.1:
            threshold=float(threshold_candidate);break
    normal_median=x.median().fillna(0)
    # Training marginal envelope is a warning heuristic, not a formal OOD detector.
    return dict(pipeline=pipeline,classes=classes,temperature=temp,threshold=threshold,
        margin_threshold=.15,columns=list(x.columns),median=normal_median,
        lower=x.quantile(.005),upper=x.quantile(.995),
        calibration=confidence_metrics(cal_y,p,classes),fit_rows=len(x),calibration_rows=len(cal_y))

def uncertainty(bundle,x,p):
    sort=np.sort(p,axis=1)
    margin=sort[:,-1]-sort[:,-2] if p.shape[1]>1 else np.ones(len(p))
    # Missing at least 20% of current features, or >10% outside robust training envelope.
    current=[c for c in x if c.endswith('__now') and 'MISSING_RATE' not in c]
    missing=x[current].isna().mean(axis=1).to_numpy()>.2
    outside=((x[current]<bundle['lower'][current])|(x[current]>bundle['upper'][current])).mean(axis=1).to_numpy()>.1
    low=p.max(axis=1)<bundle['threshold']; close=margin<bundle['margin_threshold']
    reasons=[]
    for a,b,c,d in zip(low,close,missing,outside):
        reasons.append(';'.join(s for yes,s in [(a,'low_confidence'),(b,'close_candidates'),(c,'missing_observations'),(d,'outside_training_envelope')] if yes))
    return np.array(reasons,dtype=object),margin

def evaluate(y,p,classes,scenario_ids,seed=42):
    y=np.asarray(y);pred=np.asarray(classes)[p.argmax(axis=1)]
    labels=sorted(set(y)|set(pred))
    report=classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)
    matrix=confusion_matrix(y,pred,labels=labels)
    # Bootstrap independent scenarios, rather than correlated sensor rows.
    groups=[]
    ids=np.asarray(scenario_ids)
    for sid in np.unique(ids):
        m=ids==sid;groups.append(confusion_matrix(y[m],pred[m],labels=labels))
    rng=np.random.default_rng(seed);boot=[];group_matrices=np.array(groups)
    for _ in range(200):
        cm=group_matrices[rng.integers(0,len(groups),len(groups))].sum(axis=0)
        denom=cm.sum(axis=0)+cm.sum(axis=1)
        boot.append(np.divide(2*cm.diagonal(),denom,out=np.zeros(len(labels)),where=denom!=0).mean())
    return dict(rows=len(y),scenarios=len(groups),accuracy=float(accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
        macro_f1_scenario_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
        report=report,labels=labels,confusion=matrix.tolist(),
        probability_quality=confidence_metrics(y,p,classes))

def explain(bundle,x,predicted_class,reference,max_features=3):
    """Local perturbation to a normal reference; association, not a causal explanation."""
    cls=list(bundle['classes']).index(predicted_class)
    original=probabilities(bundle,x)[0,cls]
    groups={}
    for c in bundle['columns']:
        root=c.split('__')[0]
        if root=='MISSING_RATE': continue
        groups.setdefault(root,[]).append(c)
    changed=[];names=[]
    for root,cols in groups.items():
        alt=x.copy();alt.loc[:,cols]=reference[cols].to_numpy()
        changed.append(alt);names.append(root)
    all_alt=pd.concat(changed,ignore_index=True)
    drops=original-probabilities(bundle,all_alt)[:,cls]
    order=np.argsort(-drops)[:max_features]
    return [dict(observation=names[j],current_value=None if pd.isna(x.iloc[0].get(names[j]+'__now')) else float(x.iloc[0][names[j]+'__now']),
        normal_reference=float(reference.get(names[j]+'__now',0)),score_drop_to_reference=float(drops[j])) for j in order]
