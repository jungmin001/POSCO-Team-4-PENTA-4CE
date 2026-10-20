import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jungmin import config


# 조질압연기(D, P1-CR1-SPM01) 건강 인텔리전스 모델, CH.4 폴더에서 실행
#   python jungmin/mill_d.py
#
# 세 단계로 이어진다.
#   1. 이상 감지: 5분마다 "지금 고장 진행 중인가"를 행 단위로 판정 (모터 전류 ÷ 부하, 진동 첨도·파고율 등 물리 특징)
#   2. 고장 유형 판별: 이상으로 감지된 구간이 베어링 쪽인지 롤 쪽인지, 어떤 모드인지
#   3. 고장 임박 예측: 6시간 간격 시점에서 6·24시간 안에 고장 날 확률 (1단계 감지 결과의 지속 시간을 특징으로 사용)
# 모든 특징은 그 시점 이전 데이터만 쓴다 (미래 정보 없음). 학습은 train, 평가는 validation.
#
# 저장 (jungmin/outputs/mill_D)
#   mill_d_model.joblib        세 단계 모델, 특징 열, 임계값, 성능
#   metrics.json               단계별 성능 (train 교차검증 포함)
#   validation_rows.csv        validation 시나리오의 5분 단위 감지·유형 예측 + 핵심 물리 지표
#   validation_risk.csv        validation 6시간 간격 고장 임박 확률
#   report_data.json           리포트(report.html) 그림에 쓰는 요약 자료
Tag = "P1-CR1-SPM01"
OutDir = config.OutDir / "mill_D"

Sensors = ["LOAD", "VIB-TOP_RMS", "VIB-TOP_STD", "VIB-TOP_KUR", "VIB-TOP_CRF", "VIB-BOT_RMS", "CUR-MTR_RMS"]
Physics = ["cur_load", "asym", "rms_mean"]
WindowRows = {"30분": 6, "3시간": 36}
Persistence = 3          # 연속 3행(15분) 이상 감지돼야 경보
StepHours = 5 / 60
Seed = 0

TypeNames = {"bearing::bearing-damage": "베어링 손상", "roll::roll-wear": "롤 마모",
             "bearing::background-wear": "베어링 일반 마모", "roll::background-wear": "롤 일반 마모"}


# 센서 신호로 본 두 계열: 충격형(첨도·파고율이 크게 오름)과 마찰형(전류÷부하와 상하 진동 비대칭이 오름)
# 일반 마모 구간의 부품 이름(베어링/롤)은 신호 계열과 반대로 붙어 있어서, 4유형 구분과 별도로 계열 정확도를 본다
Families = {"bearing::bearing-damage": "충격형", "roll::background-wear": "충격형",
            "roll::roll-wear": "마찰형", "bearing::background-wear": "마찰형"}


def Boost(Iter=200, Rate=0.06, Depth=5):
    return HistGradientBoostingClassifier(max_iter=Iter, learning_rate=Rate, max_depth=Depth, class_weight="balanced", random_state=Seed)


# 6시간 간격 위험 예측용 앙상블 (jungmin/models.py와 같은 구성)
def RiskModel():
    Extra = make_pipeline(SimpleImputer(strategy="median"),
                          ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=Seed))

    return VotingClassifier([("extra", Extra), ("boost", Boost(200, 0.05, 4))], voting="soft")


# ---------------------------------------------------------------------------
# 자료
# ---------------------------------------------------------------------------
def LoadData():
    Index = pd.read_csv(config.ScenarioDir / "index.csv")
    Index = Index[Index["split"].isin(["train", "validation"])]

    Frames = []
    for Id, Split in zip(Index["scenario"], Index["split"]):
        Frame = pd.read_csv(config.ScenarioDir / Id / "D.csv", usecols=lambda Col: not Col.startswith("ROW"))
        Frame["split"] = Split
        Frames.append(Frame)

    Df = pd.concat(Frames, ignore_index=True)
    Df["MEAS_DT"] = pd.to_datetime(Df["MEAS_DT"])
    Df["Abnormal"] = Df["GT_HEALTH_STATE"].isin(["degraded", "failed"]).astype(int)
    Df["Mode"] = np.where(Df["Abnormal"] == 1, Df["GT_CAUSE_PART"].fillna("") + "::" + Df["GT_FAULT_MODE"].fillna(""), "")
    Df["Bad"] = ((Df["IS_MISSING"] == 1) | (Df["IS_STOPPED"] == 1) | (Df["IS_MAINTENANCE"] == 1) | (Df["IS_OUTLIER"] == 1))

    return Df.sort_values(["SCENARIO_ID", "T_SEC"]).reset_index(drop=True)


def LoadGrid(Ids):
    Rows = []
    for Id in Ids:
        Labels = pd.read_csv(config.ScenarioDir / Id / "labels.csv")
        Labels = Labels[Labels["EQP_CD"] == Tag]
        Rows.append(Labels[["SCENARIO_ID", "T_SEC", "GT_RISK_6H", "GT_RISK_24H"]])

    return pd.concat(Rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 1단계 특징: 물리 특징 + 과거 창 요약 (모두 그 행 이전 데이터만)
# ---------------------------------------------------------------------------
def MakeFeatures(Df):
    Values = Df[Sensors].where(~Df["Bad"])
    Phys = pd.DataFrame({
        "cur_load": Values["CUR-MTR_RMS"] / Values["LOAD"],                       # 부하 대비 모터 전류 (마찰·저항 증가)
        "asym": Values["VIB-TOP_RMS"] / Values["VIB-BOT_RMS"],                    # 상·하부 진동 비대칭 (롤 쪽 이상)
        "rms_mean": (Values["VIB-TOP_RMS"] + Values["VIB-BOT_RMS"]) / 2,          # 평균 진동 크기
    }, index=Df.index)
    Base = pd.concat([Values, Phys], axis=1)
    Group = Base.groupby(Df["SCENARIO_ID"])

    Out = {Col: Base[Col] for Col in Sensors + Physics}
    Watch = ["cur_load", "VIB-TOP_KUR", "VIB-TOP_CRF", "VIB-TOP_RMS", "asym", "CUR-MTR_RMS"]

    for Name, Rows in WindowRows.items():
        Rolled = Group[Watch].rolling(Rows, min_periods=2)
        for Stat in ("mean", "std", "min", "max"):
            Part = getattr(Rolled, Stat)().reset_index(level=0, drop=True).sort_index()
            for Col in Watch:
                Out[f"{Col}_{Stat}_{Name}"] = Part[Col]

    # 시나리오 시작부터 지금까지 중앙값 대비 얼마나 벗어났는가 (그 시나리오의 평상시 기준)
    for Col in ["cur_load", "VIB-TOP_RMS", "VIB-TOP_KUR", "VIB-TOP_CRF", "asym"]:
        Median = Group[Col].expanding(min_periods=12).median().reset_index(level=0, drop=True).sort_index()
        Out[f"{Col}_vs_base"] = Base[Col] - Median

    Feat = pd.DataFrame(Out)
    Feat["bad_frac_30분"] = Df["Bad"].astype(float).groupby(Df["SCENARIO_ID"]).rolling(6, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()

    return Feat


# 3단계 특징: 감지 확률의 최근 평균·최댓값·감지 비율, 감지가 이어진 시간
def DetectionFeatures(Df, Prob):
    Frame = pd.DataFrame({"p_fault": Prob}, index=Df.index)
    Frame["on"] = (Frame["p_fault"] > 0.5).astype(float).where(Frame["p_fault"].notna())
    Group = Frame.groupby(Df["SCENARIO_ID"])
    Out = {"p_fault": Frame["p_fault"]}

    for Name, Rows in (("1h", 12), ("6h", 72), ("12h", 144), ("24h", 288)):
        Out[f"pf_mean_{Name}"] = Group["p_fault"].rolling(Rows, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
        Out[f"pf_max_{Name}"] = Group["p_fault"].rolling(Rows, min_periods=1).max().reset_index(level=0, drop=True).sort_index()
        Out[f"on_frac_{Name}"] = Group["on"].rolling(Rows, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()

    Run = np.zeros(len(Df))
    for _, Idx in Df.groupby("SCENARIO_ID", sort=False).indices.items():
        On = Frame["on"].to_numpy()[Idx] > 0
        Count = 0
        for Pos, Flag in zip(Idx, On):
            Count = Count + 1 if Flag else 0
            Run[Pos] = Count * StepHours
    Out["on_run_h"] = Run

    return pd.DataFrame(Out)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def Persistent(Prob, Threshold, Scenario):
    On = (Prob >= Threshold).astype(float)
    return On.groupby(Scenario).rolling(Persistence, min_periods=Persistence).min().reset_index(level=0, drop=True).sort_index().fillna(0).astype(bool)


# 고장 구간(에피소드) 단위: 몇 개를 잡았고, 얼마나 빨리 잡았고, 정상 구간에서 경보가 하루 몇 번 울렸는가
def EpisodeMetrics(Df, Alarm):
    Details, FalseEvents = [], 0
    Scenarios = Df["SCENARIO_ID"].nunique()

    for Id, Idx in Df.groupby("SCENARIO_ID", sort=False).indices.items():
        Part = Df.iloc[Idx]
        Flag = Alarm.iloc[Idx].to_numpy()
        Truth = Part["Abnormal"].to_numpy() == 1
        Time = Part["T_SEC"].to_numpy()
        Edges = np.flatnonzero(np.diff(np.r_[0, Truth.astype(int), 0]))

        InEpisode = np.zeros(len(Truth), bool)
        for Start, End in zip(Edges[::2], Edges[1::2]):
            Reach = min(len(Truth), End + 12)                       # 끝난 뒤 1시간까지는 정상 경보로 세지 않음
            InEpisode[max(0, Start - 12):Reach] = True
            Hits = np.flatnonzero(Flag[Start:End])
            Details.append({"scenario": Id, "mode": Part["Mode"].iloc[Start], "start_h": float(Time[Start] / 3600),
                            "length_h": float((End - Start) * StepHours), "detected": bool(len(Hits)),
                            "delay_h": float(Hits[0] * StepHours) if len(Hits) else None})

        Rise = np.flatnonzero(np.diff(np.r_[0, Flag.astype(int)]) == 1)
        FalseEvents += int(sum(not InEpisode[Pos] for Pos in Rise))

    Detected = [Item for Item in Details if Item["detected"]]
    Delays = [Item["delay_h"] for Item in Detected]

    return {"episodes": len(Details), "detected": len(Detected),
            "delay_median_h": float(np.median(Delays)) if Delays else None,
            "delay_max_h": float(np.max(Delays)) if Delays else None,
            "false_alarms": FalseEvents, "false_alarms_per_day": FalseEvents / (Scenarios * 3)}, Details


def BestThreshold(Y, P):
    Precision, Recall, Thresholds = precision_recall_curve(Y, P)
    F1 = 2 * Precision[:-1] * Recall[:-1] / np.maximum(Precision[:-1] + Recall[:-1], 1e-12)
    return float(Thresholds[int(np.argmax(F1))])


def Scores(Y, P, Threshold):
    Pred = P >= Threshold
    Tp, Fp, Fn = int((Pred & (Y == 1)).sum()), int((Pred & (Y == 0)).sum()), int((~Pred & (Y == 1)).sum())
    Precision, Recall = Tp / max(Tp + Fp, 1), Tp / max(Tp + Fn, 1)
    return {"PR_AUC": float(average_precision_score(Y, P)), "ROC_AUC": float(roc_auc_score(Y, P)),
            "threshold": round(Threshold, 4), "precision": Precision, "recall": Recall,
            "F1": 2 * Precision * Recall / max(Precision + Recall, 1e-12), "alarms": Tp + Fp, "positives": int((Y == 1).sum())}


def OutOfFold(Make, X, Y, Groups, Folds=5):
    Prob = np.zeros(len(Y))
    for Fit, Pred in GroupKFold(n_splits=Folds).split(X, groups=Groups):
        Prob[Pred] = Make().fit(X.iloc[Fit], Y.iloc[Fit]).predict_proba(X.iloc[Pred])[:, 1]
    return Prob


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def Main():
    OutDir.mkdir(parents=True, exist_ok=True)

    print("자료 읽기·특징 만들기", flush=True)
    Df = LoadData()
    Feat = MakeFeatures(Df)
    Train, Valid = (Df["split"] == "train"), (Df["split"] == "validation")
    Ok = ~Df["Bad"]
    Metrics = {"scenarios": {"train": int(Df.loc[Train, "SCENARIO_ID"].nunique()), "validation": int(Df.loc[Valid, "SCENARIO_ID"].nunique())}}

    # --- 1단계: 이상 감지 -------------------------------------------------
    print("1단계 이상 감지", flush=True)
    Y = Df["Abnormal"]
    Detector = Boost(150, 0.08, 5).fit(Feat[Train & Ok], Y[Train & Ok])
    ProbValid = Detector.predict_proba(Feat[Valid])[:, 1]

    # train 안에서는 시나리오 단위 교차검증 예측을 써서 3단계가 자기 정답을 보지 않게 함
    ProbAll = np.full(len(Df), np.nan)
    ProbAll[Valid.to_numpy()] = ProbValid
    TrainIds = Df.loc[Train, "SCENARIO_ID"].unique()
    Rng = np.random.default_rng(Seed)
    for Fold in np.array_split(Rng.permutation(TrainIds), 5):
        Test = Df["SCENARIO_ID"].isin(Fold)
        Fit = Train & ~Test & Ok
        ProbAll[Test.to_numpy()] = Boost(150, 0.08, 5).fit(Feat[Fit], Y[Fit]).predict_proba(Feat[Test])[:, 1]
    ProbAll[Df["Bad"].to_numpy()] = np.nan

    OofMask = Train & Ok
    DetThreshold = BestThreshold(Y[OofMask], ProbAll[OofMask.to_numpy()])
    Metrics["detect"] = {"validation": Scores(Y[Valid & Ok], ProbAll[(Valid & Ok).to_numpy()], DetThreshold),
                         "train_cv": Scores(Y[OofMask], ProbAll[OofMask.to_numpy()], DetThreshold)}

    AlarmSeries = Persistent(pd.Series(ProbAll, index=Df.index), DetThreshold, Df["SCENARIO_ID"])
    Metrics["detect"]["episode_validation"], EpisodeValid = EpisodeMetrics(Df[Valid].reset_index(drop=True), AlarmSeries[Valid].reset_index(drop=True))
    Metrics["detect"]["episode_train_cv"], _ = EpisodeMetrics(Df[Train].reset_index(drop=True), AlarmSeries[Train].reset_index(drop=True))
    Metrics["detect"]["persistence_rows"] = Persistence

    # --- 2단계: 고장 유형 판별 ---------------------------------------------
    print("2단계 고장 유형 판별", flush=True)
    AbTrain = Train & (Y == 1) & Ok
    Typer = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, class_weight="balanced", random_state=Seed)
    Typer.fit(Feat[AbTrain], Df.loc[AbTrain, "Mode"])
    Classes = list(Typer.classes_)

    TypeProb = pd.DataFrame(np.nan, index=Df.index, columns=Classes)
    TypeProb.loc[Ok] = Typer.predict_proba(Feat[Ok])
    TypePred = pd.Series(np.nan, index=Df.index, dtype=object)
    TypePred[Ok] = TypeProb.loc[Ok].idxmax(axis=1)

    AbValid = Valid & (Y == 1) & Ok
    Truth, Pred = Df.loc[AbValid, "Mode"], TypePred[AbValid]
    Order = [Name for Name in TypeNames if Name in Classes]
    Metrics["type"] = {
        "classes": Order, "rows": int(AbValid.sum()),
        "macro_f1": float(f1_score(Truth, Pred, average="macro")), "accuracy": float(accuracy_score(Truth, Pred)),
        "part_accuracy": float(accuracy_score(Truth.str.split("::").str[0], Pred.str.split("::").str[0])),
        "confusion": confusion_matrix(Truth, Pred, labels=Order).tolist(),
    }

    Metrics["type"]["family_accuracy"] = float(accuracy_score(Truth.map(Families), Pred.map(Families)))

    # 에피소드 다수결 (감지된 구간 안에서 가장 많이 나온 유형)과 train 교차검증
    Block = (Df["Mode"] != Df["Mode"].shift()) | (Df["SCENARIO_ID"] != Df["SCENARIO_ID"].shift())
    Episode = Df.assign(Pred=TypePred, Block=Block.cumsum())[AbValid.to_numpy() | (Valid & (Y == 1)).to_numpy()]
    Vote = Episode.groupby("Block").agg(truth=("Mode", "first"), vote=("Pred", lambda S: S.value_counts().index[0] if S.notna().any() else ""))
    Metrics["type"]["episode_vote_accuracy"] = float((Vote["truth"] == Vote["vote"]).mean())
    Metrics["type"]["episode_vote_part_accuracy"] = float((Vote["truth"].str.split("::").str[0] == Vote["vote"].str.split("::").str[0]).mean())
    Metrics["type"]["episodes"] = int(len(Vote))

    # 5-fold 교차검증 (train+validation 합침, 에피소드 수가 적어 성능 안정성 확인용)
    AbAll = (Y == 1) & Ok
    CvPred = pd.Series(index=Df.index[AbAll], dtype=object)
    Groups = Df.loc[AbAll, "SCENARIO_ID"]
    for Fit, Pred_ in GroupKFold(n_splits=5).split(Feat[AbAll], groups=Groups):
        Model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, class_weight="balanced", random_state=Seed)
        Model.fit(Feat[AbAll].iloc[Fit], Df.loc[AbAll, "Mode"].iloc[Fit])
        CvPred.iloc[Pred_] = Model.predict(Feat[AbAll].iloc[Pred_])
    Metrics["type"]["cv_macro_f1"] = float(f1_score(Df.loc[AbAll, "Mode"], CvPred, average="macro"))
    Metrics["type"]["cv_accuracy"] = float(accuracy_score(Df.loc[AbAll, "Mode"], CvPred))
    Metrics["type"]["cv_family_accuracy"] = float(accuracy_score(Df.loc[AbAll, "Mode"].map(Families), CvPred.map(Families)))
    Metrics["type"]["cv_part_accuracy"] = float(accuracy_score(Df.loc[AbAll, "Mode"].str.split("::").str[0], CvPred.str.split("::").str[0]))

    # --- 3단계: 고장 임박 예측 ---------------------------------------------
    print("3단계 고장 임박 예측", flush=True)
    DetFeat = DetectionFeatures(Df, ProbAll)
    Full = pd.concat([Feat, DetFeat.drop(columns=["p_fault"]).assign(p_fault=DetFeat["p_fault"])], axis=1)
    Grid = LoadGrid(Df["SCENARIO_ID"].unique())
    Keys = Df.reset_index()[["index", "SCENARIO_ID", "T_SEC", "split"]]
    Grid = Grid.merge(Keys, on=["SCENARIO_ID", "T_SEC"], how="inner")
    Rows = Full.loc[Grid["index"]].reset_index(drop=True)
    Metrics["risk"] = {}
    RiskOut = Grid[["SCENARIO_ID", "T_SEC", "split", "GT_RISK_6H", "GT_RISK_24H"]].copy()
    Models = {}

    for Horizon, Col in (("6H", "GT_RISK_6H"), ("24H", "GT_RISK_24H")):
        Known = Grid[Col].notna().to_numpy()
        TrainRows = Known & (Grid["split"] == "train").to_numpy()
        ValidRows = Known & (Grid["split"] == "validation").to_numpy()
        Target = Grid[Col].astype("float")

        Oof = OutOfFold(RiskModel, Rows[TrainRows].reset_index(drop=True), Target[TrainRows].astype(int).reset_index(drop=True),
                        Grid.loc[TrainRows, "SCENARIO_ID"].reset_index(drop=True))
        Threshold = BestThreshold(Target[TrainRows].astype(int), Oof)
        Model = RiskModel().fit(Rows[TrainRows], Target[TrainRows].astype(int))
        Prob = Model.predict_proba(Rows[(Grid["split"] == "validation").to_numpy()])[:, 1]
        RiskOut.loc[Grid["split"] == "validation", f"P{Horizon}"] = Prob

        Probs = pd.Series(np.nan, index=Grid.index)
        Probs[(Grid["split"] == "validation").to_numpy()] = Prob
        Metrics["risk"][Horizon] = {"validation": Scores(Target[ValidRows].astype(int), Probs[ValidRows].to_numpy(), Threshold),
                                    "train_cv": Scores(Target[TrainRows].astype(int), Oof, Threshold)}
        Models[Horizon] = {"model": Model, "threshold": Threshold}

    # --- 저장 ---------------------------------------------------------------
    print("저장", flush=True)
    DfV = Df[Valid].copy()
    PhysCols = ["cur_load", "asym", "rms_mean", "VIB-TOP_KUR", "VIB-TOP_CRF", "VIB-TOP_RMS", "CUR-MTR_RMS", "LOAD"]
    Rows_ = DfV[["SCENARIO_ID", "T_SEC", "MEAS_DT", "GT_HEALTH_STATE", "Mode", "Bad"]].copy()
    Rows_["p_fault"] = ProbAll[Valid.to_numpy()]
    Rows_["alarm"] = AlarmSeries[Valid].astype(int).to_numpy()
    Rows_["type"] = TypePred[Valid]
    Rows_["family"] = TypePred[Valid].map(Families)
    for Name in Order:
        Rows_[f"p_{Name}"] = TypeProb.loc[Valid, Name]
    for Col in PhysCols:
        Rows_[Col] = Feat.loc[Valid, Col] if Col in Feat else DfV[Col]
    Number = Rows_.select_dtypes("number").columns
    Rows_[Number] = Rows_[Number].round(4)
    Rows_.to_csv(OutDir / "validation_rows.csv", index=False, encoding="utf-8-sig")
    RiskOut[RiskOut["split"] == "validation"].round(4).to_csv(OutDir / "validation_risk.csv", index=False, encoding="utf-8-sig")

    joblib.dump({"detector": Detector, "detect_threshold": DetThreshold, "persistence": Persistence, "typer": Typer, "classes": Classes,
                 "risk": Models, "feature_cols": list(Feat.columns), "risk_cols": list(Full.columns), "metrics": Metrics},
                OutDir / "mill_d_model.joblib", compress=3)
    (OutDir / "metrics.json").write_text(json.dumps(Metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 리포트 그림용: 유형별 특징 지문 (정상 대비 중앙값 변화 배수)
    Fingerprint = {}
    Normal = Df[Ok & (Y == 0)]
    for Mode in Order:
        Part = Df[Ok & (Df["Mode"] == Mode)]
        Fingerprint[Mode] = {Col: float(Part[Col].median()) if Col in Df else float(Feat.loc[Part.index, Col].median()) for Col in ["LOAD"]}
        Fingerprint[Mode] = {Col: float(Feat.loc[Part.index, Col].median()) for Col in ["cur_load", "VIB-TOP_RMS", "VIB-TOP_KUR", "VIB-TOP_CRF", "asym", "CUR-MTR_RMS"]}
    NormalMed = {Col: float(Feat.loc[Normal.index, Col].median()) for Col in ["cur_load", "VIB-TOP_RMS", "VIB-TOP_KUR", "VIB-TOP_CRF", "asym", "CUR-MTR_RMS"]}
    (OutDir / "report_data.json").write_text(json.dumps({"fingerprint": Fingerprint, "normal": NormalMed, "episodes": EpisodeValid, "order": Order, "families": Families},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")

    Show = Metrics
    print("\n===== 조질압연기(D) 모델 요약 (validation) =====")
    print(f"1. 이상 감지  : 행 단위 AP {Show['detect']['validation']['PR_AUC']:.3f}, 정밀도 {Show['detect']['validation']['precision']:.3f}, 재현율 {Show['detect']['validation']['recall']:.3f}")
    Ep = Show["detect"]["episode_validation"]
    print(f"                고장 구간 {Ep['detected']}/{Ep['episodes']}개 감지, 감지까지 중앙값 {Ep['delay_median_h']:.2f}시간, 오경보 하루 {Ep['false_alarms_per_day']:.2f}회")
    T = Show["type"]
    print(f"2. 유형 판별  : 4유형 macro-F1 {T['macro_f1']:.3f} (교차검증 {T['cv_macro_f1']:.3f}), 부품 정확도 {T['part_accuracy']:.3f}, 신호 계열 정확도 {T['family_accuracy']:.3f}, 구간 다수결 {T['episode_vote_accuracy']:.3f} ({T['episodes']}개)")
    for Horizon in ("6H", "24H"):
        R = Show["risk"][Horizon]
        print(f"3. 임박 {Horizon:3s}  : PR-AUC {R['validation']['PR_AUC']:.3f} (교차검증 {R['train_cv']['PR_AUC']:.3f}), 정밀도 {R['validation']['precision']:.2f}, 재현율 {R['validation']['recall']:.2f}, 고장 {R['validation']['positives']}건")
    print(f"\n저장 위치: {OutDir}")


if __name__ == "__main__":
    Main()
