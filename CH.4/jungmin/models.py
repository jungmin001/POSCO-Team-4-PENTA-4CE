import joblib
import numpy as np
import pandas as pd
import matplotlib

from . import config, features

if not config.Show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def MakeModels():
    Extra = make_pipeline(SimpleImputer(strategy="median"),
                          ExtraTreesClassifier(n_estimators=config.ForestTrees, min_samples_leaf=3, class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=config.RandomState))
    Boost = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, class_weight="balanced",
                                           random_state=config.RandomState)

    return {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "forest": make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestClassifier(n_estimators=config.ForestTrees, min_samples_leaf=config.ForestLeaf,
                                                       class_weight="balanced_subsample", n_jobs=-1,
                                                       random_state=config.RandomState)),
        # ExtraTrees + 그래디언트 부스팅 평균 (시나리오 단위 5-fold 교차검증에서 forest보다 PR-AUC가 6H +0.04, 24H +0.03 높음)
        "ensemble": VotingClassifier([("extra", Extra), ("boost", Boost)], voting="soft"),
    }


# 6시간 창과 12시간 창의 차이 (최근 변화량), 시점 이전 데이터만 쓰는 기존 특징에서 계산
def AddDeltaFeatures(Table):
    Stats = ("mean", "std", "min", "max")
    Delta = {f"d_{Col[3:]}": Table[Col] - Table[f"w12_{Col[3:]}"] for Col in Table.columns
             if Col.startswith("w6_") and Col.endswith(tuple(f"_{Stat}" for Stat in Stats)) and f"w12_{Col[3:]}" in Table}

    return pd.concat([Table, pd.DataFrame(Delta, index=Table.index)], axis=1)


# 특징 표를 읽어 정답이 있는 행만 남기고 train·validation으로 나눔 (시나리오 단위 분할)
def LoadData(Equip, Horizon):
    Table = AddDeltaFeatures(pd.read_csv(features.TablePath(Equip)))
    Target = f"RISK_{Horizon}"
    Table = Table[Table[Target].notna()].reset_index(drop=True)

    FeatureCols = [Col for Col in Table.columns if Col not in config.MetaCols]

    return Table[Table["split"] == "train"], Table[Table["split"] == "validation"], FeatureCols, Target


# 학습 데이터만으로 경보 임계값 선택: 시나리오 단위 교차검증 예측에서 F1이 가장 큰 임계값
def PickThreshold(Model, Train, FeatureCols, Target):
    Oof = np.zeros(len(Train))
    Splitter = GroupKFold(n_splits=min(config.Folds, Train["scenario"].nunique()))

    for FitIdx, PredIdx in Splitter.split(Train, groups=Train["scenario"]):
        Fitted = Model.fit(Train.iloc[FitIdx][FeatureCols], Train.iloc[FitIdx][Target])
        Oof[PredIdx] = Fitted.predict_proba(Train.iloc[PredIdx][FeatureCols])[:, 1]

    Precision, Recall, Thresholds = precision_recall_curve(Train[Target], Oof)
    F1 = 2 * Precision[:-1] * Recall[:-1] / np.maximum(Precision[:-1] + Recall[:-1], 1e-12)

    return float(Thresholds[int(np.argmax(F1))])


# 임계값 하나에서의 성능
def AtThreshold(Y, P, Threshold):
    Pred = P >= Threshold
    Tn, Fp, Fn, Tp = confusion_matrix(Y.astype(int), Pred.astype(int), labels=[0, 1]).ravel()
    Precision = Tp / (Tp + Fp) if Tp + Fp else float("nan")
    Recall = Tp / (Tp + Fn) if Tp + Fn else float("nan")
    F1 = 2 * Precision * Recall / (Precision + Recall) if Tp else 0.0

    return {"threshold": round(Threshold, 3), "alarms": int(Tp + Fp), "TP": int(Tp), "FP": int(Fp), "FN": int(Fn),
            "precision": Precision, "recall": Recall, "F1": F1}


# 임계값(0.01~0.99)을 바꿔 가며 정밀도·재현율·F1을 그린 그림, 모델마다 한 칸, 점선은 train으로 고른 임계값
def PlotThresholds(Y, Probs, Chosen, Path_, Title):
    Y = np.asarray(Y).astype(bool)
    Grid = np.round(np.arange(0.01, 1.0, 0.01), 2)

    Fig, Axes = plt.subplots(1, len(Probs), figsize=(6 * len(Probs), 4.5), sharey=True, squeeze=False)

    for Ax, (Name, P) in zip(Axes[0], Probs.items()):
        Pred = np.asarray(P)[None, :] >= Grid[:, None]
        Tp, Fp = (Pred & Y).sum(axis=1), (Pred & ~Y).sum(axis=1)
        Precision = np.where(Tp + Fp > 0, Tp / np.maximum(Tp + Fp, 1), np.nan)
        Recall = Tp / max(int(Y.sum()), 1)
        F1 = 2 * Precision * Recall / np.maximum(Precision + Recall, 1e-12)

        Ax.plot(Grid, Precision, label="precision")
        Ax.plot(Grid, Recall, label="recall")
        Ax.plot(Grid, F1, label="F1", linestyle=":")
        Ax.axvline(Chosen[Name], color="gray", linestyle="--")
        Ax.set_title(f"{Name} (train-selected threshold {Chosen[Name]:.2f})")
        Ax.set_xlabel("Alarm threshold")
        Ax.set_ylim(0, 1.02)
        Ax.grid(alpha=0.3)

    Axes[0][0].set_ylabel("Score")
    Axes[0][0].legend(loc="lower left")
    Fig.suptitle(Title)
    Fig.tight_layout()
    Fig.savefig(Path_, dpi=120)

    if not config.Show:
        plt.close(Fig)


# 설비 하나, 위험 구간(6H 또는 24H) 하나의 학습·평가
def Run(Equip, Horizon, Verbose=True):
    Show = print if Verbose else (lambda *Args, **Kwargs: None)

    Train, Valid, FeatureCols, Target = LoadData(Equip, Horizon)
    YTrain, YValid = Train[Target].astype(int), Valid[Target].astype(int)

    Show(f"\n===== 설비 {Equip}, {Horizon} 고장 위험 =====")
    Show(f"train {len(Train):,}행 (고장 {int(YTrain.sum())}, {YTrain.mean() * 100:.1f}%, 시나리오 {Train['scenario'].nunique()}개) | "
          f"validation {len(Valid):,}행 (고장 {int(YValid.sum())}, {YValid.mean() * 100:.1f}%, 시나리오 {Valid['scenario'].nunique()}개) | 특징 {len(FeatureCols)}개")

    Results, Predictions, Curves, Tables = [], Valid[["scenario", "time", Target]].copy(), {}, []
    Chosen, Probs, Saved = {}, {}, {}

    # 기준선: 항상 같은 확률을 예측하면 PR-AUC는 고장 비율과 같음
    Results.append({"model": "baseline(고장 비율)", "PR_AUC": YValid.mean(), "ROC_AUC": 0.5})

    for Name, Model in MakeModels().items():
        Threshold = PickThreshold(Model, Train, FeatureCols, Target)
        Fitted = Model.fit(Train[FeatureCols], YTrain)
        P = Fitted.predict_proba(Valid[FeatureCols])[:, 1]

        Predictions[f"p_{Name}"] = P
        Chosen[Name], Probs[Name], Saved[Name] = Threshold, P, Fitted
        Curves[Name] = precision_recall_curve(YValid, P)[:2]
        Results.append({"model": Name, "PR_AUC": average_precision_score(YValid, P), "ROC_AUC": roc_auc_score(YValid, P),
                        **{f"{Key}@train선택임계값": Value for Key, Value in AtThreshold(YValid, P, Threshold).items()}})

        for Value in config.Thresholds:
            Tables.append({"model": Name, **AtThreshold(YValid, P, Value)})

        if Name == "forest":
            Importance = pd.Series(Fitted[-1].feature_importances_, index=FeatureCols).sort_values(ascending=False)

    ResultDf = pd.DataFrame(Results)
    ThresholdDf = pd.DataFrame(Tables)

    pd.set_option("display.width", 220)
    Show("\n[validation 성능] 임계값은 train 교차검증으로 선택")
    Show(ResultDf.round(3).to_string(index=False))
    Show("\n[임계값별 정밀도·재현율 (validation)]")
    Show(ThresholdDf.round(3).to_string(index=False))
    Show("\n[랜덤 포레스트 특징 중요도 상위 10]")
    Show(Importance.head(10).round(4).to_string())

    config.OutDir.mkdir(parents=True, exist_ok=True)
    Tag = f"{Equip}_{Horizon}"
    ResultDf.to_csv(config.OutDir / f"results_{Tag}.csv", index=False)
    ThresholdDf.to_csv(config.OutDir / f"thresholds_{Tag}.csv", index=False)
    Predictions.to_csv(config.OutDir / f"predictions_{Tag}.csv", index=False)
    Importance.rename("importance").to_csv(config.OutDir / f"importance_{Tag}.csv")

    plt.figure(figsize=(6, 5))
    for Name, (Precision, Recall) in Curves.items():
        plt.plot(Recall, Precision, label=Name)
    plt.axhline(YValid.mean(), color="gray", linestyle="--", label="baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (validation) {Tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OutDir / f"pr_curve_{Tag}.png", dpi=120)

    if not config.Show:
        plt.close()

    PlotThresholds(YValid, Probs, Chosen, config.OutDir / f"threshold_curve_{Tag}.png", f"Precision / recall by alarm threshold (validation) {Tag}")

    # 학습한 모델 저장: 새 데이터 추론용 (특징 열 순서, train으로 고른 임계값, validation 성능 포함)
    ModelDir = config.OutDir / "models"
    ModelDir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"equip": Equip, "horizon": Horizon, "target": Target, "feature_cols": FeatureCols,
                 "models": Saved, "thresholds": Chosen,
                 "validation": ResultDf.set_index("model")[["PR_AUC", "ROC_AUC"]].to_dict("index")},
                ModelDir / f"risk_{Tag}.joblib", compress=3)

    Forest = ResultDf[ResultDf["model"] == "forest"].iloc[0]
    Ensemble = ResultDf[ResultDf["model"] == "ensemble"].iloc[0]
    Summary = {"equip": Equip, "horizon": Horizon, "train_pos": int(YTrain.sum()), "train_n": len(Train),
               "val_pos": int(YValid.sum()), "val_n": len(Valid), "baseline_PR_AUC": ResultDf.iloc[0]["PR_AUC"],
               "logistic_PR_AUC": ResultDf[ResultDf["model"] == "logistic"].iloc[0]["PR_AUC"],
               "forest_PR_AUC": Forest["PR_AUC"], "forest_ROC_AUC": Forest["ROC_AUC"],
               "forest_precision": Forest["precision@train선택임계값"], "forest_recall": Forest["recall@train선택임계값"],
               "ensemble_PR_AUC": Ensemble["PR_AUC"], "ensemble_ROC_AUC": Ensemble["ROC_AUC"],
               "ensemble_precision": Ensemble["precision@train선택임계값"], "ensemble_recall": Ensemble["recall@train선택임계값"]}

    return ResultDf, Summary
