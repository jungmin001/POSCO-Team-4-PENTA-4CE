import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, precision_score, recall_score, f1_score)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BaseDir = os.path.dirname(os.path.abspath(__file__))
FeaturePath = os.path.join(BaseDir, "Data", "Processed", "timeseries_features.csv")
MaintenancePath = os.path.join(BaseDir, "Data", "Raw", "G-02_정비이력.csv")

KeyCols = ["EQP_CD", "SLAB_NO", "MEAS_DT"]
FailTypes = ["BM", "CBM"]              # 고장으로 보는 정비 유형 (BM: 고장 정비, CBM: 상태 기반 정비 / TBM은 정기 점검이라 제외)
DangerThresholds = [24, 72, 168]       # Validation에서 비교할 위험 기준 (다음 고장까지 1일 / 3일 / 7일 이내)
TargetRecall = 0.9                     # 최종 위험 기준 선택: Recall이 이 값 이상인 기준 중 F1 최대

# CSV 로드 (설비·슬라브·시각 순 정렬)
def LoadData(FilePath):
    Df = pd.read_csv(FilePath, parse_dates=["MEAS_DT"])
    Df = Df.sort_values(KeyCols).reset_index(drop=True)

    return Df

# 정비이력에서 주편 검사설비(CC1/CC2-INS01)의 고장 기록만 가져오기
def LoadFailures(FilePath):
    MaintenanceDf = pd.read_csv(FilePath, encoding="utf-8-sig")
    FailDf = MaintenanceDf[MaintenanceDf["EQP_TAG"].isin(["P1-CC1-INS01", "P1-CC2-INS01"])
                           & MaintenanceDf["MNT_TYPE"].isin(FailTypes)].copy()

    # P1-CC1-INS01 → CC1 (센서 데이터의 EQP_CD와 맞춤)
    FailDf["EQP_CD"] = FailDf["EQP_TAG"].str.split("-").str[1].astype(str)
    FailDf["FAIL_DT"] = pd.to_datetime(FailDf["DETC_DT"])
    FailDf = FailDf[["EQP_CD", "FAIL_DT"]].sort_values("FAIL_DT").reset_index(drop=True)

    print("\n===== 고장 기록 =====")
    print(FailDf["EQP_CD"].value_counts().to_dict())

    return FailDf

# 타깃 생성: 다음 고장까지 남은 시간 TTF (시간 단위, CH.2의 RUL과 같은 역할)
# 각 측정 시점 이후 같은 설비에서 처음 발생한 고장 시각 - 측정 시각
def CreateTTF(Df, FailDf):
    ResultDf = Df.copy()
    ResultDf["EQP_CD"] = ResultDf["EQP_CD"].astype(str)
    ResultDf = pd.merge_asof(ResultDf.sort_values("MEAS_DT"), FailDf, left_on="MEAS_DT", right_on="FAIL_DT",
                             by="EQP_CD", direction="forward")
    ResultDf["TTF"] = (ResultDf["FAIL_DT"] - ResultDf["MEAS_DT"]).dt.total_seconds() / 3600

    # 마지막 고장 이후에 측정된 행은 다음 고장을 알 수 없어서 제외
    NoFailCount = ResultDf["TTF"].isna().sum()
    ResultDf = ResultDf.dropna(subset=["TTF"]).drop(columns="FAIL_DT")
    ResultDf = ResultDf.sort_values(KeyCols).reset_index(drop=True)

    print("\n===== TTF 생성 =====")
    print("다음 고장을 알 수 없어 제외한 행 수:", NoFailCount)
    print("데이터 크기:", ResultDf.shape)
    print("슬라브 수:", ResultDf["SLAB_NO"].nunique())

    return ResultDf

# Feature 목록 (설비·슬라브·시각, TTF 제외: 센서 상태만으로 TTF를 얼마나 설명하는지 확인)
def GetFeatureColumns(Df):
    ExcludeCols = KeyCols + ["TTF"]
    FeatureCols = [Col for Col in Df.columns if Col not in ExcludeCols]

    print("\n===== Feature =====")
    print("사용 Feature 수:", len(FeatureCols))

    return FeatureCols

# 슬라브 단위 Train / Validation 분리 (같은 슬라브가 양쪽에 들어가지 않게 슬라브 번호를 80:20으로)
def SplitBySlab(Df, TestSize=0.2, RandomState=42):
    TrainSlabs, ValSlabs = train_test_split(Df["SLAB_NO"].unique(), test_size=TestSize, random_state=RandomState)
    TrainDf = Df[Df["SLAB_NO"].isin(TrainSlabs)].copy()
    ValDf = Df[Df["SLAB_NO"].isin(ValSlabs)].copy()

    print("\n===== Train / Validation =====")
    print("Train 슬라브 수:", TrainDf["SLAB_NO"].nunique())
    print("Validation 슬라브 수:", ValDf["SLAB_NO"].nunique())

    return TrainDf, ValDf

# 회귀 성능 (MAE, RMSE, R²)
def ComputeRegressionMetrics(YTrue, Predictions):
    return {"MAE": mean_absolute_error(YTrue, Predictions),
            "RMSE": np.sqrt(mean_squared_error(YTrue, Predictions)),
            "R2": r2_score(YTrue, Predictions)}

# 회귀 성능 출력
def PrintRegressionMetrics(Title, Metrics):
    print(f"\n{Title}")
    print(f"MAE : {Metrics['MAE']:.2f}")
    print(f"RMSE: {Metrics['RMSE']:.2f}")
    print(f"R²  : {Metrics['R2']:.3f}")

# RandomForestRegressor 학습
def TrainRandomForest(XTrain, YTrain):
    # 닫힌 그래프 창 객체를 메인 스레드에서 먼저 정리 (학습 스레드에서 정리되면 tkinter 경고)
    gc.collect()

    Model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    Model.fit(XTrain, YTrain)

    return Model

# 위험 판정 지표 (Precision, Recall, F1, FP, FN, TP, TN)
def ComputeRiskMetrics(ActualRisk, PredRisk):
    Tn, Fp, Fn, Tp = confusion_matrix(ActualRisk, PredRisk, labels=[0, 1]).ravel()

    return {"Precision": precision_score(ActualRisk, PredRisk, zero_division=0),
            "Recall": recall_score(ActualRisk, PredRisk, zero_division=0),
            "F1": f1_score(ActualRisk, PredRisk, zero_division=0),
            "FP": Fp, "FN": Fn, "TP": Tp, "TN": Tn}

# 하나의 위험 기준 평가 (실제 위험: 실제 TTF <= 기준 / 예측 위험: 예측 TTF <= 기준)
def EvaluateRisk(YTrue, Predictions, Threshold):
    ActualRisk = (np.asarray(YTrue) <= Threshold).astype(int)
    PredRisk = (np.asarray(Predictions) <= Threshold).astype(int)

    return {"TTF_Threshold": Threshold, **ComputeRiskMetrics(ActualRisk, PredRisk)}

# 회귀 모델 학습·비교 (DummyRegressor: 평균 TTF만 예측하는 기준 모델 / RandomForest: 메인 모델)
def TrainModels(TrainDf, ValDf, FeatureCols):
    DummyModel = DummyRegressor(strategy="mean")
    DummyModel.fit(TrainDf[FeatureCols], TrainDf["TTF"])
    DummyPred = DummyModel.predict(ValDf[FeatureCols])

    RfModel = TrainRandomForest(TrainDf[FeatureCols], TrainDf["TTF"])
    RfPred = RfModel.predict(ValDf[FeatureCols])

    print("\n===== 회귀 성능 (Validation) =====")
    PrintRegressionMetrics("DummyRegressor", ComputeRegressionMetrics(ValDf["TTF"], DummyPred))
    PrintRegressionMetrics("RandomForestRegressor", ComputeRegressionMetrics(ValDf["TTF"], RfPred))

    return RfModel, RfPred

# 위험 기준 비교 + 최종 기준 선택 (Validation에서만 수행)
def SelectRiskThreshold(YVal, RfPred):
    ValidationRiskDf = pd.DataFrame([EvaluateRisk(YVal, RfPred, Threshold) for Threshold in DangerThresholds])
    print("\n===== Validation 위험 기준 비교 =====")
    print(ValidationRiskDf.round(3).to_string(index=False))

    # 위험 시점을 놓치지 않도록 Recall 기준 이상 중 F1 최대 (없으면 Recall 최대)
    RecallOkDf = ValidationRiskDf[ValidationRiskDf["Recall"] >= TargetRecall]
    if len(RecallOkDf) > 0:
        FinalDangerThreshold = int(RecallOkDf.loc[RecallOkDf["F1"].idxmax(), "TTF_Threshold"])
    else:
        FinalDangerThreshold = int(ValidationRiskDf.loc[ValidationRiskDf["Recall"].idxmax(), "TTF_Threshold"])
    print("\n최종 위험 기준 (Validation으로 선택): TTF <=", FinalDangerThreshold, "시간")

    return ValidationRiskDf, FinalDangerThreshold

# 실제 vs 예측 TTF 산점도 (Validation)
def PlotPrediction(YVal, RfPred):
    plt.figure(figsize=(8, 7))
    plt.scatter(YVal, RfPred, alpha=0.35)
    MaxValue = max(YVal.max(), RfPred.max())
    plt.plot([0, MaxValue], [0, MaxValue], linestyle="--", label="실제값 = 예측값")
    plt.xlabel("실제 TTF (시간)")
    plt.ylabel("예측 TTF (시간)")
    plt.title("Validation 실제 TTF vs 예측 TTF")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Feature 중요도 (상위 15개)
def CheckFeatureImportance(Model, FeatureCols, TopN=15):
    ImportanceDf = pd.DataFrame({"Feature": FeatureCols, "Importance": Model.feature_importances_})
    ImportanceDf = ImportanceDf.sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\n===== Feature 중요도 상위 {TopN}개 =====")
    print(ImportanceDf.head(TopN).round(4).to_string(index=False))

    TopImportanceDf = ImportanceDf.head(TopN).sort_values("Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(TopImportanceDf["Feature"], TopImportanceDf["Importance"])
    plt.title(f"Random Forest Feature 중요도 상위 {TopN}개")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return ImportanceDf

# 슬라브 단위 GroupKFold (특정 80:20 분할에서만 성능이 좋았는지 확인하는 안정성 검증)
def RunGroupKFold(Df, FeatureCols, NSplits=5):
    Results = []
    Splitter = GroupKFold(n_splits=NSplits)
    for Fold, (TrainIdx, ValIdx) in enumerate(Splitter.split(Df, groups=Df["SLAB_NO"]), start=1):
        FoldTrain, FoldVal = Df.iloc[TrainIdx], Df.iloc[ValIdx]
        FoldModel = TrainRandomForest(FoldTrain[FeatureCols], FoldTrain["TTF"])
        Metrics = ComputeRegressionMetrics(FoldVal["TTF"], FoldModel.predict(FoldVal[FeatureCols]))
        Results.append({"Fold": Fold, "Validation_Slabs": FoldVal["SLAB_NO"].nunique(),
                        "MAE": Metrics["MAE"], "RMSE": Metrics["RMSE"]})

    GroupKFoldDf = pd.DataFrame(Results)
    print("\n===== GroupKFold =====")
    print(GroupKFoldDf.round(3).to_string(index=False))
    print("\nFold 평균")
    print(GroupKFoldDf[["MAE", "RMSE"]].mean().round(3))
    print("\nFold 표준편차")
    print(GroupKFoldDf[["MAE", "RMSE"]].std().round(3))

    return GroupKFoldDf

# 자동 실행 함수
def Main():
    Df = LoadData(FeaturePath)

    FailDf = LoadFailures(MaintenancePath)

    Df = CreateTTF(Df, FailDf)

    FeatureCols = GetFeatureColumns(Df)

    TrainDf, ValDf = SplitBySlab(Df)

    RfModel, RfPred = TrainModels(TrainDf, ValDf, FeatureCols)

    ValidationRiskDf, FinalDangerThreshold = SelectRiskThreshold(ValDf["TTF"], RfPred)

    PlotPrediction(ValDf["TTF"], RfPred)

    CheckFeatureImportance(RfModel, FeatureCols)

    RunGroupKFold(Df, FeatureCols)


if __name__ == "__main__":
    Main()
