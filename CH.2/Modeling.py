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
import Analysis

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BaseDir = os.path.dirname(os.path.abspath(__file__))
FeaturePath = os.path.join(BaseDir, "Data", "Processed", "timeseries_features.csv")
TestPath = os.path.join(BaseDir, "Data", "Raw", "test_fd001.csv")
RulPath = os.path.join(BaseDir, "Data", "Raw", "rul_fd001.csv")

DangerThresholds = [10, 20, 30, 50]   # Validation에서 비교할 위험 기준
ZThresholds = [2.0, 2.5, 3.0, 3.5]    # Z-score 경보 기준 후보
TargetRecall = 0.9                    # 최종 위험 기준 선택: Recall이 이 값 이상인 기준 중 F1 최대

# CSV 로드 (엔진·사이클 순 정렬, TRAIN·TEST 공용)
def LoadData(FilePath):
    Df = pd.read_csv(FilePath)
    Df = Df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

    return Df

# RUL 생성 (엔진별 마지막 사이클 - 현재 사이클)
def CreateRUL(Df):
    ResultDf = Df.copy()
    MaxCycle = ResultDf.groupby("unit_number")["time_in_cycles"].transform("max")
    ResultDf["RUL"] = MaxCycle - ResultDf["time_in_cycles"]

    print("\n===== TRAIN 데이터 =====")
    print("데이터 크기:", ResultDf.shape)
    print("엔진 수:", ResultDf["unit_number"].nunique())

    return ResultDf

# 핵심 센서 = rolling 특성이 있는 센서 (Analysis.py에서 선정한 센서와 자동으로 맞춰짐)
def GetKeySensors(Df):
    KeySensors = list(dict.fromkeys(Col.split("_rolling_")[0] for Col in Df.columns if "_rolling_" in Col))
    print("핵심 센서:", KeySensors)

    return KeySensors

# Feature 목록 (unit_number, time_in_cycles, RUL 제외: 센서 상태만으로 RUL을 얼마나 설명하는지 확인)
def GetFeatureColumns(Df):
    ExcludeCols = ["unit_number", "time_in_cycles", "RUL"]
    FeatureCols = [Col for Col in Df.columns if Col not in ExcludeCols]

    print("\n===== Feature =====")
    print("사용 Feature 수:", len(FeatureCols))

    return FeatureCols

# 엔진 단위 Train / Validation 분리 (같은 엔진이 양쪽에 들어가지 않게 엔진 번호를 80:20으로)
def SplitByEngine(Df, TestSize=0.2, RandomState=42):
    TrainUnits, ValUnits = train_test_split(Df["unit_number"].unique(), test_size=TestSize, random_state=RandomState)
    TrainDf = Df[Df["unit_number"].isin(TrainUnits)].copy()
    ValDf = Df[Df["unit_number"].isin(ValUnits)].copy()

    print("\n===== Train / Validation =====")
    print("Train 엔진 수:", TrainDf["unit_number"].nunique())
    print("Validation 엔진 수:", ValDf["unit_number"].nunique())

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

# 하나의 RUL 위험 기준 평가 (실제 위험: 실제 RUL <= 기준 / 예측 위험: 예측 RUL <= 기준)
def EvaluateRisk(YTrue, Predictions, Threshold):
    ActualRisk = (np.asarray(YTrue) <= Threshold).astype(int)
    PredRisk = (np.asarray(Predictions) <= Threshold).astype(int)

    return {"RUL_Threshold": Threshold, **ComputeRiskMetrics(ActualRisk, PredRisk)}

# 회귀 모델 학습·비교 (DummyRegressor: 평균 RUL만 예측하는 기준 모델 / RandomForest: 메인 모델)
def TrainModels(TrainDf, ValDf, FeatureCols):
    DummyModel = DummyRegressor(strategy="mean")
    DummyModel.fit(TrainDf[FeatureCols], TrainDf["RUL"])
    DummyPred = DummyModel.predict(ValDf[FeatureCols])

    RfModel = TrainRandomForest(TrainDf[FeatureCols], TrainDf["RUL"])
    RfPred = RfModel.predict(ValDf[FeatureCols])

    print("\n===== 회귀 성능 (Validation) =====")
    PrintRegressionMetrics("DummyRegressor", ComputeRegressionMetrics(ValDf["RUL"], DummyPred))
    PrintRegressionMetrics("RandomForestRegressor", ComputeRegressionMetrics(ValDf["RUL"], RfPred))

    return RfModel, RfPred

# 위험 기준 비교 + 최종 기준 선택 (Validation에서만 수행, 공식 TEST를 보고 고르면 안 됨)
def SelectRiskThreshold(YVal, RfPred):
    ValidationRiskDf = pd.DataFrame([EvaluateRisk(YVal, RfPred, Threshold) for Threshold in DangerThresholds])
    print("\n===== Validation 위험 기준 비교 =====")
    print(ValidationRiskDf.round(3).to_string(index=False))

    # 위험 엔진을 놓치지 않도록 Recall 기준 이상 중 F1 최대 (없으면 Recall 최대)
    RecallOkDf = ValidationRiskDf[ValidationRiskDf["Recall"] >= TargetRecall]
    if len(RecallOkDf) > 0:
        FinalDangerThreshold = int(RecallOkDf.loc[RecallOkDf["F1"].idxmax(), "RUL_Threshold"])
    else:
        FinalDangerThreshold = int(ValidationRiskDf.loc[ValidationRiskDf["Recall"].idxmax(), "RUL_Threshold"])
    print("\n최종 위험 기준 (Validation으로 선택): RUL <=", FinalDangerThreshold)

    return ValidationRiskDf, FinalDangerThreshold

# Z-score 단순 비교 모델 (정상 구간 = RUL 60 초과, 가장 크게 벗어난 센서 기준, 위험 정의는 최종 위험 기준 사용)
def CheckZScore(TrainDf, ValDf, KeySensors, DangerThreshold):
    NormalDf = TrainDf[TrainDf["RUL"] > 60]
    ZScore = ((ValDf[KeySensors] - NormalDf[KeySensors].mean()) / NormalDf[KeySensors].std()).abs()
    MaxZ = ZScore.max(axis=1)
    ActualRisk = (ValDf["RUL"] <= DangerThreshold).astype(int)

    ZResultDf = pd.DataFrame([{"Z_Threshold": ZThreshold, **ComputeRiskMetrics(ActualRisk, (MaxZ >= ZThreshold).astype(int))}
                              for ZThreshold in ZThresholds])
    print(f"\n===== Z-score 비교 (위험: RUL <= {DangerThreshold}) =====")
    print(ZResultDf.round(3).to_string(index=False))

    return ZResultDf

# 실제 vs 예측 RUL 산점도 (Validation)
def PlotPrediction(YVal, RfPred):
    plt.figure(figsize=(8, 7))
    plt.scatter(YVal, RfPred, alpha=0.35)
    MaxValue = max(YVal.max(), RfPred.max())
    plt.plot([0, MaxValue], [0, MaxValue], linestyle="--", label="실제값 = 예측값")
    plt.xlabel("실제 RUL")
    plt.ylabel("예측 RUL")
    plt.title("Validation 실제 RUL vs 예측 RUL")
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

# 엔진 단위 GroupKFold (특정 80:20 분할에서만 성능이 좋았는지 확인하는 안정성 검증)
def RunGroupKFold(Df, FeatureCols, NSplits=5):
    Results = []
    Splitter = GroupKFold(n_splits=NSplits)
    for Fold, (TrainIdx, ValIdx) in enumerate(Splitter.split(Df, groups=Df["unit_number"]), start=1):
        FoldTrain, FoldVal = Df.iloc[TrainIdx], Df.iloc[ValIdx]
        FoldModel = TrainRandomForest(FoldTrain[FeatureCols], FoldTrain["RUL"])
        Metrics = ComputeRegressionMetrics(FoldVal["RUL"], FoldModel.predict(FoldVal[FeatureCols]))
        Results.append({"Fold": Fold, "Validation_Engines": FoldVal["unit_number"].nunique(),
                        "MAE": Metrics["MAE"], "RMSE": Metrics["RMSE"]})

    GroupKFoldDf = pd.DataFrame(Results)
    print("\n===== GroupKFold =====")
    print(GroupKFoldDf.round(3).to_string(index=False))
    print("\nFold 평균")
    print(GroupKFoldDf[["MAE", "RMSE"]].mean().round(3))
    print("\nFold 표준편차")
    print(GroupKFoldDf[["MAE", "RMSE"]].std().round(3))

    return GroupKFoldDf

# 공식 TEST 로드 + Rolling Feature 생성
# TRAIN 특성과 같은 함수(Analysis.AddRollingFeatures)로 만들어 window·컬럼 이름이 항상 같게 유지
def LoadOfficialTest(FilePath, KeySensors, FeatureCols):
    OfficialTestDf = LoadData(FilePath)
    print("\n===== 공식 TEST =====")
    print("TEST 엔진 수:", OfficialTestDf["unit_number"].nunique())
    print("TEST 데이터 크기:", OfficialTestDf.shape)

    OfficialTestDf = Analysis.AddRollingFeatures(OfficialTestDf, KeySensors)
    XOfficialTest = OfficialTestDf[FeatureCols]

    return OfficialTestDf, XOfficialTest

# 공식 TEST 예측 + 정답 연결 (각 엔진의 현재 상태 = 가장 마지막 관측 시점)
def PredictOfficialTest(FinalModel, OfficialTestDf, XOfficialTest, FilePath):
    OfficialTestDf["PredRUL"] = FinalModel.predict(XOfficialTest)
    LastPredDf = OfficialTestDf.groupby("unit_number").tail(1)
    ResultDf = (LastPredDf[["unit_number", "time_in_cycles", "PredRUL"]]
                .sort_values("unit_number").reset_index(drop=True))

    # 첫 번째 컬럼이 실제 RUL
    ResultDf["ActualRUL"] = pd.read_csv(FilePath).iloc[:, 0].reset_index(drop=True)
    ResultDf["Error"] = ResultDf["PredRUL"] - ResultDf["ActualRUL"]

    return ResultDf

# 공식 TEST 최종 확인 (결과 확인만, 이 결과를 보고 모델·기준을 바꾸지 않음)
# 위험 탐지는 Validation에서 정한 기준 하나만 사용 (10/20/30/50 다시 비교하지 않음)
def EvaluateOfficialTest(ResultDf, FinalDangerThreshold):
    print("\n===== 공식 TEST RUL 성능 =====")
    PrintRegressionMetrics("RandomForestRegressor", ComputeRegressionMetrics(ResultDf["ActualRUL"], ResultDf["PredRUL"]))

    OfficialRiskDf = pd.DataFrame([EvaluateRisk(ResultDf["ActualRUL"], ResultDf["PredRUL"], FinalDangerThreshold)])
    print(f"\n===== 공식 TEST 위험 탐지 (RUL <= {FinalDangerThreshold}) =====")
    print(OfficialRiskDf.round(3).to_string(index=False))

    return OfficialRiskDf

# 정비 우선순위 (예측 RUL이 작을수록 고장이 임박했다고 보고 먼저 점검)
def RankMaintenance(ResultDf, FinalDangerThreshold):
    MaintenanceDf = ResultDf.copy()
    MaintenanceDf["Status"] = np.where(MaintenanceDf["PredRUL"] <= FinalDangerThreshold, "점검 대상", "정상")
    MaintenanceDf = MaintenanceDf.sort_values("PredRUL", ascending=True).reset_index(drop=True)
    MaintenanceDf["Maintenance_Rank"] = MaintenanceDf.index + 1
    MaintenanceDf = MaintenanceDf[["Maintenance_Rank", "unit_number", "time_in_cycles", "PredRUL",
                                   "Status", "ActualRUL", "Error"]]

    print("\n===== 정비 우선순위 상위 20대 =====")
    print(MaintenanceDf.head(20).round(2).to_string(index=False))

    DangerEngines = MaintenanceDf[MaintenanceDf["Status"] == "점검 대상"]
    print("\n===== 현재 점검 대상 엔진 =====")
    print(DangerEngines.round(2).to_string(index=False))
    print("\n점검 대상 엔진 수:", len(DangerEngines))

    return MaintenanceDf

# 자동 실행 함수
def Main():
    Df = LoadData(FeaturePath)

    Df = CreateRUL(Df)

    KeySensors = GetKeySensors(Df)

    FeatureCols = GetFeatureColumns(Df)

    TrainDf, ValDf = SplitByEngine(Df)

    RfModel, RfPred = TrainModels(TrainDf, ValDf, FeatureCols)

    # 최종 위험 기준은 여기(Validation)에서 정하고, 공식 TEST를 보기 전에 고정한다
    ValidationRiskDf, FinalDangerThreshold = SelectRiskThreshold(ValDf["RUL"], RfPred)

    CheckZScore(TrainDf, ValDf, KeySensors, FinalDangerThreshold)

    PlotPrediction(ValDf["RUL"], RfPred)

    CheckFeatureImportance(RfModel, FeatureCols)

    RunGroupKFold(Df, FeatureCols)

    # Validation으로 모델·기준 선택이 끝난 뒤 TRAIN 전체(100대)로 최종 모델 학습
    FinalModel = TrainRandomForest(Df[FeatureCols], Df["RUL"])
    print("\n===== 최종 모델 학습 완료 =====")
    print("TRAIN 전체 엔진 수:", Df["unit_number"].nunique())

    OfficialTestDf, XOfficialTest = LoadOfficialTest(TestPath, KeySensors, FeatureCols)

    ResultDf = PredictOfficialTest(FinalModel, OfficialTestDf, XOfficialTest, RulPath)

    EvaluateOfficialTest(ResultDf, FinalDangerThreshold)

    RankMaintenance(ResultDf, FinalDangerThreshold)


if __name__ == "__main__":
    Main()
