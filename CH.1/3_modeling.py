import os
import pandas as pd
from sklearn.ensemble import IsolationForest

BaseDir = os.path.dirname(os.path.abspath(__file__))

Df = pd.read_csv(os.path.join(BaseDir, "cleaned_data.csv"))
Df["timestamp"] = pd.to_datetime(Df["timestamp"])
SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_")]

BrokenTimes = Df.loc[Df["machine_status"] == "BROKEN", "timestamp"]

# 고장 24시간 전까지를 위험 구간으로 잡음
Df["label"] = 0
for T in BrokenTimes:
    Start = T - pd.Timedelta(hours=24)
    Df.loc[(Df["timestamp"] >= Start) & (Df["timestamp"] <= T), "label"] = 1

# RECOVERING은 이미 고장난 뒤라 평가에서 뺌
EvalDf = Df[Df["machine_status"] != "RECOVERING"].copy()

Model = IsolationForest(random_state=42)
EvalDf["predict"] = Model.fit_predict(EvalDf[SensorCols])
EvalDf["alert"] = (EvalDf["predict"] == -1).astype(int)

print("전체 평가 대상 행 수:", len(EvalDf))
print("경보(이상 신호) 개수:", EvalDf["alert"].sum())

Result = EvalDf[["timestamp", "machine_status", "label", "alert"]]
Result.to_csv(os.path.join(BaseDir, "model_predictions.csv"), index=False)
print("\n저장 완료:", os.path.join(BaseDir, "model_predictions.csv"))
