import os
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

BaseDir = os.path.dirname(os.path.abspath(__file__))

Df = pd.read_csv(os.path.join(BaseDir, "model_predictions.csv"))
Df["timestamp"] = pd.to_datetime(Df["timestamp"])

YTrue = Df["label"]
YPred = Df["alert"]

Cm = confusion_matrix(YTrue, YPred, labels=[0, 1])
CmDf = pd.DataFrame(
    Cm,
    index=["실제_정상", "실제_위험(고장24시간전)"],
    columns=["예측_정상", "예측_경보"],
)
print("[혼동행렬]")
print(CmDf)
CmDf.to_csv(os.path.join(BaseDir, "confusion_matrix.csv"), encoding="utf-8-sig")

EvalTable = pd.DataFrame([{
    "Precision": round(precision_score(YTrue, YPred, zero_division=0), 4),
    "Recall": round(recall_score(YTrue, YPred, zero_division=0), 4),
    "F1": round(f1_score(YTrue, YPred, zero_division=0), 4),
    "전체 경보 수": int(Df["alert"].sum()),
}])
print("\n[평가표]")
print(EvalTable.to_string(index=False))
EvalTable.to_csv(os.path.join(BaseDir, "model_evaluation_table.csv"), index=False, encoding="utf-8-sig")

BrokenTimes = Df.loc[Df["machine_status"] == "BROKEN", "timestamp"]

print("\n[고장 이벤트별 감지 여부 - 24시간 전부터]")
EventRows = []
for T in BrokenTimes:
    Start = T - pd.Timedelta(hours=24)
    Window = Df[(Df["timestamp"] >= Start) & (Df["timestamp"] <= T)]
    Hit = Window["alert"].sum()
    print(f" - {T} : 경보 {Hit}번 ({'감지' if Hit > 0 else '미탐'})")
    EventRows.append({"고장시점": T, "경보횟수": int(Hit), "감지여부": "감지" if Hit > 0 else "미탐"})

pd.DataFrame(EventRows).to_csv(os.path.join(BaseDir, "event_detection_summary.csv"), index=False, encoding="utf-8-sig")

FalsePositive = Df[(Df["label"] == 0) & (Df["alert"] == 1)]
print(f"\n오탐(실제 정상인데 경보) 개수: {len(FalsePositive)}")
print("오탐 시각 샘플(상위 5건):")
print(FalsePositive[["timestamp"]].head(5).to_string(index=False))

print("\n완료. 결과 파일은 이 스크립트와 같은 폴더에 저장됨.")
