import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# =========================================================
# 0. 설정
# =========================================================

DATA_PATH = "Data/Preprocessing_Data_File/Test2_machine_data.csv"
REPORT_DIR = "Data/Reports"
PRE_FAILURE_HOURS = 6          # 고장 직전 몇 시간을 "사전징후 구간"으로 볼지
CONTAMINATION = 0.01           # IsolationForest 학습 시 정상 데이터 중 이상치로 볼 비율

os.makedirs(REPORT_DIR, exist_ok=True)

# =========================================================
# 1. 데이터 로드 및 결측치 상태 확인
# =========================================================

Df = pd.read_csv(DATA_PATH)
Df["timestamp"] = pd.to_datetime(Df["timestamp"], errors="coerce")
Df = Df.sort_values("timestamp").reset_index(drop=True)

SensorCols = [c for c in Df.columns if c.startswith("sensor_")]

print("=" * 60)
print("1. 결측치 상태 확인")
print("=" * 60)
print("데이터 크기:", Df.shape)
remaining_na = Df[SensorCols].isna().sum()
remaining_na = remaining_na[remaining_na > 0]
if len(remaining_na) == 0:
    print("결측치 없음 (전처리 완료 상태)")
else:
    print("아직 남은 결측치:")
    print(remaining_na)
    # 남은 결측치는 baseline 모델 학습을 위해 임시로 앞/뒤 값으로 채움
    Df[SensorCols] = Df[SensorCols].ffill().bfill()

Df["machine_status"] = Df["machine_status"].astype("string").str.strip().str.upper()

# =========================================================
# 2. BROKEN 이벤트 및 사전징후 구간 라벨링
# =========================================================

broken_times = Df.loc[Df["machine_status"] == "BROKEN", "timestamp"].tolist()

print("\n" + "=" * 60)
print("2. BROKEN 이벤트")
print("=" * 60)
print(f"BROKEN 이벤트 수: {len(broken_times)}")
for t in broken_times:
    print(" -", t)

Df["pre_failure"] = False
for t in broken_times:
    window_start = t - pd.Timedelta(hours=PRE_FAILURE_HOURS)
    mask = (Df["timestamp"] >= window_start) & (Df["timestamp"] < t)
    Df.loc[mask, "pre_failure"] = True

# 학습에 쓸 "순정상" 구간: NORMAL이면서 사전징후 구간이 아닌 곳
is_train_normal = (Df["machine_status"] == "NORMAL") & (~Df["pre_failure"])

print(f"\n순정상(학습용) 구간 수: {is_train_normal.sum()} / 전체 {len(Df)}")
print(f"사전징후 구간 수(BROKEN 전 {PRE_FAILURE_HOURS}시간): {Df['pre_failure'].sum()}")

# =========================================================
# 3. 이상치(IQR) - 라벨(사전징후) 관계 확인
# =========================================================

print("\n" + "=" * 60)
print(f"3. 센서별 IQR 이상치 비율: 사전징후 구간 vs 순정상 구간")
print("=" * 60)

rows = []
for col in SensorCols:
    q1 = Df.loc[is_train_normal, col].quantile(0.25)
    q3 = Df.loc[is_train_normal, col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (Df[col] < lower) | (Df[col] > upper)

    normal_rate = outlier_mask[is_train_normal].mean() * 100
    prefail_rate = outlier_mask[Df["pre_failure"]].mean() * 100 if Df["pre_failure"].sum() > 0 else np.nan

    rows.append({
        "sensor": col,
        "outlier_rate_normal(%)": round(normal_rate, 2),
        "outlier_rate_prefailure(%)": round(prefail_rate, 2) if not np.isnan(prefail_rate) else None,
        "gap(prefail-normal)": round(prefail_rate - normal_rate, 2) if not np.isnan(prefail_rate) else None,
    })

CompareDf = pd.DataFrame(rows).sort_values("gap(prefail-normal)", ascending=False)
print(CompareDf.to_string(index=False))

CompareDf.to_csv("Data/Reports/Outlier_vs_PreFailure_Comparison.csv", index=False)
print("\n저장: Data/Reports/Outlier_vs_PreFailure_Comparison.csv")

# =========================================================
# 4. Baseline 이상탐지 모델 (IsolationForest)
# =========================================================

print("\n" + "=" * 60)
print("4. IsolationForest baseline 학습 및 이벤트 단위 평가")
print("=" * 60)

model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
)
model.fit(Df.loc[is_train_normal, SensorCols])

# score_samples: 값이 낮을수록 더 이상치에 가까움
Df["anomaly_score"] = -model.score_samples(Df[SensorCols])

# 학습 데이터(순정상) 기준 상위 1% 임계값을 경보 기준으로 사용
threshold = Df.loc[is_train_normal, "anomaly_score"].quantile(1 - CONTAMINATION)
Df["alert"] = Df["anomaly_score"] > threshold

print(f"경보 임계값(threshold): {threshold:.4f}")

# 이벤트 단위 평가: 각 BROKEN 사건마다 사전징후 구간에서 경보가 떴는지 확인
print("\n[이벤트별 사전 감지 여부]")
detected = 0
for t in broken_times:
    window_start = t - pd.Timedelta(hours=PRE_FAILURE_HOURS)
    window_mask = (Df["timestamp"] >= window_start) & (Df["timestamp"] < t)
    hit = Df.loc[window_mask, "alert"].any()
    if hit:
        detected += 1
        first_alert_time = Df.loc[window_mask & Df["alert"], "timestamp"].min()
        lead_minutes = (t - first_alert_time).total_seconds() / 60
        print(f" - BROKEN {t} : 감지 O (첫 경보 {first_alert_time}, {lead_minutes:.0f}분 전)")
    else:
        print(f" - BROKEN {t} : 감지 X (사전 {PRE_FAILURE_HOURS}시간 내 경보 없음)")

event_recall = detected / len(broken_times) if broken_times else float("nan")
print(f"\n이벤트 Recall: {detected}/{len(broken_times)} = {event_recall:.1%}")

# 오탐률: 순정상 구간에서 하루 평균 몇 번 경보가 뜨는지
normal_alert_count = Df.loc[is_train_normal, "alert"].sum()
normal_days = (Df.loc[is_train_normal, "timestamp"].max() - Df.loc[is_train_normal, "timestamp"].min()).days
false_alarms_per_day = normal_alert_count / max(normal_days, 1)
print(f"순정상 구간 오탐 횟수: {normal_alert_count}건 / 총 {normal_days}일 -> 하루 평균 {false_alarms_per_day:.1f}건")

# =========================================================
# 5. 결과 저장
# =========================================================

Df[["timestamp", "machine_status", "pre_failure", "anomaly_score", "alert"]].to_csv(
    "Data/Reports/AnomalyScore_Result.csv", index=False
)
print("\n저장: Data/Reports/AnomalyScore_Result.csv")
