import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

# =========================================================
# 1. 데이터
# =========================================================

df = pd.read_csv("sensor.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# 2. 사용할 센서
# =========================================================

candidate_sensors = [
    "sensor_04",
    "sensor_02",
    "sensor_13",
    "sensor_33",
    "sensor_32",
    "sensor_28",
]

# 실제 데이터에 존재하는 센서만 사용
sensors = [s for s in candidate_sensors if s in df.columns]

print("사용 센서:", sensors)


# =========================================================
# 3. 시계열 특성
# =========================================================

for sensor in sensors:

    df[f"{sensor}_lag10"] = df[sensor].shift(10)
    df[f"{sensor}_lag30"] = df[sensor].shift(30)

    df[f"{sensor}_mean30"] = df[sensor].rolling(30).mean()

    df[f"{sensor}_std30"] = df[sensor].rolling(30).std()

    df[f"{sensor}_diff10"] = df[sensor] - df[sensor].shift(10)


feature_cols = []

for sensor in sensors:

    feature_cols += [
        f"{sensor}_lag10",
        f"{sensor}_lag30",
        f"{sensor}_mean30",
        f"{sensor}_std30",
        f"{sensor}_diff10",
    ]


# =========================================================
# 4. 고장 발생시간
# =========================================================

failure_times = (
    df.loc[df["machine_status"] == "BROKEN", "timestamp"].sort_values().tolist()
)

print("고장 횟수:", len(failure_times))


# =========================================================
# 5. 시간순 Train / Test 기준
# =========================================================

split_time = failure_times[-2] - pd.Timedelta(hours=12)


# =========================================================
# 6. 몇 시간 전까지 잡을 수 있는지
# =========================================================

rows = []

for hour in [1, 3, 6, 12]:

    df["failure_soon"] = 0

    for t in failure_times:

        mask = (df["timestamp"] >= t - pd.Timedelta(hours=hour)) & (df["timestamp"] < t)

        df.loc[mask, "failure_soon"] = 1

    data = df.dropna(subset=feature_cols).copy()

    # BROKEN 이후 복구구간은 제외
    data = data[data["machine_status"] == "NORMAL"]

    train = data[data["timestamp"] < split_time]

    test = data[data["timestamp"] >= split_time]

    X_train = train[feature_cols]
    y_train = train["failure_soon"]

    X_test = test[feature_cols]
    y_test = test["failure_soon"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        max_depth=12,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    rows.append(
        [
            hour,
            recall_score(y_test, pred, zero_division=0),
            precision_score(y_test, pred, zero_division=0),
            f1_score(y_test, pred, zero_division=0),
            fp,
            fn,
            tp,
        ]
    )


# =========================================================
# 7. 결과
# =========================================================

result = pd.DataFrame(
    rows, columns=["고장 몇 시간 전", "재현율", "정밀도", "F1", "FP", "FN", "TP"]
)

print(result)
