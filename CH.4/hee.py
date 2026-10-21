# 1. 라이브러리·설정·공통 함수
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline

RUN_SURVIVAL = True
SHOW_PLOTS = False
# scikit-survival이 없으면 생존분석(9·10단계)만 건너뛰고 회귀모델은 그대로 실행 (Main.py all 이 여기서 멈추지 않도록)
if RUN_SURVIVAL:
    try:
        import sksurv  # noqa: F401
    except ImportError:
        print("scikit-survival이 없어 생존분석(9·10단계)을 건너뜁니다. 설치: pip install scikit-survival")
        RUN_SURVIVAL = False
folder = Path(__file__).resolve().parent / "Data" / "Processed" / "Scenarios"
index = pd.read_csv(folder / "index.csv")
index = index[index["split"].isin(["train", "validation"])]
keys = ["SCENARIO_ID", "EQP_CD", "T_SEC"]
equipment = "P1-CC1-INS01"
basic_sensors = ["WD-LOD", "MLD-LVL", "CONTROL_DELAY"]
all_sensors = [
    "LOAD", "WD-SPD", "WD-LOD", "OSC-FRQ",
    "OSC-STK", "MLD-LVL", "CONTROL_DELAY",
]
basic_columns = [
    f"{sensor}_{kind}"
    for sensor in basic_sensors
    for kind in ["현재", "1시간평균"]
]
current_columns = [f"{sensor}_현재" for sensor in all_sensors]
temporal_columns = [
    f"{sensor}_{kind}"
    for sensor in all_sensors
    for kind in ["현재", "평균", "표준편차", "변화량", "기울기"]
] + ["이력시간"]
additional_columns = [
    f"{sensor}_초기대비변화" for sensor in all_sensors
] + [
    f"{sensor}_{kind}"
    for sensor in ["WD-LOD", "WD-SPD"]
    for kind in ["부하비현재", "부하비12시간평균"]
]
augmented_columns = temporal_columns + additional_columns

def show(title, table):
    print(f"\n[{title}]")
    print(table.round(3).to_string(index=False))

def make_model(depth=6, leaf=5):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestRegressor(
            n_estimators=200, max_depth=depth,
            min_samples_leaf=leaf, random_state=42, n_jobs=2,
        ),
    )

def fit_model(data, columns, leaf=5, weight=None):
    model = make_model(leaf=leaf)
    options = {}
    if weight is not None:
        options["randomforestregressor__sample_weight"] = np.where(
            data["RUL정답"].to_numpy() <= 6, weight, 1.0
        )
    model.fit(data[columns], data["RUL정답"], **options)
    return model

def evaluate(name, truth, prediction):
    truth = np.asarray(truth)
    error = np.asarray(prediction) - truth
    near = truth <= 6
    return {
        "모델": name,
        "전체MAE": np.abs(error).mean(),
        "6시간이하MAE": np.abs(error[near]).mean(),
        "6시간이하평균예측오차": error[near].mean(),
    }

def prediction_table(data, prediction):
    result = data[["scenario", "HOUR", "RUL정답"]].copy()
    result["예측RUL"] = prediction
    result["예측오차"] = result["예측RUL"] - result["RUL정답"]
    result["절대오차"] = result["예측오차"].abs()
    return result

def make_features(sensors, target, scenario, split):
    t = target.T_SEC
    past = sensors[sensors["T_SEC"] <= t]
    current_rows = past[past["T_SEC"] == t]
    if len(current_rows) != 1 or current_rows.iloc[0]["RUNNING"] != 1:
        raise ValueError(f"{scenario}: {t}초 센서·운전 상태 확인 필요")
    current = current_rows.iloc[0]
    stopped = past.loc[past["RUNNING"] == 0, "T_SEC"]
    last_stop = stopped.max() if not stopped.empty else -1
    running = past[
        (past["T_SEC"] > last_stop) & (past["RUNNING"] == 1)
    ]
    short = running[running["T_SEC"] >= t - 3600]
    recent = running[running["T_SEC"] >= t - 12 * 3600]
    initial_end = min(running.iloc[0]["T_SEC"] + 1800, t)
    initial = running[running["T_SEC"] <= initial_end]
    metadata = {
        "scenario": scenario, "split": split, "HOUR": t / 3600,
        "RUL_EVENT": target.GT_RUL_EVENT,
        "RUL정답": target.GT_RUL_HOURS,
        "FOLLOWUP_HOURS": target.GT_FOLLOWUP_HOURS,
    }
    basic, temporal = metadata.copy(), metadata.copy()
    additional = {"scenario": scenario, "HOUR": t / 3600}
    for sensor in basic_sensors:
        basic[f"{sensor}_현재"] = current[sensor]
        basic[f"{sensor}_1시간평균"] = short[sensor].mean()
    for sensor in all_sensors:
        values = recent[["T_SEC", sensor]].dropna()
        temporal[f"{sensor}_현재"] = current[sensor]
        temporal[f"{sensor}_평균"] = values[sensor].mean()
        temporal[f"{sensor}_표준편차"] = values[sensor].std(ddof=0)
        if len(values) >= 2:
            hours = values["T_SEC"].to_numpy() / 3600
            signal = values[sensor].to_numpy()
            centered = hours - hours.mean()
            temporal[f"{sensor}_변화량"] = signal[-1] - signal[0]
            temporal[f"{sensor}_기울기"] = (
                np.sum(centered * (signal - signal.mean()))
                / np.sum(centered ** 2)
            )
        else:
            temporal[f"{sensor}_변화량"] = np.nan
            temporal[f"{sensor}_기울기"] = np.nan
        additional[f"{sensor}_초기대비변화"] = (
            current[sensor] - initial[sensor].mean()
        )
    temporal["이력시간"] = (
        recent["T_SEC"].iloc[-1] - recent["T_SEC"].iloc[0]
    ) / 3600
    load = recent["LOAD"].where(recent["LOAD"] > 0)
    for sensor in ["WD-LOD", "WD-SPD"]:
        additional[f"{sensor}_부하비현재"] = (
            current[sensor] / current["LOAD"]
            if current["LOAD"] > 0 else np.nan
        )
        additional[f"{sensor}_부하비12시간평균"] = (
            recent[sensor] / load
        ).mean()
    return basic, temporal, additional

def out_of_fold(data, columns, weight):
    prediction = np.full(len(data), np.nan)
    cv = GroupKFold(n_splits=3)
    for fit_positions, check_positions in cv.split(
        data[columns], data["RUL정답"], groups=data["scenario"]
    ):
        model = fit_model(
            data.iloc[fit_positions], columns, leaf=2, weight=weight
        )
        prediction[check_positions] = model.predict(
            data.iloc[check_positions][columns]
        )
    return prediction

def attach_diagnosis(data):
    return data.merge(
        diagnosis_data,
        on=["scenario", "HOUR"],
        how="left",
        validate="one_to_one",
    )

def diagnosis_summary(data, column):
    result = data.copy()
    result["임박表本"] = result["RUL정답"].le(6).astype(int)
    result["임박오차"] = result["절대오차"].where(result["RUL정답"] <= 6)
    return result.groupby(column).agg(
        표본수=("RUL정답", "size"),
        시나리오수=("scenario", "nunique"),
        MAE=("절대오차", "mean"),
        평균예측오차=("예측오차", "mean"),
        임박표본수=("임박表本", "sum"),
        임박MAE=("임박오차", "mean"),
    ).sort_values("MAE", ascending=False)

# 2. 전체 시나리오 연결 점검 및 공통 데이터 준비
summaries, basic_rows, temporal_rows = [], [], []
additional_rows, diagnosis_rows = [], []
for number, row in enumerate(index.itertuples(index=False), start=1):
    scenario = row.scenario
    path = folder / scenario
    sensors = pd.read_csv(
        path / "C.csv", usecols=keys + ["RUNNING"] + all_sensors
    ).sort_values("T_SEC")
    targets = pd.read_csv(path / "targets.csv")
    targets = targets[targets["EQP_CD"] == equipment]
    merged = targets.merge(
        sensors[keys + ["RUNNING"]],
        on=keys, how="left", validate="one_to_one", indicator=True,
    )
    event = merged["GT_AT_RISK"].eq(1) & merged["GT_RUL_EVENT"].eq(1)
    censored = merged["GT_AT_RISK"].eq(1) & merged["GT_RUL_EVENT"].eq(0)
    unmatched = merged["T_SEC"].lt(259200) & merged["_merge"].ne("both")
    summaries.append({
        "scenario": scenario, "split": row.split,
        "정답 행": len(merged), "고장 사건": int(event.sum()),
        "검열": int(censored.sum()),
        "평가 제외": int(merged["GT_AT_RISK"].eq(0).sum()),
        "연결 실패": int(unmatched.sum()),
    })
    if unmatched.any():
        raise ValueError(f"{scenario}: 센서 연결 실패")
    diagnosis = pd.read_csv(path / "diagnostics.csv")
    diagnosis = diagnosis[diagnosis["EQP_CD"] == equipment]
    for target in targets[targets["GT_AT_RISK"] == 1].itertuples(index=False):
        basic, temporal, additional = make_features(
            sensors, target, scenario, row.split
        )
        basic_rows.append(basic)
        temporal_rows.append(temporal)
        additional_rows.append(additional)
        matched = diagnosis[
            (diagnosis["T_START_SEC"] <= target.T_SEC)
            & (target.T_SEC < diagnosis["T_END_SEC"])
        ]
        if len(matched) != 1:
            raise ValueError(f"{scenario}: 진단 구간 연결 확인 필요")
        truth = matched.iloc[0]
        diagnosis_rows.append({
            "scenario": scenario, "HOUR": target.T_SEC / 3600,
            "진단상태": truth["GT_HEALTH_STATE"],
            "원인부품": truth["GT_CAUSE_PART"],
            "고장모드": truth["GT_FAULT_MODE"],
        })
    if number % 50 == 0:
        print(f"{number}개 시나리오 점검·특징 생성 완료")
summary = pd.DataFrame(summaries)
basic_data = pd.DataFrame(basic_rows)
expanded_data = pd.DataFrame(temporal_rows)
additional_data = pd.DataFrame(additional_rows)
diagnosis_data = pd.DataFrame(diagnosis_rows)
diagnosis_data[["진단상태", "원인부품", "고장모드"]] = (
    diagnosis_data[["진단상태", "원인부품", "고장모드"]].fillna("미지정")
)
count_columns = ["정답 행", "고장 사건", "검열", "평가 제외", "연결 실패"]
print("\n[2단계: 학습·검증별 연결 점검]")
print(summary.groupby("split")[count_columns].sum().to_string())

# 3. S0002 센서 특징 확인
show(
    "3단계: S0002 센서 특징",
    basic_data.loc[
        basic_data["scenario"] == "S0002",
        ["HOUR", "RUL정답"] + basic_columns,
    ],
)

# 4. 전체 센서 특징 확인
print(f"\n[4단계: 센서 특징 생성]\n전체 행 수: {len(basic_data)}")
print(
    basic_data.groupby(["split", "RUL_EVENT"])
    .size().unstack(fill_value=0).to_string()
)
print("\n[센서 특징 결측 개수]")
print(basic_data[basic_columns].isna().sum().to_string())

# 5. 회귀 학습·검증 데이터 준비
basic_events = basic_data[
    basic_data["RUL_EVENT"].eq(1) & basic_data["RUL정답"].notna()
].copy()
train = basic_events[basic_events["split"] == "train"].copy()
valid = basic_events[basic_events["split"] == "validation"].copy()
expanded_events = expanded_data[
    expanded_data["RUL_EVENT"].eq(1) & expanded_data["RUL정답"].notna()
].copy()
expanded_train = expanded_events[expanded_events["split"] == "train"].copy()
expanded_valid = expanded_events[expanded_events["split"] == "validation"].copy()
pd.testing.assert_frame_equal(
    valid[["scenario", "HOUR"]].reset_index(drop=True),
    expanded_valid[["scenario", "HOUR"]].reset_index(drop=True),
)
y_valid = expanded_valid["RUL정답"]
baseline_rul = train["RUL정답"].median()
predictions = {}

# 6. 센서 3종 회귀모델 학습
basic_model = fit_model(train, basic_columns)
predictions["6단계: 센서 3종"] = basic_model.predict(valid[basic_columns])
basic_comparison = prediction_table(valid, predictions["6단계: 센서 3종"])
show(
    "6단계: 첫 회귀모델 평가",
    pd.DataFrame([
        evaluate("단순 기준", y_valid, np.full(len(valid), baseline_rul)),
        evaluate("센서 3종", y_valid, predictions["6단계: 센서 3종"]),
    ]),
)

# 7. 남은 시간 구간별 오차 확인
basic_comparison["남은시간구간"] = pd.cut(
    basic_comparison["RUL정답"],
    bins=[0, 6, 12, 24, float("inf")],
    labels=["6시간 이하", "6~12시간", "12~24시간", "24시간 초과"],
    include_lowest=True,
)
print("\n[7단계: 남은 시간 구간별 오차]")
print(
    basic_comparison.groupby("남은시간구간", observed=True).agg(
        표본수=("RUL정답", "size"),
        평균절대오차=("절대오차", "mean"),
        평균예측오차=("예측오차", "mean"),
    ).round(2).to_string()
)

# 8. 센서 7종·12시간 변화 모델 비교
for name, columns in [
    ("8단계: 센서 7종 현재값", current_columns),
    ("8단계: 센서 7종+12시간 변화", temporal_columns),
]:
    candidate = fit_model(expanded_train, columns)
    predictions[name] = candidate.predict(expanded_valid[columns])
show(
    "8단계: 센서 특징별 모델 비교",
    pd.DataFrame([
        evaluate(name, y_valid, prediction)
        for name, prediction in predictions.items()
    ]),
)

# 9. 검열 데이터를 포함한 생존분석
if RUN_SURVIVAL:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored, brier_score
    from sksurv.nonparametric import kaplan_meier_estimator
    survival_train = expanded_data[expanded_data["split"] == "train"].copy()
    survival_valid = expanded_data[expanded_data["split"] == "validation"].copy()
    y_survival_train = Surv.from_arrays(
        survival_train["RUL_EVENT"].eq(1).to_numpy(),
        survival_train["FOLLOWUP_HOURS"].to_numpy(),
    )
    y_survival_valid = Surv.from_arrays(
        survival_valid["RUL_EVENT"].eq(1).to_numpy(),
        survival_valid["FOLLOWUP_HOURS"].to_numpy(),
    )
    survival_imputer = SimpleImputer(strategy="median", add_indicator=True)
    Xs_train = survival_imputer.fit_transform(survival_train[temporal_columns])
    Xs_valid = survival_imputer.transform(survival_valid[temporal_columns])
    survival_model = RandomSurvivalForest(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=2,
    )
    survival_model.fit(Xs_train, y_survival_train)
    risk_scores = survival_model.predict(Xs_valid)
    c_index = concordance_index_censored(
        survival_valid["RUL_EVENT"].eq(1).to_numpy(),
        survival_valid["FOLLOWUP_HOURS"].to_numpy(),
        risk_scores,
    )[0]
    survival_probabilities = survival_model.predict_survival_function(
        Xs_valid, return_array=True
    )
    times = survival_model.unique_times_
    if times[-1] < 72:
        raise ValueError("학습 관측 범위가 72시간보다 짧습니다.")
    inside = times < 72
    boundaries = np.r_[0.0, times[inside], 72.0]
    heights = np.column_stack([
        np.ones(len(survival_valid)), survival_probabilities[:, inside]
    ])
    restricted_mean = heights @ np.diff(boundaries)
    survival_comparison = survival_valid[
        ["scenario", "HOUR", "RUL_EVENT", "RUL정답", "FOLLOWUP_HOURS"]
    ].copy()
    survival_comparison["72시간제한평균"] = restricted_mean
    survival_events = survival_comparison[
        survival_comparison["RUL_EVENT"] == 1
    ]
    pd.testing.assert_frame_equal(
        survival_events[["scenario", "HOUR"]].reset_index(drop=True),
        expanded_valid[["scenario", "HOUR"]].reset_index(drop=True),
    )
    print(
        f"\n[9단계: 생존분석]\n학습 {len(survival_train)}행 / "
        f"검증 {len(survival_valid)}행 / C-index {c_index:.3f}"
    )
    show(
        "사건 표본에서의 보조 비교",
        pd.DataFrame([
            evaluate(
                "8단계 회귀", y_valid,
                predictions["8단계: 센서 7종+12시간 변화"],
            ),
            evaluate(
                "생존모델 72시간 제한평균",
                survival_events["RUL정답"],
                survival_events["72시간제한평균"],
            ),
        ]),
    )

# 10. 검열을 고려한 생존확률 평가
if RUN_SURVIVAL:
    evaluation_times = np.array([6.0, 24.0])
    positions = np.searchsorted(times, evaluation_times, side="right") - 1
    model_probabilities = np.ones((len(survival_valid), 2))
    for j, position in enumerate(positions):
        if position >= 0:
            model_probabilities[:, j] = survival_probabilities[:, position]
    km_times, km_probabilities = kaplan_meier_estimator(
        survival_train["RUL_EVENT"].eq(1).to_numpy(),
        survival_train["FOLLOWUP_HOURS"].to_numpy(),
    )
    baseline_probabilities = []
    for hour in evaluation_times:
        position = np.searchsorted(km_times, hour, side="right") - 1
        baseline_probabilities.append(
            km_probabilities[position] if position >= 0 else 1.0
        )
    baseline_probabilities = np.tile(
        baseline_probabilities, (len(survival_valid), 1)
    )
    _, model_scores = brier_score(
        y_survival_train, y_survival_valid,
        model_probabilities, evaluation_times,
    )
    _, baseline_scores = brier_score(
        y_survival_train, y_survival_valid,
        baseline_probabilities, evaluation_times,
    )
    probability_report = pd.DataFrame({
        "평가시점": ["6시간 후", "24시간 후"],
        "단순기준_Brier": baseline_scores,
        "생존모델_Brier": model_scores,
        "개선량": baseline_scores - model_scores,
    })
    print("\n[10단계: 검열 고려 생존확률 평가]")
    print(probability_report.round(4).to_string(index=False))

# 11. 회귀모델 예측 그래프
if SHOW_PLOTS:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    graph_data = prediction_table(
        expanded_valid, predictions["8단계: 센서 7종+12시간 변화"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    limit = max(graph_data["RUL정답"].max(), graph_data["예측RUL"].max()) + 3
    axes[0].scatter(graph_data["RUL정답"], graph_data["예측RUL"], alpha=0.6)
    axes[0].plot([0, limit], [0, limit], "r--", label="정답과 예측 일치")
    axes[0].set(
        xlabel="RUL 정답 (시간)", ylabel="예측 RUL (시간)",
        title="전체 검증 표본의 정답·예측",
    )
    scenario = graph_data["scenario"].iloc[0]
    sample = graph_data[graph_data["scenario"] == scenario].sort_values("HOUR")
    axes[1].plot(sample["HOUR"], sample["RUL정답"], "o-", label="RUL 정답")
    axes[1].plot(sample["HOUR"], sample["예측RUL"], "s--", label="예측 RUL")
    axes[1].set(
        xlabel="시나리오 경과시간", ylabel="남은 시간",
        title=f"{scenario}: 시간별 정답·예측",
    )
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 12. 학습 시나리오 내부 모델 설정 비교
search = GridSearchCV(
    estimator=make_model(),
    param_grid={
        "randomforestregressor__max_depth": [6, None],
        "randomforestregressor__min_samples_leaf": [2, 5, 10],
    },
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=3), n_jobs=1,
    refit=True, error_score="raise",
)
search.fit(
    expanded_train[temporal_columns],
    expanded_train["RUL정답"],
    groups=expanded_train["scenario"],
)
tuned_model = search.best_estimator_
predictions["12단계: 모델 설정 조정"] = tuned_model.predict(
    expanded_valid[temporal_columns]
)
cv_results = pd.DataFrame(search.cv_results_)
show(
    "12단계: 학습 내부 설정 비교",
    pd.DataFrame({
        "최대깊이": cv_results["param_randomforestregressor__max_depth"].astype(str),
        "최소잎표본수": cv_results["param_randomforestregressor__min_samples_leaf"],
        "내부평가MAE": -cv_results["mean_test_score"],
    }).sort_values("내부평가MAE"),
)
print(f"\n선택된 설정: {search.best_params_}")

# 13. 검증 시나리오별 오차 확인
tuned_comparison = prediction_table(
    expanded_valid, predictions["12단계: 모델 설정 조정"]
)
print("\n[13단계: 검증 시나리오별 오차]")
print(
    tuned_comparison.groupby("scenario").agg(
        표본수=("RUL정답", "size"),
        MAE=("절대오차", "mean"),
        평균예측오차=("예측오차", "mean"),
        최대절대오차=("절대오차", "max"),
    ).sort_values("MAE", ascending=False).round(2).to_string()
)
show(
    "고장까지 6시간 이하인 검증 표본",
    tuned_comparison[tuned_comparison["RUL정답"] <= 6],
)

# 14. 고장 임박 표본 가중치 비교
weight_results, weight_predictions = [], {}
for multiplier in [1, 2, 4]:
    prediction = out_of_fold(expanded_train, temporal_columns, multiplier)
    weight_predictions[multiplier] = prediction
    result = evaluate(multiplier, expanded_train["RUL정답"], prediction)
    weight_results.append({
        "가중치": multiplier,
        "내부전체MAE": result["전체MAE"],
        "내부6시간이하MAE": result["6시간이하MAE"],
        "선택점수": (result["전체MAE"] + result["6시간이하MAE"]) / 2,
    })
weight_report = pd.DataFrame(weight_results)
best_weight = int(
    weight_report.sort_values(["선택점수", "가중치"]).iloc[0]["가중치"]
)
weighted_model = fit_model(
    expanded_train, temporal_columns, leaf=2, weight=best_weight
)
predictions["14단계: 고장 임박 가중치"] = weighted_model.predict(
    expanded_valid[temporal_columns]
)
weighted_comparison = prediction_table(
    expanded_valid, predictions["14단계: 고장 임박 가중치"]
)
show("14단계: 학습 내부 가중치 비교", weight_report)
print(f"\n선택된 가중치: {best_weight}배")

# 15. 진단 상태·부품·고장 모드별 오차
diagnosis_errors = attach_diagnosis(weighted_comparison)
for column in ["진단상태", "원인부품", "고장모드"]:
    print(f"\n[15단계: {column}별 오차]")
    print(diagnosis_summary(diagnosis_errors, column).round(2).to_string())

# 16. 정상·고장 임박 표본의 센서·정비 확인
suspects = diagnosis_errors[
    diagnosis_errors["진단상태"].eq("normal")
    & diagnosis_errors["RUL정답"].le(6)
]
show(
    "16단계: 정상·고장 임박 표본",
    suspects[["scenario", "HOUR", "RUL정답", "예측RUL", "절대오차"]],
)
for _, case in suspects.iterrows():
    scenario, t = case["scenario"], case["HOUR"] * 3600
    sensors = pd.read_csv(
        folder / scenario / "C.csv",
        usecols=["T_SEC", "RUNNING"] + all_sensors,
    ).sort_values("T_SEC")
    check_times = [t - hour * 3600 for hour in [6, 3, 1, 0] if t >= hour * 3600]
    view = sensors[sensors["T_SEC"].isin(check_times)].copy()
    view.insert(0, "경과시간", view["T_SEC"] / 3600)
    show(f"{scenario}: 현재·과거 센서", view.drop(columns="T_SEC"))
    maintenance = pd.read_csv(folder / scenario / "history" / "maintenance.csv")
    maintenance = maintenance[
        maintenance["EQP_TAG"].eq(equipment)
        & maintenance["T_START_SEC"].le(t)
    ].copy()
    if maintenance.empty:
        print("현재까지 시작된 C 정비 없음")
    else:
        completed = maintenance["T_END_SEC"].notna() & maintenance["T_END_SEC"].le(t)
        maintenance["시작시간"] = maintenance["T_START_SEC"] / 3600
        maintenance["종료시간"] = (maintenance["T_END_SEC"] / 3600).where(completed)
        maintenance["현재상태"] = np.where(completed, "완료", "진행 중")
        show(
            f"{scenario}: 현재까지 시작된 정비",
            maintenance[
                ["WO_NO", "PART_CD", "MNT_TYPE", "시작시간", "종료시간", "현재상태"]
            ],
        )

# 17. 학습·검증의 정상·고장 임박 표본 분포
training_diagnoses = attach_diagnosis(
    expanded_train[["scenario", "HOUR", "RUL정답"]]
)
state_tables = []
for name, data in [("학습", training_diagnoses), ("검증", diagnosis_errors)]:
    table = data[["scenario", "HOUR", "RUL정답", "진단상태"]].copy()
    table["구분"] = name
    table["남은시간"] = np.where(table["RUL정답"] <= 6, "6시간 이하", "6시간 초과")
    state_tables.append(table)
state_data = pd.concat(state_tables, ignore_index=True)
print("\n[17단계: 학습·검증 상태별 표본 분포]")
print(
    state_data.groupby(["구분", "진단상태", "남은시간"]).agg(
        표본수=("RUL정답", "size"), 시나리오수=("scenario", "nunique")
    ).to_string()
)

# 18. 학습 내부 정상·고장 임박 예측 확인
internal_errors = attach_diagnosis(
    prediction_table(expanded_train, weight_predictions[best_weight])
)
internal_near = internal_errors[internal_errors["RUL정답"] <= 6]
print("\n[18단계: 학습 내부 고장 임박 오차]")
print(
    internal_near.groupby("진단상태").agg(
        표본수=("RUL정답", "size"),
        시나리오수=("scenario", "nunique"),
        MAE=("절대오차", "mean"),
        평균예측오차=("예측오차", "mean"),
    ).round(2).to_string()
)
show(
    "정상·고장 임박 학습 표본의 내부 예측",
    internal_near.loc[
        internal_near["진단상태"] == "normal",
        ["scenario", "HOUR", "RUL정답", "예측RUL", "예측오차", "절대오차"],
    ],
)

# 19. 초기 변화·부하 특징의 학습 내부 비교
augmented_data = expanded_data.merge(
    additional_data, on=["scenario", "HOUR"],
    how="left", validate="one_to_one",
)
augmented_events = augmented_data[
    augmented_data["RUL_EVENT"].eq(1) & augmented_data["RUL정답"].notna()
]
augmented_train = augmented_events[augmented_events["split"] == "train"].copy()
augmented_valid = augmented_events[augmented_events["split"] == "validation"].copy()
pd.testing.assert_frame_equal(
    augmented_train[["scenario", "HOUR"]].reset_index(drop=True),
    expanded_train[["scenario", "HOUR"]].reset_index(drop=True),
)
augmented_oof = out_of_fold(augmented_train, augmented_columns, best_weight)
normal_near = (
    internal_errors["진단상태"].eq("normal")
    & internal_errors["RUL정답"].le(6)
).to_numpy()
report19 = []
for name, prediction in [
    ("기존 특징", weight_predictions[best_weight]),
    ("초기변화·부하 특징 추가", augmented_oof),
]:
    truth = expanded_train["RUL정답"].to_numpy()
    error = np.abs(prediction - truth)
    report19.append({
        "모델": name, "전체MAE": error.mean(),
        "6시간이하MAE": error[truth <= 6].mean(),
        "정상임박6행MAE": error[normal_near].mean(),
    })
show("19단계: 학습 내부 특징 비교", pd.DataFrame(report19))

# 20. 추가 특징 모델의 기존 검증 평가
pd.testing.assert_frame_equal(
    augmented_valid[["scenario", "HOUR"]].reset_index(drop=True),
    expanded_valid[["scenario", "HOUR"]].reset_index(drop=True),
)
augmented_model = fit_model(
    augmented_train, augmented_columns, leaf=2, weight=best_weight
)
predictions["20단계: 초기변화·부하 특징 추가"] = augmented_model.predict(
    augmented_valid[augmented_columns]
)
augmented_comparison = prediction_table(
    augmented_valid, predictions["20단계: 초기변화·부하 특징 추가"]
)
show(
    "20단계: 기존 검증 데이터 비교",
    pd.DataFrame([
        evaluate(name, y_valid, predictions[name])
        for name in [
            "14단계: 고장 임박 가중치",
            "20단계: 초기변화·부하 특징 추가",
        ]
    ]),
)
normal_near_keys = suspects[["scenario", "HOUR"]]
show(
    "정상·고장 임박 표본의 새 예측",
    normal_near_keys.merge(
        augmented_comparison,
        on=["scenario", "HOUR"], validate="one_to_one",
    ),
)

# 21. 모델별 결과 종합
final_report = pd.DataFrame([
    evaluate("단순 기준: 항상 학습 중앙값", y_valid, np.full(len(y_valid), baseline_rul)),
    *[
        evaluate(name, y_valid, prediction)
        for name, prediction in predictions.items()
    ],
])
show("21단계: 회귀모델 실험 결과 종합", final_report)
print(
    f"\n[현재 후보 모델]\n20단계 추가 특징 모델\n"
    f"입력 특징 {len(augmented_columns)}개 / "
    f"학습 사건 {len(augmented_train)}행 / 검증 사건 {len(augmented_valid)}행"
)
print(
    "\n[평가 범위와 한계]\n"
    "- 회귀 MAE는 고장시간이 알려진 사건 표본만 대상으로 계산\n"
    "- 검열 활용은 별도의 생존모델 실험에서 평가\n"
    "- 정상 진단의 일부 고장 임박 사례에서 큰 과대예측이 남음\n"
    "- 반복 확인한 검증 결과이며 최종 테스트 성능은 아님\n"
    "- 다른 공정의 flow 정보는 모델 입력에 사용하지 않음"
)

# 22. 정답 파일 없이 C 센서로 RUL 예측
from types import SimpleNamespace

def predict_c_rul(csv_path, hour=None):
    required_columns = [
        "SCENARIO_ID", "EQP_CD", "T_SEC", "RUNNING",
    ] + all_sensors
    sensors = pd.read_csv(
        csv_path, usecols=required_columns
    ).sort_values("T_SEC")
    sensors = sensors[sensors["EQP_CD"] == equipment].copy()
    if sensors.empty:
        raise ValueError("C 연주설비 데이터가 없습니다.")
    if sensors["SCENARIO_ID"].nunique() != 1:
        raise ValueError("한 시나리오의 C.csv를 입력하세요.")
    scenario = sensors["SCENARIO_ID"].iloc[0]
    t = sensors["T_SEC"].max() if hour is None else float(hour) * 3600
    if not np.isfinite(t) or t < 0:
        raise ValueError("예측시각은 0 이상의 유한한 시간이어야 합니다.")
    sensors = sensors[sensors["T_SEC"] <= t]
    current = sensors[sensors["T_SEC"] == t]
    if len(current) != 1:
        raise ValueError("예측시각과 정확히 일치하는 센서 행이 필요합니다.")
    result = {
        "시나리오": scenario,
        "기준시각": t / 3600,
        "예측RUL": None,
        "상태": "",
    }
    if current.iloc[0]["RUNNING"] == 0:
        result["상태"] = "정지 중: 신규 고장 RUL 예측 보류"
        return result
    if current.iloc[0]["RUNNING"] != 1:
        raise ValueError("운전 상태를 확인하세요.")
    target = SimpleNamespace(
        T_SEC=t,
        GT_RUL_EVENT=np.nan,
        GT_RUL_HOURS=np.nan,
        GT_FOLLOWUP_HOURS=np.nan,
    )
    _, temporal, additional = make_features(
        sensors, target, scenario, "prediction"
    )
    features = {**temporal, **additional}
    X = pd.DataFrame([features])[augmented_columns]
    if X[all_sensors[0] + "_현재"].isna().any():
        raise ValueError("현재 LOAD 측정값을 확인하세요.")
    prediction = augmented_model.predict(X)[0]
    result["예측RUL"] = float(prediction)
    result["상태"] = "사건 표본으로 학습한 회귀모델의 잔여시간 추정"
    return result

prediction_result = predict_c_rul(
    folder / "S0222" / "C.csv",
    hour=30,
)
print("\n[22단계: 센서만 사용한 RUL 예측]")
print(pd.DataFrame([prediction_result]).round(3).to_string(index=False))