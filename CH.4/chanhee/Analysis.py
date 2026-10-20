import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import joblib

import numpy as np
import pandas as pd
import sklearn

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

# ==================================================
# 1. 경로 및 최종 후보 설정
# ==================================================

# 실행 파일 위치: CH.4/chanhee/Analysis.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS_ROOT = PROJECT_ROOT / "Data" / "Processed" / "Scenarios"

INDEX_PATH = SCENARIOS_ROOT / "index.csv"

# 저장 위치: 학습한 모델과 흐름별 예측 (Main.py all 의 quality 단계, Bundle.py가 Results/Unreal로 복사)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Quality"

# 앞선 학습 자료 내부 평가에서 선택한 기준
THRESHOLD = 0.13108238272783354

# 화면에 출력할 개별 사례 수
MAX_CASES = 15

INPUT_STAGES = ["F", "E1", "E2", "C"]

INSPECTION_COLUMNS = ["Q_F", "Q_E1", "Q_E2", "Q_C"]

# 재현을 위해 센서 및 특징 생성 순서를 고정
SENSOR_STAGES = ["A", "B", "D"]

SENSOR_COLUMNS = {
    "A": [
        "LOAD",
        "AMBIENT",
        "SET_TEMP",
        "TC-Z3",
        "OP-Z3",
        "RUNNING",
    ],
    "B": [
        "THICKNESS",
        "SPD",
        "PWR",
        "SET-PWR",
        "RESISTANCE",
        "RUNNING",
    ],
    "D": [
        "LOAD",
        "VIB-TOP_RMS",
        "VIB-TOP_STD",
        "VIB-TOP_KUR",
        "VIB-TOP_CRF",
        "VIB-BOT_RMS",
        "CUR-MTR_RMS",
    ],
}

# 추세를 추가할 센서
TREND_COLUMNS = {
    "A": [
        "TEMP_ERROR",
        "OP-Z3",
        "TC-Z3",
        "LOAD",
    ],
    "B": [
        "POWER_ERROR",
        "RESISTANCE",
        "PWR",
    ],
    "D": [
        "VIB-TOP_RMS",
        "VIB-TOP_CRF",
        "VIB-TOP_KUR",
        "CUR-MTR_RMS",
        "LOAD",
    ],
}

EQUIPMENT_CODES = {
    "A": "P1-CR1-CAL01",
    "B": "P1-CR1-WLD01",
    "D": "P1-CR1-SPM01",
}

ORIGIN_NAMES = {
    "P1-SP1-AGT01": "F 교반",
    "P1-SP1-EXF01": "E1 배풍",
    "P1-SP2-EXF01": "E2 배풍",
    "P1-CC1-INS01": "C 연주",
    "P1-CR1-WLD01": "B 용접",
    "P1-CR1-CAL01": "A 소둔",
    "P1-CR1-SPM01": "D 조질압연",
}


# ==================================================
# 2. 공통 기능
# ==================================================


def print_table(table):
    if table.empty:
        print("없음")
        return

    print(
        table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def to_time(value):
    return pd.to_datetime(
        value,
        utc=True,
        errors="raise",
    )


def read_flow_table(path, scenario_id, required):
    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    missing = set(required) - set(table.columns)

    if missing:
        raise ValueError(
            f"필요한 컬럼이 없습니다: {path}\n" f"누락 컬럼: {sorted(missing)}"
        )

    keys = ["SCENARIO_ID", "FLOW_ID", "STAGE"]

    if table[keys].isna().any().any():
        raise ValueError(f"연결 키에 빈칸이 있습니다: {path}")

    if not table["SCENARIO_ID"].eq(scenario_id).all():
        raise ValueError(f"다른 시나리오가 섞여 있습니다: {path}")

    if table.duplicated(["FLOW_ID", "STAGE"]).any():
        raise ValueError(f"FLOW·공정별 중복 기록이 있습니다: {path}")

    return table


# ==================================================
# 3. C 검사값과 최종 정답 연결
# ==================================================


def prepare_flow_data(scenario_id, split):
    folder = SCENARIOS_ROOT / scenario_id / "flow"

    inspections = read_flow_table(
        folder / "inspections.csv",
        scenario_id,
        [
            "SCENARIO_ID",
            "FLOW_ID",
            "STAGE",
            "EQP_CD",
            "OBSERVED_AT",
            "QUALITY_INDICATOR",
        ],
    )

    operations = read_flow_table(
        folder / "operations.csv",
        scenario_id,
        [
            "SCENARIO_ID",
            "FLOW_ID",
            "STAGE",
            "EQP_CD",
            "STATUS",
            "END_AT",
        ],
    )

    quality = read_flow_table(
        folder / "quality.csv",
        scenario_id,
        [
            "SCENARIO_ID",
            "FLOW_ID",
            "STAGE",
            "GT_STAGE_COMPLETED",
            "GT_FINAL_DEFECT_FLAG",
            "GT_FINAL_CENSORED",
            "GT_ORIGIN_EQP_CD",
        ],
    )

    inspections["QUALITY_INDICATOR"] = pd.to_numeric(
        inspections["QUALITY_INDICATOR"],
        errors="raise",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in [
        "GT_STAGE_COMPLETED",
        "GT_FINAL_DEFECT_FLAG",
        "GT_FINAL_CENSORED",
    ]:
        quality[column] = pd.to_numeric(
            quality[column],
            errors="raise",
        )

    operations["STATUS"] = operations["STATUS"].astype(str).str.strip().str.lower()

    ins = inspections.set_index(["FLOW_ID", "STAGE"])
    ops = operations.set_index(["FLOW_ID", "STAGE"])
    qua = quality.set_index(["FLOW_ID", "STAGE"])

    flows = sorted(
        set(inspections["FLOW_ID"])
        | set(operations["FLOW_ID"])
        | set(quality["FLOW_ID"])
    )

    labels_by_flow = quality.groupby("FLOW_ID")["GT_FINAL_DEFECT_FLAG"]

    records = []
    audit = Counter()

    for flow_id in flows:
        d_key = (flow_id, "D")

        # 완료된 D 작업의 정답이 있는 흐름만 평가
        if d_key not in ops.index:
            audit["D 작업 기록 없음"] += 1
            continue

        d_operation = ops.loc[d_key]

        if d_operation["STATUS"] != "completed" or pd.isna(d_operation["END_AT"]):
            audit["D 작업 미완료"] += 1
            continue

        if d_key not in qua.index:
            audit["D 품질 정답 없음"] += 1
            continue

        d_quality = qua.loc[d_key]

        if d_quality["GT_STAGE_COMPLETED"] != 1 or d_quality["GT_FINAL_CENSORED"] != 0:
            audit["최종 정답 검열 또는 완료 확인 불가"] += 1
            continue

        label = d_quality["GT_FINAL_DEFECT_FLAG"]

        if pd.isna(label) or label not in (0, 1):
            audit["최종 결함 정답 없음 또는 잘못된 값"] += 1
            continue

        label = int(label)

        repeated = labels_by_flow.get_group(flow_id).dropna()

        if not repeated.eq(label).all():
            raise ValueError(f"최종 정답 불일치: " f"{scenario_id}, {flow_id}")

        if any(
            (flow_id, stage) not in ins.index or (flow_id, stage) not in ops.index
            for stage in INPUT_STAGES
        ):
            audit["F·E·C 검사 또는 작업 기록 부족"] += 1
            continue

        stage_ins = {stage: ins.loc[(flow_id, stage)] for stage in INPUT_STAGES}

        stage_ops = {stage: ops.loc[(flow_id, stage)] for stage in INPUT_STAGES}

        if any(
            pd.isna(stage_ins[s]["OBSERVED_AT"]) or pd.isna(stage_ops[s]["END_AT"])
            for s in INPUT_STAGES
        ):
            audit["입력 공정 시각 누락"] += 1
            continue

        connection_ok = all(
            stage_ops[s]["STATUS"] == "completed"
            and stage_ops[s]["EQP_CD"] == stage_ins[s]["EQP_CD"]
            and to_time(stage_ops[s]["END_AT"]) <= to_time(stage_ins[s]["OBSERVED_AT"])
            for s in INPUT_STAGES
        )

        if not connection_ok:
            audit["작업·검사 연결 확인 필요"] += 1
            continue

        observed = {
            stage: to_time(stage_ins[stage]["OBSERVED_AT"]) for stage in INPUT_STAGES
        }

        cutoff = observed["C"]

        order_ok = observed["F"] <= min(observed["E1"], observed["E2"]) and max(
            observed["E1"], observed["E2"]
        ) <= cutoff < to_time(d_operation["END_AT"])

        if not order_ok:
            audit["검사 시각 순서 확인 필요"] += 1
            continue

        origin = d_quality["GT_ORIGIN_EQP_CD"]

        record = {
            "SCENARIO_ID": scenario_id,
            "FLOW_ID": flow_id,
            "split": split,
            "C_INSPECTED_AT": cutoff,
            "target_D": label,
            "origin_eqp": (
                str(origin).strip()
                if pd.notna(origin) and str(origin).strip()
                else "원인 정보 없음"
            ),
        }

        for stage in INPUT_STAGES:
            record[f"Q_{stage}"] = stage_ins[stage]["QUALITY_INDICATOR"]

        records.append(record)
        audit["분석에 포함"] += 1

    return records, audit


# ==================================================
# 4. 후속 설비 센서 읽기
# ==================================================


def read_sensor(scenario_id, stage):
    path = SCENARIOS_ROOT / scenario_id / f"{stage}.csv"

    metadata = [
        "SCENARIO_ID",
        "MEAS_DT",
        "EQP_CD",
    ]

    if stage == "B":
        metadata += ["COIL_ID"]

    if stage == "D":
        metadata += ["BURST_ID", "N", "FS_HZ"]

    sensor = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=metadata + SENSOR_COLUMNS[stage],
    )

    if not sensor["SCENARIO_ID"].eq(scenario_id).all():
        raise ValueError(f"센서 시나리오 혼합: {path}")

    if sensor[metadata].isna().any().any():
        raise ValueError(f"센서 식별 정보 또는 시각 누락: {path}")

    if not sensor["EQP_CD"].eq(EQUIPMENT_CODES[stage]).all():
        raise ValueError(f"센서 설비 코드 불일치: {path}")

    sensor["MEAS_DT"] = to_time(sensor["MEAS_DT"])

    for column in SENSOR_COLUMNS[stage]:
        sensor[column] = pd.to_numeric(
            sensor[column],
            errors="raise",
        )

    sensor[SENSOR_COLUMNS[stage]] = sensor[SENSOR_COLUMNS[stage]].replace(
        [np.inf, -np.inf], np.nan
    )

    if stage == "A":
        sensor["TEMP_ERROR"] = sensor["TC-Z3"] - sensor["SET_TEMP"]

    if stage == "B":
        sensor["POWER_ERROR"] = sensor["PWR"] - sensor["SET-PWR"]

    sensor = sensor.sort_values(
        "MEAS_DT",
        kind="stable",
    ).reset_index(drop=True)

    if stage == "D":
        n = pd.to_numeric(sensor["N"], errors="raise")
        fs = pd.to_numeric(sensor["FS_HZ"], errors="raise")

        if (n <= 0).any() or (fs <= 0).any():
            raise ValueError(f"N 또는 FS_HZ 오류: {path}")

        if sensor.duplicated(["EQP_CD", "BURST_ID"]).any():
            raise ValueError(f"D 버스트 중복: {path}")

        # 버스트 전체 측정이 끝난 뒤 특징량 사용
        sensor["_READY_AT"] = sensor["MEAS_DT"] + pd.to_timedelta(n / fs, unit="s")
    else:
        sensor["_READY_AT"] = sensor["MEAS_DT"]

    return sensor


# ==================================================
# 5. 과거 구간 선택
# ==================================================


def select_past_window(sensor, cutoff, hours):
    start = cutoff - pd.Timedelta(hours=hours)

    left = sensor["MEAS_DT"].searchsorted(start, side="left")

    right = sensor["MEAS_DT"].searchsorted(cutoff, side="right")

    segment = sensor.iloc[left:right]

    # C 시점까지 측정이 완료된 자료만 사용
    return segment.loc[segment["_READY_AT"].le(cutoff)]


# ==================================================
# 6. 최근 1시간 센서 요약
# ==================================================


def summarize_one_hour(segment, stage):
    columns = list(SENSOR_COLUMNS[stage])

    if stage == "A":
        columns += ["TEMP_ERROR"]

    if stage == "B":
        columns += ["POWER_ERROR"]

        # 코일 프로파일을 먼저 코일별로 요약
        grouped = segment.groupby(
            ["COIL_ID", "MEAS_DT"],
            sort=True,
        )[columns]

        values = pd.concat(
            [
                grouped.mean().add_suffix("_profile_mean"),
                grouped.std(ddof=0).add_suffix("_profile_std"),
                grouped.max().add_suffix("_profile_max"),
            ],
            axis=1,
        )
    else:
        values = segment[columns]

    features = {}

    for column in values.columns:
        observed = values[column].dropna()
        prefix = f"PAST_{stage}__{column}"

        features[f"{prefix}__mean"] = (
            float(observed.mean()) if len(observed) else np.nan
        )

        features[f"{prefix}__std"] = (
            float(observed.std(ddof=0)) if len(observed) else np.nan
        )

        features[f"{prefix}__last"] = (
            float(observed.iloc[-1]) if len(observed) else np.nan
        )

    return features


# ==================================================
# 7. 최근 1·4시간 센서 추세
# ==================================================


def summarize_trend(segment, stage, cutoff, hours):
    # 같은 시각의 B 프로파일 행은 평균으로 묶어
    # 프로파일 순서를 시간 변화로 오해하지 않도록 한다.
    values = segment.groupby("MEAS_DT")[TREND_COLUMNS[stage]].mean().sort_index()

    elapsed = (values.index - cutoff).total_seconds().to_numpy() / 3600

    features = {}

    for column in TREND_COLUMNS[stage]:
        observed = values[column].to_numpy(dtype=float)

        mask = np.isfinite(observed)

        y = observed[mask]
        t = elapsed[mask]

        if len(y) > 1:
            centered_t = t - t.mean()
            centered_y = y - y.mean()

            slope = float(
                np.sum(centered_t * centered_y)
                / max(
                    np.sum(centered_t**2),
                    1e-12,
                )
            )
        else:
            slope = np.nan

        prefix = f"NEW_{stage}_{hours}_{column}"

        # 시간당 변화량
        features[f"{prefix}_slope"] = slope

        if hours == 4:
            features[f"{prefix}_mean"] = float(y.mean()) if len(y) else np.nan

    return features


# ==================================================
# 8. 한 시나리오의 센서 특징 추가
# ==================================================


def add_sensor_features(scenario_id, records):
    if not records:
        return records, Counter()

    sensors = {stage: read_sensor(scenario_id, stage) for stage in SENSOR_STAGES}

    audit = Counter()

    for record in records:
        cutoff = record["C_INSPECTED_AT"]

        for stage in SENSOR_STAGES:
            sensor = sensors[stage]

            for hours in [1, 4]:
                segment = select_past_window(
                    sensor,
                    cutoff,
                    hours,
                )

                if segment.empty:
                    audit[f"{stage}: 과거 {hours}시간 기록 없음"] += 1

                if hours == 1:
                    record.update(summarize_one_hour(segment, stage))

                record.update(
                    summarize_trend(
                        segment,
                        stage,
                        cutoff,
                        hours,
                    )
                )

    return records, audit


# ==================================================
# 9. index.csv 기준으로 데이터 구성
# ==================================================


def build_dataset():
    index = pd.read_csv(
        INDEX_PATH,
        encoding="utf-8-sig",
    )

    if not {"scenario", "split"}.issubset(index.columns):
        raise ValueError("index.csv에 scenario, split 컬럼이 필요합니다.")

    if index[["scenario", "split"]].isna().any().any():
        raise ValueError("index.csv 연결 정보에 빈칸이 있습니다.")

    index["scenario"] = index["scenario"].astype(str).str.strip()

    index["split"] = (
        index["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "val": "validation",
                "valid": "validation",
            }
        )
    )

    if index["scenario"].duplicated().any():
        raise ValueError("index.csv에 시나리오가 중복됐습니다.")

    unknown = set(index["split"]) - {"train", "validation", "test"}

    if unknown:
        raise ValueError(f"알 수 없는 split: {sorted(unknown)}")

    print("[시나리오 구분]")
    print(index["split"].value_counts().to_string())

    selected = index.loc[index["split"].isin(["train", "validation"])]

    records = []
    flow_audit = Counter()
    sensor_audit = Counter()

    for position, row in enumerate(
        selected.itertuples(index=False),
        start=1,
    ):
        scenario_records, counts = prepare_flow_data(
            row.scenario,
            row.split,
        )

        scenario_records, sensor_counts = add_sensor_features(
            row.scenario,
            scenario_records,
        )

        records.extend(scenario_records)
        flow_audit.update(counts)
        sensor_audit.update(sensor_counts)

        if position % 25 == 0 or position == len(selected):
            print(f"자료 연결 중: " f"{position}/{len(selected)} " "시나리오 완료")

    print("\n[생산 흐름 포함·제외 내역]")
    for reason, count in flow_audit.items():
        print(f"{reason}: {count:,}개")

    print("\n[과거 센서 구간 점검]")
    if sensor_audit:
        for reason, count in sensor_audit.items():
            print(f"{reason}: {count:,}개")
    else:
        print("기록 누락 구간 없음")

    if not records:
        raise ValueError("분석 가능한 생산 흐름이 없습니다.")

    return pd.DataFrame(records)


# ==================================================
# 10. 최종 후보 모델
# ==================================================


def make_model():
    return make_pipeline(
        SimpleImputer(
            strategy="median",
            add_indicator=True,
        ),
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=5,
            early_stopping=False,
            random_state=31,
        ),
    )


# ==================================================
# 11. 평가 지표
# ==================================================


def calculate_metrics(y, probability):
    alarm = probability >= THRESHOLD

    tp = int(np.sum(alarm & (y == 1)))
    fp = int(np.sum(alarm & (y == 0)))
    fn = int(np.sum(~alarm & (y == 1)))
    tn = int(np.sum(~alarm & (y == 0)))

    return {
        "AP": average_precision_score(y, probability),
        "검사요망기준": THRESHOLD,
        "정밀도": (tp / (tp + fp) if tp + fp else 0.0),
        "재현율": (tp / (tp + fn) if tp + fn else 0.0),
        "오경보율": (fp / (fp + tn) if fp + tn else 0.0),
        "발견결함": tp,
        "오경보": fp,
        "놓친결함": fn,
        "정상안내없음": tn,
        "전체안내수": tp + fp,
        "전체안내비율": ((tp + fp) / len(y)),
    }


# ==================================================
# 12. 생산 흐름별 예측 결과
# ==================================================


def make_result_table(valid, probability):
    result = valid[
        [
            "SCENARIO_ID",
            "FLOW_ID",
            "C_INSPECTED_AT",
            "target_D",
            "origin_eqp",
        ]
    ].copy()

    result["위험점수"] = probability
    result["검사요망"] = probability >= THRESHOLD

    result["안내"] = np.where(
        result["검사요망"],
        "결함 검사 요망",
        "현재 기준 안내 없음",
    )

    actual = result["target_D"].eq(1)
    alarm = result["검사요망"]

    result["평가구분"] = np.select(
        [
            actual & alarm,
            actual & ~alarm,
            ~actual & alarm,
        ],
        [
            "결함 발견",
            "놓친 결함",
            "정상이나 검사 요망",
        ],
        default="정상·안내 없음",
    )

    result["정답의원인설비"] = (
        result["origin_eqp"].map(ORIGIN_NAMES).fillna(result["origin_eqp"])
    )

    result.loc[
        ~actual,
        "정답의원인설비",
    ] = "최종 결함 없음"

    return result


# ==================================================
# 13. 원인 설비별 탐지 결과
# ==================================================


def summarize_origins(result):
    defects = result.loc[result["target_D"].eq(1)]

    rows = []

    for origin, group in defects.groupby("정답의원인설비"):
        found = int(group["검사요망"].sum())

        rows.append(
            {
                "정답의원인설비": origin,
                "실제결함수": len(group),
                "독립시나리오수": (group["SCENARIO_ID"].nunique()),
                "발견결함": found,
                "놓친결함": len(group) - found,
                "재현율": found / len(group),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        "놓친결함",
        ascending=False,
    )


# ==================================================
# 14. 시나리오별 오류
# ==================================================


def summarize_scenarios(result):
    rows = []

    for scenario_id, group in result.groupby("SCENARIO_ID"):
        actual = group["target_D"].eq(1)
        alarm = group["검사요망"]

        rows.append(
            {
                "시나리오": scenario_id,
                "전체FLOW": len(group),
                "실제결함": int(actual.sum()),
                "발견결함": int((actual & alarm).sum()),
                "놓친결함": int((actual & ~alarm).sum()),
                "오경보": int((~actual & alarm).sum()),
            }
        )

    table = pd.DataFrame(rows)

    table["전체오류"] = table["놓친결함"] + table["오경보"]

    return table.sort_values(
        ["전체오류", "놓친결함"],
        ascending=False,
    )


# ==================================================
# 15. 실행 — 화면에만 출력
# ==================================================


# ==================================================
# 14. 모델·예측 저장 (언리얼 연동)
# ==================================================


def save_outputs(model, features, valid, scores, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    joblib.dump(
        {
            "model": model,
            "features": features,
            "threshold": THRESHOLD,
            "predict_at": "C 검사 직후",
            "generated_at": generated_at,
            "validation": metrics,
        },
        OUTPUT_DIR / "quality_model.joblib",
        compress=3,
    )

    # 흐름 하나당 한 줄: C 검사 직후의 최종(D) 결함 위험
    # actual_defect는 validation 정답이라 평가·피드백용 (실제 운영 예측에는 없음)
    table = pd.DataFrame(
        {
            "SCENARIO_ID": valid["SCENARIO_ID"],
            "FLOW_ID": valid["FLOW_ID"],
            "PREDICTED_AT": pd.to_datetime(valid["C_INSPECTED_AT"], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "defect_score": scores.round(4),
            "inspection_requested": scores >= THRESHOLD,
            "threshold": round(THRESHOLD, 4),
            "origin_eqp": valid["origin_eqp"],
            "actual_defect": valid["target_D"].astype(int),
        }
    ).sort_values(["SCENARIO_ID", "PREDICTED_AT", "FLOW_ID"])

    table.to_csv(OUTPUT_DIR / "flow_quality.csv", index=False, encoding="utf-8-sig")

    summary = {
        "Version": 1,
        "GeneratedAt": generated_at,
        "Task": "C 검사 직후 최종(D) 결함 위험 예측 (흐름 단위)",
        "Split": "validation",
        "Threshold": THRESHOLD,
        "Features": len(features),
        "Metrics": metrics,
    }

    (OUTPUT_DIR / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n저장 위치: {OUTPUT_DIR}")


def main():
    import io
    from contextlib import redirect_stdout
    from sklearn.metrics import average_precision_score

    print("자료 연결 및 모델 학습 중...", flush=True)

    # 자료 연결 과정의 상세 출력만 숨김
    with redirect_stdout(io.StringIO()):
        data = build_dataset()

    train = data.loc[data["split"].eq("train")].reset_index(drop=True)

    valid = data.loc[data["split"].eq("validation")].reset_index(drop=True)

    y_train = train["target_D"].astype(int).to_numpy()
    y_valid = valid["target_D"].astype(int).to_numpy()

    if len(np.unique(y_train)) != 2:
        raise ValueError("학습 자료에 정상과 결함이 모두 필요합니다.")

    if len(np.unique(y_valid)) != 2:
        raise ValueError("검증 자료에 정상과 결함이 모두 필요합니다.")

    # 기존 모델과 동일한 입력 및 순서 유지
    sensor_features = [
        column
        for stage in SENSOR_STAGES
        for column in data.columns
        if column.startswith(f"PAST_{stage}__")
    ]

    trend_features = [
        column
        for stage in SENSOR_STAGES
        for column in data.columns
        if column.startswith(f"NEW_{stage}_")
    ]

    candidates = INSPECTION_COLUMNS + sensor_features + trend_features

    features = [column for column in candidates if train[column].notna().any()]

    model = make_model()
    model.fit(train[features], y_train)

    scores = model.predict_proba(valid[features])[:, 1]
    alarms = scores >= THRESHOLD
    actual = y_valid == 1

    tp = int((alarms & actual).sum())
    fp = int((alarms & ~actual).sum())
    fn = int((~alarms & actual).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    ap = average_precision_score(y_valid, scores)

    print("\n[최종 D 결함 예측 — 검증 결과]")
    print("예측 시점: C 검사 직후")
    print("입력: F·E·C 검사값 + B·A·D 과거 센서")
    print(f"검사 요망 기준: {THRESHOLD:.4f}")
    print(
        f"검증 대상: {len(valid):,}개 FLOW "
        f"/ {valid['SCENARIO_ID'].nunique()}개 시나리오"
    )

    print(f"\nAP: {ap:.4f}")
    print(f"정밀도: {precision:.1%}")
    print(f"재현율: {recall:.1%}")

    print(f"\n실제 결함: {int(actual.sum()):,}개")
    print(f"발견한 결함: {tp:,}개")
    print(f"놓친 결함: {fn:,}개")
    print(f"오경보: {fp:,}개")

    save_outputs(
        model,
        features,
        valid,
        scores,
        {
            "AP": round(float(ap), 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "ActualDefects": int(actual.sum()),
            "Found": tp,
            "Missed": fn,
            "FalseAlarms": fp,
            "Flows": int(len(valid)),
        },
    )

    print("\n※ D 작업이 완료된 FLOW만 평가했습니다.")
    print("※ 검사 요망은 불량 확정이 아니며, 테스트 평가는 별도입니다.")


if __name__ == "__main__":
    main()
