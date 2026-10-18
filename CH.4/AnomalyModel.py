from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "process_file_name": "C.csv",
    "quality_relative_path": "flow/quality.csv",
    "time_column": "auto",
    "sensor_columns": "auto",
    "resample_rule": "1min",
    "short_window": 5,
    "long_window": 15,
    "top_k": 3,
    "threshold_percentiles": [95, 97, 99],
    "persistence_options": [1, 2, 3],
    "target_recall_24h": 95.0,
    "comparison_baseline_persistence": 3,
    "alarm_merge_minutes": 30,
    "max_lead_hours": 24,
    "short_lead_hours": 6,
    "train_scenario_ratio": 0.7,
    "max_train_rows_per_scenario": 800,
    "random_state": 42,
    "min_event_scenarios": 10,
    "quality": {
        "enabled": True,
        "casting_stage": "C",
        "casting_equipment": "auto"
    },
    "ground_truth": {
        "running": "RUNNING",
        "quality_flag": "QUALITY_FLAG",
        "is_missing": "IS_MISSING",
        "is_stopped": "IS_STOPPED",
        "is_maintenance": "IS_MAINTENANCE",
        "is_outlier": "IS_OUTLIER",
        "health_state": "GT_HEALTH_STATE",
        "fault_mode": "GT_FAULT_MODE"
    }
}

TIME_CANDIDATES = ["MEAS_DT", "TIMESTAMP", "TIME", "DATETIME", "DATE_TIME", "DT"]
NON_SENSOR_EXACT = {
    "ROW_ID", "T_SEC", "SCENARIO_ID", "EQP_CD", "SLAB_NO",
    "RUNNING", "QUALITY_FLAG", "IS_MISSING", "IS_STOPPED",
    "IS_MAINTENANCE", "IS_OUTLIER", "GT_HEALTH_STATE",
    "GT_CAUSE_PART", "GT_FAULT_MODE", "GT_STAGE_COMPLETED",
    "GT_DEFECT_FLAG", "GT_ORIGIN_EQP_CD", "GT_FINAL_DEFECT_FLAG",
    "GT_FINAL_CENSORED", "FLOW_ID", "STAGE"
}

QUALITY_REQUIRED = [
    "FLOW_ID", "STAGE", "EQP_CD", "GT_STAGE_COMPLETED",
    "GT_DEFECT_FLAG", "GT_ORIGIN_EQP_CD", "GT_FINAL_DEFECT_FLAG",
    "GT_FINAL_CENSORED"
]


def deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[Path]) -> dict:
    if path is None:
        return deep_update({}, DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    return deep_update(DEFAULT_CONFIG, user)


def save_json(obj: dict, path: Path):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return None if np.isnan(x) else float(x)
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2)


def read_csv(path: Path, usecols=None, nrows=None):
    return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)


def scenario_id(path: Path) -> str:
    return path.parent.name


def discover_process_files(root: Path, file_name: str):
    root = root.resolve()
    if root.is_file():
        return [(scenario_id(root), root)]
    if (root / file_name).exists():
        return [(root.name, root / file_name)]
    files = sorted(root.rglob(file_name))
    if not files:
        raise FileNotFoundError(f"{root} 아래에서 {file_name}을 찾지 못했습니다.")
    return [(scenario_id(f), f) for f in files]


def detect_time_column(sample: pd.DataFrame, configured):
    if configured != "auto":
        if configured not in sample.columns:
            raise ValueError(f"시간 컬럼이 없습니다: {configured}")
        return configured

    for col in TIME_CANDIDATES:
        if col in sample.columns:
            parsed = pd.to_datetime(sample[col], errors="coerce", utc=True)
            if parsed.notna().mean() >= 0.8:
                return col

    for col in sample.columns:
        if sample[col].dtype == "object" or pd.api.types.is_string_dtype(sample[col]):
            parsed = pd.to_datetime(sample[col], errors="coerce", utc=True)
            if parsed.notna().mean() >= 0.95:
                return col
    raise ValueError("시간 컬럼을 자동으로 찾지 못했습니다.")


def infer_sensor_columns(sample: pd.DataFrame, time_col: str, configured):
    if configured != "auto":
        missing = [c for c in configured if c not in sample.columns]
        if missing:
            raise ValueError(f"설정한 센서 컬럼이 없습니다: {missing}")
        return list(configured)

    sensors = []
    for col in sample.columns:
        upper = str(col).upper()
        if col == time_col or upper in NON_SENSOR_EXACT:
            continue
        if upper.startswith("GT_") or upper.startswith("IS_"):
            continue
        if upper.endswith("_ID") or upper.endswith("_NO") or upper.endswith("_CD"):
            continue
        num = pd.to_numeric(sample[col], errors="coerce")
        if num.notna().mean() >= 0.90 and num.nunique(dropna=True) >= 5:
            sensors.append(col)

    if not sensors:
        raise ValueError("센서 컬럼 자동 탐색 실패. config의 sensor_columns에 직접 지정하세요.")
    return sensors


def detect_schema(files, config):
    samples = [read_csv(f, nrows=3000) for _, f in files[:5]]
    sample = pd.concat(samples, ignore_index=True, sort=False)
    time_col = detect_time_column(sample, config["time_column"])
    sensors = infer_sensor_columns(sample, time_col, config["sensor_columns"])
    return {"time_column": time_col, "sensor_columns": sensors}


def gt_columns(config):
    return list(config["ground_truth"].values())


def has_value(series: pd.Series):
    text = series.astype("string").str.strip().str.lower().fillna("")
    return (~text.isin(["", "nan", "none", "null", "<na>"])).fillna(False)


def get_ground_truth(df: pd.DataFrame, config):
    gt = config["ground_truth"]
    missing_cols = [c for c in gt.values() if c not in df.columns]
    if missing_cols:
        raise ValueError(f"GT 컬럼이 없습니다: {missing_cols}")

    health = df[gt["health_state"]].astype("string").str.strip().str.lower()
    maintenance = (
        pd.to_numeric(df[gt["is_maintenance"]], errors="coerce").fillna(0).eq(1)
        | health.eq("maintenance").fillna(False)
    ).fillna(False)
    running = pd.to_numeric(df[gt["running"]], errors="coerce").fillna(0).eq(1)
    missing = pd.to_numeric(df[gt["is_missing"]], errors="coerce").fillna(0).eq(1)
    stopped = pd.to_numeric(df[gt["is_stopped"]], errors="coerce").fillna(0).eq(1)
    outlier = pd.to_numeric(df[gt["is_outlier"]], errors="coerce").fillna(0).eq(1)
    fault_mode = has_value(df[gt["fault_mode"]])
    abnormal_health = health.notna() & ~health.isin(["normal", "maintenance"]).fillna(False)
    fault = (~maintenance & (fault_mode | abnormal_health)).fillna(False)
    unplanned_stop = (stopped & ~maintenance).fillna(False)
    quality_ok = (
        df[gt["quality_flag"]].astype("string").str.strip().str.lower().eq("ok").fillna(False)
    )
    normal = (
        health.eq("normal").fillna(False) & running & ~maintenance & ~missing
        & ~outlier & ~unplanned_stop & ~fault & quality_ok
    ).fillna(False)

    return {
        "NORMAL": normal.astype(bool),
        "FAULT": fault.astype(bool),
        "UNPLANNED_STOP": unplanned_stop.astype(bool),
        "OUTLIER": (outlier & ~maintenance).fillna(False).astype(bool),
        "MAINTENANCE": maintenance.astype(bool)
    }


def profile_ground_truth(files, schema, config):
    time_col = schema["time_column"]
    usecols = [time_col] + gt_columns(config)
    rows = []

    for i, (sid, file) in enumerate(files, 1):
        df = read_csv(file, usecols=usecols)
        time = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        gt = get_ground_truth(df, config)

        def first(mask):
            temp = time[mask.fillna(False).astype(bool) & time.notna()]
            return pd.NaT if temp.empty else temp.min()

        rows.append({
            "SCENARIO_ID": sid,
            "FAULT_AT": first(gt["FAULT"]),
            "STOP_AT": first(gt["UNPLANNED_STOP"]),
            "OUTLIER_AT": first(gt["OUTLIER"])
        })
        if i % 30 == 0 or i == len(files):
            print(f"[GT] {i}/{len(files)}")

    return pd.DataFrame(rows)


def select_event_type(summary, config):
    counts = {
        "FAULT": int(summary["FAULT_AT"].notna().sum()),
        "UNPLANNED_STOP": int(summary["STOP_AT"].notna().sum()),
        "OUTLIER": int(summary["OUTLIER_AT"].notna().sum())
    }
    col_map = {"FAULT": "FAULT_AT", "UNPLANNED_STOP": "STOP_AT", "OUTLIER": "OUTLIER_AT"}
    for event_type in ["FAULT", "UNPLANNED_STOP", "OUTLIER"]:
        if counts[event_type] >= int(config["min_event_scenarios"]):
            out = summary.copy()
            out["EVENT_AT"] = out[col_map[event_type]]
            out["HAS_EVENT"] = out["EVENT_AT"].notna()
            return out, event_type, counts
    raise ValueError(f"분석 가능한 Event가 부족합니다: {counts}")


def build_features(file: Path, schema, config, event_type=None, include_gt=True):
    time_col = schema["time_column"]
    sensors = schema["sensor_columns"]
    usecols = [time_col, *sensors]
    if include_gt:
        usecols += gt_columns(config)
    usecols = list(dict.fromkeys(usecols))

    df = read_csv(file, usecols=usecols)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col])
    for c in sensors:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if include_gt:
        gt = get_ground_truth(df, config)
        df["__NORMAL__"] = gt["NORMAL"].astype(int)
        df["__MAINTENANCE__"] = gt["MAINTENANCE"].astype(int)
        df["__EVENT__"] = 0 if event_type is None else gt[event_type].astype(int)

    df = df.sort_values(time_col).set_index(time_col)
    rule = config["resample_rule"]
    sw = int(config["short_window"])
    lw = int(config["long_window"])
    minute = df[sensors].resample(rule).median()

    cur = minute.copy(); cur.columns = [f"CUR__{c}" for c in cur.columns]
    diff = minute.diff(); diff.columns = [f"DIFF__{c}" for c in diff.columns]
    mean = minute.rolling(sw, min_periods=1).mean(); mean.columns = [f"MEAN{sw}__{c}" for c in mean.columns]
    std = minute.rolling(sw, min_periods=2).std(); std.columns = [f"STD{sw}__{c}" for c in std.columns]
    slope = (minute - minute.shift(lw)) / lw; slope.columns = [f"SLOPE{lw}__{c}" for c in slope.columns]

    parts = [cur, diff, mean, std, slope]
    if include_gt:
        meta = df[["__NORMAL__", "__MAINTENANCE__", "__EVENT__"]].resample(rule).agg({
            "__NORMAL__": "min", "__MAINTENANCE__": "max", "__EVENT__": "max"
        })
        parts.append(meta)

    return pd.concat(parts, axis=1).reset_index().rename(columns={time_col: "TIME"})


def get_feature_cols(df, config):
    sw, lw = int(config["short_window"]), int(config["long_window"])
    prefixes = ("CUR__", "DIFF__", f"MEAN{sw}__", f"STD{sw}__", f"SLOPE{lw}__")
    return [c for c in df.columns if c.startswith(prefixes)]


def split_scenarios(summary, config):
    rng = np.random.default_rng(int(config["random_state"]))
    event_ids = summary.loc[summary["HAS_EVENT"], "SCENARIO_ID"].tolist()
    normal_ids = np.array(summary.loc[~summary["HAS_EVENT"], "SCENARIO_ID"].tolist())
    rng.shuffle(normal_ids)
    n = max(1, int(len(normal_ids) * float(config["train_scenario_ratio"])))
    train_ids = normal_ids[:n].tolist()
    eval_ids = event_ids + normal_ids[n:].tolist()
    return train_ids, eval_ids


def build_normal_dataset(file_map, train_ids, event_type, schema, config):
    parts, feature_cols = [], None
    max_rows = int(config["max_train_rows_per_scenario"])
    seed = int(config["random_state"])

    for i, sid in enumerate(train_ids, 1):
        df = build_features(file_map[sid], schema, config, event_type, True)
        if feature_cols is None:
            feature_cols = get_feature_cols(df, config)
        normal = df[df["__NORMAL__"] == 1].copy()
        if normal.empty:
            continue
        if len(normal) > max_rows:
            normal = normal.sample(n=max_rows, random_state=seed + i)
        parts.append(normal[feature_cols])
        if i % 20 == 0 or i == len(train_ids):
            print(f"[정상 학습] {i}/{len(train_ids)}")

    if not parts:
        raise ValueError("정상 학습 데이터를 만들 수 없습니다.")
    train = pd.concat(parts, ignore_index=True)
    feature_cols = [c for c in feature_cols if train[c].notna().any()]
    return train, feature_cols


def fit_normal_range(train, features):
    median = train[features].median()
    mad = train[features].sub(median).abs().median()
    std = train[features].std()
    scale = (1.4826 * mad).where((1.4826 * mad) > 1e-9, std).fillna(1).replace(0, 1)
    return median, scale


def anomaly_details(df, features, median, scale, top_k):
    z = (
        df[features].sub(median).div(scale).abs()
        .replace([np.inf, -np.inf], np.nan).fillna(0).clip(upper=10)
    )
    values = z.to_numpy()
    k = min(int(top_k), values.shape[1])
    scores = np.partition(values, -k, axis=1)[:, -k:].mean(axis=1)
    top_feature = z.idxmax(axis=1).astype(str)
    top_sensor = top_feature.str.split("__", n=1).str[-1]
    return scores, top_feature.to_numpy(), top_sensor.to_numpy()


def build_thresholds(train, features, median, scale, config):
    scores, _, _ = anomaly_details(train, features, median, scale, config["top_k"])
    return {int(p): float(np.percentile(scores, p)) for p in config["threshold_percentiles"]}


def alarm_mask(scores, threshold, persistence):
    above = pd.Series(np.asarray(scores) >= threshold)
    p = int(persistence)
    return above.rolling(p, min_periods=p).sum() >= p


def alarm_starts(df, mask, merge_minutes):
    start = mask & ~mask.shift(fill_value=False)
    valid = start.values
    if "__MAINTENANCE__" in df.columns:
        valid = valid & (df["__MAINTENANCE__"] == 0).to_numpy()
    times = df.loc[valid, "TIME"].dropna().sort_values().tolist()
    if not times:
        return []
    merged = [times[0]]
    gap = pd.Timedelta(minutes=int(merge_minutes))
    for t in times[1:]:
        if t - merged[-1] > gap:
            merged.append(t)
    return merged


def max_consecutive(mask):
    best = cur = 0
    for v in np.asarray(mask, dtype=bool):
        if v:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def evaluate(file_map, summary, eval_ids, event_type, features, median, scale, thresholds, schema, config):
    s_map = summary.set_index("SCENARIO_ID")
    keys = [(int(p), int(d)) for p in thresholds for d in config["persistence_options"]]
    metrics = {k: {"EVENTS": 0, "D24": 0, "D6": 0, "LEADS": [], "FP": 0, "NORMAL_MIN": 0} for k in keys}
    scenario_rows = []

    for i, sid in enumerate(eval_ids, 1):
        df = build_features(file_map[sid], schema, config, event_type, True)
        scores, top_features, top_sensors = anomaly_details(df, features, median, scale, config["top_k"])
        df["ANOMALY_SCORE"] = scores
        df["TOP_FEATURE"] = top_features
        df["TOP_SENSOR"] = top_sensors
        event_at = s_map.loc[sid, "EVENT_AT"]
        has_event = pd.notna(event_at)

        for pct, threshold in thresholds.items():
            for persistence in config["persistence_options"]:
                key = (int(pct), int(persistence)); m = metrics[key]
                mask = alarm_mask(scores, threshold, persistence)
                starts = alarm_starts(df, mask, config["alarm_merge_minutes"])

                if has_event:
                    m["EVENTS"] += 1
                    start24 = event_at - pd.Timedelta(hours=float(config["max_lead_hours"]))
                    start6 = event_at - pd.Timedelta(hours=float(config["short_lead_hours"]))
                    valid24 = [t for t in starts if start24 <= t < event_at]
                    valid6 = [t for t in starts if start6 <= t < event_at]
                    d24 = bool(valid24); d6 = bool(valid6)
                    lead = np.nan
                    if d24:
                        m["D24"] += 1
                        lead = (event_at - min(valid24)).total_seconds() / 3600
                        m["LEADS"].append(lead)
                    if d6:
                        m["D6"] += 1
                    false = [t for t in starts if t < start24]
                    m["FP"] += len(false)
                    normal_mask = (df["TIME"] < start24) & (df["__MAINTENANCE__"] == 0)
                    m["NORMAL_MIN"] += int(normal_mask.sum())

                    pre = (df["TIME"] >= start24) & (df["TIME"] < event_at) & (df["__MAINTENANCE__"] == 0)
                    pre_scores = df.loc[pre, "ANOMALY_SCORE"]
                    max_run, max_score, top_sensor, top_feature, reason = 0, np.nan, "", "", ""
                    if not pre_scores.empty:
                        idx = pre_scores.idxmax()
                        max_score = float(pre_scores.loc[idx])
                        top_sensor = str(df.loc[idx, "TOP_SENSOR"])
                        top_feature = str(df.loc[idx, "TOP_FEATURE"])
                        max_run = max_consecutive(pre_scores.to_numpy() >= threshold)
                    if not d24:
                        if max_run == 0:
                            reason = "임계값 미초과"
                        elif max_run < int(persistence):
                            reason = f"지속시간 부족({max_run}분)"
                        elif any(t < start24 for t in starts):
                            reason = "24시간보다 너무 이른 경보"
                        else:
                            reason = "유지보수/경보규칙 영향"

                    scenario_rows.append({
                        "SCENARIO_ID": sid, "THRESHOLD_%": int(pct), "PERSIST_MIN": int(persistence),
                        "EVENT_AT": event_at, "DETECTED_24H": int(d24), "DETECTED_6H": int(d6),
                        "LEAD_HOURS": lead, "MAX_PRE_EVENT_SCORE": max_score,
                        "MAX_PRE_EVENT_RUN_MIN": max_run, "TOP_SENSOR_AT_MAX": top_sensor,
                        "TOP_FEATURE_AT_MAX": top_feature, "MISS_REASON": reason,
                        "FALSE_ALARMS": len(false)
                    })
                else:
                    m["FP"] += len(starts)
                    m["NORMAL_MIN"] += int((df["__MAINTENANCE__"] == 0).sum())

        if i % 20 == 0 or i == len(eval_ids):
            print(f"[평가] {i}/{len(eval_ids)}")

    rows = []
    for (pct, p), m in metrics.items():
        events = m["EVENTS"]
        detected = m["D24"]
        days = m["NORMAL_MIN"] / 1440
        rows.append({
            "THRESHOLD_%": pct,
            "PERSIST_MIN": p,
            "THRESHOLD_SCORE": thresholds[pct],
            "EVENTS": events,
            "DETECTED_24H": detected,
            "MISSED_24H": events - detected,
            "RECALL_24H_%": 100 * detected / events if events else np.nan,
            "DETECTED_6H": m["D6"],
            "RECALL_6H_%": 100 * m["D6"] / events if events else np.nan,
            "MEDIAN_LEAD_H": float(np.median(m["LEADS"])) if m["LEADS"] else np.nan,
            "FALSE_ALARMS": m["FP"],
            "FALSE_ALARMS_DAY": m["FP"] / days if days else np.nan
        })

    return (
        pd.DataFrame(rows).sort_values(["THRESHOLD_%", "PERSIST_MIN"]).reset_index(drop=True),
        pd.DataFrame(scenario_rows)
    )


def choose_operating_point(result, config):
    target = float(config["target_recall_24h"])
    valid = result[result["RECALL_24H_%"] >= target].copy()
    if not valid.empty:
        row = valid.sort_values(["FALSE_ALARMS_DAY", "MEDIAN_LEAD_H", "THRESHOLD_%"], ascending=[True, False, False]).iloc[0]
        reason = f"탐지율 {target:.1f}% 이상 조합 중 오경보/일 최소"
    else:
        row = result.sort_values(["RECALL_24H_%", "FALSE_ALARMS_DAY"], ascending=[False, True]).iloc[0]
        reason = "목표 탐지율을 만족하는 조합이 없어 탐지율 최대 조합 선택"

    action_p = int(row["PERSIST_MIN"])
    return {
        "threshold_percentile": int(row["THRESHOLD_%"]),
        "threshold_score": float(row["THRESHOLD_SCORE"]),
        "action_persistence_min": action_p,
        "warning_persistence_min": max(1, action_p - 1),
        "selection_reason": reason,
        "selected_metrics": {
            "recall_24h_pct": float(row["RECALL_24H_%"]),
            "recall_6h_pct": float(row["RECALL_6H_%"]),
            "false_alarms_day": float(row["FALSE_ALARMS_DAY"]),
            "median_lead_h": float(row["MEDIAN_LEAD_H"]),
            "missed_24h": int(row["MISSED_24H"])
        }
    }


def infer_casting_equipment(file_map, config):
    configured = config["quality"]["casting_equipment"]
    if configured != "auto":
        return configured
    stage = str(config["quality"]["casting_stage"]).upper()
    values = []
    for process_file in list(file_map.values())[:30]:
        qfile = process_file.parent / config["quality_relative_path"]
        if not qfile.exists():
            continue
        q = read_csv(qfile)
        if not {"STAGE", "EQP_CD"}.issubset(q.columns):
            continue
        values += q.loc[q["STAGE"].astype(str).str.upper().eq(stage), "EQP_CD"].dropna().astype(str).tolist()
    return None if not values else pd.Series(values).mode().iloc[0]


def analyze_quality(sid, qfile, casting_stage, casting_eqp):
    q = read_csv(qfile)
    if any(c not in q.columns for c in QUALITY_REQUIRED):
        return None
    for c in ["GT_STAGE_COMPLETED", "GT_DEFECT_FLAG", "GT_FINAL_DEFECT_FLAG", "GT_FINAL_CENSORED"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    total = q["FLOW_ID"].dropna().nunique()
    any_defect = q.loc[q["GT_DEFECT_FLAG"] == 1, "FLOW_ID"].nunique()
    cdf = q[q["STAGE"].astype(str).str.upper().eq(str(casting_stage).upper())]
    casting_defect = cdf.loc[cdf["GT_DEFECT_FLAG"] == 1, "FLOW_ID"].nunique()
    origin = np.nan if casting_eqp is None else q.loc[q["GT_ORIGIN_EQP_CD"].astype(str).eq(str(casting_eqp)), "FLOW_ID"].nunique()

    final = q[q["GT_FINAL_CENSORED"].notna()].sort_values("FLOW_ID").drop_duplicates("FLOW_ID", keep="last")
    observed = final[(final["GT_FINAL_CENSORED"] == 0) & final["GT_FINAL_DEFECT_FLAG"].notna()]
    censored = final[final["GT_FINAL_CENSORED"] == 1]
    defect = int((observed["GT_FINAL_DEFECT_FLAG"] == 1).sum())

    return {
        "SCENARIO_ID": sid,
        "TOTAL_FLOWS": total,
        "ANY_STAGE_DEFECT_FLOWS": any_defect,
        "CASTING_DEFECT_FLOWS": casting_defect,
        "CASTING_ORIGIN_FLOWS": origin,
        "FINAL_OBSERVED_FLOWS": len(observed),
        "FINAL_CENSORED_FLOWS": len(censored),
        "FINAL_NORMAL_FLOWS": int((observed["GT_FINAL_DEFECT_FLAG"] == 0).sum()),
        "FINAL_DEFECT_FLOWS": defect,
        "FINAL_DEFECT_RATE_%": defect / len(observed) * 100 if len(observed) else np.nan,
        "HAS_FINAL_DEFECT": int(defect > 0),
        "HAS_CASTING_DEFECT": int(casting_defect > 0),
        "HAS_CASTING_ORIGIN": np.nan if pd.isna(origin) else int(origin > 0)
    }


def quality_summary(file_map, scenario_eval, op, config):
    if not config["quality"]["enabled"] or scenario_eval.empty:
        return None, None, None

    pct = int(op["threshold_percentile"])
    action_p = int(op["action_persistence_min"])
    baseline_p = int(config["comparison_baseline_persistence"])
    temp = scenario_eval[scenario_eval["THRESHOLD_%"] == pct]
    pivot = temp.pivot_table(index="SCENARIO_ID", columns="PERSIST_MIN", values="DETECTED_24H", aggfunc="max", fill_value=0)
    for c in [action_p, baseline_p]:
        if c not in pivot.columns:
            pivot[c] = 0
    recovered = set(pivot[(pivot[baseline_p] == 0) & (pivot[action_p] == 1)].index)
    missed = set(pivot[pivot[action_p] == 0].index)

    eqp = infer_casting_equipment(file_map, config)
    stage = config["quality"]["casting_stage"]
    rows = []
    for sid, pfile in file_map.items():
        qfile = pfile.parent / config["quality_relative_path"]
        if not qfile.exists():
            continue
        row = analyze_quality(sid, qfile, stage, eqp)
        if row is None:
            continue
        row["ALARM_RESULT"] = "본 경보 미탐" if sid in missed else ("기준 완화로 추가 탐지" if sid in recovered else "기타 Scenario")
        rows.append(row)

    if not rows:
        return None, None, eqp

    summary = pd.DataFrame(rows)
    comp = summary[summary["ALARM_RESULT"].isin(["본 경보 미탐", "기준 완화로 추가 탐지"])]
    if comp.empty:
        group = pd.DataFrame()
    else:
        group = comp.groupby("ALARM_RESULT").agg(
            SCENARIOS=("SCENARIO_ID", "nunique"),
            FINAL_DEFECT_SCENARIOS=("HAS_FINAL_DEFECT", "sum"),
            CASTING_DEFECT_SCENARIOS=("HAS_CASTING_DEFECT", "sum"),
            TOTAL_FINAL_OBSERVED=("FINAL_OBSERVED_FLOWS", "sum"),
            TOTAL_FINAL_DEFECT=("FINAL_DEFECT_FLOWS", "sum")
        ).reset_index()
        group["SCENARIO_FINAL_DEFECT_RATE_%"] = group["FINAL_DEFECT_SCENARIOS"] / group["SCENARIOS"] * 100
        group["FLOW_FINAL_DEFECT_RATE_%"] = group["TOTAL_FINAL_DEFECT"] / group["TOTAL_FINAL_OBSERVED"] * 100
    return summary, group, eqp



# ============================================================
# 출력 정리
# Excel 없이 JSON + CSV만 생성
# ============================================================

def prepare_output_dir(out_dir: Path):
    """
    실행할 때마다 같은 파일을 덮어써서 결과가 계속 늘어나지 않게 합니다.

    model_output/
    ├─ anomaly_model.json
    ├─ threshold_persistence_comparison.csv
    ├─ quality_group_compare.csv       # quality.csv가 있을 때
    └─ details/
       ├─ ground_truth_summary.csv
       ├─ scenario_evaluation.csv
       └─ quality_summary.csv          # quality.csv가 있을 때
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # 예전 버전에서 만들어졌을 수 있는 파일 정리
    old_top_files = [
        "model_report.xlsx",
        "run_summary.json",
        "threshold_persistence_comparison.csv",
        "quality_summary.csv",
        "quality_group_compare.csv",
        "ground_truth_summary.csv",
        "scenario_evaluation.csv",
    ]

    for name in old_top_files:
        path = out_dir / name
        if path.exists():
            path.unlink()

    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    # 상세 결과도 매번 같은 이름으로 덮어쓰기
    for name in [
        "ground_truth_summary.csv",
        "scenario_evaluation.csv",
        "quality_summary.csv",
    ]:
        path = details_dir / name
        if path.exists():
            path.unlink()

    return details_dir

def build_training_summary(
    event_counts,
    event_type,
    train_ids,
    eval_ids,
    q_group,
    casting_eqp,
):
    quality_compare = []
    if q_group is not None and not q_group.empty:
        quality_compare = q_group.to_dict(orient="records")

    return {
        "event_counts": event_counts,
        "selected_event_type": event_type,
        "train_scenarios": len(train_ids),
        "eval_scenarios": len(eval_ids),
        "casting_equipment": casting_eqp,
        "quality_group_compare": quality_compare,
    }

def save_model(path, config, schema, event_type, features, median, scale, thresholds, op, casting_eqp, training_summary=None):
    model = {
        "model_type": "RobustNormalRangeAnomalyModel",
        "version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "feature_config": {
            "resample_rule": config["resample_rule"],
            "short_window": config["short_window"],
            "long_window": config["long_window"],
            "top_k": config["top_k"]
        },
        "event_type_used_for_evaluation": event_type,
        "feature_columns": features,
        "normal_range": {
            "median": {c: float(median[c]) for c in features},
            "scale": {c: float(scale[c]) for c in features}
        },
        "candidate_thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "operating_point": op,
        "alarm_config": {"alarm_merge_minutes": config["alarm_merge_minutes"]},
        "quality": {"casting_stage": config["quality"]["casting_stage"], "casting_equipment": casting_eqp},
        "training_summary": training_summary or {}
    }
    save_json(model, path)


def load_model(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_file(file: Path, model, config):
    schema = model["schema"]
    header = read_csv(file, nrows=5)
    missing = [c for c in schema["sensor_columns"] if c not in header.columns]
    if missing:
        raise ValueError(f"학습 모델과 센서 구조가 다릅니다: {missing}\nGT가 있는 새 데이터셋이면 train 모드로 재학습하세요.")

    config = deep_update(config, model["feature_config"])
    config["alarm_merge_minutes"] = model["alarm_config"]["alarm_merge_minutes"]
    df = build_features(file, schema, config, include_gt=False)
    features = model["feature_columns"]
    median = pd.Series(model["normal_range"]["median"])[features]
    scale = pd.Series(model["normal_range"]["scale"])[features]
    scores, top_features, top_sensors = anomaly_details(df, features, median, scale, config["top_k"])
    op = model["operating_point"]
    threshold = float(op["threshold_score"])
    warning = alarm_mask(scores, threshold, int(op["warning_persistence_min"]))
    action = alarm_mask(scores, threshold, int(op["action_persistence_min"]))

    return pd.DataFrame({
        "TIME": df["TIME"],
        "ANOMALY_SCORE": scores,
        "TOP_FEATURE": top_features,
        "TOP_SENSOR": top_sensors,
        "WARNING_ALARM": warning.astype(int).to_numpy(),
        "ACTION_ALARM": action.astype(int).to_numpy()
    })



def train_pipeline(data_root: Path, config, out_dir: Path):

    # --------------------------------------------------------
    # 출력 폴더 준비
    # --------------------------------------------------------
    details_dir = prepare_output_dir(out_dir)

    # --------------------------------------------------------
    # 1. 파일 / 컬럼 자동 탐색
    # --------------------------------------------------------
    files = discover_process_files(
        data_root,
        config["process_file_name"],
    )

    file_map = dict(files)

    schema = detect_schema(
        files,
        config,
    )

    print("[자동 탐색] 시간 컬럼:", schema["time_column"])
    print("[자동 탐색] 센서 컬럼:", schema["sensor_columns"])

    # --------------------------------------------------------
    # 2. 실제 이상 Event 자동 확인
    # --------------------------------------------------------
    summary = profile_ground_truth(
        files,
        schema,
        config,
    )

    summary, event_type, event_counts = select_event_type(
        summary,
        config,
    )

    # --------------------------------------------------------
    # 3. 정상 학습 / 평가 Scenario 분리
    # --------------------------------------------------------
    train_ids, eval_ids = split_scenarios(
        summary,
        config,
    )

    # --------------------------------------------------------
    # 4. 정상 데이터 Feature 생성
    # --------------------------------------------------------
    train, features = build_normal_dataset(
        file_map,
        train_ids,
        event_type,
        schema,
        config,
    )

    # --------------------------------------------------------
    # 5. Robust 정상범위 학습
    #    Median + MAD
    # --------------------------------------------------------
    median, scale = fit_normal_range(
        train,
        features,
    )

    # --------------------------------------------------------
    # 6. 95 / 97 / 99% Threshold 자동 생성
    # --------------------------------------------------------
    thresholds = build_thresholds(
        train,
        features,
        median,
        scale,
        config,
    )

    # --------------------------------------------------------
    # 7. Threshold × 1/2/3분 자동 비교
    # --------------------------------------------------------
    result, scenario_eval = evaluate(
        file_map,
        summary,
        eval_ids,
        event_type,
        features,
        median,
        scale,
        thresholds,
        schema,
        config,
    )

    # --------------------------------------------------------
    # 8. 운영 조건 자동 선택
    # --------------------------------------------------------
    op = choose_operating_point(
        result,
        config,
    )

    # --------------------------------------------------------
    # 9. quality.csv가 있으면 품질 자동 연결
    # --------------------------------------------------------
    q_summary, q_group, casting_eqp = quality_summary(
        file_map,
        scenario_eval,
        op,
        config,
    )

    training_summary = build_training_summary(
        event_counts,
        event_type,
        train_ids,
        eval_ids,
        q_group,
        casting_eqp,
    )

    # ========================================================
    # 최종 저장
    # ========================================================

    # --------------------------------------------------------
    # A. 실제 모델 JSON
    # --------------------------------------------------------
    model_path = out_dir / "anomaly_model.json"

    save_model(
        model_path,
        config,
        schema,
        event_type,
        features,
        median,
        scale,
        thresholds,
        op,
        casting_eqp,
        training_summary=training_summary,
    )

    # --------------------------------------------------------
    # B. 핵심 비교 결과 CSV
    # --------------------------------------------------------
    comparison_path = (
        out_dir
        / "threshold_persistence_comparison.csv"
    )

    result.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # C. 품질 비교 핵심 결과 CSV
    # --------------------------------------------------------
    quality_compare_path = None

    if q_group is not None and not q_group.empty:

        quality_compare_path = (
            out_dir
            / "quality_group_compare.csv"
        )

        q_group.to_csv(
            quality_compare_path,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # D. 상세 자료
    # --------------------------------------------------------
    gt_path = (
        details_dir
        / "ground_truth_summary.csv"
    )

    summary.to_csv(
        gt_path,
        index=False,
        encoding="utf-8-sig",
    )

    scenario_path = (
        details_dir
        / "scenario_evaluation.csv"
    )

    scenario_eval.to_csv(
        scenario_path,
        index=False,
        encoding="utf-8-sig",
    )

    quality_detail_path = None

    if q_summary is not None and not q_summary.empty:

        quality_detail_path = (
            details_dir
            / "quality_summary.csv"
        )

        q_summary.to_csv(
            quality_detail_path,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("자동 모델링 완료")
    print("=" * 90)

    print(
        f"본 경보: "
        f"{op['threshold_percentile']}% + "
        f"{op['action_persistence_min']}분"
    )

    print(
        f"주의 경보: "
        f"{op['threshold_percentile']}% + "
        f"{op['warning_persistence_min']}분"
    )

    print(
        f"24시간 탐지율: "
        f"{op['selected_metrics']['recall_24h_pct']:.2f}%"
    )

    print(
        f"오경보/일: "
        f"{op['selected_metrics']['false_alarms_day']:.3f}"
    )

    print("\n생성 파일")
    print("  1.", model_path)
    print("  2.", comparison_path)

    if quality_compare_path is not None:
        print("  3.", quality_compare_path)

    print("\n상세 파일")
    print("  -", gt_path)
    print("  -", scenario_path)

    if quality_detail_path is not None:
        print("  -", quality_detail_path)

    return model_path

def predict_pipeline(input_path: Path, model_path: Path, config, out_dir: Path):
    """신규 파일이 여러 개여도 prediction_results.csv 하나로 저장합니다."""

    out_dir.mkdir(parents=True, exist_ok=True)

    # 예전 prediction 결과 정리
    for old in out_dir.glob("*_anomaly_prediction.csv"):
        old.unlink()

    old_summary = out_dir / "prediction_summary.json"
    if old_summary.exists():
        old_summary.unlink()

    model = load_model(model_path)
    files = discover_process_files(
        input_path,
        config["process_file_name"],
    )

    parts = []

    for sid, file in files:
        pred = predict_file(file, model, config)
        pred.insert(0, "SCENARIO_ID", sid)
        parts.append(pred)
        print(f"[예측] {sid}")

    if not parts:
        raise ValueError("예측할 파일이 없습니다.")

    result = pd.concat(parts, ignore_index=True)
    output_file = out_dir / "prediction_results.csv"

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("예측 결과:", output_file)
    return output_file



# ============================================================
# 실행 설정
# VS Code에서 ▶ 실행 버튼만 누르면 됩니다.
# ============================================================

BaseDir = Path(__file__).resolve().parent

# 1) 학습할 Scenario 폴더
DataDir = BaseDir / "Data" / "Processed" / "Scenarios"

# 2) 설정 JSON
# 파일이 없으면 코드 안의 DEFAULT_CONFIG를 사용합니다.
ConfigFile = BaseDir / "anomaly_config.json"

# 3) 학습 결과 저장 폴더
# 최상위에는 모델 JSON과 핵심 CSV만 생성됩니다.
ModelOutputDir = BaseDir / "model_output"

# ------------------------------------------------------------
# 실행 모드
# ------------------------------------------------------------
# "train"   : 새 데이터셋으로 조건 자동 재설정 + 모델 학습
# "predict" : 저장된 모델로 신규 C.csv 이상탐지
RunMode = "train"

# predict 모드에서만 사용
NewDataPath = BaseDir / "new_data"
ModelFile = ModelOutputDir / "anomaly_model.json"
PredictionOutputDir = BaseDir / "prediction_output"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("연속주조 자동 이상탐지 모델")
    print("=" * 90)

    # --------------------------------------------------------
    # 설정 파일
    # --------------------------------------------------------
    if ConfigFile.exists():

        print("설정 JSON 사용:", ConfigFile)

        config = load_config(
            ConfigFile
        )

    else:

        print("설정 JSON 없음 -> 기본 설정 사용")

        config = load_config(
            None
        )

    # --------------------------------------------------------
    # 학습
    # --------------------------------------------------------
    if RunMode.lower() == "train":

        print("\n[MODE] TRAIN")
        print("데이터:", DataDir)
        print("결과:", ModelOutputDir)

        if not DataDir.exists():

            raise FileNotFoundError(
                f"Scenario 폴더를 찾을 수 없습니다.\n"
                f"{DataDir}"
            )

        train_pipeline(
            DataDir,
            config,
            ModelOutputDir,
        )

    # --------------------------------------------------------
    # 신규 데이터 예측
    # --------------------------------------------------------
    elif RunMode.lower() == "predict":

        print("\n[MODE] PREDICT")
        print("신규 데이터:", NewDataPath)
        print("모델:", ModelFile)

        if not NewDataPath.exists():

            raise FileNotFoundError(
                f"신규 데이터 경로를 찾을 수 없습니다.\n"
                f"{NewDataPath}"
            )

        if not ModelFile.exists():

            raise FileNotFoundError(
                f"학습된 모델 JSON을 찾을 수 없습니다.\n"
                f"{ModelFile}\n\n"
                f"먼저 RunMode = 'train' 으로 실행하세요."
            )

        predict_pipeline(
            NewDataPath,
            ModelFile,
            config,
            PredictionOutputDir,
        )

    else:

        raise ValueError(
            "RunMode는 'train' 또는 'predict'만 가능합니다."
        )


if __name__ == "__main__":
    main()
