from pathlib import Path
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 폰트
# =========================================================
plt.rcParams["font.family"] = (
    "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
)
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 경로
# CH.4/practice.py 기준
# =========================================================
BaseDir = Path(__file__).resolve().parent
ScenarioDir = BaseDir / "Data" / "Processed" / "Scenarios"

StartScenario = 1
EndScenario = 360
TargetFileName = "C.csv"

# =========================================================
# 연속주조 센서
# GT/상태 컬럼은 이상점수 입력에 넣지 않음
# =========================================================
SensorCols = [
    "LOAD",
    "WD-SPD",
    "WD-LOD",
    "OSC-FRQ",
    "OSC-STK",
    "MLD-LVL",
    "CONTROL_DELAY",
]

# =========================================================
# 분석 설정
# =========================================================
ResampleRule = "1min"       # 5초 -> 1분
ShortWindow = 5             # 5분 평균/표준편차
LongWindow = 15             # 15분 추세
TopK = 3                    # 가장 큰 이상 Feature 3개 평균

ThresholdPercentiles = [95, 97, 99]
PersistenceOptions = [1, 2, 3]       # ★ 1분 / 2분 / 3분 비교
AlarmMergeMinutes = 30
MaxLeadHours = 24
ShortLeadHours = 6

TrainScenarioRatio = 0.7
MaxTrainRowsPerScenario = 800
RandomState = 42

# =========================================================
# 비용 시나리오
# 공개자료를 바탕으로 만든 "비교용 추정치"이며 실제 사내 원가가 아님.
# 아래 값은 필요하면 바꾸면 됨.
# =========================================================
MISS_COST_PER_EVENT_WON = 44_200_000
# 설명: 연속주조 1시간 생산 차질을 영업이익 기준으로 환산한 보수적 Proxy.

FALSE_ALARM_CHECK_COST_WON = 200_000
# 설명: 오경보 1건당 현장 확인/점검비용 가정값.
# 실제 인원, 시간, 외주비가 있으면 이 값만 교체.

# =========================================================
# Scenario 파일
# =========================================================
def GetScenarioFiles():
    Files = []

    for Number in range(StartScenario, EndScenario + 1):
        ScenarioId = f"S{Number:04d}"
        File = ScenarioDir / ScenarioId / TargetFileName

        if File.exists():
            Files.append((ScenarioId, File))

    print("=" * 100)
    print("연속주조 C.csv 탐색")
    print("=" * 100)
    print("발견 파일:", len(Files))

    if Files:
        print("첫 Scenario:", Files[0][0])
        print("마지막 Scenario:", Files[-1][0])

    return Files


def ReadCsv(File, UseCols=None):
    return pd.read_csv(File, usecols=UseCols, low_memory=False)


def HasValue(Series):
    Text = (
        Series.astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )

    return (~Text.isin(["", "nan", "none", "null", "<na>"])).fillna(False)


# =========================================================
# Ground Truth
# =========================================================
def GetGroundTruth(Df):
    Health = (
        Df["GT_HEALTH_STATE"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    Maintenance = (
        pd.to_numeric(Df["IS_MAINTENANCE"], errors="coerce")
        .fillna(0).eq(1)
        |
        Health.eq("maintenance").fillna(False)
    ).fillna(False)

    Running = (
        pd.to_numeric(Df["RUNNING"], errors="coerce")
        .fillna(0).eq(1)
    ).fillna(False)

    Missing = (
        pd.to_numeric(Df["IS_MISSING"], errors="coerce")
        .fillna(0).eq(1)
    ).fillna(False)

    Stopped = (
        pd.to_numeric(Df["IS_STOPPED"], errors="coerce")
        .fillna(0).eq(1)
    ).fillna(False)

    Outlier = (
        pd.to_numeric(Df["IS_OUTLIER"], errors="coerce")
        .fillna(0).eq(1)
    ).fillna(False)

    FaultMode = HasValue(Df["GT_FAULT_MODE"]).fillna(False)

    AbnormalHealth = (
        Health.notna()
        & ~Health.isin(["normal", "maintenance"]).fillna(False)
    ).fillna(False)

    Fault = (
        ~Maintenance
        & (FaultMode | AbnormalHealth)
    ).fillna(False)

    UnplannedStop = (
        Stopped & ~Maintenance
    ).fillna(False)

    QualityOk = (
        Df["QUALITY_FLAG"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("ok")
        .fillna(False)
    )

    StrictNormal = (
        Health.eq("normal").fillna(False)
        & Running
        & ~Maintenance
        & ~Missing
        & ~Outlier
        & ~UnplannedStop
        & ~Fault
        & QualityOk
    ).fillna(False)

    return {
        "NORMAL": StrictNormal.astype(bool),
        "FAULT": Fault.astype(bool),
        "UNPLANNED_STOP": UnplannedStop.astype(bool),
        "OUTLIER": (Outlier & ~Maintenance).fillna(False).astype(bool),
        "MAINTENANCE": Maintenance.astype(bool),
    }


def ProfileGroundTruth(Files):
    UseCols = [
        "MEAS_DT", "RUNNING", "QUALITY_FLAG",
        "IS_MISSING", "IS_STOPPED", "IS_MAINTENANCE", "IS_OUTLIER",
        "GT_HEALTH_STATE", "GT_FAULT_MODE",
    ]

    Rows = []

    for Index, (ScenarioId, File) in enumerate(Files, 1):
        Df = ReadCsv(File, UseCols)
        Time = pd.to_datetime(Df["MEAS_DT"], errors="coerce", utc=True)
        GT = GetGroundTruth(Df)

        def FirstTime(Mask):
            Mask = Mask.fillna(False).astype(bool)
            Temp = Time[Mask & Time.notna()]
            return pd.NaT if Temp.empty else Temp.min()

        Rows.append({
            "SCENARIO_ID": ScenarioId,
            "FAULT_AT": FirstTime(GT["FAULT"]),
            "STOP_AT": FirstTime(GT["UNPLANNED_STOP"]),
            "OUTLIER_AT": FirstTime(GT["OUTLIER"]),
        })

        if Index % 30 == 0 or Index == len(Files):
            print(f"Ground Truth: {Index}/{len(Files)}")

    return pd.DataFrame(Rows)


def SelectEventType(Summary):
    FaultCount = Summary["FAULT_AT"].notna().sum()
    StopCount = Summary["STOP_AT"].notna().sum()
    OutlierCount = Summary["OUTLIER_AT"].notna().sum()

    print("\n실제 이상 후보")
    print("Fault Scenario:", FaultCount)
    print("비계획 정지 Scenario:", StopCount)
    print("Outlier Scenario:", OutlierCount)

    if FaultCount >= 10:
        EventType, EventCol = "FAULT", "FAULT_AT"
    elif StopCount >= 10:
        EventType, EventCol = "UNPLANNED_STOP", "STOP_AT"
    elif OutlierCount >= 10:
        EventType, EventCol = "OUTLIER", "OUTLIER_AT"
    else:
        raise ValueError("분석 가능한 실제 이상 Event가 부족합니다.")

    Summary = Summary.copy()
    Summary["EVENT_AT"] = Summary[EventCol]
    Summary["HAS_EVENT"] = Summary["EVENT_AT"].notna()

    print("선택된 실제 이상 기준:", EventType)
    print("Event Scenario:", Summary["HAS_EVENT"].sum())

    return Summary, EventType


# =========================================================
# 1분 Feature
# =========================================================
def BuildFeatures(File, EventType):
    UseCols = [
        "MEAS_DT", *SensorCols,
        "RUNNING", "QUALITY_FLAG",
        "IS_MISSING", "IS_STOPPED", "IS_MAINTENANCE", "IS_OUTLIER",
        "GT_HEALTH_STATE", "GT_FAULT_MODE",
    ]

    Df = ReadCsv(File, UseCols)
    Df["MEAS_DT"] = pd.to_datetime(Df["MEAS_DT"], errors="coerce", utc=True)
    Df = Df.dropna(subset=["MEAS_DT"])

    for Sensor in SensorCols:
        Df[Sensor] = pd.to_numeric(Df[Sensor], errors="coerce")

    GT = GetGroundTruth(Df)

    Df["__NORMAL__"] = GT["NORMAL"].fillna(False).astype(bool).astype(int)
    Df["__MAINTENANCE__"] = GT["MAINTENANCE"].fillna(False).astype(bool).astype(int)

    EventMask = GT[EventType]
    Df["__EVENT__"] = EventMask.fillna(False).astype(bool).astype(int)

    Df = Df.sort_values("MEAS_DT").set_index("MEAS_DT")

    SensorMinute = Df[SensorCols].resample(ResampleRule).median()

    Meta = (
        Df[["__NORMAL__", "__MAINTENANCE__", "__EVENT__"]]
        .resample(ResampleRule)
        .agg({
            "__NORMAL__": "min",
            "__MAINTENANCE__": "max",
            "__EVENT__": "max",
        })
    )

    Current = SensorMinute.copy()
    Current.columns = [f"CUR__{Col}" for Col in Current.columns]

    Diff = SensorMinute.diff()
    Diff.columns = [f"DIFF__{Col}" for Col in Diff.columns]

    Mean5 = SensorMinute.rolling(ShortWindow, min_periods=1).mean()
    Mean5.columns = [f"MEAN5__{Col}" for Col in Mean5.columns]

    Std5 = SensorMinute.rolling(ShortWindow, min_periods=2).std()
    Std5.columns = [f"STD5__{Col}" for Col in Std5.columns]

    Slope15 = (SensorMinute - SensorMinute.shift(LongWindow)) / LongWindow
    Slope15.columns = [f"SLOPE15__{Col}" for Col in Slope15.columns]

    Result = pd.concat([Current, Diff, Mean5, Std5, Slope15, Meta], axis=1)
    return Result.reset_index().rename(columns={"MEAS_DT": "TIME"})


def GetFeatureCols(Df):
    Prefixes = ("CUR__", "DIFF__", "MEAN5__", "STD5__", "SLOPE15__")
    return [Col for Col in Df.columns if Col.startswith(Prefixes)]


# =========================================================
# Train / Eval
# =========================================================
def SplitScenarios(Summary):
    Rng = np.random.default_rng(RandomState)

    EventIds = Summary.loc[Summary["HAS_EVENT"], "SCENARIO_ID"].tolist()
    NormalIds = Summary.loc[~Summary["HAS_EVENT"], "SCENARIO_ID"].tolist()

    NormalIds = np.array(NormalIds)
    Rng.shuffle(NormalIds)

    TrainCount = max(1, int(len(NormalIds) * TrainScenarioRatio))
    TrainIds = NormalIds[:TrainCount].tolist()
    TestNormalIds = NormalIds[TrainCount:].tolist()
    EvalIds = EventIds + TestNormalIds

    print("\n정상범위 학습 Scenario:", len(TrainIds))
    print("평가 Scenario:", len(EvalIds))
    print("실제 Event Scenario:", len(EventIds))

    return TrainIds, EvalIds


def BuildNormalDataset(FileMap, TrainIds, EventType):
    Parts = []
    FeatureCols = None

    for Index, ScenarioId in enumerate(TrainIds, 1):
        Df = BuildFeatures(FileMap[ScenarioId], EventType)

        if FeatureCols is None:
            FeatureCols = GetFeatureCols(Df)

        Normal = Df[Df["__NORMAL__"] == 1].copy()
        if Normal.empty:
            continue

        if len(Normal) > MaxTrainRowsPerScenario:
            Normal = Normal.sample(
                n=MaxTrainRowsPerScenario,
                random_state=RandomState + Index,
            )

        Parts.append(Normal[FeatureCols])

        if Index % 20 == 0 or Index == len(TrainIds):
            print(f"정상 학습: {Index}/{len(TrainIds)}")

    if not Parts:
        raise ValueError("정상 학습 데이터를 만들 수 없습니다.")

    TrainDf = pd.concat(Parts, ignore_index=True)
    FeatureCols = [Col for Col in FeatureCols if TrainDf[Col].notna().any()]

    print("정상 학습 행:", f"{len(TrainDf):,}")
    print("Feature 수:", len(FeatureCols))

    return TrainDf, FeatureCols


# =========================================================
# 정상범위 / 이상점수
# =========================================================
def FitNormalRange(TrainDf, FeatureCols):
    Median = TrainDf[FeatureCols].median()
    MAD = TrainDf[FeatureCols].sub(Median).abs().median()
    Std = TrainDf[FeatureCols].std()

    Scale = 1.4826 * MAD
    Scale = Scale.where(Scale > 1e-9, Std)
    Scale = Scale.fillna(1).replace(0, 1)

    return Median, Scale


def CalculateAnomalyDetails(Df, FeatureCols, Median, Scale):
    """
    종합 이상점수 + 해당 시점에서 가장 크게 벗어난 Feature/Sensor를 함께 반환.
    예: STD5__WD-LOD -> WD-LOD 센서가 주요 원인 후보.
    """
    Z = (
        Df[FeatureCols]
        .sub(Median)
        .div(Scale)
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(upper=10)
    )

    Values = Z.to_numpy()
    K = min(TopK, Values.shape[1])
    TopValues = np.partition(Values, -K, axis=1)[:, -K:]
    Scores = TopValues.mean(axis=1)

    TopFeature = Z.idxmax(axis=1).astype(str)
    TopSensor = TopFeature.str.split("__", n=1).str[-1]

    return Scores, TopFeature.to_numpy(), TopSensor.to_numpy()


def CalculateAnomalyScore(Df, FeatureCols, Median, Scale):
    Scores, _, _ = CalculateAnomalyDetails(
        Df, FeatureCols, Median, Scale
    )
    return Scores


def BuildThresholds(TrainDf, FeatureCols, Median, Scale):
    Scores = CalculateAnomalyScore(TrainDf, FeatureCols, Median, Scale)

    Thresholds = {
        P: float(np.percentile(Scores, P))
        for P in ThresholdPercentiles
    }

    print("\n정상 이상점수 기준")
    for P, Threshold in Thresholds.items():
        print(f"{P}% 경계: {Threshold:.3f}")

    return Thresholds


# =========================================================
# Alarm
# =========================================================
def BuildAlarmMask(Scores, Threshold, PersistenceMinutes):
    Above = pd.Series(np.asarray(Scores) >= Threshold)

    return (
        Above.rolling(
            PersistenceMinutes,
            min_periods=PersistenceMinutes,
        ).sum()
        >= PersistenceMinutes
    )


def GetAlarmStarts(Df, AlarmMask):
    Start = AlarmMask & ~AlarmMask.shift(fill_value=False)

    Candidate = Df.loc[
        Start.values & (Df["__MAINTENANCE__"] == 0),
        "TIME",
    ]

    Times = Candidate.dropna().sort_values().tolist()
    if not Times:
        return []

    Merged = [Times[0]]
    Gap = pd.Timedelta(minutes=AlarmMergeMinutes)

    for Time in Times[1:]:
        if Time - Merged[-1] > Gap:
            Merged.append(Time)

    return Merged


def MaxConsecutiveTrue(Mask):
    Best = 0
    Current = 0

    for Value in np.asarray(Mask, dtype=bool):
        if Value:
            Current += 1
            Best = max(Best, Current)
        else:
            Current = 0

    return Best


# =========================================================
# 1/2/3분 지속시간 비교 평가
# =========================================================
def EvaluatePersistence(
    FileMap,
    Summary,
    EvalIds,
    EventType,
    FeatureCols,
    Median,
    Scale,
    Thresholds,
):
    SummaryMap = Summary.set_index("SCENARIO_ID")

    Metrics = {
        (P, Persistence): {
            "EVENTS": 0,
            "DETECTED_24H": 0,
            "DETECTED_6H": 0,
            "LEADS": [],
            "FALSE_ALARMS": 0,
            "NORMAL_MINUTES": 0,
        }
        for P in Thresholds
        for Persistence in PersistenceOptions
    }

    ScenarioRows = []

    print("\n" + "=" * 100)
    print("1분 / 2분 / 3분 경보 지속조건 비교")
    print("=" * 100)

    for Index, ScenarioId in enumerate(EvalIds, 1):
        Df = BuildFeatures(FileMap[ScenarioId], EventType)
        Scores, TopFeatures, TopSensors = CalculateAnomalyDetails(
            Df, FeatureCols, Median, Scale
        )
        Df["ANOMALY_SCORE"] = Scores
        Df["TOP_FEATURE"] = TopFeatures
        Df["TOP_SENSOR"] = TopSensors

        EventAt = SummaryMap.loc[ScenarioId, "EVENT_AT"]
        HasEvent = pd.notna(EventAt)

        for P, Threshold in Thresholds.items():
            for Persistence in PersistenceOptions:
                Key = (P, Persistence)
                M = Metrics[Key]

                AlarmMask = BuildAlarmMask(Scores, Threshold, Persistence)
                AlarmStarts = GetAlarmStarts(Df, AlarmMask)

                Detected24 = False
                Detected6 = False
                LeadHours = np.nan
                MissReason = ""
                MaxRun = np.nan
                MaxPreScore = np.nan
                TopSensorAtMax = ""
                TopFeatureAtMax = ""

                if HasEvent:
                    M["EVENTS"] += 1

                    Start24 = EventAt - pd.Timedelta(hours=MaxLeadHours)
                    Start6 = EventAt - pd.Timedelta(hours=ShortLeadHours)

                    Valid24 = [T for T in AlarmStarts if Start24 <= T < EventAt]
                    Valid6 = [T for T in AlarmStarts if Start6 <= T < EventAt]

                    if Valid24:
                        Detected24 = True
                        M["DETECTED_24H"] += 1
                        FirstAlarm = min(Valid24)
                        LeadHours = (EventAt - FirstAlarm).total_seconds() / 3600
                        M["LEADS"].append(LeadHours)

                    if Valid6:
                        Detected6 = True
                        M["DETECTED_6H"] += 1

                    FalseAlarms = [T for T in AlarmStarts if T < Start24]
                    M["FALSE_ALARMS"] += len(FalseAlarms)

                    NormalMask = (
                        (Df["TIME"] < Start24)
                        & (Df["__MAINTENANCE__"] == 0)
                    )
                    M["NORMAL_MINUTES"] += int(NormalMask.fillna(False).sum())

                    # -------------------------------------
                    # 미탐 원인 분석
                    # -------------------------------------
                    PreWindow = (
                        (Df["TIME"] >= Start24)
                        & (Df["TIME"] < EventAt)
                        & (Df["__MAINTENANCE__"] == 0)
                    )

                    PreScores = Df.loc[PreWindow, "ANOMALY_SCORE"]
                    if not PreScores.empty:
                        MaxIndex = PreScores.idxmax()
                        MaxPreScore = float(PreScores.loc[MaxIndex])
                        TopSensorAtMax = str(Df.loc[MaxIndex, "TOP_SENSOR"])
                        TopFeatureAtMax = str(Df.loc[MaxIndex, "TOP_FEATURE"])
                        AbovePre = PreScores.to_numpy() >= Threshold
                        MaxRun = MaxConsecutiveTrue(AbovePre)
                    else:
                        MaxRun = 0

                    if not Detected24:
                        EarlyAlarms = [T for T in AlarmStarts if T < Start24]

                        if MaxRun == 0:
                            MissReason = "임계값 미초과"
                        elif MaxRun < Persistence:
                            MissReason = f"지속시간 부족({MaxRun}분)"
                        elif EarlyAlarms:
                            MissReason = "24시간보다 너무 이른 경보"
                        else:
                            MissReason = "유지보수/경보규칙 영향"

                    ScenarioRows.append({
                        "SCENARIO_ID": ScenarioId,
                        "THRESHOLD_%": P,
                        "PERSIST_MIN": Persistence,
                        "EVENT_AT": EventAt,
                        "DETECTED_24H": int(Detected24),
                        "DETECTED_6H": int(Detected6),
                        "LEAD_HOURS": LeadHours,
                        "MAX_PRE_EVENT_SCORE": MaxPreScore,
                        "MAX_PRE_EVENT_RUN_MIN": MaxRun,
                        "TOP_SENSOR_AT_MAX": TopSensorAtMax,
                        "TOP_FEATURE_AT_MAX": TopFeatureAtMax,
                        "MISS_REASON": MissReason,
                        "FALSE_ALARMS": len(FalseAlarms),
                    })

                else:
                    M["FALSE_ALARMS"] += len(AlarmStarts)
                    NormalMask = Df["__MAINTENANCE__"] == 0
                    M["NORMAL_MINUTES"] += int(NormalMask.fillna(False).sum())

        if Index % 20 == 0 or Index == len(EvalIds):
            print(f"{Index}/{len(EvalIds)} Scenario 평가")

    # =====================================================
    # 최종 요약
    # =====================================================
    Rows = []

    for (P, Persistence), M in Metrics.items():
        Events = M["EVENTS"]
        Missed = Events - M["DETECTED_24H"]
        Lead = M["LEADS"]
        NormalDays = M["NORMAL_MINUTES"] / 1440

        Recall24 = 100 * M["DETECTED_24H"] / Events if Events else np.nan
        Recall6 = 100 * M["DETECTED_6H"] / Events if Events else np.nan
        FalsePerDay = M["FALSE_ALARMS"] / NormalDays if NormalDays else np.nan

        MissCost = Missed * MISS_COST_PER_EVENT_WON
        FalseCost = M["FALSE_ALARMS"] * FALSE_ALARM_CHECK_COST_WON
        TotalCost = MissCost + FalseCost

        Rows.append({
            "THRESHOLD_%": P,
            "PERSIST_MIN": Persistence,
            "THRESHOLD_SCORE": Thresholds[P],
            "EVENTS": Events,
            "DETECTED_24H": M["DETECTED_24H"],
            "MISSED_24H": Missed,
            "RECALL_24H_%": Recall24,
            "DETECTED_6H": M["DETECTED_6H"],
            "RECALL_6H_%": Recall6,
            "MEDIAN_LEAD_H": np.median(Lead) if Lead else np.nan,
            "MEAN_LEAD_H": np.mean(Lead) if Lead else np.nan,
            "FALSE_ALARMS": M["FALSE_ALARMS"],
            "FALSE_ALARMS_DAY": FalsePerDay,
            "MISS_COST_MKRW": MissCost / 1_000_000,
            "FALSE_CHECK_COST_MKRW": FalseCost / 1_000_000,
            "TOTAL_COST_MKRW": TotalCost / 1_000_000,
        })

    ResultDf = pd.DataFrame(Rows).sort_values(
        ["THRESHOLD_%", "PERSIST_MIN"]
    ).reset_index(drop=True)

    ScenarioDf = pd.DataFrame(ScenarioRows)

    return ResultDf, ScenarioDf


# =========================================================
# 비교 출력
# =========================================================
def PrintComparison(ResultDf, ScenarioDf):
    print("\n" + "=" * 150)
    print("Threshold × 1/2/3분 지속시간 비교")
    print("=" * 150)

    DisplayCols = [
        "THRESHOLD_%", "PERSIST_MIN",
        "DETECTED_24H", "MISSED_24H", "RECALL_24H_%",
        "DETECTED_6H", "RECALL_6H_%",
        "MEDIAN_LEAD_H",
        "FALSE_ALARMS", "FALSE_ALARMS_DAY",
        "TOTAL_COST_MKRW",
    ]

    print(
        ResultDf[DisplayCols]
        .round({
            "RECALL_24H_%": 2,
            "RECALL_6H_%": 2,
            "MEDIAN_LEAD_H": 2,
            "FALSE_ALARMS_DAY": 3,
            "TOTAL_COST_MKRW": 1,
        })
        .to_string(index=False)
    )

    # 97% 기준에서 3분은 놓쳤지만 1/2분에서는 잡힌 Scenario
    if not ScenarioDf.empty:
        Pivot = ScenarioDf[
            ScenarioDf["THRESHOLD_%"] == 97
        ].pivot_table(
            index="SCENARIO_ID",
            columns="PERSIST_MIN",
            values="DETECTED_24H",
            aggfunc="max",
            fill_value=0,
        )

        for Col in [1, 2, 3]:
            if Col not in Pivot.columns:
                Pivot[Col] = 0

        RecoveredBy1 = Pivot[(Pivot[3] == 0) & (Pivot[1] == 1)]
        RecoveredBy2 = Pivot[(Pivot[3] == 0) & (Pivot[2] == 1)]

        print("\n97% 기준: 3분에서는 미탐, 1분에서는 탐지된 Scenario:", len(RecoveredBy1))
        print(RecoveredBy1.index.tolist()[:30])

        print("97% 기준: 3분에서는 미탐, 2분에서는 탐지된 Scenario:", len(RecoveredBy2))
        print(RecoveredBy2.index.tolist()[:30])

        Miss3 = ScenarioDf[
            (ScenarioDf["THRESHOLD_%"] == 97)
            & (ScenarioDf["PERSIST_MIN"] == 3)
            & (ScenarioDf["DETECTED_24H"] == 0)
        ]

        if not Miss3.empty:
            print("\n97% / 3분 미탐 원인")
            print(Miss3["MISS_REASON"].value_counts().to_string())

            print("\n미탐 예시")
            print(
                Miss3[
                    [
                        "SCENARIO_ID", "MAX_PRE_EVENT_SCORE",
                        "MAX_PRE_EVENT_RUN_MIN", "TOP_SENSOR_AT_MAX",
                        "TOP_FEATURE_AT_MAX", "MISS_REASON",
                    ]
                ]
                .head(30)
                .round(3)
                .to_string(index=False)
            )


# =========================================================
# 그래프: 97% 기준 1/2/3분 비교
# =========================================================
def PlotPersistenceComparison(ResultDf):
    Temp = ResultDf[ResultDf["THRESHOLD_%"] == 97].sort_values("PERSIST_MIN")

    if Temp.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.plot(Temp["PERSIST_MIN"], Temp["RECALL_24H_%"], marker="o", label="24시간 탐지율")
    plt.plot(Temp["PERSIST_MIN"], Temp["RECALL_6H_%"], marker="o", label="6시간 탐지율")
    plt.xlabel("경보 지속조건 (분)")
    plt.ylabel("탐지율 (%)")
    plt.title("97% 이상점수 기준: 지속시간에 따른 탐지율")
    plt.xticks(PersistenceOptions)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(Temp["PERSIST_MIN"], Temp["FALSE_ALARMS_DAY"], marker="o")
    plt.xlabel("경보 지속조건 (분)")
    plt.ylabel("오경보 / day")
    plt.title("97% 이상점수 기준: 지속시간에 따른 오경보")
    plt.xticks(PersistenceOptions)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# Main
# =========================================================
def Main():
    Files = GetScenarioFiles()
    if not Files:
        raise ValueError("C.csv를 찾을 수 없습니다.")

    FileMap = dict(Files)

    Summary = ProfileGroundTruth(Files)
    Summary, EventType = SelectEventType(Summary)

    TrainIds, EvalIds = SplitScenarios(Summary)

    TrainDf, FeatureCols = BuildNormalDataset(
        FileMap,
        TrainIds,
        EventType,
    )

    Median, Scale = FitNormalRange(
        TrainDf,
        FeatureCols,
    )

    Thresholds = BuildThresholds(
        TrainDf,
        FeatureCols,
        Median,
        Scale,
    )

    ResultDf, ScenarioDf = EvaluatePersistence(
        FileMap,
        Summary,
        EvalIds,
        EventType,
        FeatureCols,
        Median,
        Scale,
        Thresholds,
    )

    PrintComparison(ResultDf, ScenarioDf)
    PlotPersistenceComparison(ResultDf)


if __name__ == "__main__":
    Main()
