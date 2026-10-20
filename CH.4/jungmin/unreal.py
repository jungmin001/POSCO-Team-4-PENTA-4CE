import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import average_precision_score

from jungmin import config, features, models


# 언리얼 디지털 트윈(정비 의사결정 솔루션)용 파일 만들기, CH.4 폴더에서 실행 (특징 표 outputs/features_{설비}.csv가 먼저 있어야 함, python jungmin/Main.py A B C D E1 E2 F)
#   python jungmin/unreal.py                         온톨로지 + 기본 가정값 + 설비 7개 위험 예측 + 계산 확인표를 outputs/unreal에 저장 (python jungmin/unreal.py C D 처럼 설비를 고르면 그 설비만 예측)
#   python jungmin/unreal.py costs=경로/costs.json    언리얼에서 수정한 가정값으로 계산 확인표만 다시 만듦
#   python jungmin/unreal.py feedback=경로/decision_log.json   언리얼의 결정 기록을 실제 결과와 맞춰 봄 (피드백)
#
# 흐름: 데이터 → 모델 예측(risk) → 언리얼에서 비교·판단(what-if) → 결정 기록(decision_log) → 실제 결과와 비교(feedback) → 모델 개선
#
# 저장 파일 (outputs/unreal)
#   ontology.json          설비·부품·공정 흐름·가능한 작업·권한 (객체, 속성, 관계, 액션)
#   costs.json             부품 가격·재고·정지 비용 등 가정값 기본값 (언리얼에서 수정하고 언리얼 Saved 폴더에 저장)
#   risk_{설비}.json       모델 정보(버전·성능·출처) + validation 시나리오별 6시간 간격 고장 확률·경보 단계·이유·정기 정비 일정
#   decisions_{설비}_check.csv  기본 가정값으로 계산한 선택지별 기대 비용 (언리얼 계산 결과와 맞춰 보는 용도)
#   feedback.csv           결정 기록 하나마다 실제 결과와 판정 (feedback 실행 때)
UnrealDir = config.OutDir / "unreal"
RiskEquips = config.Equipments

# 가정값 기본값 (임의값, 단위 만원): 정답 데이터의 비용 점수(교체 12, 복구 30, 정지 1시간 2, 고장 20)에 1점 = 100만원을 곱한 값
# PlannedDowntimeRatio: 미리 계획한 교체는 생산 일정을 조정할 수 있어 정지 비용이 이 비율만큼만 듦
Currency = "만원"
DefaultPart = {"Price": 1200, "Stock": 1, "LeadTimeHours": 24}
DefaultEquipment = {"DowntimeCostPerHour": 200, "ReplaceHours": 4, "RepairCost": 3000, "RepairHours": 8, "FailureLoss": 2000,
                    "PlannedDowntimeRatio": 0.5}
PlanHours = 6

# 경보 단계: 임계값 이상이면 alarm, 24시간 임계값의 WatchRatio배 이상이면 watch
WatchRatio = 0.5
ReasonCount = 3
ReasonFromTop = 20

# 권한: viewer 보기만, engineer 가정값 수정·정비 결정, admin 기본값 복원까지
Roles = {"viewer": "보기만", "engineer": "가정값 수정, 정비 결정, 현장 메모", "admin": "engineer 권한 + 가정값 기본값 복원"}
Actions = [
    {"Action": "EditAssumption", "Label": "가정값 수정", "Roles": ["engineer", "admin"], "Target": "Part/Equipment",
     "Params": ["Price", "Stock", "LeadTimeHours", *DefaultEquipment]},
    {"Action": "ScheduleReplacement", "Label": "부품 교체 예약", "Roles": ["engineer", "admin"], "Target": "Part",
     "Params": ["When (replace_now | replace_6h)", "Note"]},
    {"Action": "Wait", "Label": "대기 (계속 운전)", "Roles": ["engineer", "admin"], "Target": "Equipment", "Params": ["Note"]},
    {"Action": "AddFieldNote", "Label": "현장 메모 (실제 부품 상태 등)", "Roles": ["engineer", "admin"], "Target": "Equipment",
     "Params": ["Note"]},
    {"Action": "ResetAssumptions", "Label": "가정값 기본값 복원", "Roles": ["admin"], "Target": "All", "Params": []},
]


def Now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ScenarioIds(Split=None):
    with open(config.ScenarioDir / "index.csv", encoding="utf-8") as File:
        return [Row["scenario"] for Row in csv.DictReader(File) if Split is None or Row["split"] == Split]


def EqpTag(Folder, Name):
    return pd.read_csv(Folder / f"{Name}.csv", usecols=["EQP_CD"], nrows=1)["EQP_CD"].iloc[0]


# 공정 흐름: flow/operations에서 같은 FLOW_ID 안의 다음 설비 (가장 많이 나온 연결), 평균 대기 시간도 함께
def BuildFlow(Ids):
    Ops = pd.concat(pd.read_csv(config.ScenarioDir / Id / "flow" / "operations.csv") for Id in Ids)
    Ops = Ops.sort_values(["SCENARIO_ID", "FLOW_ID", "T_START_SEC"])
    Ops["NEXT"] = Ops.groupby(["SCENARIO_ID", "FLOW_ID"])["EQP_CD"].shift(-1)
    Edges = Ops.dropna(subset=["NEXT"]).groupby(["EQP_CD", "NEXT"]).size().reset_index(name="Count")
    Edges = Edges.sort_values("Count", ascending=False).drop_duplicates("EQP_CD")
    Wait = Ops.groupby("EQP_CD")["WAIT_HOURS"].mean()

    return {Row.EQP_CD: (Row.NEXT, float(Wait.get(Row.NEXT, 0.0))) for Row in Edges.itertuples()}


# 온톨로지: 객체(설비·부품) + 속성 + 관계(부품 장착, 공정 흐름) + 액션 + 권한
def BuildOntology():
    Train = ScenarioIds("train")
    First = config.ScenarioDir / Train[0]
    Parts = pd.read_csv(First / "history" / "parts.csv")
    Info = pd.read_csv(First / "history" / "equipment.csv").set_index("EQP_TAG")
    Tags = {Name: EqpTag(First, Name) for Name in config.Equipments}
    Names = {Tag: Name for Name, Tag in Tags.items()}
    Flow = BuildFlow(Train)

    Equipments, PartObjects, Relations = [], [], []
    for Name, Tag in Tags.items():
        Sample = pd.read_csv(First / f"{Name}.csv", nrows=200, usecols=lambda Col: not Col.startswith("GT_"))
        Equipments.append({"Id": Name, "EqpTag": Tag, "EqpName": Info.at[Tag, "EQP_NM"], "Plant": Info.at[Tag, "PLANT_CD"],
                           "Sensors": features.GetSensorCols(Sample),
                           "RiskFile": f"risk_{Name}.json" if Name in RiskEquips else ""})
        for Row in Parts[Parts["EQP_TAG"] == Tag].itertuples():
            PartObjects.append({"Id": f"{Name}.{Row.PART_CD}", "Equip": Name, "Part": Row.PART_CD, "PartName": Row.PART_NM})
            Relations.append({"Type": "InstalledOn", "From": f"{Name}.{Row.PART_CD}", "To": Name})
        if Tag in Flow:
            Next, Wait = Flow[Tag]
            Relations.append({"Type": "FeedsInto", "From": Name, "To": Names[Next], "AvgWaitHours": round(Wait, 2)})

    # 공정 순서: 앞 설비가 없는 설비에서 FeedsInto를 따라감
    Next = {Item["From"]: Item["To"] for Item in Relations if Item["Type"] == "FeedsInto"}
    Order = [Name for Name in Tags if Name not in Next.values()][:1]
    while Order and Order[-1] in Next and Next[Order[-1]] not in Order:
        Order.append(Next[Order[-1]])
    for Item in Equipments:
        Index = Order.index(Item["Id"]) if Item["Id"] in Order else -1
        Item["Stage"] = Index
        Item["Downstream"] = Order[Index + 1:] if Index >= 0 else []

    Manifest = json.loads((config.BaseDir.parent / "Data" / "Raw" / "manifest.json").read_text(encoding="utf-8"))

    return {"Version": 1, "GeneratedAt": Now(),
            "ObjectTypes": {
                "Equipment": "설비: Id, EqpTag, EqpName, Plant, Stage(공정 순서), Downstream(다음 공정 설비들), Sensors, RiskFile",
                "Part": "부품: Id(설비.부품), Equip, Part, PartName / 가정값(Price, Stock, LeadTimeHours)은 costs.json",
                "RiskPoint": "시점별 예측: risk_{설비}.json의 Points",
                "Decision": "정비 결정 기록: 언리얼 Saved/DSS/decision_log.json"},
            "Equipments": Equipments, "Parts": PartObjects, "Relations": Relations, "ProcessOrder": Order,
            "Actions": Actions, "Roles": Roles,
            "Lineage": {"Dataset": Manifest.get("datasetVersion"), "Synthetic": Manifest.get("synthetic"),
                        "Preprocessed": "CH.4/Data/Processed/Scenarios (CH.4/Main.py)",
                        "Sources": ["history/equipment.csv", "history/parts.csv", "flow/operations.csv"]}}


# 가정값: 설비·부품마다 임의 기본값, 학습 정답에서 가장 많이 교체한 부품을 기본 교체 부품으로
def BuildCosts(Ontology):
    Targets = pd.concat(pd.read_csv(config.ScenarioDir / Id / "targets.csv", usecols=["EQP_CD", "GT_POLICY_PART"])
                        for Id in ScenarioIds("train"))
    Common = Targets.dropna().groupby("EQP_CD")["GT_POLICY_PART"].agg(lambda Part: Part.value_counts().index[0])

    Equipments = []
    for Item in Ontology["Equipments"]:
        Parts = [Part for Part in Ontology["Parts"] if Part["Equip"] == Item["Id"]]
        Equipments.append({"Equip": Item["Id"], "EqpTag": Item["EqpTag"], "EqpName": Item["EqpName"], **DefaultEquipment,
                           "DefaultPart": Common.get(Item["EqpTag"], Parts[0]["Part"]),
                           "Parts": [{"Part": Part["Part"], "PartName": Part["PartName"], **DefaultPart} for Part in Parts]})

    return {"Version": 1, "Currency": Currency, "PlanHours": PlanHours, "UpdatedAt": Now(), "UpdatedBy": "default",
            "Note": "모든 값은 임의값(가정)입니다. 언리얼에서 수정하고 언리얼 Saved 폴더에 저장합니다.",
            "Equipments": Equipments}


# 정상 범위: train에서 24시간 안에 고장이 없던 시점의 특징 분포 (5~95%)
def NormalRange(Train, Cols):
    Normal = Train[Train["RISK_24H"] == 0][Cols]
    return Normal.quantile(0.05), Normal.quantile(0.95)


# 특징 이름 w6_MLD-LVL_std → (6, MLD-LVL, std), 센서가 아닌 특징(정비 경과 시간 등)은 센서 없이
def SplitFeature(Name):
    Head, _, Rest = Name.partition("_")
    if Head == "d":  # d_센서_통계: 6시간 창과 12시간 창의 차이 (최근 변화량)
        Sensor, _, Stat = Rest.rpartition("_")
        return 12, Sensor, f"{Stat}_change"
    if Head.startswith("w") and Head[1:].isdigit():
        Sensor, _, Stat = Rest.rpartition("_")
        if Stat in features.StatNames:
            return int(Head[1:]), Sensor, Stat
        return int(Head[1:]), "", Rest
    return None, "", Name


# 시나리오의 정기 정비 일정 (history/calendar) 중 이 설비 것
def PlannedMaintenance(ScenarioId, Tag):
    Calendar = pd.read_csv(config.ScenarioDir / ScenarioId / "history" / "calendar.csv")
    Calendar = Calendar[Calendar["EQP_TAG"] == Tag]
    return [{"StartTSec": float(Row.T_START_SEC), "EndTSec": float(Row.T_END_SEC), "Scope": Row.SCOPE} for Row in Calendar.itertuples()]


# 6시간·24시간 앙상블(ExtraTrees + 그래디언트 부스팅)을 train으로 학습해 validation의 모든 시점(정답이 비어 있는 시점 포함)을 예측
def BuildRisk(Ontology, Equip):
    Table = models.AddDeltaFeatures(pd.read_csv(features.TablePath(Equip)))
    FeatureCols = [Col for Col in Table.columns if Col not in config.MetaCols]
    Train, Valid = Table[Table["split"] == "train"], Table[Table["split"] == "validation"].reset_index(drop=True)
    Tag = next(Item["EqpTag"] for Item in Ontology["Equipments"] if Item["Id"] == Equip)
    GeneratedAt = Now()

    Probs, Thresholds, Metrics = {}, {}, {}
    for Horizon in config.Horizons:
        Target = f"RISK_{Horizon}"
        Labeled = Train[Train[Target].notna()]
        Model = models.MakeModels()["ensemble"]
        Thresholds[Horizon] = models.PickThreshold(Model, Labeled, FeatureCols, Target)
        Fitted = Model.fit(Labeled[FeatureCols], Labeled[Target].astype(int))
        Probs[Horizon] = Fitted.predict_proba(Valid[FeatureCols])[:, 1]

        Known = Valid[Target].notna().to_numpy()
        Y = Valid.loc[Known, Target].astype(int)
        At = models.AtThreshold(Y, Probs[Horizon][Known], Thresholds[Horizon])
        Metrics[Horizon] = {"PR_AUC": round(float(average_precision_score(Y, Probs[Horizon][Known])), 4),
                            "Baseline_PR_AUC": round(float(Y.mean()), 4), "Precision": round(float(At["precision"]), 4),
                            "Recall": round(float(At["recall"]), 4), "Alarms": At["alarms"], "Points": int(Known.sum())}
        if Horizon == "24H":
            Top = pd.Series(Fitted.named_estimators_["extra"][-1].feature_importances_, index=FeatureCols).nlargest(ReasonFromTop).index.tolist()
        print(f"    {Equip} {Horizon} 학습·예측 완료 (임계값 {Thresholds[Horizon]:.3f}, PR-AUC {Metrics[Horizon]['PR_AUC']:.3f})", flush=True)

    # 이유 점수: 정상 범위 밖으로 벗어난 거리 ÷ 정상 범위 폭 (범위 안이면 0이라 이유에서 빠짐)
    Low, High = NormalRange(Train, Top)
    Width = (High - Low).replace(0, np.nan)
    Z = ((Low - Valid[Top]).clip(lower=0) + (Valid[Top] - High).clip(lower=0)) / Width
    Z = Z.where(Z > 0)

    Scenarios = []
    for Id, Rows in Valid.groupby("scenario", sort=True):
        Points = []
        for Index in Rows.index:
            P6, P24 = float(Probs["6H"][Index]), float(Probs["24H"][Index])
            Alarm6, Alarm24 = P6 >= Thresholds["6H"], P24 >= Thresholds["24H"]
            Level = "alarm" if Alarm6 or Alarm24 else "watch" if P24 >= Thresholds["24H"] * WatchRatio else "normal"

            Reasons = []
            for Feature in Z.loc[Index].dropna().nlargest(ReasonCount).index:
                Hours, Sensor, Stat = SplitFeature(Feature)
                Reasons.append({"Feature": Feature, "Sensor": Sensor, "Stat": Stat, "WindowHours": Hours,
                                "Value": float(Valid.at[Index, Feature]), "NormalLow": float(Low[Feature]),
                                "NormalHigh": float(High[Feature]), "Score": round(float(Z.at[Index, Feature]), 2)})

            Points.append({"Time": Valid.at[Index, "time"], "TSec": float(Valid.at[Index, "T_SEC"]),
                           "P6H": round(P6, 4), "P24H": round(P24, 4), "Alarm6H": bool(Alarm6), "Alarm24H": bool(Alarm24),
                           "Level": Level, "Reasons": Reasons})
        Scenarios.append({"Scenario": Id, "StartAt": Points[0]["Time"], "PlannedMaintenance": PlannedMaintenance(Id, Tag),
                          "Points": Points})

    return {"Version": 1, "Equip": Equip, "EqpTag": Tag, "Split": "validation", "IntervalHours": 6,
            "Model": {"ModelVersion": f"{Equip}-ENS-{GeneratedAt[:16].replace('-', '').replace(':', '').replace('T', '-')}",
                      "Algorithm": "ExtraTrees + HistGradientBoosting (soft voting)", "GeneratedAt": GeneratedAt,
                      "TrainedOn": f"train {Train['scenario'].nunique()}개 시나리오", "Features": len(FeatureCols),
                      "FeatureWindowsHours": config.WindowHours, "ThresholdRule": "train 시나리오 단위 교차검증에서 F1 최대",
                      "Validation": Metrics,
                      "Lineage": {"Dataset": Ontology["Lineage"]["Dataset"], "Preprocessed": Ontology["Lineage"]["Preprocessed"],
                                  "FeatureTable": f"jungmin/outputs/features_{Equip}.csv", "Code": "jungmin/unreal.py"}},
            "Thresholds": {"P6H": round(Thresholds["6H"], 4), "P24H": round(Thresholds["24H"], 4)},
            "WatchRatio": WatchRatio, "Scenarios": Scenarios}


# 정비 선택지별 기대 비용 (언리얼에서도 같은 식으로 계산)
#   고장 손실 F = 복구 비용 + 복구 시간 × 정지 비용 + 고장 손실
#   즉시 교체 R = 부품 가격 + 교체 시간 × 정지 비용
#   계획 교체 Rp = 부품 가격 + 교체 시간 × 정지 비용 × 계획 정지 비율
#   대기 = P24 × F,  즉시 교체 = R,  PlanHours시간 후 계획 교체 = P6 × F + (1 − P6) × Rp
#   부품 제약: 재고가 없으면 즉시 교체 불가, 재고가 없고 납기가 PlanHours시간보다 길면 계획 교체도 불가
#   가능한 선택지 중 가장 싼 것을 추천, 단 6시간 경보면 안전 우선으로 가능한 가장 빠른 교체
#   6시간 경보인데 교체할 수 있는 부품이 없으면 긴급 발주 표시 (UrgentOrder)
def Decide(Point, Equipment, Part, PlanHours_=PlanHours):
    Down = Equipment["DowntimeCostPerHour"]
    Failure = Equipment["RepairCost"] + Equipment["RepairHours"] * Down + Equipment["FailureLoss"]
    Replace = Part["Price"] + Equipment["ReplaceHours"] * Down
    Planned = Part["Price"] + Equipment["ReplaceHours"] * Down * Equipment["PlannedDowntimeRatio"]

    Options = {"wait": Point["P24H"] * Failure, "replace_now": Replace,
               "replace_6h": Point["P6H"] * Failure + (1 - Point["P6H"]) * Planned}
    Available = {"wait": True, "replace_now": Part["Stock"] >= 1,
                 "replace_6h": Part["Stock"] >= 1 or Part["LeadTimeHours"] <= PlanHours_}

    Best = min((Name for Name in Options if Available[Name]), key=Options.get)
    Fastest = next((Name for Name in ("replace_now", "replace_6h") if Available[Name]), "wait")
    Action = Fastest if Point["Alarm6H"] else Best

    return {**{f"Cost_{Name}": round(Value, 2) for Name, Value in Options.items()},
            **{f"Available_{Name}": Value for Name, Value in Available.items() if Name != "wait"},
            "Best": Best, "Recommended": Action, "SafetyOverride": bool(Action != Best),
            "PartShortage": not Available["replace_now"],
            "UrgentOrder": bool(Point["Alarm6H"] and Fastest == "wait"),
            "Saving_vs_wait": round(Options["wait"] - Options[Action], 2)}


def BuildCheck(Risk, Costs):
    Equipment = next(Item for Item in Costs["Equipments"] if Item["Equip"] == Risk["Equip"])
    Part = next(Item for Item in Equipment["Parts"] if Item["Part"] == Equipment["DefaultPart"])

    return pd.DataFrame([{"Scenario": Scenario["Scenario"], "Time": Point["Time"], "Level": Point["Level"],
                          "P6H": Point["P6H"], "P24H": Point["P24H"], "Part": Part["Part"],
                          **Decide(Point, Equipment, Part, Costs.get("PlanHours", PlanHours))}
                         for Scenario in Risk["Scenarios"] for Point in Scenario["Points"]])


# 피드백: 언리얼 결정 기록을 실제 결과(validation 정답)와 맞춰 봄
#   교체 + 24시간 안 실제 고장 → 예방 / 교체 + 고장 없음 → 불필요 정비 가능성
#   대기 + 고장 → 놓침 / 대기 + 고장 없음 → 정상 대기 / 정답이 비어 있으면 판정 불가
def Feedback(LogPath):
    Log = json.loads(Path(LogPath).read_text(encoding="utf-8"))
    Decisions = pd.DataFrame(Log["Decisions"])
    if Decisions.empty:
        print("결정 기록이 없습니다.")
        return

    Truth = []
    for Id in Decisions["Scenario"].unique():
        Labels = pd.read_csv(config.ScenarioDir / Id / "labels.csv")
        Truth.append(Labels.rename(columns={"SCENARIO_ID": "Scenario", "EQP_CD": "EqpTag", "OBSERVED_AT": "Time"})
                     [["Scenario", "EqpTag", "Time", "GT_RISK_6H", "GT_RISK_24H"]])
    Tags = {Item["Equip"]: Item["EqpTag"] for Item in json.loads((UnrealDir / "costs.json").read_text(encoding="utf-8"))["Equipments"]}
    Decisions["EqpTag"] = Decisions["Equip"].map(Tags)
    Result = Decisions.merge(pd.concat(Truth), on=["Scenario", "EqpTag", "Time"], how="left")

    Replaced = Result["Chosen"].str.startswith("replace")
    Failed = Result["GT_RISK_24H"]
    Result["Outcome"] = np.select(
        [Failed.isna(), Replaced & (Failed == 1), Replaced & (Failed == 0), ~Replaced & (Failed == 1)],
        ["판정 불가", "예방", "불필요 정비 가능성", "놓침"], "정상 대기")
    Result["Followed"] = Result["Chosen"] == Result["Recommended"]

    Path_ = UnrealDir / "feedback.csv"
    Result.drop(columns=[Col for Col in ("Costs", "Assumptions") if Col in Result]).to_csv(Path_, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print(f"\n결정 {len(Result)}건 (추천을 따른 결정 {int(Result['Followed'].sum())}건)")
    print(pd.crosstab(Result["Followed"].map({True: "추천 따름", False: "추천과 다름"}), Result["Outcome"]).to_string())
    print(f"저장: {Path_}")


def Save(Data, Name):
    with open(UnrealDir / Name, "w", encoding="utf-8") as File:
        json.dump(Data, File, ensure_ascii=False, indent=2)


def Main(Args):
    UnrealDir.mkdir(parents=True, exist_ok=True)
    Options = dict(Arg.split("=", 1) for Arg in Args if "=" in Arg)
    Names = [Arg for Arg in Args if Arg in config.Equipments] or RiskEquips

    if "feedback" in Options:
        Feedback(Options["feedback"])
        return

    if "costs" in Options:
        Costs = json.loads(Path(Options["costs"]).read_text(encoding="utf-8"))
        Risks = [json.loads(Path_.read_text(encoding="utf-8")) for Path_ in sorted(UnrealDir.glob("risk_*.json"))]
    else:
        Ontology = BuildOntology()
        Costs = BuildCosts(Ontology)
        Risks = []
        for Equip in Names:
            Risks.append(BuildRisk(Ontology, Equip))
            Save(Risks[-1], f"risk_{Equip}.json")
        Save(Ontology, "ontology.json")
        Save(Costs, "costs.json")
        print(f"    공정 순서 {' → '.join(Ontology['ProcessOrder'])}")

    for Risk in Risks:
        Equip = Risk["Equip"]
        Check = BuildCheck(Risk, Costs)
        Check.to_csv(UnrealDir / f"decisions_{Equip}_check.csv", index=False, encoding="utf-8-sig")

        Points = sum(len(Scenario["Points"]) for Scenario in Risk["Scenarios"])
        print(f"\n[{Equip}] 모델 {Risk['Model']['ModelVersion']} | 시나리오 {len(Risk['Scenarios'])}개, 시점 {Points}개 | 경보 단계 {Check['Level'].value_counts().to_dict()}")
        print(f"    추천 {Check['Recommended'].value_counts().to_dict()} | 안전 우선 적용 {int(Check['SafetyOverride'].sum())}회 | "
              f"재고 부족 {int(Check['PartShortage'].sum())}회 | 긴급 발주 {int(Check['UrgentOrder'].sum())}회")

    print(f"\n저장 위치: {UnrealDir}")


if __name__ == "__main__":
    Main(sys.argv[1:])
